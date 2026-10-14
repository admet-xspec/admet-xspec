"""Filesystem helpers for writing shared cache files safely from multiple processes.

The core guarantee is that a reader never observes a partially written file: every
write goes to a hidden temporary file in the same directory and is then swapped into
place with ``os.replace``, which is atomic on POSIX filesystems. A reader therefore
sees either the complete old file or the complete new one.

Temporary files are named ``.<name>.<pid>.<random>.tmp``. The leading dot and the
``.tmp`` suffix are deliberate: they keep in-progress files out of every scan that
matches ``*.csv`` or ``*.yaml``, so an unfinished write can never be mistaken for a
raw dataset or a dataset/split config.
"""

from __future__ import annotations

import contextlib
import logging
import os
import pickle
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import IO, Any, Dict, Iterator

import pandas as pd
import yaml

TMP_SUFFIX = ".tmp"


def _tmp_path_for(path: Path) -> Path:
    return path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}{TMP_SUFFIX}"


def _replace(src: Path, dst: Path, retries: int = 8, base_delay: float = 0.05) -> None:
    """``os.replace`` with a retry loop for Windows.

    POSIX never needs the retry. Windows refuses to replace a file that another process
    has open, which is a transient condition when that process is only reading it.
    """
    for attempt in range(retries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if sys.platform != "win32" or attempt == retries - 1:
                raise
            time.sleep(base_delay * (2**attempt))


@contextlib.contextmanager
def atomic_write(path: Path, mode: str = "w", **open_kwargs: Any) -> Iterator[IO]:
    """Open a temporary file for writing and atomically move it to ``path`` on success.

    If the body raises, the temporary file is removed and ``path`` is left untouched.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path_for(path)
    try:
        with open(tmp, mode, **open_kwargs) as fh:
            yield fh
            fh.flush()
            os.fsync(fh.fileno())
        _replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise


def atomic_write_text(text: str, path: Path) -> None:
    with atomic_write(path, "w") as fh:
        fh.write(text)


def atomic_to_csv(df: pd.DataFrame, path: Path, **to_csv_kwargs: Any) -> None:
    to_csv_kwargs.setdefault("index", False)
    with atomic_write(path, "w", newline="") as fh:
        df.to_csv(fh, **to_csv_kwargs)


def atomic_yaml_dump(obj: Any, path: Path) -> None:
    with atomic_write(path, "w") as fh:
        yaml.dump(obj, fh)


def atomic_pickle_dump(obj: Any, path: Path) -> None:
    with atomic_write(path, "wb") as fh:
        pickle.dump(obj, fh)


def safe_load_yaml(path: Path) -> Dict:
    """Load a YAML mapping for directory scans, tolerating files that vanish or are invalid.

    A file can disappear between being listed and being opened (another process may be
    replacing the directory that holds it). That is expected under concurrency and is
    logged at debug level. Invalid YAML is a real problem with the file, so it is logged
    as a warning, but it still should not abort a scan over many unrelated files.
    """
    try:
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError:
        logging.debug(f"`{path}` disappeared during scan; skipping.")
        return {}
    except yaml.YAMLError as exc:
        logging.warning(f"Could not parse `{path}`; skipping. ({exc})")
        return {}
    return data if isinstance(data, dict) else {}


def iter_files(root: Path, suffix: str) -> Iterator[Path]:
    """Yield files under ``root`` whose names end in ``suffix``.

    Uses ``os.walk``, which silently skips directories that disappear mid-scan (for
    example a split being replaced by another process) instead of raising, as
    ``Path.rglob`` can.
    """
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=True):
        for name in filenames:
            if name.endswith(suffix):
                yield Path(dirpath) / name


def _older_than(path: Path, max_age_seconds: float, now: float) -> bool:
    try:
        return now - path.lstat().st_mtime > max_age_seconds
    except FileNotFoundError:
        return False


def remove_stale_tmp_files(root: Path, max_age_seconds: float) -> int:
    """Delete temporary files left behind by crashed writers. Returns the number removed.

    Only files older than ``max_age_seconds`` are touched, so writes that are still in
    progress in other processes are left alone.
    """
    if not Path(root).exists():
        return 0
    now = time.time()
    removed = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if not (name.startswith(".") and name.endswith(TMP_SUFFIX)):
                continue
            path = Path(dirpath) / name
            if _older_than(path, max_age_seconds, now):
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()
                    removed += 1
    return removed


def remove_stale_entries(directory: Path, max_age_seconds: float) -> int:
    """Delete direct children of ``directory`` (files or trees) older than the threshold."""
    directory = Path(directory)
    if not directory.exists():
        return 0
    now = time.time()
    removed = 0
    for entry in directory.iterdir():
        if not _older_than(entry, max_age_seconds, now):
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            with contextlib.suppress(FileNotFoundError):
                entry.unlink()
        removed += 1
    return removed
