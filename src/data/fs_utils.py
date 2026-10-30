from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import pickle
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import IO, Any, Callable, ContextManager, Dict, Iterator, Optional

import filelock
import pandas as pd
import yaml

TMP_SUFFIX = ".tmp"


def _tmp_path_for(path: Path) -> Path:
    return path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}{TMP_SUFFIX}"


def _replace(src: Path, dst: Path, retries: int = 8, base_delay: float = 0.05) -> None:
    """Call `os.replace`, retrying on Windows while another process holds `dst` open."""
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
    """Write to a temporary file and atomically move it to `path` on success.

    On error the temporary file is removed and `path` is left untouched.
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


def atomic_copy_file(src: Path, dst: Path) -> None:
    """Copy `src` to `dst` so readers never see a partial copy."""
    with open(src, "rb") as fin, atomic_write(dst, "wb") as fout:
        shutil.copyfileobj(fin, fout)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_yaml(path: Path, missing_ok: bool = False) -> Dict:
    """Load a YAML file the caller depends on; invalid YAML raises `yaml.YAMLError`.

    With `missing_ok`, a missing file yields `{}`. For scans, use `safe_load_yaml`.
    """
    try:
        with open(path, "r") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        if missing_ok:
            return {}
        raise


def read_first_line(path: Path, default: str = "") -> str:
    """Return the stripped first line of `path`, or `default` if the file is missing."""
    try:
        with open(path, "r") as fh:
            return fh.readline().strip()
    except FileNotFoundError:
        return default


def safe_load_yaml(path: Path) -> Dict:
    """Load a YAML mapping during a scan, returning `{}` for vanished or invalid files.

    Vanished files are expected under concurrency (debug log); invalid YAML is logged as
    a warning but does not abort the scan.
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
    """Yield files under `root` whose names end in `suffix`.

    Uses `os.walk`, which skips directories removed mid-scan; `Path.rglob` can raise.
    """
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=True):
        for name in filenames:
            if name.endswith(suffix):
                yield Path(dirpath) / name


def make_unique_dir(parent: Path, prefix: str) -> Path:
    """Create and return a uniquely named directory under `parent`."""
    path = Path(parent) / f"{prefix}.{os.getpid()}.{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def publish_dir(
    staging: Path,
    final: Path,
    trash_parent: Path,
    is_complete: Optional[Callable[[Path], bool]] = None,
    overwrite: bool = False,
    lock: Optional[ContextManager] = None,
) -> bool:
    """Atomically move a fully written `staging` directory to `final`.

    Return False, leaving `staging` to the caller, if a complete `final` exists
    and `overwrite` is off. A `final` rejected by `is_complete` is always replaced.
    `staging` and `trash_parent` must be on the filesystem of `final`, so renames
    are atomic. `lock`, if given, is held across check and swap; the old tree is
    deleted after its release.
    """
    staging, final = Path(staging), Path(final)
    trash: Optional[Path] = None
    with lock if lock is not None else contextlib.nullcontext():
        if final.exists():
            complete = is_complete(final) if is_complete is not None else True
            if complete and not overwrite:
                logging.info(
                    f"`{final}` was already published by another process; "
                    f"keeping the existing copy."
                )
                return False
            if not complete:
                logging.warning(f"Replacing incomplete directory `{final}`.")
            trash = Path(trash_parent) / f"trash.{uuid.uuid4().hex}"
            os.rename(final, trash)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.rename(staging, final)
    if trash is not None:
        shutil.rmtree(trash, ignore_errors=True)
    return True


def make_file_lock(
    locks_dir: Path, name: str, timeout: float, soft: bool = False
) -> filelock.BaseFileLock:
    """Create an inter-process lock in `locks_dir` keyed by an arbitrary string `name`.

    `soft` selects `SoftFileLock`, for network filesystems with unreliable OS locks.
    Reuse the returned object: it is reentrant, but two objects on one file deadlock.
    """
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:16]
    lock_cls = filelock.SoftFileLock if soft else filelock.FileLock
    return lock_cls(str(Path(locks_dir) / f"{digest}.lock"), timeout=timeout)


@contextlib.contextmanager
def hold_lock(lock: filelock.BaseFileLock, name: str = "") -> Iterator[None]:
    """Hold `lock` for the body, raising a descriptive `TimeoutError` on timeout."""
    try:
        lock.acquire()
    except filelock.Timeout as exc:
        raise TimeoutError(
            f"Timed out after {lock.timeout}s waiting for lock `{name or lock.lock_file}` "
            f"({lock.lock_file}). Another process may be stuck, or the timeout may be "
            f"too short for this operation."
        ) from exc
    try:
        yield
    finally:
        lock.release()


def _older_than(path: Path, max_age_seconds: float, now: float) -> bool:
    try:
        return now - path.lstat().st_mtime > max_age_seconds
    except FileNotFoundError:
        return False


def remove_stale_tmp_files(root: Path, max_age_seconds: float) -> int:
    """Delete temp files of crashed writers older than `max_age_seconds`; return the count.

    The age threshold leaves writes still in progress in other processes alone.
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
    """Delete direct children of `directory` older than `max_age_seconds`; return the count."""
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
