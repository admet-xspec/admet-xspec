"""Resolve the SMILES cleaner shared with the training and inference pipelines.

Contract: `src.utils.get_clean_smiles` returns `None` for any input that does not
yield a usable molecule; the whole package relies on that.
"""

from __future__ import annotations

from typing import Callable, Optional

CleanFn = Callable[..., Optional[str]]

try:  # pragma: no cover - depends on the host codebase
    from src.utils import get_clean_smiles as _project_cleaner
except Exception:  # anything raised at import time, not only ImportError
    _project_cleaner = None

_IMPORT_ERROR = (
    "Could not import `get_clean_smiles` from `src.utils`. Either run this "
    "package from the root of the project so that `src` is importable, or pass "
    "an explicit function via `BrenkChecker(clean_fn=...)`."
)


def resolve_cleaner(clean_fn: CleanFn | None = None) -> CleanFn:
    """Return the cleaning function to use, preferring an explicit override."""
    if clean_fn is not None:
        if not callable(clean_fn):
            raise TypeError("clean_fn must be callable")
        return clean_fn
    if _project_cleaner is None:
        raise ImportError(_IMPORT_ERROR)
    return _project_cleaner
