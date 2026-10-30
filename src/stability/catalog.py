"""Enumerate RDKit's BRENK catalog and derive the alert column schema from it.

The catalog is fixed, so every run emits every column regardless of hits. This
keeps the schema independent of the input, so batches concatenate cleanly.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple

from rdkit.Chem import FilterCatalog

# Spelled out, not stripped: they carry meaning (">_2_ester_groups").
_SYMBOL_WORDS = {">": "gt", "<": "lt"}

_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def slugify(description: str) -> str:
    """Turn a BRENK description into a stable column suffix.

    >>> slugify(">_2_ester_groups")
    'gt_2_ester_groups'
    >>> slugify("cyanate_/aminonitrile_/thiocyanate")
    'cyanate_aminonitrile_thiocyanate'
    """
    text = description
    for symbol, word in _SYMBOL_WORDS.items():
        text = text.replace(symbol, f"_{word}_")
    text = _NON_ALNUM.sub("_", text.lower())
    return text.strip("_")


@dataclass(frozen=True)
class CatalogSchema:
    """Immutable column contract derived from the BRENK catalog."""

    descriptions: Tuple[str, ...]
    columns: Tuple[str, ...]
    prefix: str

    @property
    def column_to_description(self) -> Dict[str, str]:
        return dict(zip(self.columns, self.descriptions))

    @property
    def description_to_column(self) -> Dict[str, str]:
        return dict(zip(self.descriptions, self.columns))

    @property
    def fingerprint(self) -> str:
        """Return a hash of the catalog contents.

        Stored in the metadata sidecar to detect catalog changes across RDKit versions.
        """
        payload = "\n".join(sorted(self.descriptions)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def __len__(self) -> int:
        return len(self.columns)


def build_catalog() -> FilterCatalog.FilterCatalog:
    """Build a BRENK-only `FilterCatalog`."""
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog.FilterCatalog(params)


@lru_cache(maxsize=8)
def build_schema(prefix: str = "brenk_") -> CatalogSchema:
    """Enumerate the catalog and derive the full, sorted column schema."""
    catalog = build_catalog()
    descriptions = [
        catalog.GetEntryWithIdx(i).GetDescription()
        for i in range(catalog.GetNumEntries())
    ]

    duplicates = _duplicates(descriptions)
    if duplicates:
        raise RuntimeError(
            f"BRENK catalog contains duplicate descriptions: {sorted(duplicates)}. "
            "The column schema would be ambiguous."
        )

    columns = [f"{prefix}{slugify(d)}" for d in descriptions]
    collisions = _duplicates(columns)
    if collisions:
        offenders = {
            c: [d for d, col in zip(descriptions, columns) if col == c]
            for c in collisions
        }
        raise RuntimeError(
            "Slugified BRENK descriptions collide, which would silently merge "
            f"distinct alerts into one column: {offenders}"
        )

    order = sorted(range(len(columns)), key=lambda i: columns[i])
    return CatalogSchema(
        descriptions=tuple(descriptions[i] for i in order),
        columns=tuple(columns[i] for i in order),
        prefix=prefix,
    )


def _duplicates(values: List[str]) -> set:
    """Return values occurring more than once."""
    seen, dupes = set(), set()
    for value in values:
        if value in seen:
            dupes.add(value)
        seen.add(value)
    return dupes
