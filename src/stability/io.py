"""Write results with a metadata sidecar and read them back with original dtypes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import rdkit


def metadata_path(csv_path: str | Path) -> Path:
    """Return the `.meta.json` sidecar path for `csv_path`."""
    csv_path = Path(csv_path)
    return csv_path.with_suffix(csv_path.suffix + ".meta.json")


def write_metadata(checker, result: pd.DataFrame, csv_path: str | Path) -> Path:
    """Write a sidecar with everything needed to interpret and reproduce the CSV."""
    path = metadata_path(csv_path)
    valid = result["valid"]
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "rdkit_version": rdkit.__version__,
        "pandas_version": pd.__version__,
        "catalog": "BRENK",
        "catalog_n_entries": len(checker.schema),
        "catalog_fingerprint": checker.schema.fingerprint,
        "column_to_brenk_description": checker.schema.column_to_description,
        "dtypes": checker.dtypes,
        "smiles_col": checker.smiles_col,
        "id_col": checker.id_col,
        "remove_salt": checker.remove_salt,
        "config": {
            "severities": {
                k: [list(p) for p in v]
                for k, v in checker.config["severities"].items()
            },
            "aggregation": checker.config["aggregation"],
            "tiers": [list(t) for t in checker.config["tiers"]],
        },
        "rows": {
            "total": int(len(result)),
            "valid": int(valid.sum()),
            "invalid": int((~valid.fillna(False)).sum()),
        },
        "tier_counts": {
            str(k): int(v) for k, v in result["tier"].value_counts().items()
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    return path


def read_results(csv_path: str | Path) -> pd.DataFrame:
    """Read a results CSV, restoring nullable dtypes from its sidecar if present.

    Plain `pd.read_csv` turns `pd.NA` into NaN, so alert columns become floats.
    """
    csv_path = Path(csv_path)
    meta_file = metadata_path(csv_path)
    frame = pd.read_csv(csv_path)
    if not meta_file.exists():
        return frame
    dtypes = json.loads(meta_file.read_text(encoding="utf-8"))["dtypes"]
    for column, dtype in dtypes.items():
        if column in frame.columns:
            frame[column] = frame[column].astype(dtype)
    return frame
