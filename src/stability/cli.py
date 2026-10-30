"""Command-line entry point.

Example: `python -m src.stability.cli input.csv -o out.csv --id-col id`
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .checker import BrenkChecker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stability",
        description=(
            "Assess chemical stability of small molecules using RDKit's BRENK "
            "alert catalog. Triage only: a clean result is not a stability claim."
        ),
    )
    parser.add_argument("input", type=Path, help="CSV or .smi file of SMILES")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output CSV")
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--config", type=Path, default=None, help="YAML/JSON overrides")
    parser.add_argument("--prefix", default="brenk_")
    parser.add_argument("--no-salt-removal", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--no-metadata", action="store_true")
    parser.add_argument(
        "--quiet-rdkit", action="store_true", help="suppress RDKit parse logging"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def read_input(path: Path, smiles_col: str) -> pd.DataFrame:
    """Read a CSV, or a whitespace-separated `.smi` file with SMILES first."""
    if path.suffix.lower() in {".smi", ".smiles", ".txt"}:
        frame = pd.read_csv(path, sep=r"\s+", header=None, dtype=str)
        frame.columns = [smiles_col] + [f"col_{i}" for i in range(1, frame.shape[1])]
        return frame
    return pd.read_csv(path)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    frame = read_input(args.input, args.smiles_col)
    checker = BrenkChecker(
        smiles_col=args.smiles_col,
        id_col=args.id_col,
        remove_salt=not args.no_salt_removal,
        config_path=args.config,
        prefix=args.prefix,
        n_jobs=args.n_jobs,
        quiet_rdkit=args.quiet_rdkit,
    )
    checker.assess_to_csv(frame, args.output, write_metadata=not args.no_metadata)

    result = pd.read_csv(args.output)
    invalid = int((~result["valid"].astype("boolean").fillna(False)).sum())
    counts = result["tier"].value_counts().to_dict()
    print(f"Wrote {args.output} ({len(result)} rows, {invalid} invalid)")
    for tier in ("clean", "watch", "labile", "avoid"):
        if tier in counts:
            print(f"  {tier:7s} {counts[tier]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
