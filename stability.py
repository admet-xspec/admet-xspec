"""Annotate a CSV of SMILES with BRENK structural alerts.

Every one of the 105 BRENK criteria becomes a binary column, alongside per-axis
stability scores and a tier. See `src/stability/ALERTS.md` for what the alerts
mean and `src/stability/README.md` for the output schema.

Example:
    python stability.py -i compounds.csv -o alerts.csv --smiles-col smiles
"""

import argparse
import logging
import pathlib
import time

import pandas as pd

from src.data.data_interface import DataInterface
from src.stability.checker import BrenkChecker

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Add BRENK structural alerts as separate columns to a CSV of SMILES. "
            "Triage only: a clean result is not a stability claim."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        type=pathlib.Path,
        required=True,
        help="Path to the input CSV containing a column of SMILES",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=pathlib.Path,
        default=None,
        help=(
            "Path of the output CSV. Default: `<input>_stability.csv`, "
            "or nothing further when `--in-place` is given"
        ),
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file with the results. Combine with -o to write both",
    )
    parser.add_argument(
        "--smiles-col",
        type=str,
        default="smiles",
        help="SMILES column name. If absent, it is detected as in the data pipeline",
    )
    parser.add_argument(
        "--alert-config",
        type=pathlib.Path,
        default=None,
        help="Optional YAML/JSON file overriding alert severities and tiers",
    )
    parser.add_argument(
        "--alert-prefix",
        type=str,
        default="brenk_",
        help="Prefix of the generated binary alert columns",
    )
    parser.add_argument(
        "--no-salt-removal",
        action="store_true",
        help="Do not strip salts and counterions before matching",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Worker processes; used only for large inputs",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Do not write the `.meta.json` sidecar next to the output",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Can be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    return parser.parse_args()


def load_dataset(dataset_path: pathlib.Path, smiles_col: str) -> pd.DataFrame:
    """Read the input CSV, renaming the SMILES column if it goes by another name."""
    dataset = pd.read_csv(dataset_path)
    if smiles_col not in dataset.columns:
        raw_smiles_col = DataInterface.get_smiles_col_in_raw(dataset)
        logging.info(f"* Using column `{raw_smiles_col}` as `{smiles_col}`")
        dataset = dataset.rename(columns={raw_smiles_col: smiles_col})
    return dataset


def drop_previous_results(df: pd.DataFrame, checker: BrenkChecker) -> pd.DataFrame:
    """Drop alert columns left by an earlier run, so re-running a file is safe."""
    previous = [c for c in checker.output_columns if c in df.columns]
    if previous:
        logging.warning(
            f"Input already contains {len(previous)} stability columns; replacing them."
        )
        df = df.drop(columns=previous)
    return df


def log_summary(alerts_df: pd.DataFrame, checker: BrenkChecker) -> None:
    n_invalid = int((~alerts_df["valid"].fillna(False)).sum())
    if n_invalid:
        logging.warning(
            f"{n_invalid} input SMILES could not be cleaned; their alert columns are left empty."
        )

    tier_counts = alerts_df["tier"].value_counts()
    logging.info("Stability tiers:")
    for tier in ("clean", "watch", "labile", "avoid"):
        logging.info(f"* {tier}: {int(tier_counts.get(tier, 0))}")

    alert_counts = alerts_df[checker.alert_columns].sum()
    alert_counts = alert_counts[alert_counts > 0].sort_values(ascending=False)
    if not alert_counts.empty:
        logging.info("Most frequent alerts:")
        for column, count in alert_counts.head(10).items():
            logging.info(f"* {column}: {int(count)}")


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(level=args.log_level, format="%(message)s")

    if not args.input.is_file():
        raise FileNotFoundError(f"Input file {args.input} not found.")
    if args.alert_config is not None and not args.alert_config.is_file():
        raise FileNotFoundError(f"Alert config file {args.alert_config} not found.")

    output_path = args.output
    if output_path is None and not args.in_place:
        output_path = args.input.with_name(f"{args.input.stem}_stability.csv")

    # `--in-place` is the only way the input file is ever written to.
    output_paths = [args.input] if args.in_place else []
    if output_path is not None:
        output_paths.append(output_path)

    time_start = time.time()

    logging.info("Structural alert configuration summary:")
    logging.info(f"* Input path: {args.input}")
    logging.info(f"* SMILES column: {args.smiles_col}")
    logging.info(f"* Remove salt: {not args.no_salt_removal}")
    logging.info(f"* Alert config: {args.alert_config}")
    logging.info(f"* Output path(s): {', '.join(str(p) for p in output_paths)}")
    if args.in_place:
        logging.warning(f"Overwriting the input file `{args.input}` with the results.")

    checker = BrenkChecker(
        smiles_col=args.smiles_col,
        remove_salt=not args.no_salt_removal,
        config_path=args.alert_config,
        prefix=args.alert_prefix,
        n_jobs=args.n_jobs,
        quiet_rdkit=True,
    )

    data = load_dataset(args.input, args.smiles_col)
    data = drop_previous_results(data, checker)

    # `input_smiles` duplicates the SMILES column already present in `data`.
    alerts_df = checker.assess(data)
    results = pd.concat([data, alerts_df.drop(columns=["input_smiles"])], axis=1)

    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(path, index=False)
        if not args.no_metadata:
            # The sidecar maps alert columns to BRENK descriptions and lets
            # `src.stability.io.read_results` restore nullable dtypes.
            from src.stability.io import write_metadata

            write_metadata(checker, alerts_df, path)

    log_summary(alerts_df, checker)

    time_elapsed = time.time() - time_start
    logging.info(
        f"Structural alert check completed in {round(time_elapsed, 2)} seconds."
        if time_elapsed < 60
        else f"Structural alert check completed in {round(time_elapsed / 60, 2)} minutes."
    )
    for path in output_paths:
        logging.info(f"Alerts file: {path}")
