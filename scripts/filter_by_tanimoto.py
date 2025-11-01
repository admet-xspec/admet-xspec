import pandas as pd
import pathlib
import argparse
import logging
from src.data.tanimoto import TanimotoCalculator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-path",
        "-p",
        type=str,
        required=True,
        help="Path to the test set CSV file",
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=str,
        default=None,
        help="Directory containing CSV files to filter "
        "(default: same directory as test set)",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.1,
        help="Tanimoto distance threshold for filtering (default: 0.1)",
    )
    parser.add_argument(
        "--discard-closer",
        "-c",
        action="store_true",
        help="If set, discard molecules with Tanimoto distance less than the threshold",
    )
    parser.add_argument(
        "--discard-further",
        "-f",
        action="store_true",
        help="If set, discard molecules with Tanimoto distance greater than the threshold",
    )
    parser.add_argument(
        "--inclusive",
        "-i",
        action="store_true",
        help="If set, use inclusive comparison (<= or >=) when filtering",
    )

    args = parser.parse_args()
    if args.discard_closer and args.discard_further:
        raise ValueError(
            "Cannot set both --discard-closer and --discard-further flags at the same time."
        )
    if not args.discard_closer and not args.discard_further:
        raise ValueError("Must set either --discard-closer or --discard-further flag.")
    test_set_path = pathlib.Path(args.test_set)
    dir = test_set_path.parent if args.dir is None else pathlib.Path(args.dir)

    # load test set smiles
    test_df = pd.read_csv(test_set_path)
    test_smiles = set(test_df["smiles"].tolist())

    # configure logging
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(dir / "drop_test_smiles.log"),
            logging.StreamHandler(),
        ],
    )

    # initialize tanimoto calculator
    tanimoto_calculator = TanimotoCalculator(smiles_list=list(test_smiles))

    # collect all .csv files in the directory where the test set is located, except the test set itself
    all_files = list(dir.glob("*.csv"))
    all_files = [f for f in all_files if f.name != test_set_path.name]

    # read each of the files file, delete smiles which are in the test set
    for f in all_files:
        logging.info(f"Processing {f.name}")
        df = pd.read_csv(f)
        original_count = len(df)
        df["tanimoto_to_test_set"] = tanimoto_calculator.run_batch(
            df["smiles"].tolist()
        )["min_distance"]

        if args.discard_closer:
            if args.inclusive:
                df = df[df["tanimoto_to_test_set"] >= args.threshold]
                save_path = (
                    dir / f"{f.stem}_further_than_{args.threshold}_inclusive{f.suffix}"
                )
            else:
                df = df[df["tanimoto_to_test_set"] >= args.threshold]
                save_path = dir / f"{f.stem}_further_than_{args.threshold}{f.suffix}"
        else:
            if args.inclusive:
                df = df[df["tanimoto_to_test_set"] < args.threshold]
                save_path = (
                    dir / f"{f.stem}_closer_than_{args.threshold}_inclusive{f.suffix}"
                )
            else:
                df = df[df["tanimoto_to_test_set"] <= args.threshold]
                save_path = dir / f"{f.stem}_closer_than_{args.threshold}{f.suffix}"
        filtered_count = len(df)
        logging.info(
            f"Processed {f.name}: {original_count} -> {filtered_count} molecules after filtering."
        )
        logging.info(f"Saving filtered dataset to {save_path}")
        df.to_csv(save_path, index=False)
    logging.info(f"Logs saved to {dir / 'drop_test_smiles.log'}")
