import pandas as pd
from pathlib import Path


def load_multiple_datasets(
    dataset_paths: list[Path],
) -> list[pd.DataFrame]:
    return [pd.read_csv(ds_path) for ds_path in dataset_paths]


def check_dataset_is_raw_chembl(dataset_path: Path) -> bool:
    with open(dataset_path, "r") as f:
        first_two_lines_str = "".join(f.readlines()[:2])
        if ";" in first_two_lines_str:
            return True

    return False
