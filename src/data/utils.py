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


def get_label_count(df: pd.DataFrame, column_name="source") -> dict:
    source_count = {}
    for name in df[column_name].unique():
        source_count[name] = len(df[df[column_name] == name])
    return source_count
