import pandas as pd
from pathlib import Path


def load_multiple_datasets(
    dataset_paths: list[Path],
) -> list[pd.DataFrame]:
    return [pd.read_csv(ds_path) for ds_path in dataset_paths]
