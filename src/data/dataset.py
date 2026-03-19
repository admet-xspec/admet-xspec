from pathlib import Path

import pandas as pd


class Dataset:
    # class invariant (under consideration)
    # there are never any NaNs in any cell

    def __init__(self):
        raw_dataframe: pd.DataFrame
        cur_dataframe: pd.DataFrame

    def get_raw_dataframe(self) -> pd.DataFrame: ...

    def get_current_dataframe(self) -> pd.DataFrame: ...

    def set_current_dataframe(self, df: pd.DataFrame): ...

    def is_ready_for_training(self) -> bool: ...

    def save_to_disk(self, target_dir: Path): ...

    def load_from_disk(self, source_dir: Path): ...

    def get_task_dataset_is_prepared_for(self):
        # one of regression, classification
        ...

    def get_dataframe_in_predictor_form(self):
        # only features and labels
        ...

    # properties
    def get_datasource_name(self): ...

    def get_endpoint_type(self): ...

    def get_species_name(self): ...
