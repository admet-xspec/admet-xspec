from pathlib import Path

import pandas as pd


class Dataset:
    # class invariant (under consideration)
    # there are never any NaNs in any cell

    def __init__(self, raw_dataframe: pd.DataFrame):
        self.raw_dataframe = raw_dataframe
        self.cur_dataframe = raw_dataframe

    def get_raw_dataframe(self) -> pd.DataFrame:
        """
        Gets the original, 'raw' dataframe, with the following properties/sequence of applied transformations:
        1. SMILES are canonicalized
        2. Dropna is applied
        So, all columns are retained, "dropping only invalid values".
        """
        ...

    def get_current_dataframe(self) -> pd.DataFrame:
        """
        Gets the dataframe as it has been transformed from its raw form thus far.
        So, it is the code handling the Dataset object that is responsible for this state.
        """
        ...

    def set_current_dataframe(self, df: pd.DataFrame):
        """
        Replaces the dataframe available through 'get_current_dataframe' with df.
        """
        ...

    def is_ready_for_training(self) -> bool:
        """
        Asserts whether the dataframe available through 'get_current_dataframe'
        """

        # check "x", "y" cols exist
        ...

    def save_to_disk(self, target_dir: Path): ...

    def load_from_disk(self, source_dir: Path): ...

    def get_task_dataset_is_prepared_for(self):
        # one of regression, classification
        ...

    def get_dataframe_in_predictor_form(self):
        # only features and labels
        # use is_ready_for_training
        ...

    # properties
    def get_datasource_name(self): ...

    def get_endpoint_type(self): ...

    def get_species_name(self): ...
