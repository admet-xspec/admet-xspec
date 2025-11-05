import gin
import logging
import pandas as pd
import yaml
from PIL.Image import Image
from pathlib import Path

from src.utils import detect_csv_delimiter, get_clean_smiles
from src.data.utils import load_multiple_datasets


@gin.configurable
class DataInterface:

    possible_smiles_cols = ["SMILES", "Smiles", "smiles", "molecule"]

    def __init__(
        self,
        dataset_dir: str,
        metrics_dir: str,
        visualizations_dir: str,
        data_config_filename: str,
        normalized_filename: str,
        handle_multiple_datasets_method: str = None,
    ):
        self.dataset_dir: Path = Path(dataset_dir)
        self.metrics_dir: Path = Path(metrics_dir)
        self.visualizations_dir: Path = Path(visualizations_dir)
        self.data_config_filename: str = data_config_filename
        self.normalized_filename: str = normalized_filename
        self.handle_multiple_datasets_method: str = handle_multiple_datasets_method

        self._init_create_dirs()

    def _init_create_dirs(self):
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)

    def _find_dataset_dir(self, friendly_name: str) -> Path:
        dataset_dir = None
        for yaml_path in Path(self.dataset_dir).rglob("*.yaml"):
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
                if data and data.get("friendly_name") == friendly_name:
                    dataset_dir = yaml_path.parent
                    break

        if dataset_dir:
            return dataset_dir
        raise FileNotFoundError(
            f"No dataset directory with yaml containing friendly_name: '{friendly_name}' found"
        )

    def _check_normalized_dataset_exists(self, dataset_dir_path: Path) -> bool:
        normalized_dataset_path = dataset_dir_path / self.normalized_filename
        return normalized_dataset_path.exists()

    def _load_df(self, dataset_path: Path, permit_chembl: bool = True) -> pd.DataFrame:
        """permit_chembl: whether to handle semicolon-delimited csvs with quotation marks around data"""
        raw_chembl_detected = (
            True
            if permit_chembl and detect_csv_delimiter(dataset_path) == ";"
            else False
        )

        if raw_chembl_detected:
            loaded_df = pd.read_csv(dataset_path, delimiter=";")
        else:
            loaded_df = pd.read_csv(dataset_path, delimiter=",")

        if loaded_df is not None and loaded_df.count() > 2:
            pre_dropna_len = len(loaded_df)
            smiles_col = self.get_smiles_col_in_raw(loaded_df)
            df = loaded_df.dropna(subset=[smiles_col]).reset_index(drop=True)

            if pre_dropna_len != len(df):
                logging.info(
                    f"Loading dataset {dataset_path} resulted in {pre_dropna_len - len(df)} "
                    f"'nan' SMILES, all were dropped."
                )
            return loaded_df
        elif loaded_df is not None:
            raise RuntimeError(
                f"Loading dataset from {dataset_path} resulted in df with {loaded_df.count()} rows."
            )
        raise RuntimeError(
            f"Failed to load dataset from {dataset_path} into dataframe."
        )

    def _save_df(self, df: pd.DataFrame, dir: Path) -> None:
        df.to_csv(dir / self.normalized_filename, index=False)

    def _generate_normalized_dataset(self, dataset_dir_path: Path) -> None:
        """
        Take whatever raw dataset in 'dataset_dir_path' and output
        normalized dataset under 'self.normalized_filename' in that dir
        """
        multiple_raw_datasets = False

        # note that check against existence of normalized dataset should have occured before this
        datasets_in_dir: list[Path] = [
            Path(globbed_ds) for globbed_ds in dataset_dir_path.rglob("*.csv")
        ]
        if len(datasets_in_dir) > 1:
            multiple_raw_datasets = True

        raw_dfs = [self._load_df(ds) for ds in datasets_in_dir]
        if multiple_raw_datasets:
            match self.handle_multiple_datasets_method:
                case "naive_aggregate":
                    aggregate_df = self._naive_aggregate_multiple_datasets(raw_dfs)
                    normalized_df = self.get_normalized_df(aggregate_df)
                case None:
                    raise ValueError(
                        f"Found multiple raw datasets to be processed in {dataset_dir_path}, "
                        "however, parameter 'self.handle_multiple_datasets_method' is not specified"
                    )
                case _:
                    raise NotImplementedError(
                        "Aggregation method 'self.handle_multiple_datasets_method' is not implemented"
                    )
        else:
            normalized_df = self.get_normalized_df(raw_dfs[0])

        if normalized_df is not None:
            self._save_df(normalized_df, dataset_dir_path)
        else:
            raise RuntimeError(
                f"Failed to generate a normalized dataset within dataset directory '{dataset_dir_path}'"
            )

    def _load_normalized_dataset(self, dataset_dir_path: Path) -> pd.DataFrame:
        return pd.read_csv(dataset_dir_path)

    @staticmethod
    def get_clean_smiles_df(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
        """Consolidate pandas, our-internal NaN-dropping into one function"""
        pre_cleaning_len = len(df)
        df[smiles_col].apply(get_clean_smiles)

        df = df.dropna(subset=[smiles_col]).reset_index(drop=True)

        if pre_cleaning_len != len(df):
            logging.info(
                f"Applying 'get_clean_smiles' resulted in {pre_cleaning_len - len(df)} "
                f"'nan' SMILES, all were dropped."
            )

        return df

    @staticmethod
    def get_canon_smiles_df(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
        pre_canonicalization_len = len(df)
        df[smiles_col].apply(get_clean_smiles)

        df = df.dropna(subset=[smiles_col]).reset_index(drop=True)

        if pre_canonicalization_len != len(df):
            logging.info(
                f"Canonicalization resulted in {pre_canonicalization_len - len(df)} "
                "'None' SMILES, all were dropped."
            )

        return df

    @classmethod
    def get_smiles_col_in_raw(cls, raw_df) -> str:
        for smiles_col_variant in cls.possible_smiles_cols:
            if smiles_col_variant in raw_df.columns:
                return smiles_col_variant

        # Failed
        raise ValueError(
            "Failed to find one of SMILES column name variants:",
            str(cls.possible_smiles_cols),
            "in dataframe:",
            str(raw_df.head()),
        )

    def get_normalized_df(self, df_to_normalize: pd.DataFrame) -> pd.DataFrame:
        """Get ready-to-save df without NaNs and with canonical SMILES"""
        logging.debug(f"Raw dataset size: {len(df_to_normalize)}")
        logging.debug(f"Raw dataset columns: {df_to_normalize.columns.tolist()}")

        current_smiles_col = self.get_smiles_col_in_raw(df_to_normalize)
        if current_smiles_col != "smiles":
            df_to_normalize.rename(
                columns={current_smiles_col: "smiles"},
                inplace=True,
            )
        else:
            logging.warning(
                "While running 'get_normalized_df' on raw df, found column 'smiles'. "
                "This is not expected from a raw ChEMBL dataset. Proceeding anyway."
            )

        df_to_normalize = self.get_clean_smiles_df(df_to_normalize, smiles_col="smiles")

        df_to_normalize = self.get_canon_smiles_df(df_to_normalize, smiles_col="smiles")

        return df_to_normalize

    def get_by_friendly_name(self, friendly_name: str) -> pd.DataFrame:
        dataset_dir_path: Path = self._find_dataset_dir(friendly_name)

        if not self._check_normalized_dataset_exists(dataset_dir_path):
            self._generate_normalized_dataset(dataset_dir_path)

        dataset_df = self._load_normalized_dataset(dataset_dir_path)

        return dataset_df

    def save_metrics(
        self, friendly_name: str, metrics: pd.DataFrame | dict
    ) -> None: ...

    def save_visualization(self, friendly_name: str, visualization: Image) -> None: ...
