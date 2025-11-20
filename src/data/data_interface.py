import gin
import logging
import pandas as pd
import yaml
from PIL.Image import Image
from pathlib import Path
import numpy as np
from datetime import datetime
from collections import namedtuple

from src.utils import detect_csv_delimiter, get_clean_smiles
from src.data.utils import load_multiple_datasets


@gin.configurable
class DataInterface:

    possible_smiles_cols = ["SMILES", "Smiles", "smiles", "molecule"]
    possible_label_cols = ["LABEL", "Label", "label", "Y", "y", "Standard Value"]

    def __init__(
        self,
        dataset_dir: str,
        splits_dir: str,
        visualizations_dir: str,
        data_config_filename: str,
        prepared_filename: str,
        metrics_dir: str | None = None,
        handle_multiple_datasets_method: str = None,
        registry_filename: str = "registry.txt",
    ):
        self.dataset_dir: Path = Path(dataset_dir)
        self.splits_dir: Path = Path(splits_dir)
        self.metrics_dir: Path = (
            Path(metrics_dir) if metrics_dir else Path("data/metrics")
        )
        self.visualizations_dir: Path = Path(visualizations_dir)
        self.data_config_filename: str = data_config_filename
        self.prepared_filename: str = prepared_filename
        self.handle_multiple_datasets_method: str = handle_multiple_datasets_method
        self.registry_filename: str = registry_filename
        self.split_datafile_name = "data.csv"

        self._init_create_dirs()
        # Update splits and datasets registries
        self.update_registries()

    def _init_create_dirs(self):
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        (self.splits_dir / "regression").mkdir(parents=True, exist_ok=True)
        (self.splits_dir / "classification").mkdir(parents=True, exist_ok=True)

    def _find_dataset_dir(self, friendly_name: str) -> Path:
        dataset_dir = None
        for yaml_path in Path(self.dataset_dir).rglob("*.yaml"):
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
                if data and (
                    data.get("friendly_name") == friendly_name
                    and data.get("task_setting") == self.task_setting
                ):
                    dataset_dir = yaml_path.parent
                    break

        if dataset_dir:
            return dataset_dir
        raise FileNotFoundError(
            f"No dataset directory with yaml containing friendly_name: '{friendly_name}' found"
        )

    def _find_split_dir(self, friendly_name: str) -> Path:
        split_dir = None
        for yaml_path in Path(self.splits_dir).rglob("*.yaml"):
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
                if data and data.get("friendly_name") == friendly_name:
                    split_dir = yaml_path.parent
                    break

        if split_dir:
            return split_dir
        raise FileNotFoundError(
            f"No split directory with yaml containing friendly_name: '{friendly_name}' found"
        )

    def _check_prepared_dataset_exists(self, dataset_dir_path: Path) -> bool:
        prepared_dataset_path = dataset_dir_path / self.prepared_filename
        return prepared_dataset_path.exists()

    def _parse_filter_criteria(self, dataset_dir_path: Path) -> dict | None:
        config_path = dataset_dir_path / self.data_config_filename
        if config_path.exists():
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
                if data and data.get("filter_criteria"):
                    logging.debug(f"Filter criteria: {data['filter_criteria']}")
                    return data["filter_criteria"]
        logging.debug("No filter criteria found in data config.")
        return None

    def _parse_label_transformations(self, dataset_dir_path: Path) -> list | None:
        config_path = dataset_dir_path / self.data_config_filename
        if config_path.exists():
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
                if data and data.get("label_transformations"):
                    logging.debug(
                        f"Label transformations: {data['label_transformations']}"
                    )
                    return data["label_transformations"]
        logging.debug("No label transformations found in data config.")
        return None

    def _apply_filter_criteria(
        self, df: pd.DataFrame, dataset_dir_path: Path
    ) -> pd.DataFrame:
        filter_criteria = self._parse_filter_criteria(dataset_dir_path)
        if filter_criteria is not None:
            for column, criteria in filter_criteria.items():
                if column in df.columns:
                    pre_filter_len = len(df)
                    df = df[df[column].isin(criteria)].reset_index(drop=True)
                    if pre_filter_len != len(df):
                        logging.debug(
                            f"Applying filter criteria on column '{column}' resulted in "
                            f"{pre_filter_len - len(df)} rows being dropped."
                        )
                else:
                    logging.warning(
                        f"Filter criteria specified for column '{column}', "
                        "but column not found in dataframe. Skipping."
                    )
        return df

    def _apply_label_transformations(
        self, df: pd.DataFrame, dataset_dir_path: Path
    ) -> pd.DataFrame:
        label_transformations = self._parse_label_transformations(dataset_dir_path)
        logging.debug(f"Label transformations: {label_transformations}")

        if label_transformations is not None:
            for transformation in label_transformations:
                match transformation:
                    case "log10":
                        pre_transform_len = len(df)
                        df = df[df["y"] > 0].reset_index(drop=True)
                        if pre_transform_len != len(df):
                            logging.warning(
                                f"Applying log_10 resulted in "
                                f"{pre_transform_len - len(df)} rows being dropped due to non-positive labels."
                            )
                        df["y"] = df["y"].apply(np.log10)
                    case "negate":
                        df["y"] = df["y"] * -1
                    case _:
                        logging.warning(
                            f"Label transformation type '{transformation}' is not recognized. Skipping."
                        )
        return df

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

        if loaded_df is not None and len(loaded_df) > 2:
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
        df.to_csv(dir / self.prepared_filename, index=False)

    def _generate_prepared_dataset(self, dataset_dir_path: Path) -> None:
        """
        Take whatever raw dataset in 'dataset_dir_path' and output
        prepared dataset under 'self.prepared_filename' in that dir
        """
        multiple_raw_datasets = False

        # note that check against existence of prepared dataset should have occured before this
        datasets_in_dir: list[Path] = [
            Path(globbed_ds) for globbed_ds in dataset_dir_path.rglob("*.csv")
        ]
        # check if self.prepared_filename is among them, and remove it
        datasets_in_dir = [
            ds for ds in datasets_in_dir if ds.name != self.prepared_filename
        ]

        if len(datasets_in_dir) > 1:
            multiple_raw_datasets = True

        raw_dfs = [self._load_df(ds) for ds in datasets_in_dir]
        if multiple_raw_datasets:
            match self.handle_multiple_datasets_method:
                case "naive_aggregate":
                    aggregate_df = self._naive_aggregate_multiple_datasets(raw_dfs)
                    prepared_df = self.get_prepared_df(aggregate_df)
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
            # Normalize SMILES and drop NaNs
            logging.debug(f"Normalizing smiles")
            prepared_df = self.get_normalized_df(raw_dfs[0])

            # Apply any filter criteria from data_config.yaml
            logging.debug(f"Applying filters on data columns")
            prepared_df = self._apply_filter_criteria(prepared_df, dataset_dir_path)

            # Apply any label transformations from data_config.yaml
            logging.debug(f"Applying label transformations on labels column")
            prepared_df = self._apply_label_transformations(
                prepared_df, dataset_dir_path
            )

            # If classification task, assign classes based on continuous labels
            if self.task_setting == "classification":
                logging.debug(
                    f"Assigning classes based on continuous labels for classification task"
                )
                prepared_df = self._assign_classes_based_on_continous_labels(
                    prepared_df, dataset_dir_path, is_chembl=True
                )

        if prepared_df is not None:
            self._save_df(prepared_df, dataset_dir_path)
        else:
            raise RuntimeError(
                f"Failed to generate a prepared dataset within dataset directory '{dataset_dir_path}'"
            )

    def _load_prepared_dataset(self, dataset_dir_path: Path) -> pd.DataFrame:
        return pd.read_csv(dataset_dir_path / self.prepared_filename)

    def set_task_setting(self, task_setting: str) -> None:
        self.task_setting = task_setting

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

    @classmethod
    def get_label_col_in_raw(cls, raw_df) -> str:
        for label_col_variant in cls.possible_label_cols:
            if label_col_variant in raw_df.columns:
                return label_col_variant

        # Failed
        raise ValueError(
            "Failed to find one of label column name variants:",
            str(cls.possible_label_cols),
            "in datafsrame:",
            str(raw_df.head()),
        )

    def get_normalized_df(self, df_to_prepare: pd.DataFrame) -> pd.DataFrame:
        """Get ready-to-save df without NaNs and with canonical SMILES"""
        logging.debug(f"Raw dataset size: {len(df_to_prepare)}")
        logging.debug(f"Raw dataset columns: {df_to_prepare.columns.tolist()}")

        current_smiles_col = self.get_smiles_col_in_raw(df_to_prepare)
        current_label_col = self.get_label_col_in_raw(df_to_prepare)
        if current_smiles_col != "smiles":
            df_to_prepare.rename(
                columns={current_smiles_col: "smiles", current_label_col: "y"},
                inplace=True,
            )
        else:
            logging.warning(
                "While running 'get_prepared_df' on raw df, found column 'smiles'. "
                "This is not expected from a raw ChEMBL dataset. Proceeding anyway."
            )

        df_to_prepare = self.get_clean_smiles_df(df_to_prepare, smiles_col="smiles")

        # Remove NaNs in labels column
        pre_label_nan_len = len(df_to_prepare)
        df_to_prepare = df_to_prepare.dropna(subset=["y"]).reset_index(drop=True)
        if pre_label_nan_len != len(df_to_prepare):
            logging.debug(
                f"Dropping 'nan' labels resulted in {pre_label_nan_len - len(df_to_prepare)} "
                f"rows being dropped."
            )

        return df_to_prepare

    def get_by_friendly_name(self, friendly_name: str) -> pd.DataFrame:
        dataset_dir_path: Path = self._find_dataset_dir(friendly_name)

        if not self._check_prepared_dataset_exists(dataset_dir_path):
            self._generate_prepared_dataset(dataset_dir_path)

        logging.info(
            f"Loading dataset {friendly_name} from {dataset_dir_path / self.prepared_filename}"
        )
        dataset_df = self._load_prepared_dataset(dataset_dir_path)
        logging.info(f"Dataset size: {len(dataset_df)}")

        return dataset_df

    def get_split_by_friendly_name(self, friendly_name: str) -> pd.DataFrame:
        split_dir_path: Path = self._find_split_dir(friendly_name)
        split_df = pd.read_csv(split_dir_path / self.split_datafile_name)
        return split_df

    def save_metrics(
        self, friendly_name: str, metrics: pd.DataFrame | dict
    ) -> None: ...

    def save_visualization(self, friendly_name: str, visualization: Image) -> None:
        output_path = self.visualizations_dir / f"vis_{friendly_name}.png"
        visualization.save(output_path)

    def save_train_test_split(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        subdir_name: str,
        split_friendly_name: str,
        classification_or_regression: str,
        console_log: str | None = None,
    ) -> None:
        """
        Save the given train and test DataFrames to the appropriate locations and generate params.yaml files.
        The function searches for the dataset directory using the friendly name provided. It then saves the
        train and test DataFrames to .csvs in the specified subdirectory.
        """

        split_dir = self.splits_dir / classification_or_regression / subdir_name

        def save_split_component(df: pd.DataFrame, train_or_test: str):
            component_path = split_dir / train_or_test / self.split_datafile_name
            component_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(component_path, index=False)
            params_path = split_dir / train_or_test / "params.yaml"
            component_friendly_name = f"{classification_or_regression[:3]}_{train_or_test}_{split_friendly_name}"
            with open(params_path, "w") as f:
                yaml.dump(
                    {
                        "friendly_name": component_friendly_name,
                        "raw_or_derived": "derived",
                        "task": classification_or_regression,
                    },
                    f,
                )

        save_split_component(train_df, "train")
        save_split_component(test_df, "test")

        operative_config_path = split_dir / "operative_config.txt"
        with open(operative_config_path, "w") as f:
            f.write(gin.operative_config_str())

        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        if console_log:
            logs_path = split_dir / f"console.log"
            with open(logs_path, "w") as f:
                f.write(timestamp + "\n")
                f.write(console_log)

        logging.info(f"Train-test split was saved to {split_dir}.")
        logging.info(f"Generated friendly names:")
        logging.info(
            f"- Train: {classification_or_regression[:3]}_train_{split_friendly_name}"
        )
        logging.info(
            f"- Test: {classification_or_regression[:3]}_test_{split_friendly_name}"
        )

    def update_registries(self):
        self.update_splits_registry()

    def update_datasets_registry(self) -> None:
        """
        This method looks for .yaml files in the dataset directory and creates/updates a list
        of all friendly names, which is stores in a text file for convenient presentation to the user.
        """
        friendly_names = []
        for yaml_path in Path(self.dataset_dir).rglob("*.yaml"):
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
                if data and data.get("friendly_name"):
                    friendly_names.append(data["friendly_name"])

        registry_path = self.dataset_dir / self.registry_filename
        with open(registry_path, "w") as f:
            for name in friendly_names:
                f.write(f"{name}\n")

        logging.debug(f"Updated datasets registry at {registry_path}")

    def update_splits_registry(self) -> None:
        """
        This method looks for .yaml files in the dataset directory and creates/updates a list
        of all friendly names, which is stores in a text file for convenient presentation to the user.
        """

        Split = namedtuple("Split", ["friendly_name", "timestamp"])

        splits = []
        for yaml_path in Path(self.splits_dir).rglob("*.yaml"):
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
                if data and data.get("friendly_name"):
                    friendly_name = data["friendly_name"]

            timestamp_path = yaml_path.parent.parent / "console.log"
            with open(timestamp_path, "r") as f:
                timestamp = f.readline().strip()

            splits.append(Split(friendly_name, timestamp))

        splits.sort(key=lambda x: x.timestamp)

        registry_path = self.splits_dir / self.registry_filename
        with open(registry_path, "w") as f:
            for split in splits:
                f.write(f"{split.timestamp} {split.friendly_name}\n")

        logging.debug(f"Updated splits registry at {registry_path}")

    def _parse_classification_threshold(self, dataset_dir_path: Path):
        """Parse threshold for classification tasks from yaml config."""
        config_path = dataset_dir_path / self.data_config_filename
        if config_path.exists():
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)

            if data and data.get("threshold_source"):
                logging.info(
                    f"Using {data.get('threshold_source')} metadata to establish classification threshold."
                )
                dataset = self.get_by_friendly_name(data.get("threshold_source"))
                threshold_value = dataset["y_cont"].median()

            elif data and data.get("threshold"):
                threshold_value = data.get("threshold")

            else:
                raise RuntimeError(
                    "No threshold or threshold_source found in data config yaml."
                )

            return threshold_value
        raise RuntimeError(
            "No data config yaml found to parse threshold for classification task."
        )

    def _assign_classes_based_on_continous_labels(
        self, df: pd.DataFrame, dataset_dir_path: Path, is_chembl=True
    ) -> pd.DataFrame:
        """Assign classes based on threshold for classification tasks."""

        # Read yaml config to get threshold if not provided
        threshold_value = self._parse_classification_threshold(dataset_dir_path)

        if threshold_value == "median":
            threshold_value = df["y"].median()
            logging.info(
                f"Using median value {threshold_value} as threshold for class assignment."
            )
        elif isinstance(threshold_value, (int, float)):
            logging.info(
                f"Using provided threshold value {threshold_value} for class assignment."
            )
        else:
            raise ValueError(f"Threshold value '{threshold_value}' is not valid.")

        initial_len = len(df)
        if is_chembl:
            conditions = [
                (df["Standard Relation"] == "'='") & (df["y"] >= threshold_value),
                (df["Standard Relation"] == "'='") & (df["y"] < threshold_value),
                (df["Standard Relation"] == "'>'") & (df["y"] >= threshold_value),
                (df["Standard Relation"] == "'<'") & (df["y"] < threshold_value),
                (df["Standard Relation"] == "'>='") & (df["y"] >= threshold_value),
                (df["Standard Relation"] == "'<='") & (df["y"] < threshold_value),
            ]

            # Corresponding class labels for the conditions
            choices = [1, 0, 1, 0, 1, 0]

            df["class"] = np.select(conditions, choices, default=np.nan)
            df.dropna(subset=["class"], inplace=True)
            df["class"] = df["class"].astype(int)

            rows_dropped = initial_len - len(df)
            if rows_dropped > 0:
                logging.info(
                    f"Dropped {rows_dropped} rows with ambiguous relations for classification."
                )

        else:
            # For other datasets, we assume a simple thresholding
            df["class"] = (df["y"] >= threshold_value).astype(int)

        # The original 'y' column is no longer needed for classification
        df.rename(columns={"y": "y_cont"}, inplace=True)
        df.rename(columns={"class": "y"}, inplace=True)

        return df
