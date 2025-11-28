# python
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from collections import namedtuple
import pickle
import logging
import yaml
import gin
import pandas as pd
import numpy as np
from PIL.Image import Image

from src.utils import detect_csv_delimiter, get_clean_smiles
from src.predictor.predictor_base import PredictorBase


@gin.configurable
class DataInterface:
    """
    Manage dataset loading, normalization and train/test split saving.
    """

    # TODO: add support for multi-class classification datasets
    # TODO: handle model saving/loading here?

    possible_smiles_cols = ["SMILES", "Smiles", "smiles", "molecule"]
    possible_label_cols = ["LABEL", "Label", "label", "Y", "y", "Standard Value"]

    def __init__(
        self,
        dataset_dir: str,
        cache_dir: str,
        visualizations_dir: str,
        data_config_filename: str,
        prepared_filename: str,
        registry_filename: str = "registry.txt",
    ):
        self.dataset_dir = Path(dataset_dir)
        self.cache_dir = Path(cache_dir)
        self.visualizations_dir = Path(visualizations_dir)
        self.data_config_filename = data_config_filename
        self.prepared_filename = prepared_filename
        self.registry_filename = registry_filename
        self.taks_setting: Optional[str] = None  # must be set externally before use
        self.logfile: Optional[Path] = None  # must be set externally before use
        self.override_cache = False

        # Those may not need to be configurable
        self.splits_dir = self.cache_dir / "splits"
        self.models_dir = self.cache_dir / "models"
        self.split_datafile_name = "data.csv"
        self.model_filename = "model.pkl"
        self.model_refit_filename = "model_final_refit.pkl"
        self.model_metrics_filename = "metrics.yaml"
        self.params_filename = "params.yaml"
        self.model_params_filename = "hyperparams.yaml"
        self.model_metadata_filename = "model_metadata.yaml"

        self._init_create_dirs()
        self.update_registries()
        # task_setting must be set externally via set_task_setting before use where required

    def _init_create_dirs(self) -> None:
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)

    def set_logfile(self, logfile: str) -> None:
        self.logfile = Path(logfile)

    def set_override_cache(self, override_cache: bool) -> None:
        self.override_cache = override_cache

    def dump_logs(self, path: Path) -> None:
        if self.logfile:
            with open(self.logfile, "r") as fh:
                contents = fh.read()
            with open(path, "w") as fh:
                fh.write(contents)

    # --- discovery helpers -------------------------------------------------
    def _find_dataset_dir(self, friendly_name: str) -> Path:
        for yaml_path in self.dataset_dir.rglob("*.yaml"):
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f) or {}
                if (
                    data.get("friendly_name") == friendly_name
                    and data.get("task_setting") == self.task_setting
                ):
                    return yaml_path.parent
        raise FileNotFoundError(
            f"No dataset directory with yaml containing friendly_name: `{friendly_name}` found"
        )

    def _find_split_dir(self, friendly_name: str) -> Path:
        for yaml_path in self.splits_dir.rglob("*.yaml"):
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f) or {}
                if data.get("friendly_name") == friendly_name:
                    return yaml_path.parent
        raise FileNotFoundError(
            f"No split directory with yaml containing friendly_name: `{friendly_name}` found"
        )

    def _check_prepared_dataset_exists(self, dataset_dir_path: Path) -> bool:
        return (dataset_dir_path / self.prepared_filename).exists()

    # --- yaml parsing small helpers ---------------------------------------
    def _read_data_config(self, dataset_dir_path: Path) -> Dict:
        config_path = dataset_dir_path / self.data_config_filename
        if not config_path.exists():
            return {}
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}

    def _parse_filter_criteria(self, dataset_dir_path: Path) -> Optional[Dict]:
        return self._read_data_config(dataset_dir_path).get("filter_criteria")

    def _parse_label_transformations(self, dataset_dir_path: Path) -> Optional[List]:
        return self._read_data_config(dataset_dir_path).get("label_transformations")

    def _parse_is_chembl(self, dataset_dir_path: Path) -> bool:
        return self._read_data_config(dataset_dir_path).get("is_chembl", False)

    # --- dataframe transformations ----------------------------------------
    def _apply_filter_criteria(
        self, df: pd.DataFrame, dataset_dir_path: Path
    ) -> pd.DataFrame:
        criteria = self._parse_filter_criteria(dataset_dir_path)
        if not criteria:
            return df
        for column, allowed in criteria.items():
            if column in df.columns:
                pre = len(df)
                df = df[df[column].isin(allowed)].reset_index(drop=True)
                if len(df) != pre:
                    logging.info(f"Filter `{column}` dropped {pre - len(df)} rows.")
            else:
                logging.warning(
                    f"Filter criteria field `{column}` not present on dataframe; skipping."
                )
        return df

    def _apply_label_transformations(
        self, df: pd.DataFrame, dataset_dir_path: Path
    ) -> pd.DataFrame:
        transformations = self._parse_label_transformations(dataset_dir_path) or []
        for t in transformations:
            if t == "log10":
                pre = len(df)
                df = df[df["y"] > 0].reset_index(drop=True)
                if len(df) != pre:
                    logging.warning(
                        f"log10 dropped {pre - len(df)} rows (non-positive labels)."
                    )
                df["y"] = np.log10(df["y"])
            elif t == "negate":
                df["y"] = -df["y"]
            else:
                logging.warning(f"Unknown label transformation `{t}`; skipping.")
        return df

    def _load_df(self, dataset_path: Path) -> pd.DataFrame:
        # If raw dataset is from ChEMBL, use ';' as delimiter, else detect
        delimiter = (
            ";"
            if self._parse_is_chembl(dataset_path.parent)
            else detect_csv_delimiter(dataset_path)
        )
        loaded = pd.read_csv(dataset_path, delimiter=delimiter)
        logging.info(f"Loaded raw dataset from `{dataset_path}`")
        if loaded is None or len(loaded) <= 2:
            raise RuntimeError(
                f"Loading dataset from `{dataset_path}` produced {0 if loaded is None else len(loaded)} rows."
            )
        # detect SMILES and drop NaNs
        smiles_col = self.get_smiles_col_in_raw(loaded)
        pre = len(loaded)
        df = loaded.dropna(subset=[smiles_col]).reset_index(drop=True)
        if len(df) != pre:
            logging.info(
                f"Dropped {pre - len(df)} rows with NaN SMILES when loading `{dataset_path}`."
            )
        return df

    def _save_prepared_df(self, df: pd.DataFrame, dir_path: Path) -> None:

        (dir_path).mkdir(parents=True, exist_ok=True)
        # Save prepared dataset
        df.to_csv(dir_path / self.prepared_filename, index=False)
        # Save preparation logs
        self.dump_logs(dir_path / "preparation.log")

    def _generate_prepared_dataset(self, dataset_dir_path: Path) -> None:
        csvs = [
            p
            for p in dataset_dir_path.rglob("*.csv")
            if p.name != self.prepared_filename
        ]
        if not csvs:
            raise RuntimeError(f"No raw dataset found in `{dataset_dir_path}`")
        if len(csvs) > 1:
            raise RuntimeError(f"Multiple raw datasets found in `{dataset_dir_path}`")
        logging.info(f"Generating prepared dataset for raw {csvs[0]}")
        raw_df = self._load_df(csvs[0])
        logging.info(f"Raw dataset size: {len(raw_df)}")

        # normalize smiles
        prepared = self.get_normalized_df(raw_df)
        # apply filtering by columns (mainly for ChEMBL datasets)
        prepared = self._apply_filter_criteria(prepared, dataset_dir_path)
        # apply pre-defined label transformations (e.g., log10, negate)
        prepared = self._apply_label_transformations(prepared, dataset_dir_path)
        # for classification tasks, assign binary classes based on continuous labels
        if self.task_setting == "binary_classification":
            prepared = self._assign_binary_classes_based_on_continuous_labels(
                prepared, dataset_dir_path, is_chembl=True
            )
        elif self.task_setting == "multi_class_classification":
            raise NotImplementedError(
                "Multi-class classification datasets are not yet supported."
            )
        if prepared is None:
            raise RuntimeError(
                f"Failed to generate prepared dataset for `{dataset_dir_path}`"
            )
        self._save_prepared_df(prepared, dataset_dir_path)

    def _load_prepared_dataset(self, dataset_dir_path: Path) -> pd.DataFrame:
        return pd.read_csv(dataset_dir_path / self.prepared_filename)

    def _load_split_component(
        self, split_dir_path: Path) -> pd.DataFrame:
        return pd.read_csv(split_dir_path / self.split_datafile_name)

    def set_task_setting(self, task_setting: str) -> None:
        assert task_setting in [
            "regression",
            "binary_classification",
            "multi_class_classification",
        ], f"Unknown task_setting parsed: {task_setting}"
        self.task_setting = task_setting

    @staticmethod
    def get_clean_smiles_df(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
        # apply canonicalization and drop rows where canonicalization produced NaN
        pre = len(df)
        df[smiles_col] = df[smiles_col].apply(get_clean_smiles)
        df = df.dropna(subset=[smiles_col]).reset_index(drop=True)
        if len(df) != pre:
            logging.info(
                f"get_clean_smiles dropped {pre - len(df)} rows with invalid SMILES."
            )
        return df

    @classmethod
    def get_smiles_col_in_raw(cls, raw_df: pd.DataFrame) -> str:
        for c in cls.possible_smiles_cols:
            if c in raw_df.columns:
                return c
        raise ValueError(
            "No SMILES column found. Looked for: " + ", ".join(cls.possible_smiles_cols)
        )

    @classmethod
    def get_label_col_in_raw(cls, raw_df: pd.DataFrame) -> str:
        for c in cls.possible_label_cols:
            if c in raw_df.columns:
                return c
        raise ValueError(
            "No label column found. Looked for: " + ", ".join(cls.possible_label_cols)
        )

    def get_normalized_df(self, df_to_prepare: pd.DataFrame) -> pd.DataFrame:
        logging.debug(f"Normalizing dataset with {len(df_to_prepare)}")
        smiles_col = self.get_smiles_col_in_raw(df_to_prepare)
        label_col = self.get_label_col_in_raw(df_to_prepare)

        # rename to canonical names
        rename_map = {}
        if smiles_col != "smiles":
            rename_map[smiles_col] = "smiles"
        if label_col != "y":
            rename_map[label_col] = "y"
        if rename_map:
            df_to_prepare = df_to_prepare.rename(columns=rename_map)

        if "smiles" not in df_to_prepare.columns or "y" not in df_to_prepare.columns:
            raise RuntimeError("Expected `smiles` and `y` columns after normalization.")

        df_to_prepare = self.get_clean_smiles_df(df_to_prepare, "smiles")
        pre = len(df_to_prepare)
        df_to_prepare = df_to_prepare.dropna(subset=["y"]).reset_index(drop=True)
        if len(df_to_prepare) != pre:
            logging.debug(f"Dropped {pre - len(df_to_prepare)} rows with NaN labels.")
        return df_to_prepare

    # --- public dataset getters ------------------------------------------
    def get_by_friendly_name(self, friendly_name: str, is_in_splits = False) -> pd.DataFrame:
        if is_in_splits:
            dataset_dir_path = self._find_split_dir(friendly_name)
            dataset = self._load_split_component(dataset_dir_path)
        else:
            dataset_dir_path = self._find_dataset_dir(friendly_name)
            if not self._check_prepared_dataset_exists(dataset_dir_path) or self.override_cache:
                self._generate_prepared_dataset(dataset_dir_path)
            logging.debug(f"Loading prepared dataset `{friendly_name}`")
            dataset = self._load_prepared_dataset(dataset_dir_path)
        return dataset

    def get_split_friendly_names(self, cache_key: str) -> Tuple[str, str]:
        split_dir = self.splits_dir / cache_key
        train_params_path = split_dir / "train" / self.params_filename
        test_params_path = split_dir / "test" / self.params_filename
        if not train_params_path.exists():
            raise FileNotFoundError(
                f"No {self.params_filename} found for train split with cache_key `{cache_key}`"
            )
        if not test_params_path.exists():
            raise FileNotFoundError(
                f"No {self.params_filename} found for test split with cache_key `{cache_key}`"
            )
        with open(train_params_path, "r") as fh:
            train_params = yaml.safe_load(fh) or {}
            train_friendly_name = train_params.get("friendly_name")
            if not train_friendly_name:
                raise RuntimeError(
                    f"No `friendly_name` found in train {self.params_filename} for split with cache_key `{cache_key}`"
                )
        with open(test_params_path, "r") as fh:
            test_params = yaml.safe_load(fh) or {}
            test_friendly_name = test_params.get("friendly_name")
            if not test_friendly_name:
                raise RuntimeError(
                    f"No `friendly_name` found in test {self.params_filename} for split with cache_key `{cache_key}`"
                )
        return train_friendly_name, test_friendly_name

    def get_train_test_friendly_names(self, cache_key: str) -> str:
        split_dir = self.splits_dir / cache_key
        params_path = split_dir / "train" / self.params_filename
        if not params_path.exists():
            raise FileNotFoundError(
                f"No {self.params_filename} found for split with cache_key `{cache_key}`"
            )
        with open(params_path, "r") as fh:
            params = yaml.safe_load(fh) or {}
            friendly_name = params.get("friendly_name")
            if not friendly_name:
                raise RuntimeError(
                    f"No `friendly_name` found in {self.params_filename} for split with cache_key `{cache_key}`"
                )
            return friendly_name

    def save_visualization(self, friendly_name: str, visualization: Image) -> None:
        out = self.visualizations_dir / f"vis_{friendly_name}.png"
        visualization.save(out)

    def save_train_test_split(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        cache_key: str,
        split_friendly_name: str,
        classification_or_regression: str,
    ) -> None:
        split_dir = self.splits_dir / cache_key

        def _save_component(df: pd.DataFrame, which: str) -> None:
            path = split_dir / which / self.split_datafile_name
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            params = {
                "friendly_name": f"{classification_or_regression[:3]}_{which}_{split_friendly_name}",
                "raw_or_derived": "derived",
                "task": classification_or_regression,
            }
            with open(path.parent / self.params_filename, "w") as fh:
                yaml.dump(params, fh)

        _save_component(train_df, "train")
        _save_component(test_df, "test")
        split_dir.mkdir(parents=True, exist_ok=True)

        # dump gin config
        with open(split_dir / "operative_config.txt", "w") as fh:
            fh.write(gin.operative_config_str())

        # dump console log
        self.dump_logs(split_dir / "console.log")

        logging.info(f"Saved split at `{split_dir}`")

    def check_train_test_split_exists(self, cache_key: str) -> bool:
        split_dir = self.splits_dir / cache_key
        train_path = split_dir / "train" / self.split_datafile_name
        test_path = split_dir / "test" / self.split_datafile_name
        return train_path.exists() and test_path.exists()

    # --- models saving/loading -------------------------------------------

    def pickle_model(
        self,
        model: PredictorBase,
        model_cache_key: str,
        data_cache_key: str,
        save_as_refit: bool = False,
    ) -> None:
        path = self.models_dir / model_cache_key / data_cache_key
        path = (
            path / self.model_refit_filename
            if save_as_refit
            else path / self.model_filename
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Saving trained model to `{path}`")
        # Pickle the model
        with open(path, "wb") as f:
            pickle.dump(model, f)

    def save_model_metadata(
        self,
        metadata: Dict,
        model_cache_key: str,
        data_cache_key: str,
    ) -> None:
        path = self.models_dir / model_cache_key / data_cache_key / self.model_metadata_filename
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.dump(metadata, fh)

    def unpickle_model(
        self, model_cache_key: str, data_cache_key: str
    ) -> PredictorBase:
        # Check if the path exists
        path = self.models_dir / model_cache_key / data_cache_key / self.model_filename
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")
        # Load the model
        with open(path, "rb") as f:
            loaded_model = pickle.load(f)
            return loaded_model

    def save_metrics(
        self, metrics: Dict, model_cache_key: str, data_cache_key: str
    ) -> None:
        path = (
            self.models_dir
            / model_cache_key
            / data_cache_key
            / self.model_metrics_filename
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.dump(metrics, fh)

    def save_hyperparams(
        self, params: Dict, model_cache_key: str, data_cache_key: str
    ) -> None:
        path = (
            self.models_dir
            / model_cache_key
            / data_cache_key
            / self.model_params_filename
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.dump(params, fh)

    def load_hyperparams(self, model_cache_key: str, data_cache_key: str) -> Dict:
        path = (
            self.models_dir
            / model_cache_key
            / data_cache_key
            / self.model_params_filename
        )
        if not path.exists():
            raise FileNotFoundError(f"Model params file not found at {path}")
        with open(path, "r") as fh:
            return yaml.safe_load(fh) or {}

    def dump_training_logs(self, model_cache_key: str, data_cache_key: str) -> None:
        path = self.models_dir / model_cache_key / data_cache_key / "training.log"
        self.dump_logs(path)

    def dump_gin_config_to_model_dir(self, model_cache_key: str, data_cache_key: str) -> None:
        path = self.models_dir / model_cache_key / data_cache_key / "operative_config.gin"
        with open(path, "w") as fh:
            fh.write(gin.operative_config_str())

    # --- registries ------------------------------------------------------
    def update_registries(self) -> None:
        self.update_splits_registry()

    def update_datasets_registry(self) -> None:
        names = []
        for yaml_path in self.dataset_dir.rglob("*.yaml"):
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f) or {}
                if data.get("friendly_name"):
                    names.append(data["friendly_name"])
        with open(self.dataset_dir / self.registry_filename, "w") as fh:
            for n in names:
                fh.write(f"{n}\n")
        logging.debug("Updated datasets registry.")

    def update_splits_registry(self) -> None:
        Split = namedtuple("Split", ["friendly_name", "timestamp"])
        splits: List[Tuple[str, str]] = []
        for yaml_path in self.splits_dir.rglob("*.yaml"):
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f) or {}
                fname = data.get("friendly_name")
            ts_path = yaml_path.parent.parent / "console.log"
            ts = ""
            if ts_path.exists():
                with open(ts_path, "r") as fh:
                    ts = fh.readline().strip()
            if fname:
                splits.append(Split(fname, ts))
        splits.sort(key=lambda s: s.timestamp)
        with open(self.splits_dir / self.registry_filename, "w") as fh:
            for s in splits:
                fh.write(f"{s.timestamp} {s.friendly_name}\n")
        logging.debug("Updated splits registry.")

    # --- classification helpers -----------------------------------------
    def _parse_classification_threshold(self, dataset_dir_path: Path) -> float:
        cfg = self._read_data_config(dataset_dir_path)
        if not cfg:
            raise RuntimeError(
                "No data config yaml found to parse threshold for classification task."
            )
        if cfg.get("threshold_source"):
            ds = self.get_by_friendly_name(cfg["threshold_source"])
            if "y_cont" not in ds.columns:
                raise RuntimeError("threshold_source dataset must contain `y_cont`.")
            return float(ds["y_cont"].median())
        if "threshold" in cfg:
            return cfg["threshold"]
        raise RuntimeError("No `threshold` or `threshold_source` in data config.")

    def _clean_relation_series(
        self, df: pd.DataFrame
    ) -> Tuple[pd.Series, Optional[str]]:
        if "Standard Relation" in df.columns:
            cleaned = (
                df["Standard Relation"]
                .astype(str)
                .str.replace(r"^['\"]|['\"]$", "", regex=True)
                .str.strip()
            )
            return cleaned
        # no column found -> return empty strings
        return pd.Series([""] * len(df), index=df.index), None

    def _assign_binary_classes_based_on_continuous_labels(
        self, df: pd.DataFrame, dataset_dir_path: Path, is_chembl: bool = True
    ) -> pd.DataFrame:
        threshold_value = self._parse_classification_threshold(dataset_dir_path)
        if threshold_value == "median":
            threshold_value = float(df["y"].median())
            logging.info(
                "Assigning binary classes based on continuous labels and given threshold."
            )
            logging.info(f"Using median {threshold_value} as threshold.")
        else:
            threshold_value = float(threshold_value)

        initial_len = len(df)
        if is_chembl:
            relation_series = self._clean_relation_series(df)
            eq = relation_series == "="
            gt = relation_series == ">"
            lt = relation_series == "<"
            ge = relation_series == ">="
            le = relation_series == "<="

            conditions = [
                (eq | ge | gt) & (df["y"] >= threshold_value),
                (eq | le | lt) & (df["y"] < threshold_value),
            ]
            choices = [1, 0]
            df["class"] = np.select(conditions, choices, default=np.nan)
            df = df.dropna(subset=["class"]).reset_index(drop=True)
            df["class"] = df["class"].astype(int)
            dropped = initial_len - len(df)
            if dropped:
                logging.info(
                    f"Dropped {dropped} rows with ambiguous/unsupported relations."
                )
        else:
            df["class"] = (df["y"] >= threshold_value).astype(int)

        # keep continuous label in `y_cont`, classification in `y`
        df = df.rename(columns={"y": "y_cont", "class": "y"})
        return df
