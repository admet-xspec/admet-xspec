import logging
import shutil
from collections import namedtuple
from pathlib import Path
from typing import ContextManager, Dict, List, Optional, Tuple

import filelock
import gin
import numpy as np
import pandas as pd

from src.data.fs_utils import (
    atomic_copy_file,
    atomic_pickle_dump,
    atomic_to_csv,
    atomic_write_text,
    atomic_yaml_dump,
    hold_lock,
    iter_files,
    load_pickle,
    load_yaml,
    make_file_lock,
    make_unique_dir,
    publish_dir,
    read_first_line,
    remove_stale_entries,
    remove_stale_tmp_files,
    safe_load_yaml,
)
from src.data.utils import get_label_counts
from src.utils import detect_csv_delimiter, get_clean_smiles
from src.predictor.predictor_base import PredictorBase

_Split = namedtuple("_Split", ["friendly_name", "timestamp"])


@gin.configurable
class DataInterface:
    """Manage dataset loading, normalization and train/test split saving."""

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
        lock_timeout: float = 3600.0,
        use_soft_locks: bool = False,
        stale_artifact_max_age_hours: float = 24.0,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.cache_dir = Path(cache_dir)
        self.visualizations_dir = Path(visualizations_dir)
        self.data_config_filename = data_config_filename
        self.prepared_filename = prepared_filename
        self.registry_filename = registry_filename
        # Set externally via `set_task_setting` / `set_logfile`.
        self.task_setting: Optional[str] = None
        self.logfile: Optional[Path] = None
        # Regenerate cached prepared datasets and splits.
        self.override_cache = False

        self.lock_timeout = lock_timeout
        self.use_soft_locks = use_soft_locks
        self.stale_artifact_max_age_seconds = stale_artifact_max_age_hours * 3600.0
        self._locks: Dict[str, filelock.BaseFileLock] = {}

        self.splits_dir = self.cache_dir / "splits"
        self.models_dir = self.cache_dir / "models"
        # Outside scanned trees, on the `splits_dir` filesystem for atomic renames.
        self.locks_dir = self.cache_dir / ".locks"
        self.staging_dir = self.cache_dir / ".staging"
        self.split_datafile_name = "data.csv"
        self.model_filename = "model.pkl"
        self.model_refit_filename = "model_final_refit.pkl"
        self.model_metrics_filename = "metrics.yaml"
        self.params_filename = "params.yaml"
        self.model_params_filename = "hyperparams.yaml"
        self.model_metadata_filename = "model_metadata.yaml"
        self.console_log_filename = "console.log"
        self.operative_config_filename = "operative_config.gin"
        # Fitted domains live in the split dir, calibrations in the model dir.
        self.domain_dirname = "applicability_domain"
        self.domain_filename = "domain.pkl"
        self.domain_refit_filename = "domain_refit.pkl"
        self.domain_calibration_filename = "calibration.yaml"

        self._init_create_dirs()
        self._cleanup_stale_artifacts()
        self.update_registries()

    def _init_create_dirs(self) -> None:
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        self.locks_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)

    def _cleanup_stale_artifacts(self) -> None:
        """Remove leftovers of crashed processes: temp files and abandoned staging dirs."""
        max_age = self.stale_artifact_max_age_seconds
        removed = remove_stale_entries(self.staging_dir, max_age)
        removed += remove_stale_tmp_files(self.cache_dir, max_age)
        removed += remove_stale_tmp_files(self.dataset_dir, max_age)
        if removed:
            logging.info(f"Removed {removed} stale temporary file(s)/directories.")

    # File locks cannot be pickled; drop them and recreate lazily in the new process.
    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        state["_locks"] = {}
        return state

    def __setstate__(self, state: Dict) -> None:
        self.__dict__.update(state)

    # --- locking ---
    def _get_lock(self, name: str) -> filelock.BaseFileLock:
        # One object per name: reuse makes nested acquisition reentrant.
        lock = self._locks.get(name)
        if lock is None:
            lock = make_file_lock(
                self.locks_dir, name, self.lock_timeout, soft=self.use_soft_locks
            )
            self._locks[name] = lock
        return lock

    def _locked(self, name: str) -> ContextManager[None]:
        return hold_lock(self._get_lock(name), name)

    def split_lock(self, cache_key: str) -> ContextManager[None]:
        """Return the lock for one split; hold it across check, compute and save.

            with data_interface.split_lock(key):
                if not data_interface.check_train_test_split_exists(key):
                    train_df, test_df = compute_split(...)
                    data_interface.save_train_test_split(train_df, test_df, key, ...)
        """
        return self._locked(f"split:{cache_key}")

    def domain_lock(
        self, data_cache_key: str, domain_cache_key: str
    ) -> ContextManager[None]:
        """Return the lock for one split's domain; hold it across check, fit and save."""
        return self._locked(f"domain:{data_cache_key}/{domain_cache_key}")

    # --- setters / log dumping ---
    def set_logfile(self, logfile: str) -> None:
        self.logfile = Path(logfile)

    def set_override_cache(self, override_cache: bool) -> None:
        self.override_cache = override_cache

    def dump_logs(self, path: Path) -> None:
        if self.logfile:
            atomic_copy_file(self.logfile, path / self.console_log_filename)

    def dump_gin_config(self, path: Path) -> None:
        atomic_write_text(gin.operative_config_str(), path / self.operative_config_filename)

    # --- discovery ---
    def _find_dataset_dir(self, friendly_name: str) -> Path:
        found_paths = []
        for yaml_path in iter_files(self.dataset_dir, ".yaml"):
            data = safe_load_yaml(yaml_path)
            if (
                data.get("friendly_name") == friendly_name
                and data.get("task_setting") == self.task_setting
            ):
                found_paths.append(yaml_path.parent)
        if len(found_paths) == 1:
            return found_paths[0]
        if len(found_paths) > 1:
            raise RuntimeError(
                f"Multiple dataset directories with yaml containing friendly_name: `{friendly_name}` for the task of {self.task_setting} found: {found_paths}"
            )
        raise FileNotFoundError(
            f"No dataset directory with yaml containing friendly_name: `{friendly_name}` for the task of {self.task_setting} found"
        )

    def _find_split_dir(self, friendly_name: str) -> Path:
        found_paths = []
        for yaml_path in iter_files(self.splits_dir, ".yaml"):
            data = safe_load_yaml(yaml_path)
            if data.get("friendly_name") == friendly_name:
                found_paths.append(yaml_path.parent)
        if len(found_paths) == 1:
            return found_paths[0]
        if len(found_paths) > 1:
            raise RuntimeError(
                f"Multiple split directories with yaml containing friendly_name: `{friendly_name}` found: {found_paths}"
            )
        raise FileNotFoundError(
            f"No split directory with yaml containing friendly_name: `{friendly_name}` found"
        )

    def _check_prepared_dataset_exists(self, dataset_dir_path: Path) -> bool:
        return (dataset_dir_path / self.prepared_filename).exists()

    # --- data config parsing ---
    def _read_data_config(self, dataset_dir_path: Path) -> Dict:
        return load_yaml(dataset_dir_path / self.data_config_filename, missing_ok=True)

    def _parse_filter_criteria(self, dataset_dir_path: Path) -> Optional[Dict]:
        return self._read_data_config(dataset_dir_path).get("filter_criteria")

    def _parse_label_transformations(self, dataset_dir_path: Path) -> Optional[List]:
        return self._read_data_config(dataset_dir_path).get("label_transformations")

    def _parse_is_chembl(self, dataset_dir_path: Path) -> bool:
        return self._read_data_config(dataset_dir_path).get("is_chembl", False)

    # --- dataframe transformations ---
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
        # ChEMBL exports use ';'.
        delimiter = (
            ";"
            if self._parse_is_chembl(dataset_path.parent)
            else detect_csv_delimiter(dataset_path)
        )
        loaded = pd.read_csv(dataset_path, delimiter=delimiter)
        logging.info(f"Loaded raw dataset from `{dataset_path}`")
        if len(loaded) <= 2:
            raise RuntimeError(
                f"Loading dataset from `{dataset_path}` produced {len(loaded)} rows."
            )
        smiles_col = self.get_smiles_col_in_raw(loaded)
        pre = len(loaded)
        df = loaded.dropna(subset=[smiles_col]).reset_index(drop=True)
        if len(df) != pre:
            logging.info(
                f"Dropped {pre - len(df)} rows with NaN SMILES when loading `{dataset_path}`."
            )
        return df

    def _save_prepared_df(self, df: pd.DataFrame, dir_path: Path) -> None:
        # Logs first: other processes treat the visible CSV as "done".
        self.dump_logs(dir_path / "preparation_log")
        atomic_to_csv(df, dir_path / self.prepared_filename)

    def _prepared_lock_name(self, dataset_dir_path: Path) -> str:
        # Relative key, so hosts mounting shared storage at different paths agree.
        resolved = dataset_dir_path.resolve()
        try:
            key = resolved.relative_to(self.dataset_dir.resolve()).as_posix()
        except ValueError:
            key = str(resolved)
        return f"prepared:{key}"

    def _ensure_prepared_dataset(self, dataset_dir_path: Path) -> None:
        """Generate the prepared dataset unless it exists (double-checked locking).

        Readers need no lock, as the prepared CSV is written atomically.
        """
        if self._check_prepared_dataset_exists(dataset_dir_path) and not self.override_cache:
            return
        with self._locked(self._prepared_lock_name(dataset_dir_path)):
            if (
                self._check_prepared_dataset_exists(dataset_dir_path)
                and not self.override_cache
            ):
                return
            self._generate_prepared_dataset(dataset_dir_path)

    def _load_transformed_raw_df(self, dataset_dir_path: Path) -> pd.DataFrame:
        """Load the raw dataset, clean SMILES, apply filter criteria and label transformations."""
        csvs = [
            p
            for p in dataset_dir_path.rglob("*.csv")
            if p.name != self.prepared_filename
        ]
        if not csvs:
            raise RuntimeError(f"No raw dataset found in `{dataset_dir_path}`")
        if len(csvs) > 1:
            raise RuntimeError(f"Multiple raw datasets found in `{dataset_dir_path}`")
        raw_df = self._load_df(csvs[0])
        logging.info(f"Raw dataset size: {len(raw_df)}")

        df = self.get_normalized_df(raw_df)
        df = self._apply_filter_criteria(df, dataset_dir_path)
        return self._apply_label_transformations(df, dataset_dir_path)

    def _generate_prepared_dataset(self, dataset_dir_path: Path) -> None:
        # Caller must hold the dataset's "prepared" lock.
        logging.info(f"Generating prepared dataset in `{dataset_dir_path}`")
        prepared = self._load_transformed_raw_df(dataset_dir_path)
        if self.task_setting == "binary_classification":
            label_counts = get_label_counts(prepared, column_name="y")
            if len(label_counts) > 2:
                logging.warning(
                    f"Dataset has {len(label_counts)} unique labels. The labels are assumed to be continuous and will be converted to binary classes."
                )
                prepared = self._assign_binary_classes_based_on_continuous_labels(
                    prepared,
                    dataset_dir_path,
                    is_chembl=self._parse_is_chembl(dataset_dir_path),
                )
            else:
                logging.info(f"Binary labels detected: {label_counts.keys()}.")
                unique_labels = sorted(label_counts.keys())
                if unique_labels != [0, 1]:
                    logging.warning(
                        "The binary labels are not 0 and 1; remapping to 0 and 1."
                    )
                    mapping = {old: new for new, old in enumerate(unique_labels)}
                    prepared["y"] = prepared["y"].map(mapping)
        elif self.task_setting == "multi_class_classification":
            raise NotImplementedError(
                "Multi-class classification datasets are not yet supported."
            )
        pre = len(prepared)
        prepared = prepared.dropna(subset=["smiles", "y"]).reset_index(drop=True)
        if len(prepared) != pre:
            logging.warning(
                f"After all transformations, dropped additional {pre - len(prepared)} rows with NaN values in 'smiles' or 'y' columns."
            )
        logging.info(f"Prepared dataset size: {len(prepared)}")
        self._save_prepared_df(prepared, dataset_dir_path)

    def _load_prepared_dataset(self, dataset_dir_path: Path) -> pd.DataFrame:
        df = pd.read_csv(dataset_dir_path / self.prepared_filename)
        pre_nan_len = len(df)
        df = df.dropna(subset=["smiles", "y"]).reset_index(drop=True)
        if len(df) != pre_nan_len:
            logging.warning(
                f"Unexpected behaviour: dropped {pre_nan_len - len(df)} rows with NaN labels when loading prepared dataset. Check data preparation step."
            )
        return df

    def _load_split_component(self, split_dir_path: Path) -> pd.DataFrame:
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

    # --- public dataset getters ---
    def get_by_friendly_name(
        self, friendly_name: str, is_in_splits=False
    ) -> pd.DataFrame:
        if is_in_splits:
            dataset_dir_path = self._find_split_dir(friendly_name)
            dataset = self._load_split_component(dataset_dir_path)
        else:
            dataset_dir_path = self._find_dataset_dir(friendly_name)
            self._ensure_prepared_dataset(dataset_dir_path)
            logging.debug(f"Loading prepared dataset `{friendly_name}`")
            dataset = self._load_prepared_dataset(dataset_dir_path)
        return dataset

    def get_split_friendly_names(self, cache_key: str) -> Tuple[str, str]:
        """Return the (train, test) friendly names of a cached split."""
        params_paths = {
            which: self.splits_dir / cache_key / which / self.params_filename
            for which in ("train", "test")
        }
        for which, path in params_paths.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"No {self.params_filename} found for {which} split with cache_key `{cache_key}`"
                )
        names = []
        for which, path in params_paths.items():
            friendly_name = load_yaml(path).get("friendly_name")
            if not friendly_name:
                raise RuntimeError(
                    f"No `friendly_name` found in {which} {self.params_filename} for split with cache_key `{cache_key}`"
                )
            names.append(friendly_name)
        return names[0], names[1]

    def _split_dir_is_complete(self, split_dir: Path) -> bool:
        return all(
            (split_dir / which / name).exists()
            for which in ("train", "test")
            for name in (self.split_datafile_name, self.params_filename)
        )

    def save_train_test_split(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        cache_key: str,
        split_friendly_name: str,
        classification_or_regression: str,
    ) -> None:
        """Write the split to a staging directory, then publish it with one rename.

        Other processes never see a partial split. If another process published it first,
        theirs is kept unless `override_cache` is set.
        """
        final_dir = self.splits_dir / cache_key
        staging_dir = make_unique_dir(self.staging_dir, "split")

        def _save_component(df: pd.DataFrame, which: str) -> None:
            component_dir = staging_dir / which
            atomic_to_csv(df, component_dir / self.split_datafile_name)
            params = {
                "friendly_name": f"{classification_or_regression[:3]}_{which}_{split_friendly_name}",
                "raw_or_derived": "derived",
                "task": classification_or_regression,
            }
            atomic_yaml_dump(params, component_dir / self.params_filename)

        try:
            _save_component(train_df, "train")
            _save_component(test_df, "test")
            self.dump_gin_config(staging_dir)
            self.dump_logs(staging_dir)
            published = publish_dir(
                staging_dir,
                final_dir,
                trash_parent=self.staging_dir,
                is_complete=self._split_dir_is_complete,
                overwrite=self.override_cache,
                lock=self.split_lock(cache_key),
            )
        finally:
            # No-op after a successful publish.
            shutil.rmtree(staging_dir, ignore_errors=True)

        if published:
            logging.info(f"Saved split at `{final_dir}`")
            self.update_splits_registry()

    def check_train_test_split_exists(self, cache_key: str) -> bool:
        return self._split_dir_is_complete(self.splits_dir / cache_key)

    def load_train_test_split(
        self, cache_key: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return the (train, test) data of the split saved under `cache_key`."""
        if not self.check_train_test_split_exists(cache_key):
            raise FileNotFoundError(
                f"No complete split with cache_key `{cache_key}` in `{self.splits_dir}`"
            )
        return tuple(
            self._load_split_component(self.splits_dir / cache_key / which)
            for which in ("train", "test")
        )

    # --- models saving/loading ---
    # Files are written atomically.
    def _model_dir(self, model_cache_key: str, data_cache_key: str) -> Path:
        return self.models_dir / model_cache_key / data_cache_key

    def pickle_model(
        self,
        model: PredictorBase,
        model_cache_key: str,
        data_cache_key: str,
        save_as_refit: bool = False,
    ) -> None:
        filename = self.model_refit_filename if save_as_refit else self.model_filename
        path = self._model_dir(model_cache_key, data_cache_key) / filename
        logging.info(f"Saving trained model to `{path}`")
        atomic_pickle_dump(model, path)

    def save_model_metadata(
        self,
        metadata: Dict,
        model_cache_key: str,
        data_cache_key: str,
    ) -> None:
        path = (
            self._model_dir(model_cache_key, data_cache_key)
            / self.model_metadata_filename
        )
        atomic_yaml_dump(metadata, path)

    def check_model_exists(
        self, model_cache_key: str, data_cache_key: str, refit: bool = False
    ) -> bool:
        filename = self.model_refit_filename if refit else self.model_filename
        return (self._model_dir(model_cache_key, data_cache_key) / filename).exists()

    def unpickle_model(
        self, model_cache_key: str, data_cache_key: str, refit: bool = False
    ) -> PredictorBase:
        """Load the model trained on the train split, or with `refit` the one refit on train + test."""
        filename = self.model_refit_filename if refit else self.model_filename
        path = self._model_dir(model_cache_key, data_cache_key) / filename
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")
        return load_pickle(path)

    def save_metrics(
        self, metrics: Dict, model_cache_key: str, data_cache_key: str
    ) -> None:
        path = (
            self._model_dir(model_cache_key, data_cache_key)
            / self.model_metrics_filename
        )
        atomic_yaml_dump(metrics, path)

    def save_hyperparams(
        self, params: Dict, model_cache_key: str, data_cache_key: str
    ) -> None:
        path = (
            self.models_dir
            / model_cache_key
            / data_cache_key
            / self.model_params_filename
        )
        atomic_yaml_dump(params, path)

    def load_hyperparams(self, model_cache_key: str, data_cache_key: str) -> Dict:
        path = (
            self.models_dir
            / model_cache_key
            / data_cache_key
            / self.model_params_filename
        )
        if not path.exists():
            raise FileNotFoundError(f"Model params file not found at {path}")
        return load_yaml(path)

    # --- applicability domains ---
    # A fitted domain depends only on the split, so all its models share it.
    def _domain_path(
        self, data_cache_key: str, domain_cache_key: str, refit: bool
    ) -> Path:
        filename = self.domain_refit_filename if refit else self.domain_filename
        return (
            self.splits_dir
            / data_cache_key
            / self.domain_dirname
            / domain_cache_key
            / filename
        )

    def check_domain_exists(
        self, data_cache_key: str, domain_cache_key: str, refit: bool = False
    ) -> bool:
        return self._domain_path(data_cache_key, domain_cache_key, refit).exists()

    def save_domain(self, domain, data_cache_key: str, refit: bool = False) -> None:
        """Save a fitted domain of `model.pkl`, or with `refit` of `model_final_refit.pkl`."""
        path = self._domain_path(data_cache_key, domain.get_cache_key(), refit)
        logging.info(f"Saving applicability domain to `{path}`")
        atomic_pickle_dump(domain, path)

    def load_domain(
        self, data_cache_key: str, domain_cache_key: str, refit: bool = False
    ):
        path = self._domain_path(data_cache_key, domain_cache_key, refit)
        if not path.exists():
            raise FileNotFoundError(f"Applicability domain not found at {path}")
        return load_pickle(path)

    def _domain_calibration_path(
        self, model_cache_key: str, data_cache_key: str, domain_cache_key: str
    ) -> Path:
        return (
            self._model_dir(model_cache_key, data_cache_key)
            / self.domain_dirname
            / domain_cache_key
            / self.domain_calibration_filename
        )

    def save_domain_calibration(
        self,
        calibration: Dict,
        model_cache_key: str,
        data_cache_key: str,
        domain_cache_key: str,
    ) -> None:
        path = self._domain_calibration_path(
            model_cache_key, data_cache_key, domain_cache_key
        )
        atomic_yaml_dump(calibration, path)

    def load_domain_calibration(
        self, model_cache_key: str, data_cache_key: str, domain_cache_key: str
    ) -> Dict:
        """Return the calibration report of a model's domain, or {} if there is none."""
        path = self._domain_calibration_path(
            model_cache_key, data_cache_key, domain_cache_key
        )
        return load_yaml(path, missing_ok=True)

    def dump_training_logs(self, model_cache_key: str, data_cache_key: str) -> None:
        path = self._model_dir(model_cache_key, data_cache_key) / "training_log"
        self.dump_logs(path)

    def dump_gin_config_to_model_dir(
        self, model_cache_key: str, data_cache_key: str
    ) -> None:
        self.dump_gin_config(self._model_dir(model_cache_key, data_cache_key))

    # --- registries ---
    # The lock spans scan and write, so a slow process cannot write a stale listing.
    def update_registries(self) -> None:
        self.update_datasets_registry()
        self.update_splits_registry()

    def update_datasets_registry(self) -> None:
        with self._locked("registry:datasets"):
            names = []
            for yaml_path in iter_files(self.dataset_dir, ".yaml"):
                data = safe_load_yaml(yaml_path)
                if data.get("friendly_name"):
                    names.append(data["friendly_name"])
            atomic_write_text(
                "".join(f"{n}\n" for n in names),
                self.dataset_dir / self.registry_filename,
            )

    def update_splits_registry(self) -> None:
        with self._locked("registry:splits"):
            splits: List[_Split] = []
            for yaml_path in iter_files(self.splits_dir, ".yaml"):
                fname = safe_load_yaml(yaml_path).get("friendly_name")
                ts = read_first_line(yaml_path.parent.parent / self.console_log_filename)
                if fname:
                    splits.append(_Split(fname, ts))
            splits.sort(key=lambda s: s.timestamp)
            atomic_write_text(
                "".join(f"{s.timestamp} {s.friendly_name}\n" for s in splits),
                self.splits_dir / self.registry_filename,
            )

    # --- classification helpers ---
    def _parse_classification_threshold(
        self, dataset_dir_path: Path, _visited: Tuple[Path, ...] = ()
    ) -> float | str:
        """Return the dataset's `threshold`, or the one of its `threshold_source`.

        A source's `"median"` threshold resolves to the median of the source's own
        transformed labels, i.e. the value used to classify the source itself.
        """
        cfg = self._read_data_config(dataset_dir_path)
        if not cfg:
            raise RuntimeError(
                "No data config yaml found to parse threshold for classification task."
            )
        source_name = cfg.get("threshold_source")
        if source_name:
            source_dir = self._find_dataset_dir(source_name)
            visited = (*_visited, dataset_dir_path.resolve())
            if source_dir.resolve() in visited:
                raise RuntimeError(
                    f"Cyclic `threshold_source` reference to `{source_name}` in `{dataset_dir_path}`."
                )
            threshold = self._parse_classification_threshold(source_dir, visited)
            if threshold == "median":
                source_df = self._load_transformed_raw_df(source_dir)
                threshold = float(source_df["y"].median())
            logging.info(f"Using threshold {threshold} of `threshold_source` `{source_name}`.")
            return threshold
        threshold = cfg.get("threshold")
        if threshold is None:
            raise RuntimeError(
                f"No `threshold` or `threshold_source` in data config of `{dataset_dir_path}`."
            )
        return threshold

    def _clean_relation_series(self, df: pd.DataFrame) -> pd.Series:
        if "Standard Relation" in df.columns:
            return (
                df["Standard Relation"]
                .astype(str)
                .str.replace(r"^['\"]|['\"]$", "", regex=True)
                .str.strip()
            )
        return pd.Series([""] * len(df), index=df.index)

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
            logging.info(f"Using value {threshold_value} as threshold.")

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

        # Keep the continuous label in `y_cont`.
        return df.rename(columns={"y": "y_cont", "class": "y"})

    def save_bootstrap_metrics(self, metrics_dict, predictor_key, split_key):
        path = self._model_dir(predictor_key, split_key) / "bootstrap_data.yaml"
        atomic_yaml_dump(metrics_dict, path)
