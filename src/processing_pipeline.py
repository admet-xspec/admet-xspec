import gin
import pandas as pd
from datetime import datetime
import logging
from typing import List, Optional, Tuple

from src.data.data_interface import DataInterface
from src.data.featurizer import FeaturizerBase
from src.data.reducer import ReducerBase
from src.data.visualizer import VisualizerBase
from src.data.split import DataSplitterBase
from src.data.sim_filter import SimilarityFilterBase
from src.predictor.predictor_base import PredictorBase
from src.utils import log_markdown_table
from src.data.utils import get_label_counts
import hashlib


@gin.configurable
class ProcessingPipeline:
    """
    Orchestrates dataset loading, splitting, optional similarity filtering,
    visualization, training and evaluation.

    The design aims for:
    - Small, focused helper methods
    - Clear validation of configuration errors early
    - Minimal duplication between automatic and manual split flows
    """

    def __init__(
        self,
        # Execution flags
        do_load_datasets: bool,
        do_visualize_datasets: bool,
        do_load_train_test: bool,
        do_dump_train_test: bool,
        do_visualize_train_test: bool,
        do_load_optimized_hyperparams: bool,
        do_train_model: bool,
        do_train_optimize: bool,
        do_refit_final_model: bool,
        # Core components
        data_interface: DataInterface,
        predictor: Optional[PredictorBase] = None,
        featurizer: Optional[FeaturizerBase] = None,
        reducer: Optional[ReducerBase] = None,
        splitter: Optional[DataSplitterBase] = None,
        sim_filter: Optional[SimilarityFilterBase] = None,
        # Dataset configuration
        datasets: Optional[List[str]] = None,
        manual_train_splits: Optional[List[str]] = None,
        manual_test_splits: Optional[List[str]] = None,
        test_origin_dataset: Optional[str] = None,
        # Other
        task_setting: str = "regression",
        smiles_col: str = "smiles",
        source_col: str = "source",
        target_col: str = "y",
        logfile: Optional[str] = None,
        override_cache: bool = False
    ):
        # Execution flags
        self.do_load_datasets = do_load_datasets
        self.do_visualize_datasets = do_visualize_datasets
        self.do_load_train_test = do_load_train_test
        self.do_dump_train_test = do_dump_train_test
        self.do_visualize_train_test = do_visualize_train_test
        self.do_load_optimized_hyperparams = do_load_optimized_hyperparams
        self.do_train_optimize = do_train_optimize
        self.do_train_model = do_train_model
        self.do_refit_final_model = do_refit_final_model

        # Core components and settings
        self.data_interface = data_interface
        self.predictor = predictor
        self.featurizer = featurizer
        self.reducer = reducer
        self.splitter = splitter
        self.sim_filter = sim_filter

        # Dataset & column config
        self.datasets = datasets or []
        self.manual_train_splits = manual_train_splits or []
        self.manual_test_splits = manual_test_splits or []
        self.test_origin_dataset = test_origin_dataset
        self.task_setting = task_setting
        self.smiles_col = smiles_col
        self.source_col = source_col
        self.target_col = target_col
        self.logfile = logfile
        self.override_cache = override_cache

        # Let the data interface know global settings
        self.data_interface.set_task_setting(task_setting)
        self.data_interface.set_logfile(logfile)
        self.data_interface.set_override_cache(override_cache)

        # If the predictor object does not implement internal featurization,
        # inject the configured featurizer
        if self.predictor and not self.predictor.uses_internal_featurizer:
            self.predictor.set_featurizer(self.featurizer)

        # Derived identifiers / caches
        self.split_key = self._get_split_key(self.datasets)
        self.predictor_key = self._get_predictor_key()
        self.optimized_hyperparameters = None

        # Validate configuration early
        self._validate_configuration()

    def run(self) -> None:

        # Log pipeline start
        self._log_pipeline_start()

        # Step 1: Load datasets
        augmentation_dfs, origin_df = self._load_all_datasets()

        # Step 2: Visualize raw datasets if requested
        if self.do_visualize_datasets:
            self._visualize_raw_datasets(augmentation_dfs, origin_df)

        # Step 3: Create train/test splits if requested
        train_df, test_df = pd.DataFrame(), pd.DataFrame()
        if self.do_load_train_test:
            train_df, test_df = self._get_train_test_splits(augmentation_dfs, origin_df)

            # Step 4: Save train/test splits if requested
            if self.do_dump_train_test:
                self._save_splits(train_df, test_df)

            # Step 5: Visualize train/test splits if requested
            if self.do_visualize_train_test:
                self._visualize_splits(train_df, test_df)

        # Update registries regardless of train/test decisions
        self._update_registries()

        # Step 6: Load optimized hyperparameters if requested
        if self.do_load_optimized_hyperparams:
            self._load_hyperparams_optimized_on_test_origin()

        # Step 7: Train and evaluate the model if requested
        if self.do_train_model:
            self._train(train_df, optimize=self.do_train_optimize)
            self._evaluate(test_df)

            # Step 8: Refit final model on full dataset if requested
            if self.do_refit_final_model:
                self._train_final_model(train_df, test_df)

            self._dump_training_info()

    # --------------------- Dataset loading & visualization --------------------- #

    def _load_datasets(self, friendly_names: List[str]) -> List[pd.DataFrame]:
        """Load datasets by friendly name and add a `source` column for provenance."""
        if not friendly_names:
            return []

        dfs = []
        for name in friendly_names:
            df = self.data_interface.get_by_friendly_name(name)
            df = df[[self.smiles_col, self.target_col]].copy()
            df[self.source_col] = name
            dfs.append(df)
        return dfs

    def _load_split_datasets(self, friendly_names: List[str]) -> List[pd.DataFrame]:
        """Load datasets by friendly name for split datasets (no source column added)."""
        if not friendly_names:
            return []

        dfs = []
        for name in friendly_names:
            df = self.data_interface.get_by_friendly_name(name, is_in_splits=True)
            df = df[[self.smiles_col, self.target_col, self.source_col]].copy()
            dfs.append(df)
        return dfs

    def _load_all_datasets(self) -> Tuple[List[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load configured datasets and separate augmentation vs origin (if any)."""
        if not self.do_load_datasets:
            return [], None

        if not self.datasets:
            raise ValueError("do_load_datasets=True but `datasets` is empty")

        augmentation_names = [n for n in self.datasets if n != self.test_origin_dataset]
        augmentation_dfs = self._load_datasets(augmentation_names)
        logging.info(
            "Loaded %d augmentation datasets: %s",
            len(augmentation_dfs),
            augmentation_names,
        )

        origin_df = None
        if self.test_origin_dataset:
            origin_list = self._load_datasets([self.test_origin_dataset])
            origin_df = origin_list[0] if origin_list else self._empty_dataframe()
            logging.info("Loaded split-origin dataset: %s", self.test_origin_dataset)

        return augmentation_dfs, origin_df

    def _visualize_raw_datasets(
        self, augmentation_dfs: List[pd.DataFrame], origin_df: Optional[pd.DataFrame]
    ) -> None:
        """Featurize and visualize loaded datasets before splitting (if reducer is available)."""
        if not self.reducer:
            return

        dataset_names = [n for n in self.datasets if n != self.test_origin_dataset]
        dfs = []

        if augmentation_dfs:
            dfs.extend(self._featurize_datasets(augmentation_dfs))

        if origin_df is not None:
            dfs.extend(self._featurize_datasets([origin_df]))
            dataset_names.append(self.test_origin_dataset)

        if dfs:
            self._visualize_and_save(dfs, dataset_names, "datasets")

    def _visualize_splits(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Featurize (if configured) and visualize train/test splits."""
        if not self.reducer:
            return

        train_feat = (
            self._featurize_datasets([train_df])[0] if self.featurizer else train_df
        )
        test_feat = (
            self._featurize_datasets([test_df])[0] if self.featurizer else test_df
        )
        self._visualize_and_save([train_feat, test_feat], ["train", "test"], "split")

    def _visualize_and_save(
        self, dfs: List[pd.DataFrame], names: List[str], suffix: str
    ) -> None:
        """Delegate to reducer/visualizer to create an image and persist it."""
        if not self.reducer:
            return

        visualizer: VisualizerBase = self.reducer.get_associated_visualizer()
        df_map = {name: df for name, df in zip(names, dfs)}
        img = visualizer.get_visualization(df_map)
        timestamp = datetime.now().strftime("%d_%H_%M_%S")
        save_path = self.data_interface.save_visualization(f"{timestamp}_{suffix}", img)
        img.save(save_path)
        logging.info("Saved visualization to: %s", save_path)

    # --------------------- Splitting logic --------------------- #

    def _get_train_test_splits(
        self, augmentation_dfs: List[pd.DataFrame], origin_df: Optional[pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/test splits.
        - If `datasets` is set the pipeline assumes automatic splitting from `test_origin_dataset`.
        - Otherwise manual splits are used from `manual_train_splits` / `manual_test_splits`.
        """
        if self.datasets:
            # Check for cached automatic splits and load if available
            if self._check_if_already_splitted(self.split_key) and not self.override_cache:
                logging.info(
                    "Train/test split with key %s already exists; loading from cache",
                    self.split_key,
                )
                train_friendly_name, test_friendly_name = (
                    self.data_interface.get_split_friendly_names(self.split_key)
                )

                return self._get_cached_splits(
                    [train_friendly_name], [test_friendly_name]
                )
            # Create automatic splits
            else:
                return self._create_automatic_splits(augmentation_dfs, origin_df)
        # Read manual splits
        return self._get_cached_splits(
            self.manual_train_splits, self.manual_test_splits
        )

    def _create_automatic_splits(
        self, augmentation_dfs: List[pd.DataFrame], origin_df: Optional[pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Automatic split: split origin, optionally filter augmentations, combine for training."""

        split_train_df, split_test_df = self._split_dataset(origin_df)
        logging.info(
            "Split origin into train=%d, test=%d",
            len(split_train_df),
            len(split_test_df),
        )

        augmentation_df = self._aggregate_dataframes(
            augmentation_dfs, empty_if_none=True
        )
        logging.info("Aggregated %d augmentation samples", len(augmentation_df))

        if self.sim_filter:
            combined_pre = pd.concat(
                [augmentation_df, split_train_df], ignore_index=True
            )
            pre_counts = get_label_counts(combined_pre, self.source_col)
            train_df, test_df = self.sim_filter.get_filtered_train_test(
                augmentation_df, split_train_df, split_test_df
            )
            post_counts = get_label_counts(train_df, self.source_col)
            for src, pre in pre_counts.items():
                if src == self.test_origin_dataset:
                    continue  # Skip origin dataset
                post = post_counts.get(src, 0)
                logging.info("Source '%s': %d -> %d after filtering", src, pre, post)
            logging.info(
                "After filtering: train=%d, test=%d", len(train_df), len(test_df)
            )
            return train_df, test_df

        # No filtering: concatenate augmentation with split train
        train_df = pd.concat([split_train_df, augmentation_df], ignore_index=True)
        logging.info("No filtering applied. Combined train=%d", len(train_df))
        return train_df, split_test_df

    def _get_cached_splits(
        self, train_fiendly_names: List[str], test_firendly_names: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and aggregate DataFrames listed in manual split lists."""
        train_dfs = (
            self._load_split_datasets(train_fiendly_names) if train_fiendly_names else []
        )
        test_dfs = (
            self._load_split_datasets(test_firendly_names) if test_firendly_names else []
        )

        train = self._aggregate_dataframes(train_dfs, empty_if_none=True)
        test = self._aggregate_dataframes(test_dfs, empty_if_none=True)

        logging.info("Caches split loaded: train=%d, test=%d", len(train), len(test))
        return train, test

    def _save_splits(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Persist train/test split using the data interface."""
        friendly = (
            self.splitter.get_friendly_name(self.datasets)
            if self.splitter
            else "manual_split"
        )
        self.data_interface.save_train_test_split(
            train_df,
            test_df,
            cache_key=self.split_key,
            split_friendly_name=friendly,
            classification_or_regression=self.task_setting,
        )
        logging.info("Saved split with cache key: %s", self.split_key)

    def _check_if_already_splitted(self, split_key: str) -> bool:
        """Check if a train/test split with the given key already exists."""
        return self.data_interface.check_train_test_split_exists(split_key)

    def _split_dataset(
        self, df: Optional[pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Use configured splitter to split a single DataFrame; return empty frames for empty input."""
        if df is None or df.empty:
            empty = self._empty_dataframe()
            return empty, empty

        if not self.splitter:
            raise ValueError("No splitter configured for automatic splitting")

        X_train, X_test, y_train, y_test = self.splitter.split(
            df[self.smiles_col], df[self.target_col]
        )

        train_df = pd.DataFrame(
            {
                self.smiles_col: X_train,
                self.target_col: y_train,
                self.source_col: df.loc[X_train.index, self.source_col],
            }
        )
        test_df = pd.DataFrame(
            {
                self.smiles_col: X_test,
                self.target_col: y_test,
                self.source_col: df.loc[X_test.index, self.source_col],
            }
        )
        return train_df, test_df

    # --------------------- Small helpers --------------------- #

    def _featurize_datasets(
        self, dataset_dfs: List[pd.DataFrame]
    ) -> List[pd.DataFrame]:
        """Apply featurizer to each DataFrame, preserving row counts and adding feature column."""
        if not self.featurizer:
            return dataset_dfs

        feature_name = self.featurizer.feature_name
        featurized = []
        for df in dataset_dfs:
            df_copy = df.copy()
            # featurize expects list input; keep per-row mapping for clarity
            df_copy[feature_name] = df_copy[self.smiles_col].apply(
                lambda s: self.featurizer.featurize([s])
            )
            if len(df_copy) != len(df):
                raise RuntimeError(
                    f"Featurization failed: {len(df) - len(df_copy)} samples lost using {self.featurizer.name}"
                )
            featurized.append(df_copy)
        return featurized

    def _aggregate_dataframes(
        self, dfs: List[pd.DataFrame], empty_if_none: bool = False
    ) -> pd.DataFrame:
        """Concatenate multiple DataFrames, return empty standardized frame if requested."""
        if not dfs:
            return self._empty_dataframe() if empty_if_none else pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    def _empty_dataframe(self) -> pd.DataFrame:
        """Return an empty DataFrame with the pipeline's expected columns."""
        return pd.DataFrame(columns=[self.smiles_col, self.target_col, self.source_col])

    def _update_registries(self) -> None:
        """Attempt to update back-end registries; log exceptions rather than failing the whole run."""
        try:
            self.data_interface.update_registries()
            logging.info("Successfully updated data registries")
        except Exception as exc:
            logging.exception("Failed to update registries: %s", exc)

    def _log_pipeline_start(self) -> None:
        """Log initial pipeline configuration for easy debugging."""
        logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M%-S"))
        logging.info("# =================== Processing Pipeline ==================== #")
        logging.info("Starting processing pipeline with datasets: %s", self.datasets)
        if self.splitter:
            logging.info("Using splitter: %s", self.splitter.name)
        if self.sim_filter:
            logging.info(
                "Filtering augmented datasets with: %s against %s",
                self.sim_filter.name,
                self.sim_filter.against,
            )
        if self.featurizer:
            logging.info("Using featurizer: %s", self.featurizer.name)
        if self.predictor:
            logging.info("Using predictor: %s", self.predictor.name)
        logging.info("Task setting: %s", self.task_setting)

    # --------------------- Identification / caching --------------------- #

    def _get_split_key(self, datasets: List[str]) -> str:
        """Generate a compact, deterministic identifier for the split configuration."""
        splitter_key = self.splitter.get_cache_key() if self.splitter else "nosplit"
        filter_key = self.sim_filter.get_cache_key() if self.sim_filter else "nofilter"
        datasets_params = (
            tuple(sorted(datasets)),
            self.test_origin_dataset,
            self.task_setting,
        )
        datasets_hash = hashlib.md5(str(datasets_params).encode()).hexdigest()[:5]
        return f"{splitter_key}_{filter_key}_{datasets_hash}"

    def _get_predictor_key(self) -> str:
        """Return predictor cache key or placeholder if missing."""
        return self.predictor.get_cache_key() if self.predictor else "nopredictor"

    # --------------------- Model training / evaluation --------------------- #

    def _load_hyperparams_optimized_on_test_origin(self) -> None:
        if not self.test_origin_dataset:
            raise ValueError(
                "test_origin_dataset must be set to load optimized hyperparameters"
            )
        test_origin_split_key = self._get_split_key([self.test_origin_dataset])
        model_key = self.predictor.get_cache_key()
        self.optimized_hyperparameters = self.data_interface.load_hyperparams(
            model_key, test_origin_split_key
        )
        logging.warning(
            "Loaded hyperparameters optimized previously on %s",
            self.test_origin_dataset,
        )
        logging.warning("Optimized hyperparameters: %s", self.optimized_hyperparameters)
        logging.warning(
            "This configuration will override hyperparameters provided in the predictor config file."
        )
        # Inject loaded hyperparameters into predictor
        self.predictor.set_hyperparameters(self.optimized_hyperparameters)

    def _train(self, train_df: pd.DataFrame, optimize: bool) -> None:
        """Train (and optionally optimize) the predictor and persist model + hyperparams."""
        X_train = train_df[self.smiles_col].tolist()
        y_train = train_df[self.target_col].tolist()

        if optimize:
            logging.info("Optimizing hyperparameters and training the model")
            self.predictor.train_optimize(X_train, y_train)
        else:
            logging.info("Training the model on set hyperparameters")
            self.predictor.train(X_train, y_train)

        # Save trained model
        self.data_interface.pickle_model(
            self.predictor, self.predictor_key, self.split_key
        )

        # Save hyperparameters
        hyperparams = self.predictor.get_hyperparameters()
        self.data_interface.save_hyperparams(
            hyperparams, self.predictor_key, self.split_key
        )

        # Save model metadata
        metadata_dict = {
            "Datasets": self.datasets,
            "Test Origin Dataset": self.test_origin_dataset,
            "Task Setting": self.task_setting,
            "Splitter": self.splitter.name if self.splitter else "None",
            "Similarity Filter": self.sim_filter.name if self.sim_filter else "None",
            "Featurizer": self.featurizer.name if self.featurizer else "None",
            "Predictor": self.predictor.name,
            "Optimized Hyperparameters": optimize
        }
        self.data_interface.save_model_metadata(metadata_dict, self.predictor_key, self.split_key)

    def _evaluate(self, test_df: pd.DataFrame) -> None:
        """Evaluate trained predictor, log metrics and persist them."""
        logging.info("Evaluating the model on test dataset")
        X_test = test_df[self.smiles_col].tolist()
        y_test = test_df[self.target_col].tolist()
        metrics = self.predictor.evaluate(X_test, y_test)
        logging.info("Evaluation metrics: %s", metrics)
        logging.info("Metrics (markdown):")
        log_markdown_table(metrics)
        self.data_interface.save_metrics(metrics, self.predictor_key, self.split_key)

    def _train_final_model(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Retrain the predictor on the combined train+test set and save as refit."""
        logging.info("Retraining the final model on the full dataset (train + test)")
        full_df = pd.concat([train_df, test_df], ignore_index=True)
        X_full = full_df[self.smiles_col].tolist()
        y_full = full_df[self.target_col].tolist()
        self.predictor.train(X_full, y_full)
        self.data_interface.pickle_model(
            self.predictor, self.predictor_key, self.split_key, save_as_refit=True
        )

    def _dump_training_info(self) -> None:
        self.data_interface.dump_training_logs(self.predictor_key, self.split_key)
        self.data_interface.dump_gin_config_to_model_dir(self.predictor_key, self.split_key)
        self.data_interface

    # --------------------- Configuration validation --------------------- #

    def _validate_configuration(self) -> None:
        """
        Basic sanity checks to fail early on common misconfigurations.
        - predictor must be present for training/evaluation
        - splitter must be present for automatic splitting
        - if do_load_train_test is True ensure either automatic or manual splits exist
        """

        # TODO: expand with more checks

        if self.do_train_model and not self.predictor:
            raise ValueError("do_train_model=True but no predictor provided")

        if self.datasets and self.do_load_train_test and not self.test_origin_dataset:
            # If datasets is present and we're creating train/test automatically, require a test_origin_dataset
            raise ValueError(
                "Automatic splitting requested but `test_origin_dataset` is not set"
            )

        if self.do_load_train_test and not (
            self.datasets or self.manual_train_splits or self.manual_test_splits
        ):
            raise ValueError(
                "do_load_train_test=True but no datasets or manual splits are configured"
            )

        # splitter requirement only when automatic splitting will actually be used
        if self.datasets and self.do_load_train_test and not self.splitter:
            raise ValueError(
                "Automatic splitting requested but no splitter is configured"
            )

        # Ensure data_interface is present
        if not self.data_interface:
            raise ValueError("ProcessingPipeline requires a valid data_interface")
