from __future__ import annotations

import gin
import pandas as pd
from datetime import datetime
import logging
from typing import List, Optional, Tuple
import time
import optuna
from tqdm import tqdm
from copy import deepcopy
import numpy as np

from src.data.data_interface import DataInterface
from src.data.featurizer import FeaturizerBase
from src.utils import sample_optuna_params
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
    """

    def __init__(
        self,
        # Execution flags
        do_load_datasets: bool,
        do_load_train_test: bool,
        do_dump_train_test: bool,
        do_load_optimized_hyperparams: bool,
        do_optimize_hyperparams: bool,
        do_train_model: bool,
        do_get_metrics_confidence_interval: bool,
        do_save_trained_model: bool,
        do_refit_final_model: bool,
        # Core components
        data_interface: DataInterface,
        predictor: PredictorBase | None = None,
        featurizer: FeaturizerBase | None = None,
        splitter: DataSplitterBase | None = None,
        sim_filter: SimilarityFilterBase | None = None,
        hyperparams_source_sim_filter: SimilarityFilterBase | None = None,
        # Dataset configuration
        datasets: List[str] | None = None,
        manual_train_splits: List[str] | None = None,
        manual_test_splits: List[str] | None = None,
        test_origin_dataset: str | None = None,
        # Hyperparameter optimization
        params_distribution: dict | None = None,
        n_optim_cv_folds: int | None = None,
        n_optim_iter: int | None = None,
        n_optim_jobs: int = -1,
        target_metric: str | None = None,
        ci_n_bootstraps: int = 100,
        ci_percentiles: float = 95.0,
        # Other
        task_setting: str = "regression",  # "regression" or "binary_classification"
        smiles_col: str = "smiles",
        source_col: str = "source",
        target_col: str = "y",
        logfile: str | None = None,
        override_cache: bool = False,
        show_progress_bar: bool = True,
    ):
        # Execution flags
        self.do_load_datasets = do_load_datasets
        self.do_load_train_test = do_load_train_test
        self.do_dump_train_test = do_dump_train_test
        self.do_load_optimized_hyperparams = do_load_optimized_hyperparams
        self.do_optimize_hyperparams = do_optimize_hyperparams
        self.do_train_model = do_train_model
        self.do_get_metrics_confidence_interval = do_get_metrics_confidence_interval
        self.do_save_trained_model = do_save_trained_model
        self.do_refit_final_model = do_refit_final_model

        # Core components and settings
        self.data_interface = data_interface
        self.predictor = predictor
        self.featurizer = featurizer
        self.splitter = splitter
        self.sim_filter = sim_filter
        self.hyperparams_source_sim_filter = hyperparams_source_sim_filter

        self.predictor.set_column_ids(smiles_col, source_col, target_col)
        self.predictor.set_task_name(test_origin_dataset)

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
        self.ci_n_bootstraps = ci_n_bootstraps
        self.ci_percentiles = ci_percentiles

        # Hyperparameter optimization
        self.params_distribution = params_distribution
        self.n_optim_cv_folds = n_optim_cv_folds
        self.n_optim_iter = n_optim_iter
        self.n_optim_jobs = n_optim_jobs
        self.target_metric = target_metric
        self.optuna_sampler = optuna.samplers.TPESampler(seed=42)

        # Let the data interface know global settings
        self.data_interface.set_task_setting(task_setting)
        self.data_interface.set_logfile(logfile)
        self.data_interface.set_override_cache(override_cache)

        # Inject featurizer into the model
        self.predictor.set_featurizer(self.featurizer)

        # Derived identifiers / caches
        self.split_key = self._get_split_key(self.datasets)
        self.predictor_key = self._get_predictor_key()
        self.optimized_hyperparameters = None

        # Other
        self.show_progress_bar = show_progress_bar

        # Validate configuration early
        self._validate_configuration()

    def run(self) -> None:
        # Log pipeline start
        self._log_pipeline_start()

        # Step 1: Load datasets
        augmentation_dfs, origin_df = self._load_all_datasets()

        # Step 2: Create train/test splits if requested
        train_df, test_df = pd.DataFrame(), pd.DataFrame()
        if self.do_load_train_test:
            train_df, test_df = self._get_train_test_splits(augmentation_dfs, origin_df)

            # Step 3: Save train/test splits if requested
            if self.do_dump_train_test:
                self._save_splits(train_df, test_df)

        # Update registries
        self._update_registries()

        # Step 4: Load optimized hyperparameters if requested
        if self.do_load_optimized_hyperparams:
            self._load_hyperparams_optimized_on_test_origin()

        # Step 5: Optimize hyperparameters if requested
        if self.do_optimize_hyperparams:
            self._optimize_hyperparams(train_df)

        # Step 6: Train and evaluate the model if requested
        if self.do_train_model:
            self._train(train_df)

            if self.do_get_metrics_confidence_interval:
                self._evaluate(test_df, get_CIs=True)
            else:
                self._evaluate(test_df)

            # Pickle the trained model if requested
            if self.do_save_trained_model:
                self._pickle_trained_model(as_refit=False)

            # Step 9: Refit final model on full dataset if requested
            if self.do_refit_final_model:
                self._train_final_model(train_df, test_df)

                # Pickle the refitted model if requested
                if self.do_save_trained_model:
                    self._pickle_trained_model(as_refit=True)

            self._dump_training_info()

    # --------------------- Dataset loading & visualization --------------------- #

    def _load_datasets(self, friendly_names: List[str]) -> List[pd.DataFrame]:
        """Load datasets by friendly name and add a `source` column for provenance."""
        if not friendly_names:
            return []

        dfs: List[pd.DataFrame] = []
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

        dfs: List[pd.DataFrame] = []
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
            f"\nLoaded {len(augmentation_dfs)} augmentation datasets: {augmentation_names}"
        )

        origin_df: Optional[pd.DataFrame] = None
        if self.test_origin_dataset:
            origin_list = self._load_datasets([self.test_origin_dataset])
            origin_df = origin_list[0] if origin_list else self._empty_dataframe()
            logging.info(f"Loaded split-origin dataset: {self.test_origin_dataset}")

        return augmentation_dfs, origin_df

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
            if (
                self._check_if_already_splitted(self.split_key)
                and not self.override_cache
            ):
                logging.info(
                    f"Train/test split with key {self.split_key} already exists; loading from cache"
                )
                train_friendly_name, test_friendly_name = (
                    self.data_interface.get_split_friendly_names(self.split_key)
                )
                return self._get_cached_splits(
                    [train_friendly_name], [test_friendly_name]
                )
            # Create automatic splits
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
            f"Split origin into train={len(split_train_df)}, test={len(split_test_df)}"
        )

        augmentation_df = self._aggregate_dataframes(
            augmentation_dfs, empty_if_none=True
        )
        logging.info(f"Aggregated {len(augmentation_df)} augmentation samples")

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
                logging.info(f"Source {src}: {pre} -> {post} samples after filtering")
            logging.info(f"After filtering: train={len(train_df)}, test={len(test_df)}")
            return train_df, test_df

        # No filtering: concatenate augmentation with split train
        train_df = pd.concat([split_train_df, augmentation_df], ignore_index=True)
        logging.info(f"No filtering applied. Combined train={len(train_df)}")
        return train_df, split_test_df

    def _get_cached_splits(
        self, train_friendly_names: List[str], test_friendly_names: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and aggregate DataFrames listed in manual split lists."""
        train_dfs = (
            self._load_split_datasets(train_friendly_names)
            if train_friendly_names
            else []
        )
        test_dfs = (
            self._load_split_datasets(test_friendly_names)
            if test_friendly_names
            else []
        )

        train = self._aggregate_dataframes(train_dfs, empty_if_none=True)
        test = self._aggregate_dataframes(test_dfs, empty_if_none=True)

        logging.info(f"Cached split loaded: train={len(train)}, test={len(test)}")
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
        logging.info(f"Saved split with cache key: {self.split_key}")

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
        except Exception as exc:
            logging.exception(f"Failed to update registries: {exc}")

    def _log_pipeline_start(self) -> None:
        """Log initial pipeline configuration for easy debugging."""
        logging.info(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        self.log_bar(w_text="ADMET-XSpec")
        logging.info("Configuration summary:")
        logging.info(f"* Datasets: {self.datasets}")
        if self.splitter:
            logging.info(f"* Splitter: {self.splitter.name}")
        if self.sim_filter:
            logging.info(
                f"* Filtering aug. data with {self.sim_filter.name} against {self.sim_filter.against}"
            )
        if self.featurizer:
            logging.info(f"* Featurizer: {self.featurizer.name}")
            logging.info(f"* Split key: {self.split_key}")
        if self.predictor:
            logging.info(f"* Predictor: {self.predictor.name}")
            logging.info(f"* Predictor key: {self.predictor_key}")
        if self.hyperparams_source_sim_filter:
            logging.info(
                f"* Loading hyperparams filtered by {self.hyperparams_source_sim_filter.name} against {self.hyperparams_source_sim_filter.against}"
            )
        logging.info(f"* Task setting: {self.task_setting}")

    # --------------------- Identification / caching --------------------- #

    def _get_split_key(
        self, datasets: List[str], custom_filter: SimilarityFilterBase | None = None
    ) -> str:
        """Generate a compact, deterministic identifier for the split configuration."""
        splitter_key = self.splitter.get_cache_key() if self.splitter else "nosplit"
        if custom_filter:
            filter_key = custom_filter.get_cache_key()
        else:
            filter_key = (
                self.sim_filter.get_cache_key() if self.sim_filter else "nofilter"
            )
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

        test_origin_split_key = self._get_split_key(
            [self.test_origin_dataset], custom_filter=self.hyperparams_source_sim_filter
        )
        model_key = self.predictor.get_cache_key()
        self.optimized_hyperparameters = self.data_interface.load_hyperparams(
            model_key, test_origin_split_key
        )
        logging.warning(
            f"Loaded hyperparameters optimized previously on {self.test_origin_dataset}"
        )
        logging.warning(
            f"* Optimized hyperparameters: {self.optimized_hyperparameters}"
        )
        logging.warning(
            "This configuration will override hyperparameters provided in the predictor config file!"
        )
        # Inject loaded hyperparameters into predictor
        self.predictor.set_hyperparameters(self.optimized_hyperparameters)

    def _train(self, train_df: pd.DataFrame) -> None:
        """Train the predictor and persist model + hyperparams."""

        logging.info(f"* Training {self.predictor.__class__.__name__}")
        logging.info(f"* Dataset size: {len(train_df)}")
        logging.debug(f"* Hyperparameters: {self.predictor.get_hyperparameters()}")

        self.predictor.train(train_df)

        if self.predictor.is_multi_endpoint:
            logging.info(f"* Endpoint OHE encoding:")
            for key, value in self.predictor.get_endpoint_OHE_map().items():
                logging.info(f"  - {key}: {np.array2string(value.astype(int))}")

        # Save hyperparameters
        hyperparams = self.predictor.get_hyperparameters()
        self.data_interface.save_hyperparams(
            hyperparams, self.predictor_key, self.split_key
        )

        # Save model metadata
        metadata_dict = {
            "Datasets": self.datasets,
            "Test Origin Dataset": self.test_origin_dataset,
            "Training set size": len(train_df),
            "Training set sources": get_label_counts(train_df, self.source_col),
            "Task Setting": self.task_setting,
            "Splitter": self.splitter.name if self.splitter else "None",
            "Similarity Filter": self.sim_filter.name if self.sim_filter else "None",
            "Featurizer": self.featurizer.name if self.featurizer else "None",
            "Predictor": self.predictor.name,
            "Optimized Hyperparameters": self.do_optimize_hyperparams,
        }

        if self.predictor.is_multi_endpoint:
            metadata_dict["Endpoints"] = self.predictor.get_endpoint_OHE_map()

        self.data_interface.save_model_metadata(
            metadata_dict, self.predictor_key, self.split_key
        )

    def _pickle_trained_model(self, as_refit=False) -> None:
        self.data_interface.pickle_model(
            self.predictor, self.predictor_key, self.split_key, save_as_refit=as_refit
        )

    def _evaluate(self, test_df: pd.DataFrame, get_CIs=False) -> None:
        self.log_bar(w_text="Evaluation")

        # Evaluate the model on a holdout test set
        metrics = self.predictor.evaluate(test_df)
        logging.info("Metrics (markdown):")
        log_markdown_table(metrics)

        if get_CIs:
            # Estimate confidence intervals for metrics using bootstrapping
            logging.info(f"* Estimating confidence intervals")
            logging.info(
                f"* Bootstrap parameters: n={self.ci_n_bootstraps}, percentiles={self.ci_percentiles}"
            )
            ci_lower, ci_upper = self._estimate_confidence_intervals(test_df)
            confidence_intervals = {
                metric: [ci_lower[metric], ci_upper[metric]]
                for metric, value in metrics.items()
            }
            logging.info(f"\nConfidence intervals (markdown):")
            logging.info("* lower")
            log_markdown_table(ci_lower)
            logging.info("* upper")
            log_markdown_table(ci_upper)

            metrics_dict = {
                "metrics": metrics,
                "percentile_95_ci": confidence_intervals,
            }
        else:
            metrics_dict = {"metrics": metrics}

        self.data_interface.save_metrics(
            metrics_dict, self.predictor_key, self.split_key
        )

    def _estimate_confidence_intervals(
        self, test_df: pd.DataFrame
    ) -> Tuple[dict, dict]:
        """Estimate confidence intervals for evaluation metrics using bootstrapping."""
        # sample with replacement from test_df, evaluate on each bootstrap sample, compute statistics
        metrics_list = []
        for _ in tqdm(
            range(self.ci_n_bootstraps), disable=(not self.show_progress_bar)
        ):
            bootstrap_sample = test_df.sample(frac=1.0, replace=True)
            metrics = self.predictor.evaluate(bootstrap_sample)
            metrics_list.append(metrics)
        # Compute confidence intervals (percentile) for each metric
        metrics = pd.DataFrame(metrics_list)
        ci_lower = metrics.quantile((100 - self.ci_percentiles) / 200)
        ci_upper = metrics.quantile(1 - (100 - self.ci_percentiles) / 200)
        return ci_lower.to_dict(), ci_upper.to_dict()

    def _train_final_model(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Retrain the predictor on the combined train+test set and save as refit."""
        if not self.predictor:
            raise ValueError("No predictor configured; cannot retrain final model")

        logging.info("* Retraining the final model on the full dataset (train + test)")
        full_df = pd.concat([train_df, test_df], ignore_index=True)
        self.predictor.train(full_df)

    def _optimize_hyperparams(self, train_df: pd.DataFrame) -> None:
        """Optimize hyperparameters of the predictor."""
        self.log_bar()
        logging.info(
            f"\nOptimizing hyperparameters with Optuna for target metric: {self.target_metric}"
        )
        logging.info(f"* Optimization CV folds: {self.n_optim_cv_folds}")
        logging.info(f"* Optimization iterations: {self.n_optim_iter}")
        logging.info(f"* Hyperparameter search space: {self.params_distribution}")

        # TODO: move this somewhere else
        _METRICS_TO_MAXIMIZE = {
            "roc_auc",
            "accuracy",
            "f1",
            "precision",
            "recall",
            "r2",
        }
        _METRICS_TO_MINIMIZE = {"rmse", "mae", "mse"}
        if self.target_metric in _METRICS_TO_MAXIMIZE:
            direction = "maximize"
        elif self.target_metric in _METRICS_TO_MINIMIZE:
            direction = "minimize"
        else:
            raise ValueError(
                f"Unknown target metric {self.target_metric} for optimization"
            )

        # Define an objective funcion for the optimization process
        # Here it is a k-fold cross-validation on some target metric
        def objective(trial: optuna.Trial) -> float:
            # Instantiate a fresh predictor for the trial
            trial_predictor = deepcopy(self.predictor)
            # Inject hyperparams
            params = sample_optuna_params(trial, self.params_distribution)
            logging.debug(f"Trial {trial.number} sampled hyperparameters: {params}")
            trial_predictor.set_hyperparameters(params)
            trial_predictor.train(train_df)
            # Evaluate with cross-validation
            scores = trial_predictor.cross_validate(
                train_df, n_folds=self.n_optim_cv_folds
            )
            mean_score = scores[self.target_metric].mean()
            return mean_score

        start_time = time.time()

        # Set up optuna study and run optimization
        study = optuna.create_study(
            direction=direction,
            study_name="_".join([self.predictor_key, self.split_key]),
            sampler=self.optuna_sampler,
        )
        study.optimize(objective, n_trials=self.n_optim_iter, n_jobs=self.n_optim_jobs)
        elapsed_time = time.time() - start_time
        logging.info(
            f"Hyperparameter optimization completed in {elapsed_time/60:.2f} minutes"
        )
        logging.info(f"* Retaining best hyperparams:")
        logging.info(study.best_params)

        # Retain best hyperparameters
        self.predictor.set_hyperparameters(study.best_params)

    def _dump_training_info(self) -> None:
        self.data_interface.dump_training_logs(self.predictor_key, self.split_key)
        self.data_interface.dump_gin_config_to_model_dir(
            self.predictor_key, self.split_key
        )

    def log_bar(self, w_text: str = "") -> None:
        """Log a horizontal bar with optional centered text for visual separation."""
        if w_text:
            w_text = " " + w_text + " "
        bar = f"\n# ================================{w_text}================================ #\n"
        logging.info(bar)

    # --------------------- Configuration validation --------------------- #

    def _validate_configuration(self) -> None:
        """
        Basic sanity checks to fail early on common misconfigurations.
        - predictor must be present for training/evaluation
        - splitter must be present for automatic splitting
        - if do_load_train_test is True ensure either automatic or manual splits exist
        """
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
