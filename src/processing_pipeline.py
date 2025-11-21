import gin
import pandas as pd
from datetime import datetime
import logging
from typing import List, Optional, Tuple, Hashable

from src.data.data_interface import DataInterface
from src.data.featurizer import FeaturizerBase
from src.data.reducer import ReducerBase
from src.data.visualizer import VisualizerBase
from src.data.split import DataSplitterBase
from src.data.sim_filter import SimilarityFilterBase
from src.utils import read_logfile
from src.data.utils import get_label_counts


@gin.configurable
class ProcessingPipeline:
    """
    Orchestrates the molecular data processing workflow.

    Handles:
    - Loading and aggregating datasets
    - Train/test splitting (automatic or manual)
    - Similarity filtering of augmentation data
    - Visualization of datasets and splits
    - Persistence of splits
    """

    def __init__(
            self,
            # Execution flags
            do_load_datasets: bool,
            do_visualize_datasets: bool,
            do_load_train_test: bool,
            do_dump_train_test: bool,
            do_visualize_train_test: bool,
            # Core components
            data_interface: DataInterface,
            featurizer: Optional[FeaturizerBase] = None,
            reducer: Optional[ReducerBase] = None,
            splitter: Optional[DataSplitterBase] = None,
            sim_filter: Optional[SimilarityFilterBase] = None,
            # Dataset configuration
            datasets: Optional[List[str]] = None,
            manual_train_splits: Optional[List[str]] = None,
            manual_test_splits: Optional[List[str]] = None,
            test_origin_dataset: Optional[str] = None,
            # Task configuration
            task_setting: str = "regression",
            smiles_col: str = "smiles",
            source_col: str = "source",
            target_col: str = "y",
            logfile: Optional[str] = None,
    ):
        # Execution flags
        self.do_load_datasets = do_load_datasets
        self.do_visualize_datasets = do_visualize_datasets
        self.do_load_train_test = do_load_train_test
        self.do_dump_train_test = do_dump_train_test
        self.do_visualize_train_test = do_visualize_train_test

        # Core components
        self.data_interface = data_interface
        self.data_interface.set_task_setting(task_setting)
        self.data_interface.set_logfile(logfile)
        self.task_setting = task_setting
        self.featurizer = featurizer
        self.reducer = reducer
        self.splitter = splitter
        self.sim_filter = sim_filter

        # Dataset configuration
        self.datasets = datasets or []
        self.manual_train_splits = manual_train_splits or []
        self.manual_test_splits = manual_test_splits or []
        self.test_origin_dataset = test_origin_dataset
        self.logfile = logfile

        # Column names
        self.smiles_col = smiles_col
        self.source_col = source_col
        self.target_col = target_col

        # Computed properties
        self.split_cache_key = self._generate_split_key()

    def run(self) -> None:
        """Execute the full processing pipeline."""
        self._log_pipeline_start()

        # Load datasets
        augmentation_dfs, origin_df = self._load_all_datasets()

        # Visualize raw datasets
        if self.do_visualize_datasets:
            self._visualize_raw_datasets(augmentation_dfs, origin_df)

        # Create train/test splits
        if self.do_load_train_test:
            train_df, test_df = self._create_train_test_splits(augmentation_dfs, origin_df)

            # Save splits
            if self.do_dump_train_test:
                self._save_splits(train_df, test_df)

            # Visualize splits
            if self.do_visualize_train_test:
                self._visualize_splits(train_df, test_df)

        # Update data registries
        self._update_registries()

    # ==================== Main Pipeline Steps ==================== #

    def _load_all_datasets(self) -> Tuple[List[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Load all configured datasets.

        Returns:
            Tuple of (augmentation_dfs, origin_df) where:
            - augmentation_dfs: List of DataFrames to augment training data
            - origin_df: DataFrame used for splitting (if applicable)
        """
        if not self.do_load_datasets:
            return [], None

        if not self.datasets:
            raise ValueError("do_load_datasets=True but `datasets` is empty")

        # Separate augmentation datasets from split-origin dataset
        augmentation_names = [
            name for name in self.datasets
            if name != self.test_origin_dataset
        ]

        augmentation_dfs = self._load_datasets(augmentation_names)
        logging.info(f"Loaded {len(augmentation_dfs)} augmentation datasets: {augmentation_names}")

        # Load split-origin dataset separately if specified
        origin_df = None
        if self.test_origin_dataset:
            origin_dfs = self._load_datasets([self.test_origin_dataset])
            origin_df = origin_dfs[0] if origin_dfs else self._empty_dataframe()
            logging.info(f"Loaded split-origin dataset: {self.test_origin_dataset}")

        return augmentation_dfs, origin_df

    def _visualize_raw_datasets(
            self,
            augmentation_dfs: List[pd.DataFrame],
            origin_df: Optional[pd.DataFrame]
    ) -> None:
        """Visualize loaded datasets before splitting."""
        if not self.reducer:
            return

        logging.info(f"Visualizing datasets: {self.datasets}")

        # Featurize all datasets
        dfs_to_visualize = []
        dataset_names = []

        if augmentation_dfs:
            dfs_to_visualize.extend(self._featurize_datasets(augmentation_dfs))
            dataset_names.extend([
                name for name in self.datasets
                if name != self.test_origin_dataset
            ])

        if origin_df is not None:
            dfs_to_visualize.extend(self._featurize_datasets([origin_df]))
            dataset_names.append(self.test_origin_dataset)

        if dfs_to_visualize:
            self._visualize_and_save(dfs_to_visualize, dataset_names, "datasets")

    def _create_train_test_splits(
            self,
            augmentation_dfs: List[pd.DataFrame],
            origin_df: Optional[pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/test splits using either automatic splitting or manual specification.

        Returns:
            Tuple of (train_df, test_df)
        """
        if self.datasets:
            # Get automatic split
            return self._create_automatic_splits(augmentation_dfs, origin_df)
        else:
            # Get manual split
            return self._create_manual_splits()

    def _create_automatic_splits(
            self,
            augmentation_dfs: List[pd.DataFrame],
            origin_df: Optional[pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create splits automatically by:
        1. Splitting the origin dataset
        2. Filtering augmentation data by similarity
        3. Combining filtered augmentation with split train data
        """
        if not self.test_origin_dataset:
            raise ValueError("test_origin_dataset must be set for automatic splitting")

        # Split the origin dataset
        split_train_df, split_test_df = self._split_dataset(origin_df)
        logging.info(f"Split origin dataset into train={len(split_train_df)}, test={len(split_test_df)}")

        # Aggregate augmentation data
        augmentation_df = self._aggregate_dataframes(
            augmentation_dfs,
            empty_if_none=True
        )
        logging.info(f"Aggregated {len(augmentation_df)} augmentation samples")

        # Apply similarity filtering
        if self.sim_filter:
            pre_filter_label_counts = get_label_counts(pd.concat([augmentation_df, split_train_df]), self.source_col)
            train_df, test_df = self.sim_filter.get_filtered_train_test(
                split_train_df,
                split_test_df,
                augmentation_df
            )
            post_filter_label_counts = get_label_counts(augmentation_df, self.source_col)
            for source_name, pre_count in pre_filter_label_counts.items():
                post_count = post_filter_label_counts.get(source_name, 0)
                logging.info(f"Source '{source_name}': {pre_count} -> {post_count} samples remaining after filtering")
            logging.info(f"After filtering: train={len(train_df)}, test={len(test_df)}")
        else:
            # No filtering - just concatenate
            train_df = pd.concat([split_train_df, augmentation_df], ignore_index=True)
            test_df = split_test_df
            logging.info(f"No filtering applied. Combined train={len(train_df)}")

        return train_df, test_df

    def _create_manual_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create splits from manually specified dataset lists."""
        train_dfs = self._load_datasets(self.manual_train_splits) if self.manual_train_splits else []
        test_dfs = self._load_datasets(self.manual_test_splits) if self.manual_test_splits else []

        train_df = self._aggregate_dataframes(train_dfs, empty_if_none=True)
        test_df = self._aggregate_dataframes(test_dfs, empty_if_none=True)

        logging.info(f"Manual splits created: train={len(train_df)}, test={len(test_df)}")
        return train_df, test_df

    def _visualize_splits(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Visualize train/test splits."""
        if not self.reducer:
            return

        # Featurize if needed
        train_feat = self._featurize_datasets([train_df])[0] if self.featurizer else train_df
        test_feat = self._featurize_datasets([test_df])[0] if self.featurizer else test_df

        self._visualize_and_save([train_feat, test_feat], ["train", "test"], "split")

    def _save_splits(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Persist train/test splits to disk."""
        split_friendly_name = (
            self.splitter.get_friendly_name(self.datasets)
            if self.splitter else "manual_split"
        )

        self.data_interface.save_train_test_split(
            train_df,
            test_df,
            subdir_name=self.split_name,
            split_friendly_name=split_friendly_name,
            classification_or_regression=self.task_setting,
            console_log=read_logfile(self.logfile),
        )
        logging.info(f"Saved split to: {self.split_name}")

    # ==================== Helper Methods ==================== #

    def _load_datasets(self, friendly_names: List[str]) -> List[pd.DataFrame]:
        """Load datasets by friendly name and add source column."""
        if not friendly_names:
            return []

        dfs = []
        for name in friendly_names:
            df = self.data_interface.get_by_friendly_name(name)
            # Keep only required columns
            df = df[[self.smiles_col, self.target_col]].copy()
            # Add source column (dataset identifier, friendly name)
            df[self.source_col] = name
            dfs.append(df)

        return dfs

    def _featurize_datasets(self, dataset_dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """Apply featurization to datasets."""
        if not self.featurizer:
            return dataset_dfs

        featurized = []
        for df in dataset_dfs:
            df_copy = df.copy()
            feature_col = self.featurizer.feature_name

            # Apply featurization
            df_copy[feature_col] = df_copy[self.smiles_col].apply(
                lambda s: self.featurizer.featurize([s])
            )

            # Validate no data loss
            if len(df_copy) != len(df):
                raise RuntimeError(
                    f"Featurization failed: {len(df) - len(df_copy)} samples lost "
                    f"using {self.featurizer.name}"
                )

            featurized.append(df_copy)

        return featurized

    def _split_dataset(self, df: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split a single dataset into train/test."""
        if df is None or df.empty:
            empty = self._empty_dataframe()
            return empty, empty

        if not self.splitter:
            raise ValueError("No splitter configured for automatic splitting")

        X_train, X_test, y_train, y_test = self.splitter.split(
            df[self.smiles_col],
            df[self.target_col]
        )

        train_df = pd.DataFrame({
            self.smiles_col: X_train,
            self.target_col: y_train,
            self.source_col: df.loc[X_train.index, self.source_col],
        })

        test_df = pd.DataFrame({
            self.smiles_col: X_test,
            self.target_col: y_test,
            self.source_col: df.loc[X_test.index, self.source_col],
        })

        return train_df, test_df

    def _visualize_and_save(
            self,
            dfs: List[pd.DataFrame],
            names: List[str],
            suffix: str
    ) -> None:
        """Create and save visualization."""
        if not self.reducer:
            return

        visualizer: VisualizerBase = self.reducer.get_associated_visualizer()
        df_dict = {name: df for name, df in zip(names, dfs)}

        img = visualizer.get_visualization(df_dict)
        timestamp = datetime.now().strftime("%d_%H_%M_%S")
        save_path = self.data_interface.save_visualization(f"{timestamp}_{suffix}", img)
        img.save(save_path)
        logging.info(f"Saved visualization to: {save_path}")

    def _aggregate_dataframes(
            self,
            dfs: List[pd.DataFrame],
            empty_if_none: bool = False
    ) -> pd.DataFrame:
        """Concatenate multiple DataFrames."""
        if not dfs:
            return self._empty_dataframe() if empty_if_none else pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)

    def _empty_dataframe(self) -> pd.DataFrame:
        """Create empty DataFrame with standard columns."""
        return pd.DataFrame(columns=[self.smiles_col, self.target_col, self.source_col])

    def _update_registries(self) -> None:
        """Update data interface registries."""
        try:
            self.data_interface.update_registries()
            logging.info("Successfully updated data registries")
        except Exception as e:
            logging.exception(f"Failed to update registries: {e}")

    def _log_pipeline_start(self) -> None:
        """Log pipeline initialization info."""
        logging.info("# =================== Processing Pipeline ==================== #")
        logging.info(f"Starting processing pipeline with datasets: {self.datasets}")
        if self.splitter:
            logging.info(f"Using splitter: {self.splitter.name}")
        if self.sim_filter:
            logging.info(f"Filtering augmented datasets with: {self.sim_filter.name} to {self.sim_filter.against}")


    # ==================== Hashing and Identification ==================== #

    def _generate_split_key(self) -> str:
        """Generate unique identifier for this split configuration."""
        splitter_key = self.splitter.get_cache_key() if self.splitter else "nosplit"
        filter_key = self.sim_filter.get_cache_key() if self.sim_filter else "nofilter"
        datasets_params = [
            tuple(sorted(self.datasets)) if self.datasets else None,
            self.test_origin_dataset,
            self.task_setting,
        ]
        datasets_hash = abs(hash(frozenset(datasets_params))) % (10 ** 5)
        return f"{splitter_key}_{filter_key}_{datasets_hash:05d}"