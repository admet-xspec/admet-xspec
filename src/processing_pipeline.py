import gin
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging

from src.data.data_interface import DataInterface
from src.data.featurizer import FeaturizerBase
from src.data.reducer import ReducerBase
from src.data.visualizer import VisualizerBase
from src.data.split import DataSplitterBase
from src.data.filter import FilterBase


@gin.configurable
class ProcessingPipeline:

    def __init__(
        self,
        do_load_datasets: bool,
        do_visualize_datasets: bool,
        do_load_train_test: bool,
        do_dump_train_test: bool,
        do_visualize_train_test: bool,
        data_interface: DataInterface,
        featurizer: FeaturizerBase | None = None,
        reducer: ReducerBase | None = None,
        splitter: DataSplitterBase | None = None,
        datasets: list[str] | None = None,  # friendly_name
        manual_train_splits: list[str] | None = None,  # [friendly_name]
        manual_test_splits: list[str] | None = None,  # [friendly_name]
        test_filtering_origin_dataset: str | None = None,
        task_setting: str = "regression",
        smiles_col: str = "smiles",
        target_col: str = "y",
        logfile: str | None = None,
    ):
        self.do_load_datasets = do_load_datasets
        self.do_visualize_datasets = do_visualize_datasets
        self.do_load_train_test = do_load_train_test
        self.do_dump_train_test = do_dump_train_test
        self.do_visualize_train_test = do_visualize_train_test

        self.task_setting = task_setting
        self.data_interface = data_interface
        self.data_interface.set_task_setting(self.task_setting)

        self.featurizer = featurizer
        self.reducer = reducer
        self.splitter = splitter
        self.split_name = self.splitter.get_cache_key()

        self.datasets = datasets
        self.manual_train_splits = manual_train_splits
        self.manual_test_splits = manual_test_splits
        self.test_filtering_origin_dataset = test_filtering_origin_dataset
        self.logfile = logfile

        self.target_col = target_col
        self.smiles_col = smiles_col
        self.source_col = "source"

    def run(self):
        assert self.datasets, "No datasets were provided to be processed."

        nosplit_datasets = self.datasets.copy()
        nosplit_datasets.remove(self.test_filtering_origin_dataset)

        if self.do_load_datasets:
            nosplit_dataset_dfs = self.load_datasets(nosplit_datasets)
            split_dataset_df = self.load_datasets([self.test_filtering_origin_dataset])[
                0
            ]

            nosplit_featurized_dataset_dfs = self.featurize_datasets(
                nosplit_dataset_dfs
            )
            split_featurized_dataset_df = self.featurize_datasets([split_dataset_df])[0]

            if self.do_visualize_datasets:
                self.visualize_datasets(
                    nosplit_featurized_dataset_dfs + split_featurized_dataset_df
                )

        if self.do_load_datasets and self.do_load_train_test:
            if self.datasets:
                aggregate_train_df = self.get_aggregate_trainable_form(
                    nosplit_dataset_dfs
                )

                # Split the dataset
                split_train_df, split_test_df = self.get_train_test([split_dataset_df])

                # Filter the train set to ensure no data leakage
                logging.info(
                    f"Filtering train set against test set using {self.splitter.get_filter().name} filter"
                )
                pre_filter_source_count = self.get_label_count(
                    aggregate_train_df, column_name=self.source_col
                )
                aggregate_train_df = self.splitter.filter(
                    pd.concat([aggregate_train_df, split_train_df]), split_test_df
                )
                logging.info(
                    f"Post-filtering train set size: {len(aggregate_train_df)}"
                )
                post_filter_source_count = self.get_label_count(
                    aggregate_train_df, column_name=self.source_col
                )

                for source in pre_filter_source_count.keys():
                    pre_count = pre_filter_source_count.get(source, 0)
                    post_count = post_filter_source_count.get(source, 0)
                    logging.info(
                        f"- {source}: dropped {pre_count - post_count} samples. Remaining: {post_count}"
                    )

                self.save_split(aggregate_train_df, split_test_df)
            else:
                train_df, test_df = self.get_train_test(
                    {
                        "train_sets": self.manual_train_splits,
                        "test_sets": self.manual_test_splits,
                    }
                )

            if self.do_visualize_train_test:
                self.visualize_train_test(train_df, test_df)

        # TODO: implement
        self.data_interface.update_registries()

    def load_datasets(self, friendly_names: list[str]) -> list[pd.DataFrame]:
        dataset_dfs = [
            self.data_interface.get_by_friendly_name(friendly_name)
            for friendly_name in friendly_names
        ]

        # add souce column
        for friendly_name, df in zip(friendly_names, dataset_dfs):
            df[self.source_col] = friendly_name

        return dataset_dfs

    def featurize_datasets(self, dataset_dfs):
        featurized_dataset_dfs = []
        for df in dataset_dfs:
            len_before_feat = len(df)

            feature_col_name = self.featurizer.feature_name
            df[feature_col_name] = df["smiles"].apply(
                lambda smiles: self.featurizer.featurize([smiles])
            )

            len_after_feat = len(df)
            assert len_before_feat == len_after_feat, (
                f"{len_before_feat - len_after_feat} SMILES failed to featurize with"
                f"{self.featurizer.name}. 'get_featurized_dataset_df' expects featurizable SMILES."
            )

            featurized_dataset_dfs.append(df)

        return featurized_dataset_dfs

    def visualize_datasets(self, featurized_dataset_dfs) -> None:
        visualizer: VisualizerBase = self.reducer.get_associated_visualizer()
        df_dict = {
            ds_name: df for ds_name, df in zip(self.datasets, featurized_dataset_dfs)
        }
        visualization_img = visualizer.get_visualization(df_dict)

        visualization_img.save(
            self.data_interface.save_visualization(
                datetime.now().strftime("%d_%H_%M_%S"),
                visualization_img,
            )
        )

    def get_aggregate_trainable_form(self, dfs: list[pd.DataFrame]) -> pd.DataFrame:
        """Aggregate a list of final (clean, featurized) dfs vertically, leaving only cols needed for training"""

        if not dfs:
            return pd.DataFrame([])
        # Concatenate all dataframes
        concatenated_df = pd.concat(dfs, ignore_index=True)

        return concatenated_df[[self.smiles_col, self.target_col, self.source_col]]

    def get_train_test(
        self, featurized_dataset_dfs
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns a tuple of final (clean, featurized) train and test set.
        If 'featurized_dataset_dfs' in source_dict.keys, then splitting is applied (no featurization).
        Else, when 'train_sets' and 'test_sets' in source_dict.keys, then featurization is applied (no splitting).
        """

        concatenated_df = self.get_aggregate_trainable_form(featurized_dataset_dfs)

        X_train, X_test, y_train, y_test = self.splitter.split(
            concatenated_df[self.smiles_col], concatenated_df[self.target_col]
        )
        logging.info(
            f"Train-test split completed. Train size: {len(X_train)}, Test size: {len(X_test)}"
        )

        # Combine X and y back into DataFrames
        train_df = pd.DataFrame(
            {
                self.smiles_col: X_train,
                self.target_col: y_train,
                self.source_col: concatenated_df.loc[X_train.index, self.source_col],
            }
        )
        test_df = pd.DataFrame(
            {
                self.smiles_col: X_test,
                self.target_col: y_test,
                self.source_col: concatenated_df.loc[X_test.index, self.source_col],
            }
        )

        return train_df, test_df

    def visualize_train_test(self, train_df, test_df): ...

    def save_split(self, train_df, test_df) -> None:
        assert self.datasets is not None
        self.data_interface.save_train_test_split(
            train_df,
            test_df,
            subdir_name=self.get_split_hash(),
            split_friendly_name=self.splitter.get_friendly_name(self.datasets),
            classification_or_regression=self.task_setting,
            console_log=self.read_logfile(),
        )

    def read_logfile(self) -> str | None:
        if self.logfile and Path(self.logfile).exists():
            with open(self.logfile, "r") as f:
                return f.read()
        return None

    def get_split_hash(self):
        """
        Generate a unique hash for the current split configuration to use in saving train-test splits.
        Format: {splitter_name}_{hash} (10-digit)
        """
        splitter_hashable = self.splitter.get_hashable_params_values()
        pipeline_hashable = [
            self.task_setting,
        ]
        datasets_hashable = self.datasets if self.datasets else []
        train_splits_hashable = (
            self.manual_train_splits if self.manual_train_splits else []
        )
        test_splits_hashable = (
            self.manual_test_splits if self.manual_test_splits else []
        )
        total_hashable = (
            splitter_hashable
            + pipeline_hashable
            + datasets_hashable
            + train_splits_hashable
            + test_splits_hashable
        )
        hash_key = abs(hash(tuple(total_hashable))) % (10**10)
        return f"{self.splitter.name}_{hash_key}"

    def get_label_count(self, df: pd.DataFrame, column_name="source") -> dict:
        source_count = {}
        for name in df[column_name].unique():
            source_count[name] = len(df[df[column_name] == name])
        return source_count
