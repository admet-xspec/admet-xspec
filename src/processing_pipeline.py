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

        self.target_col = target_col
        self.smiles_col = smiles_col

    def run(self):
        assert self.datasets, "No datasets were provided to be processed."

        nosplit_datasets = self.datasets
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
                    nosplit_featurized_dataset_dfs
                )

                split_train_df, split_test_df = self.get_train_test(
                    [split_featurized_dataset_df]
                )

                aggregate_train_df = self.splitter.filter(
                    aggregate_train_df + split_train_df, split_test_df
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
        # self.data_interface.update_registries()

    def load_datasets(self, friendly_names: list[str]) -> list[pd.DataFrame]:
        dataset_dfs = [
            self.data_interface.get_by_friendly_name(friendly_name)
            for friendly_name in friendly_names
        ]

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
        concatenated_df = pd.concat(dfs, ignore_index=True)

        # leaving this form of "split -> rejoin" for readability/understanding during PR
        features, labels = (
            concatenated_df[self.smiles_col],
            concatenated_df[self.target_col],
        )

        return pd.concat([features, labels], axis=1)

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

        return pd.concat([X_train, y_train], axis=1), pd.concat(
            [X_test, y_test], axis=1
        )

    def visualize_train_test(self, train_df, test_df): ...

    def save_split(self, train_df, test_df) -> None:
        assert self.datasets is not None
        self.data_interface.save_train_test_split(
            train_df,
            test_df,
            subdir_name=self.splitter.get_cache_key(),
            split_friendly_name=self.splitter.get_friendly_name(self.datasets),
            classification_or_regression=self.task_setting,
        )

    def dump_logs_to_data_dir(
        self, contents: str, filename: str, dump_to_split_subdir: bool = False
    ) -> None:
        """Dumps logs or config contents to the data under specified subdirectory."""
        if self.datasets is None or len(self.datasets) != 1:
            logging.warning(
                "dump_logs_to_data_dir currently only supports single dataset inputs."
            )
            logging.warning(
                "Logs will be dumped to general processing_logs subdirectory."
            )
            self.data_interface.dump_logs_to_general_dir(contents, filename)
            return

        if dump_to_split_subdir:
            # We dump to the split-specific subdirectory
            subdir_name = self.split_name
        else:
            # We dump to a general processing logs subdirectory
            subdir_name = "processing_logs"
        self.data_interface.dump_logs_to_data_dir(
            contents, filename, dataset_name=self.dataset_name, subdir_name=subdir_name
        )
