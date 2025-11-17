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


@gin.configurable
class ProcessingPipeline:

    def __init__(
        self,
        do_load_datasets: bool,
        do_visualize_datasets: bool,
        do_train_test_split: bool,
        do_visualize_train_test: bool,
        do_train_model: bool,
        data_interface: DataInterface,
        featurizer: FeaturizerBase | None = None,
        reducer: ReducerBase | None = None,
        splitter: DataSplitterBase | None = None,
        datasets: list[str] | None = None,  # friendly_name
        manual_train_splits: list[str] | None = None,  # [friendly_name]
        manual_test_splits: list[str] | None = None,  # [friendly_name]
        smiles_col: str = "smiles",
        target_col: str = "y",
    ):
        self.do_load_datasets = do_load_datasets
        self.do_visualize_datasets = do_visualize_datasets
        self.do_train_test_split = do_train_test_split
        self.do_visualize_train_test = do_visualize_train_test
        self.do_train_model = do_train_model

        self.data_interface = data_interface
        self.featurizer = featurizer
        self.reducer = reducer
        self.splitter = splitter
        self.split_name = self.splitter.get_cache_key()

        self.datasets = datasets
        self.manual_train_splits = manual_train_splits
        self.manual_test_splits = manual_test_splits

        self.target_col = target_col
        self.smiles_col = smiles_col

    def run(self):
        assert (
            self.datasets
            and not self.manual_train_splits
            and not self.manual_test_splits
        ) or (
            not self.datasets and self.manual_train_splits and self.manual_test_splits
        ), (
            "Both ProcessingPipeline.datasets and .manual_{train,test}_splits were provided, "
            "but aggregating 'full' datasets with 'pre-split' datasets in this manner is not "
            "supported due to its complexity."
        )

        if self.do_load_datasets:
            dataset_dfs = self.load_datasets(self.datasets)
            featurized_dataset_dfs = self.featurize_datasets(dataset_dfs)
            if self.do_visualize_datasets:
                self.visualize_datasets(featurized_dataset_dfs)

        if self.do_train_test_split:
            if self.datasets:
                train_df, test_df = self.get_train_test(
                    {"featurized_dataset_dfs": featurized_dataset_dfs}
                )
                self.save_split(train_df, test_df)
            else:
                train_df, test_df = self.get_train_test(
                    {
                        "train_sets": self.manual_train_splits,
                        "test_sets": self.manual_test_splits,
                    }
                )

            if self.do_visualize_train_test:
                self.visualize_train_test(train_df, test_df)

        if self.do_train_model and self.do_train_test_split:
            metrics = self.train_model(train_df, test_df)
            self.save_model()
            self.save_metrics(metrics)

        self.data_interface.update_registries()

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

    def get_train_test(
        self, source_dict: dict[str, list]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns a tuple of final (clean, featurized) train and test set.
        If 'featurized_dataset_dfs' in source_dict.keys, then splitting is applied (no featurization).
        Else, when 'train_sets' and 'test_sets' in source_dict.keys, then featurization is applied (no splitting).
        """
        if "featurized_dataset_dfs" in source_dict.keys():
            concatenated_df = pd.concat(
                source_dict["featurized_dataset_dfs"], ignore_index=True
            )

            features, labels = (
                concatenated_df[self.smiles_col],
                concatenated_df[self.target_col],
            )

            X_train, X_test, y_train, y_test = self.splitter.split(features, labels)
            logging.info(
                f"Train-test split completed. Train size: {len(X_train)}, Test size: {len(X_test)}"
            )

            return pd.concat([X_train, y_train], axis=1), pd.concat(
                [X_test, y_test], axis=1
            )
        elif "train_sets" in source_dict.keys() and "test_sets" in source_dict.keys():
            train_dfs = self.load_datasets(source_dict["train_sets"])
            test_dfs = self.load_datasets(source_dict["test_sets"])

            assert all(self.target_col in df.columns for df in train_dfs) and all(
                self.target_col in df.columns for df in test_dfs
            ), "Loaded manual train and test splits but target (label) column does not match expected"

            featurized_train_dfs = self.featurize_datasets(train_dfs)
            featurized_test_dfs = self.featurize_datasets(test_dfs)

            return pd.concat(featurized_train_dfs), pd.concat(featurized_test_dfs)
        else:
            raise ValueError(
                "In passing 'source_dict' object to 'get_train_test', either 'featurized_datasets_dfs' key"
                "must be present or both 'train_sets' and 'test_sets' keys must be present"
            )

    def visualize_train_test(self, train_df, test_df): ...

    def save_split(self, train_df, test_df) -> None:
        assert self.datasets is not None
        self.data_interface.save_train_test_split(
            train_df,
            test_df,
            subdir_name=self.splitter.get_cache_key(),
            split_friendly_name=self.splitter.get_friendly_name(self.datasets),
        )

        # After saving new datasets, update the registry of dataset friendly names
        # TODO: move all registry updates to the end of run
        # self.data_interface.update_dataset_names_registry()

    def dump_logs_to_data_dir(
        self, contents: str, filename: str, dump_to_split_subdir: bool = False
    ) -> None:
        """Dumps logs or config contents to the data  under spdirectoryecified subdirectory."""
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
