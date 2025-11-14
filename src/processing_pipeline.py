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

        data_interface: DataInterface,
        featurizer: FeaturizerBase | None = None,
        reducer: ReducerBase | None = None,
        splitter: DataSplitterBase | None = None,
        datasets: list[str] | None = None,  # friendly_name
        # TODO: possibly bad source of truth, placed it here for sanity!
        smiles_col: str = "smiles",
        target_col: str = "y"
    ):
        self.do_load_datasets = do_load_datasets
        self.do_visualize_datasets = do_visualize_datasets
        self.do_train_test_split = do_train_test_split
        self.do_visualize_train_test = do_visualize_train_test

        self.data_interface = data_interface
        self.featurizer = featurizer
        self.reducer = reducer
        self.splitter = splitter
        self.split_name = self.splitter.get_cache_key()

        self.datasets = datasets
        # Quick fix for single dataset case
        if len(datasets) == 1:
            self.dataset_name = datasets[0]
        else:
            logging.warning("Multiple datasets provided; the currebt implementation does not handle this case well.")

        self.target_col = target_col
        self.smiles_col = smiles_col

    def run(self):
        if self.do_load_datasets:
            dataset_dfs = self.load_datasets(self.datasets)

        if self.do_visualize_datasets:
            featurized_dataset_dfs = self.featurize_datasets(dataset_dfs)
            self.visualize_datasets(featurized_dataset_dfs)

        if self.do_train_test_split and self.do_load_datasets:
            # Perform train-test split
            train_df, test_df = self.get_train_test(dataset_dfs)
            # Save the train-test split
            self.save_split(train_df, test_df)

        if self.do_visualize_train_test:
            self.visualize_train_test(train_df, test_df)

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
        self, dataset_dfs: list[pd.DataFrame]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        concatenated_df = pd.concat(dataset_dfs, ignore_index=True)

        features, labels = (
            concatenated_df[self.smiles_col],
            concatenated_df[self.target_col],
        )

        X_train, X_test, y_train, y_test = self.splitter.split(features, labels)
        logging.info(
            f"Train-test split completed. Train size: {len(X_train)}, Test size: {len(X_test)}"
        )

        return pd.concat([X_train, y_train], axis=1), pd.concat([X_test, y_test], axis=1)

    def visualize_train_test(self, train_df, test_df): ...

    def save_split(self, train_df, test_df) -> None:
        # Use splitter's cache key as subdirectory name
        self.data_interface.save_train_test_split(
            train_df, test_df, subdir_name=self.split_name, dataset_name=self.dataset_name)

        # After saving new datasets, update the registry of dataset friendly names
        self.data_interface.update_dataset_names_registry()

    def dump_logs_to_data_dir(self, contents: str, filename: str, dump_to_split_subdir: bool = False) -> None:
        """Dumps logs or config contents to the data  under spdirectoryecified subdirectory."""
        if self.datasets is None or len(self.datasets) != 1:
            logging.warning("dump_logs_to_data_dir currently only supports single dataset inputs.")
            logging.warning("Logs will be dumped to general processing_logs subdirectory.")
            self.data_interface.dump_logs_to_general_dir(contents, filename)
            return

        if dump_to_split_subdir:
            # We dump to the split-specific subdirectory
            subdir_name = self.split_name
        else:
            # We dump to a general processing logs subdirectory
            subdir_name = 'processing_logs'
        self.data_interface.dump_logs_to_data_dir(contents, filename, dataset_name=self.dataset_name,
                                                  subdir_name=subdir_name)
