import gin
import pandas as pd
from datetime import datetime

from src import PredictorBase
from src.data.data_interface import DataInterface
from src.data.featurizer import FeaturizerBase
from src.data.reducer import ReducerBase
from src.data.visualizer import VisualizerBase
from src.data.split import DataSplitterBase


@gin.configurable
class ProcessingPipeline:

    def __init__(
        self,
        model_name: str,
        do_load_datasets: bool,
        do_visualize_datasets: bool,
        do_load_train_test: bool,
        do_visualize_train_test: bool,
        do_train_model: bool,
        data_interface: DataInterface,
        featurizer: FeaturizerBase,
        reducer: ReducerBase,
        splitter: DataSplitterBase,
        predictor: PredictorBase,
        datasets: list[str],  # friendly_name
        train_sets: list[str],  # friendly_name
        test_sets: list[str],  # friendly_name
    ):
        assert (datasets and not train_sets and not test_sets) or (
            not datasets and train_sets and test_sets
        ), (
            "Invariant violation. Either the friendly names of datasets to be cleaned, featurized and loaded "
            "can be provided, or the friendly names of datasets to be featurized and loaded can be provided. "
        )
        if not datasets:
            assert train_sets and test_sets, (
                "Invariant violation. When providing already-split .csvs manually, both"
                "train and test must be specified for 'self.train_sets', 'self.test_sets'."
            )

        self.model_name = model_name

        self.do_load_datasets = do_load_datasets
        self.do_visualize_datasets = do_visualize_datasets
        self.do_load_train_test = do_load_train_test
        self.do_visualize_train_test = do_visualize_train_test
        self.do_train_model = do_train_model

        self.data_interface = data_interface
        self.featurizer = featurizer
        self.reducer = reducer
        self.splitter = splitter
        self.predictor = predictor

        self.datasets = datasets

    def run(self):
        if self.do_load_datasets:
            dataset_dfs = self.load_datasets(self.datasets)
            featurized_dataset_dfs = self.featurize_datasets(dataset_dfs)

        if self.do_visualize_datasets:
            self.visualize_datasets(featurized_dataset_dfs)

        # TODO: sort out unecessary "if manual split then we have conditional behaviour in get_train_test
        # TODO: which also duplicates the process of loading and featurizing datasets" complexity present here
        # idea: make 'run' call a different function corresponding 1:1 with each 'plan' present in 'configs/plans' (?)
        if self.do_load_train_test and self.do_load_datasets:
            train_df, test_df = self.get_train_test(
                {"featurized_dataset_dfs": featurized_dataset_dfs}
            )
        elif self.do_load_train_test and (self.train_sets and self.test_sets):
            train_df, test_df = self.get_train_test(
                {"train_sets": self.train_sets, "test_sets": self.test_sets}
            )

        if self.do_visualize_train_test:
            self.visualize_train_test(train_df, test_df)

        if self.do_train_model:
            self.train_model(train_df, test_df)
            self.save_model()
            self.save_metrics()

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
                lambda smiles: self.featurizer.feature_to_str(
                    self.featurizer.featurize([smiles])
                )
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
            ...
        elif "train_sets" in source_dict.keys() and "test_sets" in source_dict.keys():
            ...

    def visualize_train_test(self, train_df, test_df): ...

    def train_model(self, train_df, test_df): ...

    def save_model(self): ...

    def save_metrics(self): ...
