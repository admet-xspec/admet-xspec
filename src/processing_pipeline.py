import gin

from src import PredictorBase
from src.data.data_interface import DataInterface
from src.data.featurizer import FeaturizerBase
from src.data.reducer import ReducerBase
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
        datasets: list[str],
    ):
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
        if self.do_load_datasets or self.do_load_train_test or self.do_train_model:

            dataset_dfs = self.load_datasets()
            featurized_dataset_dfs = self.featurize_datasets(dataset_dfs)

        if self.do_visualize_datasets or self.do_load_train_test or self.do_train_model:

            self.visualize_datasets(featurized_dataset_dfs)

        if self.do_load_train_test or self.do_train_model:

            train_df, test_df = self.get_train_test(featurized_dataset_dfs)

        if self.do_visualize_train_test or self.do_train_model:

            self.visualize_train_test(train_df, test_df)

        if self.do_train_model:

            self.train_model(train_df, test_df)
            self.save_model()
            self.save_metrics()

    def load_datasets(self):
        dataset_dfs = [
            self.data_interface.get_by_friendly_name(friendly_name)
            for friendly_name in self.datasets
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

    def visualize_datasets(self, featurized_dataset_dfs): ...

    def get_train_test(self, featurized_dataset_dfs): ...

    def visualize_train_test(self, train_df, test_df): ...

    def train_model(self, train_df, test_df): ...

    def save_model(self): ...

    def save_metrics(self): ...
