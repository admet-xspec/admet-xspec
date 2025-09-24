from typing import List
import abc
import pathlib
from sklearn import metrics
import pickle


class PredictorBase(abc.ABC):
    def __init__(self):
        self.model = self._init_model()
        self.working_dir = None  # working directory for training and inference
        self.evaluation_metrics = []  # will be set by the child

    @abc.abstractmethod
    def _init_model(self):
        """Initialize the model."""
        pass

    @abc.abstractmethod
    def predict(self, smiles_list: List[str]) -> List[float]:
        """Predict the target values for the given smiles list."""
        pass

    @abc.abstractmethod
    def train(self, smiles_list: List[str], target_list: List[float]):
        """Train the model with the given smiles and target list."""
        pass

    @abc.abstractmethod
    def evaluate(self, smiles_list: List[str], target_list: List[float]) -> dict:
        """
        Evaluate the model on the given smiles list and target list.
        Returns a dictionary of metrics appropriate for the task.
        """
        pass

    def save(self):
        """
        Pickle the model to working_dir/model.pkl
        """
        path = self.working_dir / "model.pkl"
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Pickle the model
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path: str):
        """Load the model from the given path."""
        # Check if the path exists
        if not pathlib.Path(path).exists():
            raise FileNotFoundError(f"Model file not found at {path}")
        # Load the model
        with open(path, "rb") as f:
            loaded_model = pickle.load(f)
            self.__dict__.update(loaded_model.__dict__)

    def set_working_dir(self, path: str):
        """
        Set working directory path for the model
        :param path: Path to the working directory
        """
        self.working_dir = path

    @staticmethod
    def get_scikit_metric_callable(metric_name: str):
        metrics_dict = {
            "accuracy": metrics.accuracy_score,
            "roc_auc": metrics.roc_auc_score,
            "f1": metrics.f1_score,
            "precision": metrics.precision_score,
            "recall": metrics.recall_score,
            "mse": metrics.mean_squared_error,
            "mae": metrics.mean_absolute_error,
            "r2": metrics.r2_score,
            "rmse": metrics.root_mean_squared_error,
        }
        if metric_name not in metrics_dict.keys():
            raise ValueError(f"Unknown metric: {metric_name}.")
        return metrics_dict[metric_name]
