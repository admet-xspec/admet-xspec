import abc
from typing import List, Tuple
import pathlib
from src.utils import get_scikit_metric_callable
import pickle


class PredictorBase(abc.ABC):
    def __init__(self):
        self.working_dir = None  # working directory for training and inference
        self.model = self._init_model()
        self.evaluation_metrics = []  # list of metrics to be used for model evaluation

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

    def save(self, path: str):
        """Save the model to the given path."""
        # Ensure the directory exists
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

    def evaluate(self, smiles_list: List[str], target_list: List[float]) -> dict:
        """
        Evaluate the model on the given smiles list and target list.
        Returns a metric (e.g., accuracy, ROC AUC) based on the model's predict method.
        """
        predictions = self.predict(smiles_list)
        metrics_dict = {}
        for metric in self.evaluation_metrics:
            metric_callable = get_scikit_metric_callable(metric)
            if metric == "accuracy":
                # For accuracy, we need to convert predictions to binary labels
                score = metric_callable(
                    target_list, [1 if pred >= 0.5 else 0 for pred in predictions]
                )
            else:
                score = metric_callable(target_list, predictions)
            metrics_dict[metric] = score
        return metrics_dict

    def set_working_dir(self, path: str):
        """
        Set working directory path for the model
        :param path: Path to the working directory
        """
        self.working_dir = path
