from typing import List
import abc
from pathlib import Path
import pickle


class PredictorBase(abc.ABC):
    def __init__(self, task: str):
        self.model = self._init_model()
        self.task = task
        self.evaluation_metrics = []  # will be set by the child

    @property
    def get_ml_task(self):
        return self.task

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

    def save(self, dir_path: str | Path, name: str = "model"):
        """
        Pickle the model to dir_path/model.pkl
        """
        path = Path(dir_path) / f"{name}.pkl"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Pickle the model
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path: str):
        """Load the model from the given path."""
        # Check if the path exists
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found at {path}")
        # Load the model
        with open(path, "rb") as f:
            loaded_model = pickle.load(f)
            self.__dict__.update(loaded_model.__dict__)
