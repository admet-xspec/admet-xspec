import abc
from typing import List, Tuple


class PredictorBase(abc.ABC):
    def __init__(self):
        self.working_dir = None  # working directory for training and inference
        self.model = self._init_model()

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
    def save(self, path: str):
        """Save the model to the given path"""
        pass

    @abc.abstractmethod
    def load(self, path: str):
        """Load the model from the given path."""
        pass

    def set_working_dir(self, path: str):
        """
        Set working directory path for the model
        :param path: Path to the working directory
        """
        self.working_dir = path
