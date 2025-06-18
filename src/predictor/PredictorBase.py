import abc
from typing import List, Tuple
import pathlib
import pickle


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
            import pickle

            loaded_model = pickle.load(f)
            self.__dict__.update(loaded_model.__dict__)

    def set_working_dir(self, path: str):
        """
        Set working directory path for the model
        :param path: Path to the working directory
        """
        self.working_dir = path
