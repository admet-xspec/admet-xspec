from typing import List
import abc
import numpy as np

from src.data.featurizer import FeaturizerBase

class PredictorBase(abc.ABC):
    def __init__(self, random_state: int = 42):
        self.model = self._init_model()
        self.evaluation_metrics = []  # will be set by the child
        self.featurizer: FeaturizerBase | None = None  # will be set if applicable
        self.random_state = random_state
        # Set random seed for reproducibility
        np.random.seed(random_state)

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return the name of the predictor."""
        pass

    @property
    def uses_internal_featurizer(self) -> bool:
        """Return True if the model uses a propriotary featurizer."""
        return False

    @abc.abstractmethod
    def get_hyperparameters(self) -> dict:
        """Return the hyperparameters of the model."""
        pass

    @abc.abstractmethod
    def set_hyperparameters(self, hyperparams: dict):
        """Inject hyperparameters into the model."""
        pass

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
        """Train the model with set hyperparameters."""
        pass

    @abc.abstractmethod
    def train_optimize(self, smiles_list: List[str], target_list: List[float]):
        """Train the model and optimize hyperparameters"""
        pass

    @abc.abstractmethod
    def evaluate(self, smiles_list: List[str], target_list: List[float]) -> dict:
        """
        Evaluate the model on the given smiles list and target list.
        Returns a dictionary of metrics appropriate for the task.
        """
        pass

    def get_featurizer(self) -> FeaturizerBase | None:
        """Return the featurizer if set."""
        return self.featurizer if self.featurizer else None

    def set_featurizer(self, featurizer: FeaturizerBase):
        """Inject featurizer into the model."""
        assert isinstance(featurizer, FeaturizerBase), "Featurizer must be an instance of FeaturizerBase"
        self.featurizer = featurizer

    def get_cache_key(self) -> str:
        """Return a unique cache key for the predictor configuration.
        Hash is based on:
        - Predictor name
        - Featurizer name and its parameters (if any)
        Does not include model hyperparameters.
        """
        feturizer_key = self.featurizer.get_cache_key() if self.featurizer else 'nofeaturizer'
        return f"{self.name}_{feturizer_key}"
