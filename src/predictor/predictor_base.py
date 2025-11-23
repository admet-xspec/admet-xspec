from typing import List
import abc

from src import FeaturizerBase


class PredictorBase(abc.ABC):
    def __init__(self):
        self.model = self._init_model()
        self.hyperparams = {}
        self.evaluation_metrics = []  # will be set by the child
        self.featurizer: FeaturizerBase | None = None  # will be set if applicable

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return the name of the predictor."""
        pass

    def get_featurizer(self) -> FeaturizerBase | None:
        """Return the featurizer if set."""
        return self.featurizer

    def get_hyperparameters(self) -> dict:
        """Return the hyperparameters of the model."""
        return self.hyperparams

    def set_hyperparameters(self, hyperparams: dict):
        """Inject hyperparameters into the model."""
        self.hyperparams = hyperparams

    def get_cache_key(self) -> str:
        """Return a unique cache key for the predictor configuration.
        Hash is based on:
        - Predictor name
        - Featurizer name and its parameters (if any)
        Does not include model hyperparameters.
        """
        featurizer_hashables = (
            self.get_featurizer().get_hashable_params_values()
            if self.get_featurizer
            else []
        )
        feturizer_name = (
            self.get_featurizer().name if self.get_featurizer else "nofeaturizer"
        )
        model_name = self.name
        return f"{model_name}_{feturizer_name}_{abs(hash(frozenset(featurizer_hashables))) % (10 ** 5):05d}"

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
