from abc import ABC
from typing import List
import abc
import numpy as np
from sklearn import metrics

from src.data.featurizer import FeaturizerBase


class PredictorBase(abc.ABC):
    def __init__(self, random_state: int = 42):
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
        """Return True if the model uses a proprietary featurizer."""
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
    def predict(self, smiles_list: List[str]) -> List[float]:
        """
        Predict the target values for the given smiles list.
        Returns a list of floats - either regression values or class probabilities.
        """
        pass

    @abc.abstractmethod
    def train(self, smiles_list: List[str], target_list: List[float]):
        """Train the model with set hyperparameters."""
        pass

    @abc.abstractmethod
    def optimize(self, smiles_list: List[str], target_list: List[float]):
        """Optimize hyperparameters of the model and set them internally."""
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
        assert isinstance(
            featurizer, FeaturizerBase
        ), "Featurizer must be an instance of FeaturizerBase"
        self.featurizer = featurizer

    def get_cache_key(self) -> str:
        """Return a unique cache key for the predictor configuration.
        Hash is based on:
        - Predictor name
        - Featurizer name and its parameters (if any)
        Does not include model hyperparameters.
        """
        feturizer_key = (
            self.featurizer.get_cache_key() if self.featurizer else "nofeaturizer"
        )
        return f"{self.name}_{feturizer_key}"

    @staticmethod
    def get_metric_callable(metric_name: str):
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
            raise ValueError(f"Invalid metric name: '{metric_name}'. Supported metrics: {list(metrics_dict.keys())}")
        return metrics_dict[metric_name]

class BinaryClassifierBase(PredictorBase, ABC):
    evaluation_metrics = ["accuracy", "roc_auc", "f1", "precision", "recall"]
    """
    Base class for binary classification predictors. Implements common evaluation metrics 
    and classification thresholding.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self, smiles_list: List[str], target_list: List[float]) -> dict:
        preds = self.predict(smiles_list)
        metrics_dict = {}
        for m in self.evaluation_metrics:
            if m == "roc_auc":
                # roc_auc needs class probabilities
                metrics_dict[m] = self.get_metric_callable(m)(target_list, preds)
            else:
                binary_preds = self.classify(preds)
                metrics_dict[m] = self.get_metric_callable(m)(target_list, binary_preds)
        return metrics_dict

    def classify(self, preds):
        return np.array(preds) > self.class_threshold

    @property
    def class_threshold(self) -> float:
        """Return the classification threshold value. Default is 0.5."""
        return 0.5

class RegressorBase(PredictorBase, ABC):
    evaluation_metrics = ["mse", "rmse", "mae", "r2"]
    """
    Base class for regression predictors. Implements common evaluation metrics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self, smiles_list: List[str], target_list: List[float]) -> dict:
        preds = self.predict(smiles_list)
        metrics_dict = {}
        for m in self.evaluation_metrics:
            metrics_dict[m] = self.get_metric_callable(m)(target_list, preds)
        return metrics_dict