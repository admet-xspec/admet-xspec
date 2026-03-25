from abc import ABC
from typing import List
import abc
from src.utils import compute_sklearn_metric
import numpy as np
from sklearn.model_selection import KFold
from typing import Dict, List, Any
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
    def predict(self, X: np.ndarray) -> List[float]:
        """Predict the target values for the given feature matrix."""
        pass

    @abc.abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the model with set hyperparameters."""
        pass

    @abc.abstractmethod
    def evaluate(self, X: np.ndarray, target_list: List[float]) -> dict:
        """Evaluate the model on the given feature matrix and targets."""
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

    def cross_validate(
        self, smiles_list: List[str], target_list: List[float], n_folds: int = 1
    ) -> Dict[str, Any]:
        """Perform a k-fold cross-validation on the given dataset and return the average metrics across folds."""
        metrics_dicts = []
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        kf.get_n_splits(smiles_list, target_list)
        for train_index, test_index in kf.split(smiles_list, target_list):
            X_train, X_test = [smiles_list[i] for i in train_index], [
                smiles_list[i] for i in test_index
            ]
            y_train, y_test = [target_list[i] for i in train_index], [
                target_list[i] for i in test_index
            ]
            self.train(X_train, y_train)
            fold_metrics = self.evaluate(X_test, y_test)
            metrics_dicts.append(fold_metrics)
        # Average metrics across folds
        out_dict = {}
        for dict in metrics_dicts:
            for k, v in dict.items():
                if k not in out_dict:
                    out_dict[k] = []
                out_dict[k].append(v)
        for k, v in out_dict.items():
            out_dict[k] = np.mean(v)
        return out_dict


class BinaryClassifierBase(PredictorBase, ABC):
    evaluation_metrics = ["accuracy", "roc_auc", "f1", "precision", "recall"]
    """
    Base class for binary classification predictors. Implements common evaluation metrics
    and classification thresholding.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self, X: np.ndarray, target_list: List[float]) -> dict:
        preds = self.predict(X)
        metrics_dict = {}
        for m in self.evaluation_metrics:
            if m == "roc_auc":
                # roc_auc needs class probabilities
                metrics_dict[m] = compute_sklearn_metric(m)(target_list, preds)
            else:
                binary_preds = self.classify(preds)
                metrics_dict[m] = compute_sklearn_metric(m)(target_list, binary_preds)
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

    def evaluate(self, X: np.ndarray, target_list: List[float]) -> dict:
        preds = self.predict(X)
        metrics_dict = {}
        for m in self.evaluation_metrics:
            metrics_dict[m] = compute_sklearn_metric(m)(target_list, preds)
        return metrics_dict
