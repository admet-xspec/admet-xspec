import abc
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.data.featurizer import FeaturizerBase
from src.utils import compute_sklearn_metric


HyperParams = dict[str, Any]
MetricsDict = dict[str, Any]


class PredictorBase(abc.ABC):
    """Shared predictor API used by training and inference pipelines."""

    def __init__(
        self,
        random_state: int = 42,
        task: str | None = None,
        multi_endpoint: bool = False,
    ) -> None:
        self.featurizer: FeaturizerBase | None = None
        self.random_state = random_state
        self.smiles_col = "smiles"
        self.source_col = "source"
        self.target_col = "y"
        self.task_name = task
        self.multi_endpoint = multi_endpoint
        self.endpoint_ohe_map: dict[Any, np.ndarray] | None = None
        # Set random seed for reproducibility
        np.random.seed(random_state)

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return the name of the predictor."""
        raise NotImplementedError

    @property
    def task(self) -> str | None:
        """Semantic task/endpoint name associated with this model instance."""
        return self.task_name

    @property
    def is_multi_endpoint(self) -> bool:
        return self.multi_endpoint

    @abc.abstractmethod
    def get_hyperparameters(self) -> HyperParams:
        """Return current model hyperparameters."""
        raise NotImplementedError

    @abc.abstractmethod
    def set_hyperparameters(self, hyperparams: HyperParams) -> None:
        """Inject hyperparameters into the model."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict the target values for the given dataset.
        Returns a series of floats - either regression values or class probabilities.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, df: pd.DataFrame) -> None:
        """Train the model with set hyperparameters."""
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, df: pd.DataFrame) -> MetricsDict:
        """
        Evaluate the model and return task-appropriate metrics.

        Implementations may return plain floats or richer values that are serializable.
        """
        raise NotImplementedError

    def get_featurizer(self) -> FeaturizerBase | None:
        """Return the featurizer if set."""
        return self.featurizer if self.featurizer else None

    def set_featurizer(self, featurizer: FeaturizerBase) -> None:
        """Inject featurizer into the model."""
        if not isinstance(featurizer, FeaturizerBase):
            raise TypeError("Featurizer must be an instance of FeaturizerBase")
        self.featurizer = featurizer

    def set_task_name(self, name: str | None) -> None:
        """Set semantic task/endpoint name used in cache keys and logs."""
        self.task_name = name

    def get_cache_key(self) -> str:
        """Return a unique cache key for predictor + featurizer configuration.

        Hash is based on:
        - Predictor name
        - Featurizer name and its parameters (if any)
        Does not include model hyperparameters.
        """
        featurizer_key = (
            self.featurizer.get_cache_key() if self.featurizer else "nofeaturizer"
        )
        key = f"{self.task_name}_{self.name}_{featurizer_key}"
        if self.is_multi_endpoint:
            key = "multiend_" + key
        return key

    def cross_validate(self, df: pd.DataFrame, n_folds: int = 1) -> MetricsDict:
        """Run k-fold cross-validation and average the resulting metrics."""
        if n_folds < 2:
            raise ValueError("`n_folds` must be at least 2 for cross-validation.")
        if len(df) < n_folds:
            raise ValueError(
                f"Not enough rows for {n_folds}-fold CV: got {len(df)} rows."
            )

        metrics_dicts: list[MetricsDict] = []
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        for train_index, test_index in kf.split(df):
            train_df = df.iloc[train_index]
            test_df = df.iloc[test_index]
            self.train(train_df)
            fold_metrics = self.evaluate(test_df)
            metrics_dicts.append(fold_metrics)
        return self._average_metrics(metrics_dicts)

    def set_column_ids(
        self,
        smiles_col: str | None = None,
        source_col: str | None = None,
        target_col: str | None = None,
    ) -> None:
        """Set dataframe column identifiers used by predictor methods."""
        if smiles_col is not None:
            self.smiles_col = smiles_col
        if source_col is not None:
            self.source_col = source_col
        if target_col is not None:
            self.target_col = target_col

    def get_endpoint_OHE_map(self) -> dict[Any, np.ndarray] | None:
        """Return endpoint one-hot map (legacy method name kept for compatibility)."""
        return self.endpoint_ohe_map if self.endpoint_ohe_map else None

    def _create_endpoint_map(self, endpoints: pd.Series) -> None:
        """Create endpoint -> one-hot mapping used for multi-endpoint predictors."""
        endpoint_map: dict[Any, np.ndarray] = {}
        unique_endpoints = sorted(endpoints.unique())
        for i, endpoint in enumerate(unique_endpoints):
            ohe = np.zeros(len(unique_endpoints), dtype=np.float32)
            ohe[i] = 1.0
            endpoint_map[endpoint] = ohe
        self.endpoint_ohe_map = endpoint_map

    @staticmethod
    def _average_metrics(metrics_dicts: list[MetricsDict]) -> MetricsDict:
        """Average each metric across CV folds."""
        grouped: dict[str, list[Any]] = defaultdict(list)
        for fold_metrics in metrics_dicts:
            for metric_name, metric_value in fold_metrics.items():
                grouped[metric_name].append(metric_value)
        return {metric_name: np.mean(values) for metric_name, values in grouped.items()}

    @staticmethod
    def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
        """Fail fast with a clear message when input schema is incomplete."""
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


class BinaryClassifierBase(PredictorBase, abc.ABC):
    """
    Base class for binary classification predictors. Implements common evaluation metrics
    and classification thresholding.
    """

    evaluation_metrics = ["accuracy", "roc_auc", "f1", "precision", "recall"]

    def evaluate(self, df: pd.DataFrame) -> MetricsDict:
        """Evaluate binary predictions using a fixed metric set."""
        self._require_columns(df, [self.target_col])
        preds = self.predict(df)
        binary_preds = self.classify(preds)
        metrics_dict: MetricsDict = {}
        for m in self.evaluation_metrics:
            if m == "roc_auc":
                # roc_auc needs class probabilities
                metrics_dict[m] = compute_sklearn_metric(m)(df[self.target_col], preds)
            else:
                metrics_dict[m] = compute_sklearn_metric(m)(
                    df[self.target_col], binary_preds
                )
        return metrics_dict

    def classify(self, preds: Any) -> np.ndarray:
        """Convert model scores/probabilities into hard labels using threshold."""
        return np.array(preds) > self.class_threshold

    @property
    def class_threshold(self) -> float:
        """Return the classification threshold value. Default is 0.5."""
        return 0.5


class RegressorBase(PredictorBase, abc.ABC):
    """
    Base class for regression predictors. Implements common evaluation metrics.
    """

    evaluation_metrics = ["mse", "rmse", "mae", "r2"]

    def evaluate(self, df: pd.DataFrame) -> MetricsDict:
        """Evaluate regression predictions using a fixed metric set."""
        self._require_columns(df, [self.target_col])
        preds = self.predict(df)
        return {
            m: compute_sklearn_metric(m)(df[self.target_col], preds)
            for m in self.evaluation_metrics
        }

