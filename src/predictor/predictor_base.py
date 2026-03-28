import abc
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.data.featurizer import FeaturizerBase
from src.utils import compute_sklearn_metric


class PredictorBase(abc.ABC):
    """Abstract predictor API used by training and inference pipelines."""

    def __init__(
        self,
        random_state: int = 42,
        task: str | None = None,
        multi_endpoint: bool = False,
    ):
        self.featurizer: FeaturizerBase | None = None  # will be set if applicable
        self.random_state = random_state
        self.smiles_col = "smiles"
        self.source_col = "source"
        self.target_col = "y"
        self.task_name = task
        self.multi_endpoint = multi_endpoint
        self.endpoint_ohe_map = None
        # Set random seed for reproducibility
        np.random.seed(random_state)

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return the name of the predictor."""
        pass

    @property
    def task(self) -> str:
        return self.task_name

    @property
    def is_multi_endpoint(self) -> bool:
        return self.multi_endpoint

    @abc.abstractmethod
    def get_hyperparameters(self) -> dict:
        """Return the hyperparameters of the model."""
        pass

    @abc.abstractmethod
    def set_hyperparameters(self, hyperparams: dict):
        """Inject hyperparameters into the model."""
        pass

    @abc.abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict the target values for the given dataset.
        Returns a series of floats - either regression values or class probabilities.
        """
        pass

    @abc.abstractmethod
    def train(self, df: pd.DataFrame):
        """Train the model with set hyperparameters."""
        pass

    @abc.abstractmethod
    def evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
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

    def set_task_name(self, name):
        """Set the task name based on the target column."""
        self.task_name = name

    def get_cache_key(self) -> str:
        """Return a unique cache key for the predictor configuration.
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

    def cross_validate(self, df: pd.DataFrame, n_folds: int = 1) -> Dict[str, Any]:
        """Perform k-fold CV and return mean metrics across folds."""
        if n_folds < 2:
            raise ValueError("`n_folds` must be at least 2 for cross-validation.")
        if len(df) < n_folds:
            raise ValueError(
                f"Not enough rows for {n_folds}-fold CV: got {len(df)} rows."
            )

        metrics_dicts = []
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        kf.get_n_splits(df)
        for train_index, test_index in kf.split(df):
            train_df = df.iloc[train_index]
            test_df = df.iloc[test_index]
            self.train(train_df)
            fold_metrics = self.evaluate(test_df)
            metrics_dicts.append(fold_metrics)
        # Average metrics across folds
        out_dict = {}
        for fold_dict in metrics_dicts:
            for k, v in fold_dict.items():
                if k not in out_dict:
                    out_dict[k] = []
                out_dict[k].append(v)
        for k, v in out_dict.items():
            out_dict[k] = np.mean(v)
        return out_dict

    def set_column_ids(
        self,
        smiles_col: str | None = None,
        source_col: str | None = None,
        target_col: str | None = None,
    ):
        """Set dataframe column identifiers used by predictor methods."""
        if smiles_col is not None:
            self.smiles_col = smiles_col
        if source_col is not None:
            self.source_col = source_col
        if target_col is not None:
            self.target_col = target_col

    def inject_smiles_col_ID(self, name: str):
        """Backward-compatible alias for legacy call sites."""
        self.smiles_col = name

    def inject_source_col_ID(self, name: str):
        """Backward-compatible alias for legacy call sites."""
        self.source_col = name

    def inject_target_col_ID(self, name: str):
        """Backward-compatible alias for legacy call sites."""
        self.target_col = name

    def get_endpoint_OHE_map(self) -> dict:
        """Return the endpoint OHE map if set."""
        return self.endpoint_ohe_map if self.endpoint_ohe_map else None

    def _create_endpoint_map(self, endpoints: pd.Series):
        """Create endpoint -> one-hot mapping used for multi-endpoint predictors."""
        endpoint_map = {}
        unique_endpoints = sorted(endpoints.unique())
        for i, endpoint in enumerate(unique_endpoints):
            ohe = np.zeros(len(unique_endpoints), dtype=np.float32)
            ohe[i] = 1.0
            endpoint_map[endpoint] = ohe
        self.endpoint_ohe_map = endpoint_map


class BinaryClassifierBase(PredictorBase, abc.ABC):
    """
    Base class for binary classification predictors. Implements common evaluation metrics
    and classification thresholding.
    """

    evaluation_metrics = ["accuracy", "roc_auc", "f1", "precision", "recall"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self, df: pd.DataFrame) -> dict:
        preds = self.predict(df)
        metrics_dict = {}
        for m in self.evaluation_metrics:
            if m == "roc_auc":
                # roc_auc needs class probabilities
                metrics_dict[m] = compute_sklearn_metric(m)(df[self.target_col], preds)
            else:
                binary_preds = self.classify(preds)
                metrics_dict[m] = compute_sklearn_metric(m)(
                    df[self.target_col], binary_preds
                )
        return metrics_dict

    def classify(self, preds):
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self, df: pd.DataFrame) -> dict:
        preds = self.predict(df)
        metrics_dict = {}
        for m in self.evaluation_metrics:
            metrics_dict[m] = compute_sklearn_metric(m)(df[self.target_col], preds)
        return metrics_dict
