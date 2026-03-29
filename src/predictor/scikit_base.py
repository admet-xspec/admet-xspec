import abc
import logging
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_array
from sklearn.base import BaseEstimator

from src.data.featurizer import FeaturizerBase


class ScikitPredictorBase(abc.ABC):
    """
    Mixin providing scikit-learn based training, optimization and prediction helpers.

    Intended to be combined with PredictorBase-derived classes:
      class MyRegressor(ScikitPredictorBase, RegressorBase): ...
      class MyClassifier(ScikitPredictorBase, BinaryClassifierBase): ...
    """

    # Attributes are provided by PredictorBase in cooperative multiple inheritance.
    smiles_col: str = "smiles"
    source_col: str = "source"
    target_col: str = "y"
    featurizer: FeaturizerBase | None = None
    endpoint_ohe_map: dict[str, np.ndarray] | None = None

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        multi_endpoint: bool = False,
        **kwargs,
    ) -> None:
        # Let other bases initialize (RegressorBase/BinaryClassifierBase -> PredictorBase)
        super().__init__(**kwargs)
        # PredictorBase usually initializes this; keep explicit for type checkers.
        self.featurizer: FeaturizerBase | None = getattr(self, "featurizer", None)
        # Keep local flag in sync with PredictorBase in multiple inheritance.
        self.multi_endpoint: bool = bool(multi_endpoint)
        self.model: BaseEstimator | None = None
        self.params: dict[str, Any] = params or {}

    @abc.abstractmethod
    def _init_model(self) -> BaseEstimator:
        """Return a fresh, untrained sklearn estimator (subclass must implement)."""
        ...

    def _featurize(self, df: pd.DataFrame) -> np.ndarray:
        """
        Convert model input dataframe into a numerical feature matrix.

        If ``multi_endpoint`` is enabled, append endpoint one-hot vectors to molecule
        features so a single estimator can distinguish endpoint context.
        """
        if self.featurizer is None:
            raise ValueError(
                "Featurizer is not set. Inject a FeaturizerBase object before calling this method."
            )
        if self.smiles_col not in df.columns:
            raise ValueError(
                f"Missing required smiles column `{self.smiles_col}` in input dataframe."
            )

        X = self.featurizer.featurize(df[self.smiles_col].tolist())
        # Multi-endpoint predictors append endpoint identity to molecular features.
        if self.multi_endpoint:
            X = np.hstack([X, self._endpoint_features(df)])
        return X.astype(np.float32)

    def _endpoint_features(self, df: pd.DataFrame) -> np.ndarray:
        """Return endpoint one-hot features aligned with ``df`` row order."""
        if self.source_col not in df.columns:
            raise ValueError(
                f"Missing required source column `{self.source_col}` for multi-endpoint prediction."
            )
        if self.endpoint_ohe_map is None:
            raise ValueError(
                "Endpoint OHE map is not initialized. Train first or load a model with endpoint metadata."
            )

        missing_sources = sorted(
            set(df[self.source_col].unique()) - set(self.endpoint_ohe_map.keys())
        )
        if missing_sources:
            raise ValueError(
                f"Unknown source values not seen during training: {missing_sources}"
            )

        return np.array([self.endpoint_ohe_map[src] for src in df[self.source_col]])

    def train(self, df: pd.DataFrame) -> None:
        """Fit the estimator on a dataframe containing features and targets."""
        if self.target_col not in df.columns:
            raise ValueError(
                f"Missing required target column `{self.target_col}` in training dataframe."
            )

        # Always create a fresh estimator for each train call.
        self.model = self._init_model()
        # For multi-endpoint tasks, create endpoint map for OHE encoding
        if self.endpoint_ohe_map is None and self.multi_endpoint:
            cast(Any, self)._create_endpoint_map(df[self.source_col])
        if self.params:
            self.set_hyperparameters(self.params)
        X = self._featurize(df)
        y = np.array(df[self.target_col], dtype=np.float32)
        estimator = cast(Any, self.model)
        estimator.fit(X, y)

    def predict(self, df: pd.DataFrame) -> list[float]:
        """Predict scores/values for every input row in ``df``."""
        if self.model is None:
            raise ValueError(
                "Model is not initialized. Call `train` or `optimize` first."
            )
        X = self._featurize(df)

        if np.isnan(X).any() or np.isinf(X).any():
            num_nan = int(np.sum(np.isnan(X)))
            num_inf = int(np.sum(np.isinf(X)))
            logging.warning(
                f"Input contains {num_nan} NaN(s) and {num_inf} infinite(s). Replacing with 0."
            )
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        X = check_array(X, ensure_all_finite=True, dtype=np.float32)

        model = cast(Any, self.model)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                preds = proba[:, 1]
            else:
                preds = proba.ravel()
        else:
            preds = model.predict(X)
        return list(map(float, np.asarray(preds)))

    def get_hyperparameters(self) -> dict[str, Any]:
        """Return estimator hyperparameters with numpy scalar values normalized."""
        if self.model is None:
            return {
                k: (v.item() if isinstance(v, np.generic) else v)
                for k, v in self.params.items()
            }
        hyperparams = self.model.get_params()
        for k, v in list(hyperparams.items()):
            if isinstance(v, np.generic):
                hyperparams[k] = v.item()
        return hyperparams

    def set_hyperparameters(self, params: dict[str, Any]) -> None:
        """Validate and apply estimator hyperparameters."""
        if self.model is None:
            self.model = self._init_model()
        # Validate supported params against model.get_params()
        model_params = self.model.get_params()
        unknown = [key for key in params if key not in model_params]
        if unknown:
            raise ValueError(
                f"Model {self.model.__class__.__name__} does not accept hyperparameter(s) {unknown}."
            )
        # Persist validated params so they survive before/after model initialization.
        self.params = dict(params)
        self.model.set_params(**params)
