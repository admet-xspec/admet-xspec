import abc
import logging
from typing import List, Dict, Optional, Any

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

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        multi_endpoint: bool = False,
        **kwargs,
    ):
        # Let other bases initialize (RegressorBase/BinaryClassifierBase -> PredictorBase)
        super().__init__(**kwargs)
        self.featurizer: Optional[FeaturizerBase] = getattr(self, "featurizer", None)
        self.multi_endpoint: bool = multi_endpoint
        self.model: Optional[BaseEstimator] = None
        self.params: Dict[str, Any] = params or {}

    @abc.abstractmethod
    def _init_model(self) -> BaseEstimator:
        """Return an uninitialized sklearn estimator (subclass must implement)."""
        ...

    def _featurize(self, df: pd.DataFrame) -> np.ndarray:
        """
        Accepts dataframe with columns: smiles, y, source.
        Returns a numpy array of shape (n_samples, n_features).
        If multi-endpoint training, concatenates OHE endpoint encoding to features.
        """
        if self.featurizer is None:
            raise ValueError(
                "Featurizer is not set. Inject a FeaturizerBase object before calling this method."
            )
        X = self.featurizer.featurize(df[self.smiles_col].tolist())
        # If multi-endpoint, concatenate OHE endpoint encoding to features
        if self.multi_endpoint:
            endpoint_ohe = np.array(
                [self.endpoint_ohe_map[src] for src in df[self.source_col]]
            )
            X = np.hstack([X, endpoint_ohe])
        return X.astype(np.float32)

    def _create_endpoint_map(self, endpoints: pd.Series):
        # count unique endpoints, sort them and create a mapping to OHE arrays
        map = {}
        unique_endpoints = sorted(endpoints.unique())
        for i, endpoint in enumerate(unique_endpoints):
            ohe = np.zeros(len(unique_endpoints), dtype=np.float32)
            ohe[i] = 1.0
            map[endpoint] = ohe
        # log the map
        logging.info(f"Mapped endpoints:")
        for name, encoding in map.items():
            logging.info(f"{name}: {encoding.tolist()}")
        self.endpoint_ohe_map = map

    def train(self, df: pd.DataFrame) -> None:
        """
        Accepts dataframe with columns: smiles, y, source.
        """
        # Initialize model
        self.model = self._init_model()
        # For multi-endpoint tasks, create endpoint map for OHE encoding
        if self.endpoint_ohe_map is None and self.multi_endpoint:
            self._create_endpoint_map(df[self.source_col])
        if self.params:
            self.set_hyperparameters(self.params)
        X = self._featurize(df)
        y = np.array(df[self.target_col], dtype=np.float32)
        self.model.fit(X, y)

    def predict(self, df: pd.DataFrame) -> List[float]:
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

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                preds = proba[:, 1]
            else:
                preds = proba.ravel()
        else:
            preds = self.model.predict(X)
        return list(map(float, np.asarray(preds)))

    def get_hyperparameters(self) -> Dict[str, Any]:
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

    def set_hyperparameters(self, params: Dict[str, Any]) -> None:
        if self.model is None:
            self.model = self._init_model()
        # Validate supported params against model.get_params()
        model_params = self.model.get_params()
        for key in params:
            if key not in model_params:
                raise ValueError(
                    f"Model {self.model.__class__.__name__} does not accept hyperparameter `{key}`. Supported hyperparameters: {list(model_params.keys())}"
                )
        self.model.set_params(**params)
