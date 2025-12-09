import abc
import logging
from typing import List, Dict, Optional, Any

import numpy as np
from sklearn.utils.validation import check_array
from sklearn.model_selection import RandomizedSearchCV
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
        target_metric: Optional[str] = None,
        params_distribution: Optional[Dict[str, Any]] = None,
        optimization_iterations: Optional[int] = None,
        n_folds: Optional[int] = None,
        n_jobs: Optional[int] = None,
        **kwargs,
    ):
        # Let other bases initialize (RegressorBase/BinaryClassifierBase -> PredictorBase)
        super().__init__(**kwargs)
        self.featurizer: Optional[FeaturizerBase] = getattr(self, "featurizer", None)
        self.model: Optional[BaseEstimator] = None
        self.params: Dict[str, Any] = params or {}
        self.target_metric = target_metric
        self.hyper_opt = {
            "n_iter": optimization_iterations or 10,
            "n_folds": n_folds or 3,
            "n_jobs": n_jobs or 1,
            "params_distribution": params_distribution,
        }

    @abc.abstractmethod
    def _init_model(self) -> BaseEstimator:
        """Return an uninitialized sklearn estimator (subclass must implement)."""
        ...

    def _featurize(self, smiles_list: List[str]) -> np.ndarray:
        if self.featurizer is None:
            raise ValueError("Featurizer is not set. Inject a FeaturizerBase before calling this method.")
        X = self.featurizer.featurize(smiles_list)
        arr = np.array(X, dtype=np.float32)
        arr = np.atleast_2d(arr)
        return arr

    def train(self, smiles_list: List[str], target_list: List[float]) -> None:
        self.model = self._init_model()
        if self.params:
            self.set_hyperparameters(self.params)

        X = self._featurize(smiles_list)
        y = np.array(target_list, dtype=np.float32)

        logging.info(f"Training {self.model.__class__.__name__} with hyperparameters: {self.get_hyperparameters()}")
        self.model.fit(X, y)
        logging.info(f"Training complete for {self.model.__class__.__name__}")

    def optimize(self, smiles_list: List[str], target_list: List[float]) -> None:
        if not self.hyper_opt["params_distribution"]:
            raise ValueError("params_distribution must be provided to run optimization.")
        self.model = self._init_model()

        X = self._featurize(smiles_list)
        y = np.array(target_list, dtype=np.float32)

        logging.info(f"Starting hyperparameter optimization for {self.model.__class__.__name__}")
        rs = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.hyper_opt["params_distribution"],
            n_iter=self.hyper_opt["n_iter"],
            cv=self.hyper_opt["n_folds"],
            verbose=1,
            n_jobs=self.hyper_opt["n_jobs"],
            refit=True,
            scoring=self.target_metric,
        )
        rs.fit(X, y)

        self.params = rs.best_params_.copy()
        self.model = rs.best_estimator_
        logging.info(f"Optimization complete. Best params: {self.params}")

    def predict(self, smiles_list: List[str]) -> List[float]:
        if self.model is None:
            raise ValueError("Model is not initialized. Call `train` or `optimize` first.")
        X = self._featurize(smiles_list)

        if np.isnan(X).any() or np.isinf(X).any():
            num_nan = int(np.sum(np.isnan(X)))
            num_inf = int(np.sum(np.isinf(X)))
            logging.warning(f"Input contains {num_nan} NaN(s) and {num_inf} infinite(s). Replacing with 0.")
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
            return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in self.params.items()}
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
                raise ValueError(f"Model {self.model.__class__.__name__} does not accept hyperparameter `{key}`. Supported hyperparameters: {list(model_params.keys())}")
        self.model.set_params(**params)