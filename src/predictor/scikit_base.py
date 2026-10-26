import abc
import logging
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

from src.data.featurizer import FeaturizerBase


class ScikitPredictorBase(abc.ABC):
    """Mixin with scikit-learn training and prediction for `PredictorBase` subclasses.

    Usage: `class MyRegressor(ScikitPredictorBase, RegressorBase): ...`
    """

    # Provided by `PredictorBase` via cooperative multiple inheritance.
    smiles_col: str = "smiles"
    source_col: str = "source"
    target_col: str = "y"
    featurizer: FeaturizerBase | None = None
    endpoint_ohe_map: dict[str, np.ndarray] | None = None

    # Estimator kwarg setting its thread count (e.g. "n_jobs"); None if single-threaded.
    thread_param: str | None = None

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        do_attributed_learning: bool = False,
        n_threads: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # Re-declared for type checkers.
        self.featurizer: FeaturizerBase | None = getattr(self, "featurizer", None)
        self.do_attributed_learning: bool = bool(do_attributed_learning)
        self.model: BaseEstimator | None = None
        # Fitted in `train` on the training features; pickled with the model for inference.
        self.feature_scaler: TransformerMixin | None = None
        self.params: dict[str, Any] = self._without_thread_param(params or {})
        self.n_threads: int | None = n_threads

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state, taking over the scaler that older featurizers fitted."""
        self.__dict__.update(state)
        if "feature_scaler" not in state:
            self.feature_scaler = (
                self.featurizer.pop_legacy_scaler() if self.featurizer else None
            )

    @property
    def supports_threading(self) -> bool:
        """Return whether the estimator exposes a thread-count parameter."""
        return self.thread_param is not None

    def set_n_threads(self, n_threads: int | None) -> None:
        """Set the thread count for the current and future models."""
        self.n_threads = n_threads
        if self.model is not None:
            self._apply_n_threads(self.model)

    def _apply_n_threads(self, model: BaseEstimator) -> None:
        """Apply `n_threads` to `model`, if supported and set."""
        if self.thread_param is not None and self.n_threads is not None:
            model.set_params(**{self.thread_param: self.n_threads})

    def _without_thread_param(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of `params` without the thread-count parameter.

        Thread count is a machine setting, not a hyperparameter, so saved values
        must not override the current budget.
        """
        return {k: v for k, v in params.items() if k != self.thread_param}

    def _new_model(self) -> BaseEstimator:
        """Return a fresh estimator with the thread count applied."""
        model = self._init_model()
        self._apply_n_threads(model)
        return model

    @abc.abstractmethod
    def _init_model(self) -> BaseEstimator:
        """Return a fresh, untrained sklearn estimator."""
        ...

    def _featurize(self, df: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        """Convert `df` into a float32 feature matrix.

        Features are scaled with `feature_scaler`, which `fit_scaler` first refits
        on `df` (from the featurizer's template). Attributed models get endpoint
        one-hot vectors appended, so one estimator can tell endpoints apart.
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
        if fit_scaler:
            self.feature_scaler = self.featurizer.make_scaler()
            if self.feature_scaler is not None:
                self.feature_scaler.fit(X)
        if self.feature_scaler is not None:
            X = self.feature_scaler.transform(X)
        if self.is_attributed:
            X = np.hstack([X, self._endpoint_features(df)])
        return X.astype(np.float32)

    def _endpoint_features(self, df: pd.DataFrame) -> np.ndarray:
        """Return endpoint one-hot features aligned with `df` rows."""
        if self.source_col not in df.columns:
            raise ValueError(
                f"Missing required source column `{self.source_col}` for attributed learning prediction. "
                f"This column should contain the endpoint identifier (data source) for each row."
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
        """Fit a fresh estimator on `df`."""
        if self.target_col not in df.columns:
            raise ValueError(
                f"Missing required target column `{self.target_col}` in training dataframe."
            )

        self.model = self._new_model()
        if self.endpoint_ohe_map is None and self.is_attributed:
            cast(Any, self)._create_endpoint_map(df[self.source_col])
        if self.params:
            self.set_hyperparameters(self.params)
        X = self._featurize(df, fit_scaler=True)
        y = np.array(df[self.target_col], dtype=np.float32)
        cast(Any, self.model).fit(X, y)

    def predict(self, df: pd.DataFrame) -> list[float]:
        """Predict values (or positive-class probabilities) for every row of `df`."""
        if self.model is None:
            raise ValueError(
                "Model is not initialized. Call `train` or `optimize` first."
            )
        X = self._featurize(df)

        num_nan = int(np.isnan(X).sum())
        num_inf = int(np.isinf(X).sum())
        if num_nan or num_inf:
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
        """Return hyperparameters with numpy scalars converted to Python types."""
        if self.model is None:
            hyperparams = self.params
        else:
            hyperparams = self._without_thread_param(self.model.get_params())
        return {
            k: (v.item() if isinstance(v, np.generic) else v)
            for k, v in hyperparams.items()
        }

    def set_hyperparameters(self, params: dict[str, Any]) -> None:
        """Validate and apply estimator hyperparameters."""
        if self.model is None:
            self.model = self._new_model()
        params = self._without_thread_param(params)
        model_params = self.model.get_params()
        unknown = [key for key in params if key not in model_params]
        if unknown:
            raise ValueError(
                f"Model {self.model.__class__.__name__} does not accept hyperparameter(s) {unknown}."
            )
        # Persist so params survive re-creating the model in `train`.
        self.params = dict(params)
        self.model.set_params(**params)