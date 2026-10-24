import logging
import re
import warnings
from typing import Any, Type

import gin
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import sklearn
import sklearn.neural_network
import xgboost as xgb
from sklearn.exceptions import ConvergenceWarning

from src.predictor.predictor_base import BinaryClassifierBase, RegressorBase
from src.predictor.scikit_base import ScikitPredictorBase
from src.utils import sample_optuna_params


class _ScikitEstimatorBase(ScikitPredictorBase):
    """Common estimator-construction logic for sklearn-style predictors."""

    estimator_cls: Type[Any] | None = None

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        random_state: int = 42,
        do_attributed_learning: bool = False,
        n_threads: int | None = None,
    ) -> None:
        super().__init__(
            params=params,
            random_state=random_state,
            do_attributed_learning=do_attributed_learning,
            n_threads=n_threads,
        )

    def _base_estimator_kwargs(self) -> dict[str, Any]:
        """Return default constructor kwargs for `estimator_cls`."""
        return {}

    def _init_model(self) -> Any:
        """Instantiate `estimator_cls`."""
        if self.estimator_cls is None:
            raise NotImplementedError("Subclasses must define estimator_cls")
        return self.estimator_cls(**self._base_estimator_kwargs())


class _ScikitRegressorBase(_ScikitEstimatorBase, RegressorBase):
    """Shared behavior for sklearn regressors."""


class _ScikitClassifierBase(_ScikitEstimatorBase, BinaryClassifierBase):
    """Shared behavior for sklearn binary classifiers."""


@gin.configurable()
class RfRegressor(_ScikitRegressorBase):
    """RandomForest regressor wrapper."""

    estimator_cls = sklearn.ensemble.RandomForestRegressor
    thread_param = "n_jobs"

    @property
    def name(self) -> str:
        return "RF_reg"


@gin.configurable()
class RfClassifier(_ScikitClassifierBase):
    """RandomForest classifier wrapper."""

    estimator_cls = sklearn.ensemble.RandomForestClassifier
    thread_param = "n_jobs"

    @property
    def name(self) -> str:
        return "RF_clf"


@gin.configurable()
class SvmRegressor(_ScikitRegressorBase):
    """SVM regressor wrapper."""

    estimator_cls = sklearn.svm.SVR

    @property
    def name(self) -> str:
        return "SVM_reg"


@gin.configurable()
class SvmClassifier(_ScikitClassifierBase):
    """SVM classifier wrapper with probability output enabled."""

    estimator_cls = sklearn.svm.SVC

    def _base_estimator_kwargs(self) -> dict[str, Any]:
        # Enables `predict_proba`.
        return {"probability": True}

    @property
    def name(self) -> str:
        return "SVM_clf"


@gin.configurable()
class XGBoostRegressor(_ScikitRegressorBase):
    """XGBoost regressor wrapper."""

    estimator_cls = xgb.XGBRegressor
    thread_param = "n_jobs"

    @property
    def name(self) -> str:
        return "XGB_reg"


@gin.configurable()
class XGBoostClassifier(_ScikitClassifierBase):
    """XGBoost classifier wrapper."""

    estimator_cls = xgb.XGBClassifier
    thread_param = "n_jobs"

    @property
    def name(self) -> str:
        return "XGB_clf"


@gin.configurable()
class LightGbmClassifier(_ScikitClassifierBase):
    """LightGBM classifier wrapper."""

    estimator_cls = lgb.LGBMClassifier
    thread_param = "n_jobs"

    def _base_estimator_kwargs(self) -> dict[str, Any]:
        return {"random_state": self.random_state, "verbosity": -1}

    @property
    def name(self) -> str:
        return "LightGBM_clf"


@gin.configurable()
class LightGbmRegressor(_ScikitRegressorBase):
    """LightGBM regressor wrapper."""

    estimator_cls = lgb.LGBMRegressor
    thread_param = "n_jobs"

    def _base_estimator_kwargs(self) -> dict[str, Any]:
        return {"random_state": self.random_state, "verbosity": -1}

    @property
    def name(self) -> str:
        return "LightGBM_reg"


class _ScikitMlpBase(_ScikitEstimatorBase):
    """Shared behavior for sklearn MLPs.

    Optuna cannot sample tuples, and tuples do not survive the hyperparams YAML
    round-trip, so the architecture can also be given as the scalars
    `n_hidden_layers`, `hidden_layer_width` (the first layer's width) and
    `layer_shape`, which sets the widths of the following layers:

    - `constant`: all layers are `hidden_layer_width` wide (the default);
    - `funnel`: each layer is half as wide as the previous one;
    - `free`: layer `i` (from 2) is `hidden_layer_width_<i>` wide.

    `hidden_layer_sizes` is always stored as a list of ints.

    The stopping criterion is set through the constructor (and so gin):
    `early_stopping` stops on the score of a held-out `validation_fraction` of
    the training data (R² for regressors, accuracy for classifiers) and keeps
    the best epoch's weights; otherwise training stops on the training loss and
    keeps the last epoch's weights. Either way, training stops after
    `n_iter_no_change` epochs without an improvement greater than `tol`, or
    after `max_iter` epochs, which is logged as a convergence failure. Values
    in `params` take precedence.
    """

    arch_params = ("n_hidden_layers", "hidden_layer_width")
    layer_shapes = ("constant", "funnel", "free")
    _layer_width_re = re.compile(r"hidden_layer_width_\d+")

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        random_state: int = 42,
        do_attributed_learning: bool = False,
        n_threads: int | None = None,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
        tol: float = 1e-4,
        max_iter: int = 1000,
    ) -> None:
        super().__init__(
            params=params,
            random_state=random_state,
            do_attributed_learning=do_attributed_learning,
            n_threads=n_threads,
        )
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.max_iter = max_iter

    def _base_estimator_kwargs(self) -> dict[str, Any]:
        return {
            "random_state": self.random_state,
            "early_stopping": self.early_stopping,
            "validation_fraction": self.validation_fraction,
            "n_iter_no_change": self.n_iter_no_change,
            "tol": self.tol,
            "max_iter": self.max_iter,
        }

    def train(self, df: pd.DataFrame) -> None:
        """Fit a fresh estimator on `df`, logging a failure to converge.

        Other warnings raised during fitting are re-emitted unchanged.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            super().train(df)
        for warning in caught:
            if issubclass(warning.category, ConvergenceWarning):
                self._log_convergence_error(warning.message)
            else:
                warnings.warn_explicit(
                    warning.message, warning.category, warning.filename, warning.lineno
                )

    def _log_convergence_error(self, message: Warning | str) -> None:
        """Log a convergence failure of the fitted model with its training state."""
        model: Any = self.model
        details = [
            f"hidden_layer_sizes={self._as_layer_list(model.hidden_layer_sizes)}",
            f"learning_rate_init={model.learning_rate_init}",
            f"alpha={model.alpha}",
        ]
        loss = getattr(model, "loss_", None)
        if loss is not None:
            details.append(f"final loss={loss:.4g}")
        # Recorded only with early stopping.
        best_score = getattr(model, "best_validation_score_", None)
        if best_score is not None:
            details.append(f"best validation score={best_score:.4g}")
        logging.warning(
            f"{self.name} (task: {self.task}) did not converge after "
            f"{getattr(model, 'n_iter_', '?')}/{model.max_iter} iterations "
            f"({', '.join(details)}): {message}"
        )

    def sample_hyperparameters(
        self, trial: optuna.Trial, params_distribution: dict[str, Any]
    ) -> dict[str, Any]:
        """Sample hyperparameters, adding per-layer widths for the `free` shape.

        The per-layer widths reuse the `hidden_layer_width` distribution and are
        sampled only by trials with a `free` shape and enough layers.
        """
        distribution = dict(params_distribution)
        shape_spec = distribution.pop("layer_shape", None)
        params = sample_optuna_params(trial, distribution)
        if shape_spec is None:
            return params
        missing = [key for key in self.arch_params if key not in distribution]
        if missing:
            raise ValueError(
                f"Sampling `layer_shape` requires distributions for {missing}."
            )
        params.update(sample_optuna_params(trial, {"layer_shape": shape_spec}))
        if params["layer_shape"] == "free":
            width_spec = distribution["hidden_layer_width"]
            width_names = self._layer_width_names(params["n_hidden_layers"])
            params.update(
                sample_optuna_params(trial, {name: width_spec for name in width_names})
            )
        return params

    def set_hyperparameters(self, params: dict[str, Any]) -> None:
        """Translate architecture scalars into `hidden_layer_sizes`, then apply."""
        params = dict(params)
        layer_widths = {
            key: params.pop(key)
            for key in list(params)
            if self._layer_width_re.fullmatch(key)
        }
        shape = params.pop("layer_shape", None)
        given = [key for key in self.arch_params if key in params]
        if given or shape is not None or layer_widths:
            if len(given) != len(self.arch_params):
                raise ValueError(
                    f"Hyperparameters {list(self.arch_params)} must be given together "
                    f"(also with `layer_shape` or per-layer widths), got only {given}."
                )
            if "hidden_layer_sizes" in params:
                raise ValueError(
                    f"Give either `hidden_layer_sizes` or {list(self.arch_params)}, not both."
                )
            params["hidden_layer_sizes"] = self._build_layer_sizes(
                n_layers=int(params.pop("n_hidden_layers")),
                width=int(params.pop("hidden_layer_width")),
                shape=shape or "constant",
                layer_widths=layer_widths,
            )
        if "hidden_layer_sizes" in params:
            params["hidden_layer_sizes"] = self._as_layer_list(
                params["hidden_layer_sizes"]
            )
        super().set_hyperparameters(params)

    def get_hyperparameters(self) -> dict[str, Any]:
        """Return hyperparameters with `hidden_layer_sizes` as a list of ints."""
        hyperparams = super().get_hyperparameters()
        if "hidden_layer_sizes" in hyperparams:
            hyperparams["hidden_layer_sizes"] = self._as_layer_list(
                hyperparams["hidden_layer_sizes"]
            )
        return hyperparams

    def _build_layer_sizes(
        self,
        n_layers: int,
        width: int,
        shape: str,
        layer_widths: dict[str, Any],
    ) -> list[int]:
        """Return `hidden_layer_sizes` for the given architecture scalars."""
        if n_layers < 1 or width < 1:
            raise ValueError(
                f"`n_hidden_layers` and `hidden_layer_width` must be >= 1, got {n_layers} and {width}."
            )
        if shape not in self.layer_shapes:
            raise ValueError(
                f"Unknown `layer_shape` '{shape}'. Supported: {list(self.layer_shapes)}."
            )
        expected = self._layer_width_names(n_layers) if shape == "free" else []
        if sorted(layer_widths) != sorted(expected):
            raise ValueError(
                f"`layer_shape` '{shape}' with {n_layers} layer(s) takes per-layer widths "
                f"{expected}, got {sorted(layer_widths)}."
            )
        if shape == "constant":
            return [width] * n_layers
        if shape == "funnel":
            return [max(1, width // 2**i) for i in range(n_layers)]
        sizes = [width] + [int(layer_widths[name]) for name in expected]
        if min(sizes) < 1:
            raise ValueError(f"Per-layer widths must be >= 1, got {sizes}.")
        return sizes

    @staticmethod
    def _layer_width_names(n_layers: int) -> list[str]:
        """Return the per-layer width names of the `free` shape (from layer 2)."""
        return [f"hidden_layer_width_{i}" for i in range(2, n_layers + 1)]

    @staticmethod
    def _as_layer_list(sizes: Any) -> list[int]:
        """Return layer sizes (an int or a sequence) as a list of Python ints."""
        return [int(size) for size in np.atleast_1d(sizes)]


@gin.configurable()
class MlpRegressor(_ScikitMlpBase, RegressorBase):
    """Multi-layer perceptron regressor wrapper."""

    estimator_cls = sklearn.neural_network.MLPRegressor

    @property
    def name(self) -> str:
        return "MLP_reg"


@gin.configurable()
class MlpClassifier(_ScikitMlpBase, BinaryClassifierBase):
    """Multi-layer perceptron classifier wrapper."""

    estimator_cls = sklearn.neural_network.MLPClassifier

    @property
    def name(self) -> str:
        return "MLP_clf"


def validate_lgbm_specific_params(params: dict[str, Any] | None) -> None:
    """Warn about potentially inconsistent LightGBM depth/leaf combinations."""
    if params is None or "max_depth" not in params or "num_leaves" not in params:
        return
    max_depth = params["max_depth"]
    num_leaves = params["num_leaves"]
    if max_depth != -1 and num_leaves > 2**max_depth:
        logging.warning(
            f"num_leaves ({num_leaves}) should not be greater than 2^max_depth ({2 ** max_depth})."
        )