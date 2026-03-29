import logging
from typing import Any, Type

import gin
import lightgbm as lgb
import sklearn
import xgboost as xgb

from src.predictor.predictor_base import BinaryClassifierBase, RegressorBase
from src.predictor.scikit_base import ScikitPredictorBase


class _ScikitEstimatorBase(ScikitPredictorBase):
    """Common estimator-construction logic for sklearn-style predictors."""

    estimator_cls: Type[Any] | None = None

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        random_state: int = 42,
        multi_endpoint: bool = False,
    ) -> None:
        super().__init__(
            params=params,
            random_state=random_state,
            multi_endpoint=multi_endpoint,
        )

    def _base_estimator_kwargs(self) -> dict[str, Any]:
        """Default constructor kwargs for the underlying estimator class."""
        return {}

    def _init_model(self) -> Any:
        """Instantiate the configured estimator class."""
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

    @property
    def name(self) -> str:
        return "RF_reg"


@gin.configurable()
class RfClassifier(_ScikitClassifierBase):
    """RandomForest classifier wrapper."""

    estimator_cls = sklearn.ensemble.RandomForestClassifier

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
        # Needed so inference can output probabilities, not only labels.
        return {"probability": True}

    @property
    def name(self) -> str:
        return "SVM_clf"


@gin.configurable()
class XGBoostRegressor(_ScikitRegressorBase):
    """XGBoost regressor wrapper."""

    estimator_cls = xgb.XGBRegressor

    @property
    def name(self) -> str:
        return "XGB_reg"


@gin.configurable()
class XGBoostClassifier(_ScikitClassifierBase):
    """XGBoost classifier wrapper."""

    estimator_cls = xgb.XGBClassifier

    @property
    def name(self) -> str:
        return "XGB_clf"


@gin.configurable()
class LightGbmClassifier(_ScikitClassifierBase):
    """LightGBM classifier wrapper."""

    estimator_cls = lgb.LGBMClassifier

    def _base_estimator_kwargs(self) -> dict[str, Any]:
        return {"random_state": self.random_state, "verbosity": -1}

    @property
    def name(self) -> str:
        return "LightGBM_clf"


@gin.configurable()
class LightGbmRegressor(_ScikitRegressorBase):
    """LightGBM regressor wrapper."""

    estimator_cls = lgb.LGBMRegressor

    def _base_estimator_kwargs(self) -> dict[str, Any]:
        return {"random_state": self.random_state, "verbosity": -1}

    @property
    def name(self) -> str:
        return "LightGBM_reg"


def validate_lgbm_specific_params(params: dict[str, Any] | None) -> None:
    """Warn about potentially inconsistent LightGBM depth/leaf combinations."""
    if params is None:
        return
    if "max_depth" in params and "num_leaves" in params and params["max_depth"] != -1:
        max_depth = params["max_depth"]
        num_leaves = params["num_leaves"]
        if num_leaves > 2**max_depth:
            logging.warning(
                f"num_leaves ({num_leaves}) should not be greater than 2^max_depth ({2 ** max_depth})."
            )
