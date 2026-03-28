import logging
from typing import Any, Type

import gin
import lightgbm as lgb
import sklearn
import xgboost as xgb

from src.predictor.predictor_base import BinaryClassifierBase, RegressorBase
from src.predictor.scikit_base import ScikitPredictorBase


class _ScikitRegressorBase(ScikitPredictorBase, RegressorBase):
    """Shared init for sklearn regressors."""

    estimator_cls: Type[Any] | None = None

    def __init__(
        self,
        params: dict | None = None,
        random_state: int = 42,
        multi_endpoint: bool = False,
    ):
        super().__init__(
            params=params,
            random_state=random_state,
            multi_endpoint=multi_endpoint,
        )

    def _init_model(self):
        if self.estimator_cls is None:
            raise NotImplementedError("Subclasses must define estimator_cls")
        return self.estimator_cls()


class _ScikitClassifierBase(ScikitPredictorBase, BinaryClassifierBase):
    """Shared init for sklearn binary classifiers."""

    estimator_cls: Type[Any] | None = None

    def __init__(
        self,
        params: dict | None = None,
        random_state: int = 42,
        multi_endpoint: bool = False,
    ):
        super().__init__(
            params=params,
            random_state=random_state,
            multi_endpoint=multi_endpoint,
        )

    def _init_model(self):
        if self.estimator_cls is None:
            raise NotImplementedError("Subclasses must define estimator_cls")
        return self.estimator_cls()


@gin.configurable()
class RfRegressor(_ScikitRegressorBase):
    estimator_cls = sklearn.ensemble.RandomForestRegressor

    @property
    def name(self) -> str:
        return "RF_reg"


@gin.configurable()
class RfClassifier(_ScikitClassifierBase):
    estimator_cls = sklearn.ensemble.RandomForestClassifier

    @property
    def name(self) -> str:
        return "RF_clf"


@gin.configurable()
class SvmRegressor(_ScikitRegressorBase):
    estimator_cls = sklearn.svm.SVR

    @property
    def name(self) -> str:
        return "SVM_reg"


@gin.configurable()
class SvmClassifier(_ScikitClassifierBase):
    def _init_model(self):
        return sklearn.svm.SVC(probability=True)

    @property
    def name(self) -> str:
        return "SVM_clf"


@gin.configurable()
class XGBoostRegressor(_ScikitRegressorBase):
    # TODO: What hyperparameters does this accept?
    def _init_model(self):
        return xgb.XGBRegressor()

    @property
    def name(self) -> str:
        return "XGB_reg"


@gin.configurable()
class XGBoostClassifier(_ScikitClassifierBase):
    # TODO: What hyperparameters does this accept?
    def _init_model(self):
        return xgb.XGBClassifier()

    @property
    def name(self) -> str:
        return "XGB_clf"


@gin.configurable()
class LightGbmClassifier(_ScikitClassifierBase):

    def _init_model(self):
        return lgb.LGBMClassifier(random_state=self.random_state, verbosity=-1)

    @property
    def name(self) -> str:
        return "LightGBM_clf"


@gin.configurable()
class LightGbmRegressor(_ScikitRegressorBase):

    def _init_model(self):
        return lgb.LGBMRegressor(random_state=self.random_state, verbosity=-1)

    @property
    def name(self) -> str:
        return "LightGBM_reg"


def validate_lgbm_specific_params(params: dict):
    if params is None:
        return
    if "max_depth" in params and "num_leaves" in params and params["max_depth"] != -1:
        max_depth = params["max_depth"]
        num_leaves = params["num_leaves"]
        if num_leaves > 2**max_depth:
            logging.warning(
                f"WARNING: num_leaves ({num_leaves}) should not be greater than 2^max_depth ({2 ** max_depth}). Adjusting num_leaves to {2 ** max_depth}."
            )
