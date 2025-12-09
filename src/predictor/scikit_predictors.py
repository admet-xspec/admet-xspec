import sklearn
import gin
from src.predictor.scikit_base import ScikitPredictorBase
from src.predictor.predictor_base import RegressorBase, BinaryClassifierBase
import xgboost as xgb
import lightgbm as lgb
import logging


@gin.configurable()
class RfRegressor(ScikitPredictorBase, RegressorBase):
    def __init__(
        self,
        params: dict | None = None,
        params_distribution: dict | None = None,
        target_metric: str | None = None,
        optimization_iterations: int | None = None,
        n_folds: int | None = None,
        n_jobs: int | None = None,
        random_state: int = 42,
    ):
        super().__init__(
            params=params,
            params_distribution=params_distribution,
            optimization_iterations=optimization_iterations,
            target_metric=target_metric,
            n_folds=n_folds,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def _init_model(self):
        return sklearn.ensemble.RandomForestRegressor()

    @property
    def name(self) -> str:
        return "RF_reg"


@gin.configurable()
class RfClassifier(ScikitPredictorBase, BinaryClassifierBase):
    def __init__(
        self,
        params: dict | None = None,
        target_metric: str | None = None,
        params_distribution: dict | None = None,
        optimization_iterations: int | None = None,
        n_folds: int | None = None,
        n_jobs: int | None = None,
        random_state: int = 42,
    ):
        super().__init__(
            params=params,
            target_metric=target_metric,
            params_distribution=params_distribution,
            optimization_iterations=optimization_iterations,
            n_folds=n_folds,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def _init_model(self):
        return sklearn.ensemble.RandomForestClassifier()

    @property
    def name(self) -> str:
        return "RF_clf"


@gin.configurable()
class SvmRegressor(ScikitPredictorBase, RegressorBase):
    def __init__(
        self,
        params: dict | None = None,
        target_metric: str | None = None,
        params_distribution: dict | None = None,
        optimization_iterations: int | None = None,
        n_folds: int | None = None,
        n_jobs: int | None = None,
        random_state: int = 42,
    ):
        super().__init__(
            params=params,
            target_metric=target_metric,
            params_distribution=params_distribution,
            optimization_iterations=optimization_iterations,
            n_folds=n_folds,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def _init_model(self):
        return sklearn.svm.SVR()

    @property
    def name(self) -> str:
        return "SVM_reg"


@gin.configurable()
class SvmClassifier(ScikitPredictorBase, BinaryClassifierBase):
    def __init__(
        self,
        params: dict | None = None,
        target_metric: str | None = None,
        params_distribution: dict | None = None,
        optimization_iterations: int | None = None,
        n_folds: int | None = None,
        n_jobs: int | None = None,
        random_state: int = 42,
    ):
        super().__init__(
            params=params,
            target_metric=target_metric,
            params_distribution=params_distribution,
            optimization_iterations=optimization_iterations,
            n_folds=n_folds,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def _init_model(self):
        return sklearn.svm.SVC(probability=True)

    @property
    def name(self) -> str:
        return "SVM_clf"


@gin.configurable()
class XGBoostRegressor(ScikitPredictorBase, RegressorBase):
    # TODO: What hyperparameters does this accept?
    def __init__(
        self,
        params: dict | None = None,
        target_metric: str | None = None,
        params_distribution: dict | None = None,
        optimization_iterations: int | None = None,
        n_folds: int | None = None,
        n_jobs: int | None = None,
        random_state: int = 42,
    ):
        super().__init__(
            params=params,
            target_metric=target_metric,
            params_distribution=params_distribution,
            optimization_iterations=optimization_iterations,
            n_folds=n_folds,
            n_jobs=n_jobs,
            random_state=42,
        )

    def _init_model(self):
        return xgb.XGBRegressor()

    @property
    def name(self) -> str:
        return "XGB_reg"


@gin.configurable()
class XGBoostClassifier(ScikitPredictorBase, BinaryClassifierBase):
    # TODO: What hyperparameters does this accept?
    def __init__(
        self,
        params: dict | None = None,
        target_metric: str | None = None,
        params_distribution: dict | None = None,
        optimization_iterations: int | None = None,
        n_folds: int | None = None,
        n_jobs: int | None = None,
        random_state: int = 42,
    ):
        super().__init__(
            params=params,
            target_metric=target_metric,
            params_distribution=params_distribution,
            optimization_iterations=optimization_iterations,
            n_folds=n_folds,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def _init_model(self):
        return xgb.XGBClassifier()

    @property
    def name(self) -> str:
        return "XGB_clf"

@gin.configurable()
class LightGbmClassifier(ScikitPredictorBase, BinaryClassifierBase):

    def __init__(
        self,
        params: dict | None = None,
        target_metric: str | None = None,
        params_distribution: dict | None = None,
        optimization_iterations: int | None = None,
        n_folds: int | None = None,
        n_jobs: int | None = None,
        random_state: int = 42,
    ):
        validate_lgbm_specific_params(params)
        super().__init__(
            params=params,
            target_metric=target_metric,
            params_distribution=params_distribution,
            optimization_iterations=optimization_iterations,
            n_folds=n_folds,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def _init_model(self):
        return lgb.LGBMClassifier()

    @property
    def name(self) -> str:
        return "LightGBM_clf"

    @staticmethod
    def _validate_predictor_specific_params(params: dict):
        if 'max_depth' in params and 'num_leaves' in params:
            max_depth = params['max_depth']
            num_leaves = params['num_leaves']
            if num_leaves > 2 ** max_depth:
                logging.warning(
                    f"WARNING: num_leaves ({num_leaves}) should not be greater than 2^max_depth ({2 ** max_depth}). Adjusting num_leaves to {2 ** max_depth}."
                )

@gin.configurable()
class LightGbmRegressor(ScikitPredictorBase, RegressorBase):

    def __init__(
            self,
            params: dict | None = None,
            target_metric: str | None = None,
            params_distribution: dict | None = None,
            optimization_iterations: int | None = None,
            n_folds: int | None = None,
            n_jobs: int | None = None,
            random_state: int = 42,
    ):
        validate_lgbm_specific_params(params)
        super().__init__(
            params=params,
            target_metric=target_metric,
            params_distribution=params_distribution,
            optimization_iterations=optimization_iterations,
            n_folds=n_folds,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def _init_model(self):
        return lgb.LGBMRegressor(random_state=self.random_state)

    @property
    def name(self) -> str:
        return "LightGBM_reg"


# Helper function to validate LightGBM parameters

def validate_lgbm_specific_params(params: dict):
    if params is None:
        return
    if 'max_depth' in params and 'num_leaves' in params and params['max_depth'] != -1:
        max_depth = params['max_depth']
        num_leaves = params['num_leaves']
        if num_leaves > 2 ** max_depth:
            logging.warning(
                f"WARNING: num_leaves ({num_leaves}) should not be greater than 2^max_depth ({2 ** max_depth}). Adjusting num_leaves to {2 ** max_depth}."
            )