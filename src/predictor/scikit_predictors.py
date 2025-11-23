import sklearn
import gin
from typing import List
from src.predictor.scikit_base import ScikitRegressor, ScikitBinaryClassifier
import xgboost as xgb


@gin.configurable()
class RfRegressor(ScikitRegressor):
    def __init__(
        self,
        params: dict | None = None,
        optimize_hyperparameters: bool = False,
        params_distribution: dict | None = None,
        target_metric: str | None = None,
        evaluation_metrics: List[str] | None = None,
        optimization_iterations: int | None = None,
        n_folds: int | None = None,
        n_jobs: int | None = None,
    ):
        super().__init__(
            params=params,
            optimize_hyperparameters=optimize_hyperparameters,
            params_distribution=params_distribution,
            optimization_iterations=optimization_iterations,
            target_metric=target_metric,
            evaluation_metrics=evaluation_metrics,
            n_folds=n_folds,
            n_jobs=n_jobs,
        )

    def _init_model(self):
        return sklearn.ensemble.RandomForestRegressor()

    @property
    def name(self) -> str:
        return "RF_reg"


@gin.configurable()
class RfClassifier(ScikitBinaryClassifier):
    def __init__(
        self,
        params: dict | None = None,
        optimize_hyperparameters: bool = False,
        target_metric: str | None = None,
        evaluation_metrics: List[str] | None = None,
        params_distribution: dict | None = None,
        optimization_iterations: int | None = None,
        n_folds: int | None = None,
        n_jobs: int | None = None,
    ):
        super().__init__(
            params=params,
            optimize_hyperparameters=optimize_hyperparameters,
            target_metric=target_metric,
            evaluation_metrics=evaluation_metrics,
            params_distribution=params_distribution,
            optimization_iterations=optimization_iterations,
            n_folds=n_folds,
            n_jobs=n_jobs,
        )

    def _init_model(self):
        return sklearn.ensemble.RandomForestClassifier()

    @property
    def name(self) -> str:
        return "RF_clf"


@gin.configurable()
class SvmRegressor(ScikitRegressor):
    def __init__(
        self,
        params: dict | None = None,
        optimize_hyperparameters: bool = False,
        target_metric: str | None = None,
        evaluation_metrics: List[str] | None = None,
        params_distribution: dict | None = None,
        optimization_iterations: int | None = None,
        n_folds: int | None = None,
        n_jobs: int | None = None,
    ):
        super().__init__(
            params=params,
            optimize_hyperparameters=optimize_hyperparameters,
            target_metric=target_metric,
            evaluation_metrics=evaluation_metrics,
            params_distribution=params_distribution,
            optimization_iterations=optimization_iterations,
            n_folds=n_folds,
            n_jobs=n_jobs,
        )

    def _init_model(self):
        return sklearn.svm.SVR()

    @property
    def name(self) -> str:
        return "SVM_reg"


@gin.configurable()
class SvmClassifier(ScikitBinaryClassifier):
    def __init__(
        self,
        params: dict | None = None,
        optimize_hyperparameters: bool = False,
        target_metric: str | None = None,
        evaluation_metrics: List[str] | None = None,
        params_distribution: dict | None = None,
        optimization_iterations: int | None = None,
        n_folds: int | None = None,
        n_jobs: int | None = None,
    ):
        super().__init__(
            params=params,
            optimize_hyperparameters=optimize_hyperparameters,
            target_metric=target_metric,
            evaluation_metrics=evaluation_metrics,
            params_distribution=params_distribution,
            optimization_iterations=optimization_iterations,
            n_folds=n_folds,
            n_jobs=n_jobs,
        )

    def _init_model(self):
        return sklearn.svm.SVC(probability=True)

    @property
    def name(self) -> str:
        return "SVM_clf"


@gin.configurable()
class XGBoostRegressor(ScikitRegressor):
    # TODO: What hyperparameters does this accept?
    def __init__(
        self,
        params: dict | None = None,
        optimize_hyperparameters: bool = False,
        target_metric: str | None = None,
        evaluation_metrics: List[str] | None = None,
        params_distribution: dict | None = None,
        optimization_iterations: int | None = None,
        n_folds: int | None = None,
        n_jobs: int | None = None,
    ):
        super().__init__(
            params=params,
            optimize_hyperparameters=optimize_hyperparameters,
            target_metric=target_metric,
            evaluation_metrics=evaluation_metrics,
            params_distribution=params_distribution,
            optimization_iterations=optimization_iterations,
            n_folds=n_folds,
            n_jobs=n_jobs,
        )

    def _init_model(self):
        return xgb.XGBRegressor()

    @property
    def name(self) -> str:
        return "XGB_reg"


@gin.configurable()
class XGBoostClassifier(ScikitBinaryClassifier):
    # TODO: What hyperparameters does this accept?
    def __init__(
        self,
        params: dict | None = None,
        optimize_hyperparameters: bool = False,
        target_metric: str | None = None,
        evaluation_metrics: List[str] | None = None,
        params_distribution: dict | None = None,
        optimization_iterations: int | None = None,
        n_folds: int | None = None,
        n_jobs: int | None = None,
    ):
        super().__init__(
            params=params,
            optimize_hyperparameters=optimize_hyperparameters,
            target_metric=target_metric,
            evaluation_metrics=evaluation_metrics,
            params_distribution=params_distribution,
            optimization_iterations=optimization_iterations,
            n_folds=n_folds,
            n_jobs=n_jobs,
        )

    def _init_model(self):
        return xgb.XGBClassifier()

    @property
    def name(self) -> str:
        return "XGB_clf"
