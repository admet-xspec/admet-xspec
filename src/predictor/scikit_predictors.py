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
