import logging

import numpy as np
from src.utils import get_nice_class_name
from src.predictor.base import PredictorBase
from src.featurizer import FeaturizerBase
import sklearn
import gin
from typing import List, Tuple


class ScikitPredictor(PredictorBase):
    """
    Represents a Scikit-learn predictive model
    :param params: Hyperparameters for the model
    :param optimize_hyperparameters: If True, hyperparameters will be optimized
    :param target_metric: Metric to optimize for
    :param evaluation_metrics: List of metrics for model evaluation
    :param params_distribution: Distribution of hyperparameters for optimization
    :param optimization_iterations: Number of iterations for optimization
    :param n_folds: Number of folds for cross-validation
    :param n_jobs: Number of jobs for parallel processing
    """

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
        super(ScikitPredictor, self).__init__()

        # Initialize the model
        self.model = self._init_model()

        # Initialize the featurizer
        self.featurizer = None

        # Set the hyperparameters
        if not (params is None or optimize_hyperparameters):
            # Check if params will be recognized by the model
            self._check_params(self.model, params)

            self.model.set_params(**params)

        # Params for hyperparameter optimalization with randomized search CV
        self.optimize = optimize_hyperparameters

        # Set the target metric for optimization and metrics for final evaluation
        self.target_metric = target_metric
        self.evaluation_metrics = evaluation_metrics

        self.hyper_opt = {
            "n_iter": optimization_iterations,
            "n_folds": n_folds,
            "n_jobs": n_jobs,
            "params_distribution": params_distribution,
        }

    def _init_model(self):
        """
        Initialize a scikit-learn model
        """
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    def inject_featurizer(self, featurizer):
        """
        Inject a featurizer into the model
        :param featurizer: Featurizer object
        """
        if not isinstance(featurizer, FeaturizerBase):
            raise ValueError("Featurizer must be an instance of FeaturizerBase")
        logging.info(f"Using {get_nice_class_name(featurizer)} for featurization")
        self.featurizer = featurizer

    def train(self, smiles_list: List[str], target_list: List[float]):

        # Featurize the smiles
        if self.featurizer is None:
            raise ValueError("Featurizer is not set. Please inject a featurizer first.")
        X = self.featurizer.featurize(smiles_list)
        y = target_list

        # Train the model
        if self.optimize:
            # Use random search to optimize hyperparameters
            self.train_optimize(X, y)
        else:
            # Use a set of fixed hyperparameters
            self.model.fit(X, y)

        logging.info(f"Fitting of {get_nice_class_name(self.model)} has converged")

    def train_optimize(self, smiles_list: List[str], target_list: List[float]):

        # Use random search to optimize hyperparameters
        random_search = sklearn.model_selection.RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.hyper_opt["params_distribution"],
            n_iter=self.hyper_opt["n_iter"],
            cv=self.hyper_opt["n_folds"],
            verbose=2,
            n_jobs=self.hyper_opt["n_jobs"],
            refit=True,
            scoring=self.target_metric,
        )

        # Fit the model
        random_search.fit(smiles_list, target_list)

        # Save only the best model after refitting to the whole training data
        self.model = random_search.best_estimator_

        logging.info(
            f"RandomSearchCV: Fitting converged. Keeping the best model, with params: "
            f"{random_search.best_params_}"
        )

    def predict(self, smiles_list: List[str]) -> Tuple[np.array, np.array]:
        # Featurize the smiles
        X = self.featurizer.featurize(smiles_list)
        if hasattr(self.model, "predict_proba"):
            y_pred = self.model.predict_proba(X)
            y_pred = np.array([y[1] for y in y_pred])
        else:
            y_pred = self.model.predict(X)
        return y_pred

    @staticmethod
    def _check_params(model, params):
        model_params = model.get_params()
        for key in params:
            if key not in model_params:
                raise ValueError(
                    f"Model {type(model).__name__} does not accept hyperparameter {key}"
                )


@gin.configurable()
class RfRegressor(ScikitPredictor):
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
    def task_type(self) -> str:
        return "regressor"


@gin.configurable()
class RfClassifier(ScikitPredictor):
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
    def task_type(self) -> str:
        return "classifier"


@gin.configurable()
class SvmRegressor(ScikitPredictor):
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
    def task_type(self) -> str:
        return "regressor"


@gin.configurable()
class SvmClassifier(ScikitPredictor):
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
    def task_type(self) -> str:
        return "classifier"
