import logging

import numpy as np
from src.utils import get_nice_class_name
from src.predictor.predictor_base import PredictorBase
from src.data.featurizer import FeaturizerBase
import sklearn
from typing import List
from sklearn.utils.validation import check_array
import abc
from src.utils import get_metric_callable


class ScikitPredictor(PredictorBase):
    """
    Common interface for scikit-learn-based regressors and classifiers.
    """

    def __init__(
        self,
        params: dict | None = None,
        optimize_hyperparameters: bool = False,
        target_metric: str | None = None,
        params_distribution: dict | None = None,
        optimization_iterations: int | None = None,
        n_folds: int | None = None,
        n_jobs: int | None = None,
        **kwargs,
    ):
        super().__init__()

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

        self.hyper_opt = {
            "n_iter": optimization_iterations,
            "n_folds": n_folds,
            "n_jobs": n_jobs,
            "params_distribution": params_distribution,
        }

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
            logging.info(
                f"Starting {get_nice_class_name(self.model)} hyperparameter optimization"
            )
            self.train_optimize(X, y)
        else:
            # Use a set of fixed hyperparameters
            logging.info(
                f"Training {get_nice_class_name(self.model)} with fixed hyperparameters"
            )
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

    def predict(self, smiles_list: List[str]) -> List[float]:
        # Featurize the smiles
        X = self.featurizer.featurize(smiles_list)
        # Cast to numpy array
        X = np.array(X, dtype=np.float32)
        # Check for and log the number of data points with NaN or infinite features
        if np.isnan(X).any() or np.isinf(X).any():
            num_nan = np.sum(np.isnan(X))
            num_inf = np.sum(np.isinf(X))
            logging.warning(
                f"Input data contains {num_nan} NaN values and {num_inf} infinite values. "
                "These will be replaced with 0s."
            )
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        # Ensure the input is a 2D array with finite values
        X = check_array(X, ensure_all_finite=True, dtype=np.float32)
        if hasattr(self.model, "predict_proba"):
            # If the model has a predict_proba method, return probabilities
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
                    f"Model {type(model).__name__} does not accept hyperparameter {key}."
                )


class ScikitRegressor(ScikitPredictor):
    """
    A class to interface with scikit-learn-based regression models.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # evaluation metrics for regressor models
        self.evaluation_metrics = ["mse", "rmse", "mae", "r2"]

    def evaluate(self, smiles_list: List[str], target_list: List[float]) -> dict:
        preds = self.predict(smiles_list)
        metrics_dict = {}
        for m in self.evaluation_metrics:
            metrics_dict[m] = get_metric_callable(m)(target_list, preds)
        return metrics_dict


class ScikitBinaryClassifier(ScikitPredictor):
    """
    A class to interface with scikit-learn-based binary classification models.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_threshold = 0.5
        # evaluation metrics for classifier models
        self.evaluation_metrics = ["accuracy", "roc_auc", "f1", "precision", "recall"]

    def evaluate(self, smiles_list: List[str], target_list: List[float]) -> dict:
        preds = self.predict(smiles_list)
        binary_preds = self.classify(preds)
        metrics_dict = {}
        for m in self.evaluation_metrics:
            if m == "roc_auc":
                # roc_auc needs class probabilities
                metrics_dict[m] = get_metric_callable(m)(target_list, preds)
            else:
                metrics_dict[m] = get_metric_callable(m)(target_list, binary_preds)
        return metrics_dict

    def classify(self, preds):
        return np.array(preds) > self.class_threshold
