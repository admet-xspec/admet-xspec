import logging

import numpy as np
from src.utils import get_nice_class_name
from src.predictor.PredictorBase import PredictorBase
from src.featurizer.FeaturizerBase import FeaturizerBase
from typing import List
from pathlib import Path
import sklearn
import pickle as pkl
import gin
from typing import List, Dict


@gin.configurable()
class ScikitPredictorBase(PredictorBase):
    """
    Represents a Scikit-learn predictive model

    :param model: Scikit-learn model
    :param params: Hyperparameters for the model as a dictionary
    :param metric: Primary metric for the model as a string
        ("mean_squared_error", "r2_score", "roc_auc_score", "accuracy_score", "f1_score", "precission_score", "recall_score")
    :param optimize_hyperparameters: Whether to optimize hyperparameters using CV random search strategy

    """

    def __init__(
        self,
        model,
        params: dict | None = None,
        metrics: List[str] | None = None,
        primary_metric: str | None = None,
        optimize_hyperparameters: bool = False,
        params_distribution: dict | None = None,
        optimization_iterations: int | None = None,
        n_folds: int | None = None,
        n_jobs: int | None = None,
    ):

        # Initialize the model
        super(ScikitPredictorBase, self).__init__(
            model=model, metrics=metrics, primary_metric=primary_metric
        )

        # Set the hyperparameters
        if params is not None:
            self._check_params(
                self.model, params
            )  # Check if params will be recognized by the model
            self.model.set_params(**params)

        # Params for hyperparameter optimalization with randomised search CV
        self.optimize = optimize_hyperparameters
        self.hyper_opt = {
            "n_iter": optimization_iterations,
            "n_folds": n_folds,
            "n_jobs": n_jobs,
            "params_distribution": params_distribution,
        }

        # Prepare sklearn metrics to use in model evaluation
        self.metrics = [self.supported_metrics[metric_name] for metric_name in metrics]
        self.primary_metric = primary_metric

    def inject_featurizer(self, featurizer):
        """
        Inject a featurizer into the model
        :param featurizer: Featurizer object
        """
        if not isinstance(featurizer, FeaturizerBase):
            raise ValueError("Featurizer must be an instance of FeaturizerBase!")
        logging.info(f"Using {get_nice_class_name(featurizer)} for featurization")
        self.featurizer = featurizer

    def train(self, smiles_list: List[str], target_list: List[float]):

        # Featurize the smiles
        X = self.featurizer.featurize(smiles_list)
        y = target_list

        # Train the model
        if self.optimize:
            # Use random search to optimize hyperparameters
            self.train_CV(X, y)
        else:
            # Use a set of fixed hyperparameters
            self.model.fit(X, y)

        # Get the metrics on the training set
        y_hat = self.model.predict(X)
        train_primary_metric = self.calc_primary_metric(y, y_hat)

        # Signal that the model has been trained
        self.ready_flag = True

        logging.info(f"Fitting of {get_nice_class_name(self.model)} has converged.")
        logging.debug(
            f"Primary metric: {get_nice_class_name(self.primary_metric)} on the training set = {train_primary_metric}"
        )

    def train_CV(self, X, y):

        # Use random search to optimize hyperparameters
        random_search = sklearn.model_selection.RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.hyper_opt["params_distribution"],
            n_iter=self.hyper_opt["n_iter"],
            cv=self.hyper_opt["n_folds"],
            verbose=1,
            n_jobs=self.hyper_opt["n_jobs"],
            refit=True,
        )

        # Fit the model
        logging.info(f"Optimizing hyperparams with RandomSearchCV.")
        logging.info(f"Hyperparameter distribution:")
        for key, value in self.hyper_opt["params_distribution"].items():
            if isinstance(value, list):
                logging.info(f"{key}: {value}")
            else:
                logging.info(f"{key}: {value().__str__()}")

        random_search.fit(X, y)

        # Save only the best model after refitting to the whole training data
        self.model = random_search.estimator

        logging.info(
            f"RandomSearchCV: Fitting converged. Keeping the best model, with params: "
            f"{random_search.best_params}"
        )
        logging.debug(
            f"{self.primary_metric.__name__()} on the training set: {self.calc_p}"
        )

    def predict(self, smiles_list: List[str], ignore_flag=False) -> np.array:
        if not self.ready_flag:
            raise ValueError(
                f"The model has not been fitted to data. Train the model or load a saved "
                f"state first. Alternatively, pass ingore_flag=True to disable this error."
            )
        # Featurize the smiles
        X = self.featurizer.featurize(smiles_list)
        # Predict the target values
        return self.model.predict(X).reshape(-1, 1)

    def save(self, out_dir: str):
        # Check if the output directory exists
        if not Path.is_dir(Path(out_dir)):
            raise FileNotFoundError(f"Directory {out_dir} does not exist")
        # Save the model
        with open(out_dir + "/model.pkl", "wb") as fileout:
            pkl.dump(obj=self.model, file=fileout)
        logging.info(f"Model saved to {out_dir}/model.pkl")

    def load(self, path: str):
        # Check if the file exists
        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist")
        # Check if the file is a pickle file
        if not path.endswith(".pkl"):
            raise ValueError(f"File {path} is not a pickle file")
        # Load the model
        self.model = pkl.load(path)
        # Signal that a trained model has been loaded
        self.ready_flag = True

    @staticmethod
    def _check_params(model, params):
        model_params = model.get_params()
        for key in params:
            if key not in model_params:
                raise ValueError(
                    f"Model {type(model).__name__} does not accept hyperparameter {key}"
                )


@gin.configurable()
class RandomForestRegressor(ScikitPredictorBase):
    def __init__(self, params: dict | None = None):
        super(RandomForestRegressor, self).__init__(
            model=sklearn.ensemble.RandomForestRegressor, params=parmas
        )


@gin.configurable()
class RandomForestClassifier(ScikitPredictorBase):
    def __init__(self, params: dict | None = None):
        super(RandomForestClassifier, self).__init__(
            model=sklearn.ensemble.RandomForestClassifier, params=params
        )


@gin.configurable()
class SvmRegressor(ScikitPredictorBase):
    def __init__(self, params: dict | None = None):
        super(SvmRegressor, self).__init__(model=sklearn.svm.Svm, params=params)


@gin.configurable()
class SvmClassifier(ScikitPredictorBase):
    def __init__(self, params: dict | None = None):
        super(SvmClassifier, self).__init__(model=sklearn.svm.SVC, params=params)
