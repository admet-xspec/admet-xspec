import logging

import numpy as np
from src.utils import get_nice_class_name
from src.predictor.PredictorBase import PredictorBase
from src.featurizer.FeaturizerBase import FeaturizerBase
from pathlib import Path
import sklearn
import pickle as pkl
import gin
from typing import List, Tuple


@gin.configurable()
class ScikitPredictor(PredictorBase):
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
        super(ScikitPredictor, self).__init__(
            metrics=metrics, primary_metric=primary_metric
        )
        self.model = model

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
            self.train_optimize(X, y)
        else:
            # Use a set of fixed hyperparameters
            self.model.fit(X, y)

        # Signal that the model has been trained
        self._ready()

        logging.info(f"Fitting of {get_nice_class_name(self.model)} has converged.")

    def train_optimize(self, smiles_list: List[str], target_list: List[float]):

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
        random_search.fit(smiles_list, target_list)

        # Save only the best model after refitting to the whole training data
        self.model = random_search.estimator

        logging.info(
            f"RandomSearchCV: Fitting converged. Keeping the best model, with params: "
            f"{random_search.best_params_}"
        )

        # Signal that the model has been trained
        self._ready()

    def _predict(self, smiles_list: List[str]) -> Tuple[np.array, np.array]:
        # Featurize the smiles
        X = self.featurizer.featurize(smiles_list)
        # Predict the target values
        y_pred = self.model.predict(X)
        # Predict class probabilities
        y_probabilities = self.model.predict_proba(X)
        # Retain probabilities of the predicted class
        correct_class_probabilities = np.zeros(y_pred.shape)
        for i in range(len(y_pred)):
            correct_class_probabilities[i] = y_probabilities[i][y_pred[i]]
        # Return the predicted values and probabilities
        return y_pred, correct_class_probabilities

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
        self._ready()

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
    def __init__(self, params: dict | None = None):
        super(RfRegressor, self).__init__(
            model=sklearn.ensemble.RandomForestRegressor(), params=params
        )


@gin.configurable()
class RfClassifier(ScikitPredictor):
    def __init__(self, params: dict | None = None):
        super(RfClassifier, self).__init__(
            model=sklearn.ensemble.RandomForestClassifier(), params=params
        )


@gin.configurable()
class SvmRegressor(ScikitPredictor):
    def __init__(self, params: dict | None = None):
        super(SvmRegressor, self).__init__(model=sklearn.svm.SVR(), params=params)


@gin.configurable()
class SvmClassifier(ScikitPredictor):
    def __init__(self, params: dict | None = None):
        super(SvmClassifier, self).__init__(model=sklearn.svm.SVC(probability=True), params=params)
