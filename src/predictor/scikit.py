import logging

import numpy as np

from src.predictor.PredictorBase import PredictorBase
from src.featurizer.FeaturizerBase import FeaturizerBase
from typing import List
from pathlib import Path
import sklearn
import pickle as pkl
import gin


class ScikitPredictorBase(PredictorBase):
    """
    Represents a Scikit-learn predictive model

    :param model: Scikit-learn model
    :param params: Hyperparameters for the model as a dictionary
    :param metric: Primary metric for the model as a string
        ("mean_squared_error", "roc_auc_score", "accuracy_score", "f1_score")
    :param optimize_hyperparameters: Whether to optimize hyperparameters using CV random search strategy

    """

    def __init__(
        self,
        model,
        params: dict | None = None,
        metric: str | None = None,
        optimize_hyperparameters: bool = False,
        params_distribution: dict | None = None,
        optimization_iterations: int | None = None,
        n_folds: int | None = None,
        n_jobs: int | None = None,
    ):

        super(ScikitPredictorBase, self).__init__()

        # Initialize the model
        self.model = model()

        # Set the hyperparameters
        if params is not None:
            self._check_params(self.model, params)
            self.model.set_params(**params)

        # Set the hyperparameter distribution for random search CV
        self.optimize_hyperparameters = optimize_hyperparameters
        self.optimization_iterations = optimization_iterations
        self.n_folds = n_folds
        self.n_jobs = n_jobs
        if params_distribution is not None:
            self.params_distribution = params_distribution

        metrics = {
            "mean_squared_error": sklearn.metrics.mean_squared_error,
            "roc_auc_score": sklearn.metrics.roc_auc_score,
            "accuracy_score": sklearn.metrics.accuracy_score,
            "f1_score": sklearn.metrics.f1_score,
        }

        # Set primary metric
        self.primary_metric = metrics[metric]

    def name(self):
        """
        Returns the name of the model
        """
        return type(self.model).__name__

    def inject_featurizer(self, featurizer):
        """
        Inject a featurizer into the model
        :param featurizer: Featurizer object
        """
        if not isinstance(featurizer, FeaturizerBase):
            raise ValueError("Featurizer must be an instance of FeaturizerBase!")
        self.featurizer = featurizer

    def train(self, smiles_list: List[str], target_list: List[float]):

        # Featurize the smiles
        X = self.featurizer.featurize(smiles_list)
        y = target_list

        # Train the model
        if self.optimize_hyperparameters:
            # Use random search to optimize hyperparameters
            self.train_CV(X, y)
        else:
            # Use pre-defined hyperparameters
            self.model.fit(X, y)

        # Return the primary metric
        y_pred = self.model.predict(X)

        return self.primary_metric(y, y_pred)

    def train_CV(self, X, y):

        # Use random search to optimize hyperparameters
        random_search = sklearn.model_selection.RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.params_distribution,
            n_iter=self.optimization_iterations,
            cv=self.n_folds,
            verbose=2,
            n_jobs=self.n_jobs,
        )

        # Fit the model
        logging.info("Fitting model with random search CV")
        logging.info(f"Hyperparameter distribution: {self.params_distribution}")
        random_search.fit(X, y)

        # Get the best model
        self.model = random_search.best_estimator_

        # Get the best hyperparameters
        best_params = random_search.best_params_
        logging.info(f"Loading best parameters: {best_params}")
        self.model = self.model.set_params(**best_params)
        self.model.fit(X, y)  # Train the model with the best hyperparameters

    def predict(self, smiles_list: List[str]) -> np.array:

        # Featurize the smiles
        X = self.featurizer.featurize(smiles_list)

        # Predict the target values
        return self.model.predict(X)

    def score(self, y_true, y_pred):
        metric = self.primary_metric(y_true, y_pred)
        if self.verbose:
            print("Primary metric:", self.primary_metric.__name__)
            print("Values:", round(metric, 3))
        return metric

    def save(self, out_dir: str):

        # Check if the output directory exists
        if not Path(out_dir).exists():
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
    def __init__(self):

        model = sklearn.ensemble.RandomForestRegressor
        super(RandomForestRegressor, self).__init__(model)


@gin.configurable()
class RandomForestClassifier(ScikitPredictorBase):
    def __init__(self):

        model = sklearn.ensemble.RandomForestClassifier
        super(RandomForestClassifier, self).__init__(model)


@gin.configurable()
class SvrRegressor(ScikitPredictorBase):
    def __init__(self):

        model = sklearn.svm.SVR
        super(SvrRegressor, self).__init__(model=model)


@gin.configurable()
class SvrClassifier(ScikitPredictorBase):
    def __init__(self, metric: str):

        model = sklearn.svm.SVC
        metric = metric
        super(SvrClassifier, self).__init__(model=model)
