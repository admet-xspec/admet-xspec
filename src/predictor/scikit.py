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
    :param optimize_hyperparameters: Wether to optimize hyperparameters using CV random search strategy

    """

    def __init__(
        self, model, params: dict, metric: str, optimize_hyperparameters: bool
    ):

        super(ScikitPredictorBase, self).__init__()

        # Initialize the model
        self.model = model()

        # Set the hyperparameters
        if params is not None:
            self._check_params(self.model, params)
            self.model.set_params(**params)

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
        self.model.fit(X, y)

        # Return the primary metric
        y_pred = self.model.predict(X)

        return self.primary_metric(y, y_pred)

    def train_CV(
        self, smiles_list: List[str], target_list: List[float], cv: int = 5
    ) -> List[float]:

        # Featurize the smiles
        X = np.array(self.featurizer.featurize(smiles_list))
        y = np.array(target_list)

        # Train the model
        kf = sklearn.model_selection.KFold(n_splits=cv)
        kf.get_n_splits(X)

        metrics = []
        for i, (train_index, val_index) in enumerate(kf.split(X)):
            logging.debug(f"Fitting fold {i} of {cv}")
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_val)
            metrics.append(self.primary_metric(y_val, y_pred))

        print("Primary metric:", self.primary_metric)
        print("Values:", [round(x, 3) for x in metrics])

        return metrics

    def predict(self, smiles_list: List[str]) -> np.array:

        # Featurize the smiles
        X = self.featurizer.featurize(smiles_list)

        # Predict the target value
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
    def __init__(self, metric: str, optimize_hyperparameters: bool, params: dict):

        model = sklearn.ensemble.RandomForestRegressor
        super(RandomForestRegressor, self).__init__(
            model, params, metric, optimize_hyperparameters
        )


@gin.configurable()
class RandomForestClassifier(ScikitPredictorBase):
    def __init__(self, metric: str, optimize_hyperparameters: bool, params: dict):

        model = sklearn.ensemble.RandomForestClassifier
        super(RandomForestClassifier, self).__init__(
            model, params, metric, optimize_hyperparameters
        )


@gin.configurable()
class SvrRegressor(ScikitPredictorBase):
    def __init__(self, metric: str, optimize_hyperparameters: bool, params: dict):

        model = sklearn.svm.SVR
        super(SvrRegressor, self).__init__(
            model, params, metric, optimize_hyperparameters
        )


@gin.configurable()
class SvrClassifier(ScikitPredictorBase):
    def __init__(self, metric: str, optimize_hyperparameters: bool, params: dict):

        model = sklearn.svm.SVC
        metric = metric
        super(SvrClassifier, self).__init__(
            model, params, metric, optimize_hyperparameters
        )
