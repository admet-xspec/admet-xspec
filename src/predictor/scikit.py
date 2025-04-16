import numpy as np

from src.predictor.PredictorBase import PredictorBase
from typing import List, Dict, Any
from pathlib import Path
import sklearn
import pickle as pkl

class ScikitPredictorBase(PredictorBase):
    """
    Represents a Scikit-learn predictive model

    :param model: Scikit-learn model
    :param params: Hyperparameters for the model as a dictionary
    :param metric: Primary metric for the model as a string
        ("mean_squared_error", "roc_auc_score", "accuracy_score", "f1_score")

    """
    def __init__(self, model, params: dict, metric: str, featurizer=None):

        super(ScikitPredictorBase, self).__init__()

        # Initialize the model
        self.model = model()

        # Initialize the featurizer
        self.featurizer = featurizer

        # Set the hyperparameters
        if params is not None:
            self._check_params(self.model, params)
            self.model.set_params(**params)

        metrics = {
            "mean_squared_error": sklearn.metrics.mean_squared_error,
            "roc_auc_score": sklearn.metrics.roc_auc_score,
            "accuracy_score": sklearn.metrics.accuracy_score,
            "f1_score": sklearn.metrics.f1_score        }

        # Set primary metric
        self.primary_metric = metrics[metric]

    def train(self, smiles_list: List[str], target_list: List[float]):

        # Featurize the smiles
        print("Featurizing the smiles...") if self.verbose else None
        X = self.featurizer.featurize(smiles_list)
        y = target_list

        # Train the model
        print("Training the model...") if self.verbose else None
        self.model.fit(X, y)

        # Return the primary metric
        y_pred = self.model.predict(X)

        if self.verbose:
            print(f"Primary metric: {self.primary_metric.__name__}")
            print(f"Value: {self.primary_metric(y, y_pred)}")

        return self.primary_metric(y, y_pred)

    def train_CV(self, smiles_list: List[str], target_list: List[float], cv: int = 5) -> List[float]:

        # Featurize the smiles
        print("Featurizing the smiles...") if self.verbose else None
        X = self.featurizer.featurize(smiles_list)
        y = target_list

        # Train the model
        print(f"Training {self.get_name()} model...") if self.verbose else None
        cv = sklearn.model_selection.KFold(n_splits=cv)

        metrics = []
        for train_index, val_index in cv.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_val)
            metrics.append(self.primary_metric(y_val, y_pred))

        # Choose the best hyperparameters
        best_params = self.model.best_params_
        self.model.set_params(**best_params)

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

    def save(self, path: str):

        # Check if the parent directory exists
        if not Path(path).parent.exists():
            raise FileNotFoundError(f"Directory {Path(path).parent} does not exist")

        # Save the model
        pkl.dump(self.model, path)

    def load(self, path: str):

        # Check if the file exists
        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist")

        # Load the model
        self.model = pkl.load(path)

    @staticmethod
    def _check_params(model, params):
        model_params = model.get_params()
        for key in params:
            if key not in model_params:
                raise ValueError(f"Model {type(model).__name__} does not accept hyperparameter {key}")

class RandomForestRegressor(ScikitPredictorBase):
    def __init__(self, params=None):

        model = sklearn.ensemble.RandomForestRegressor
        metric = "mean_squared_error"
        super(RandomForestRegressor, self).__init__(model, params, metric)

class RandomForestClassifier(ScikitPredictorBase):
    def __init__(self, params=None):

        model = sklearn.ensemble.RandomForestClassifier
        metric = "roc_auc_score"
        super(RandomForestClassifier, self).__init__(model, params, metric)

class SvrRegressor(ScikitPredictorBase):
    def __init__(self, params=None):

        model = sklearn.svm.SVR
        metric = "mean_squared_error"
        super(SvrRegressor, self).__init__(model, params, metric)

class SvrClassifier(ScikitPredictorBase):
    def __init__(self, params=None):

        model = sklearn.svm.SVC
        metric = "roc_auc_score"
        super(SvrClassifier, self).__init__(model, params, metric)