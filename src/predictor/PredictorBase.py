import abc
import logging
from abc import abstractmethod
import numpy as np
from typing import List, Dict

from src.utils import get_nice_class_name

import sklearn.metrics
from rdkit import Chem


class PredictorBase(abc.ABC):
    def __init__(
        self, model, metrics: List[str] | None = None, primary_metric: str | None = None
    ):

        # load all supported merics as a dict of names and corresponding sklearn functions
        self.model = model()
        self.metrics = metrics
        self.primary_metric = primary_metric
        self.supported_metrics = self._get_supported_metrics_dict()

        self.ready_flag = (
            False  # only raised after either .train() or .load() method is called
        )

    @abc.abstractmethod
    def train(self, smiles_list, target_list):
        """Train the model with the given smiles and target list.
        This method should also raise the trained_falg"""
        pass

    @abc.abstractmethod
    def predict(self, smiles_list, ignore_flag=False):
        """Predict the target value for the given list of smiles.
        This should throw an exception unless the ready_flag is raised."""
        pass

    @abc.abstractmethod
    def save(self, path):
        """Save the model to the given path"""
        pass

    @abc.abstractmethod
    def load(self, path):
        """Load the model from the given path. This method should also raise the ready_flag"""
        pass

    def calc_metrics(self, target_list, pred_list):
        """Get metrics for a list of targets and a list of predictions"""
        logging.debug(
            f"Calculating metrics: {[get_nice_class_name(x) for x in self.supported_metrics]}"
        )
        metrics_dict = {}
        target_list, pred_list = np.array(target_list), np.array(pred_list)
        for metric in self.metrics:
            metrics_dict[metric] = metric(target_list, pred_list)
        return metrics_dict

    def calc_primary_metric(self, target_list, pred_list):
        if self.primary_metric is not None:
            logging.debug(f"Calculating primary metric: {self.primary_metric}")
            target_list, pred_list = np.array(target_list), np.array(pred_list)
            metric_fn = self.supported_metrics[primary_metric]
            result = metric_fn(target_list, pred_list)
            return result
        else:
            raise ValueError("primary_metric attribute of PredictorBase is None!")

    def list_supported_metrics(self):
        """
        List the names of all supported metrics.
        """
        return self.supported_metrics.keys()

    @staticmethod
    def _get_supported_metrics_dict():
        """
        Get a dictionary of all supported metrics, with string (metrics names) as the keys and sklearn
        function references as values.
        """
        return {
            "mean_squared_error": sklearn.metrics.mean_squared_error,
            "r2_score": sklearn.metrics.r2_score,
            "roc_auc_score": sklearn.metrics.roc_auc_score,
            "accuracy_score": sklearn.metrics.accuracy_score,
            "f1_score": sklearn.metrics.f1_score,
            "precision_score": sklearn.metrics.precision_score,
            "recall_score": sklearn.metrics.recall_score,
        }
