import abc
import logging
import numpy as np
from typing import List

from src.utils import get_nice_class_name

import sklearn.metrics


class PredictorBase(abc.ABC):
    def __init__(
        self, metrics: List[str] | None = None, primary_metric: str | None = None
    ):

        # load all supported merics as a dict of names and corresponding sklearn functions
        self.metrics = metrics
        self.primary_metric = primary_metric
        self.supported_metrics = self._get_supported_metrics_dict()
        self.working_dir = None  # working directory for training and inference
        self.model = None  # model will be set in the child class

        self.ready_flag = (
            False  # only raised after either .train() or .load() method is called
        )

    def infer(self, smiles_list, ignore_flag=False):
        """Predict the target values for the given smiles list. Raises ModelNotTrainedError if the model is not trained or loaded"""
        if self.ready_flag or ignore_flag:
            # Predict the target values
            return self._predict(smiles_list)
        else:
            raise Exception(
                "The model is not fitted to data. Use .train() or .load() methods before inference."
            )

    @abc.abstractmethod
    def _predict(self, smiles_list):
        """Predict the target values for the given smiles list."""
        pass

    @abc.abstractmethod
    def train(self, smiles_list, target_list):
        """Train the model with the given smiles and target list.
        This method should also raise the ready_flag by calling _ready() method"""
        pass

    @abc.abstractmethod
    def save(self, path):
        """Save the model to the given path"""
        pass

    @abc.abstractmethod
    def load(self, path):
        """Load the model from the given path. This method should also raise the ready_flag by calling _ready() method"""
        pass

    def calc_metrics(self, target_list, pred_list):
        """Get metrics for a list of targets and a list of predictions"""
        logging.debug(
            f"Calculating metrics: {[get_nice_class_name(x) for x in self.supported_metrics]}"
        )
        out_dict = {}
        target_list, pred_list = np.array(target_list), np.array(pred_list)
        for metric in self.metrics:
            result = self.supported_metrics[metric](target_list, pred_list)
            if isinstance(result, np.ndarray):
                result = result.item()
            out_dict[metric] = result
        return out_dict

    def calc_primary_metric(self, target_list, pred_list):
        logging.debug(f"Calculating primary metric: {self.primary_metric}")
        target_list, pred_list = np.array(target_list), np.array(pred_list)
        metric_fn = self.supported_metrics[self.primary_metric]
        result = metric_fn(target_list, pred_list)
        return result.item() if isinstance(result, np.ndarray) else result

    def list_supported_metrics(self):
        """
        List the names of all supported metrics (list of strings).
        """
        return self.supported_metrics.keys()

    def _ready(
        self,
    ):
        """
        Raise the ready flag. This method should be called after the model is trained or loaded.
        :return:
        """
        self.ready_flag = True

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

    def set_working_dir(self, path: str):
        """
        Set working directory path for the model
        :param path: Path to the working directory
        """
        self.working_dir = path
