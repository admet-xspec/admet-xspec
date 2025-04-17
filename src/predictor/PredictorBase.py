import abc
import logging
from rdkit import Chem


class PredictorBase(abc.ABC):
    def __init__(self, verbose: bool = False):
        self.model = None
        self.verbose = False

    @abc.abstractmethod
    def name(self):
        """Return the name of the model"""
        pass

    @abc.abstractmethod
    def train(self, smiles_list, target_list):
        """Train the model with the given smiles and target list."""
        pass

    @abc.abstractmethod
    def predict(self, smiles_list):
        """Predict the target value for the given list of smiles"""
        pass

    @abc.abstractmethod
    def save(self, path):
        """Save the model to the given path"""
        pass

    @abc.abstractmethod
    def load(self, path):
        """Load the model from the given path"""
        pass

    def set_verbose(self, verbose: bool):
        # Set the verbose flag
        if not isinstance(verbose, bool):
            raise ValueError("You must provide a boolean value")
        self.verbose = verbose
