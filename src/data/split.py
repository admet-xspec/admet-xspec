import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np
import abc
import gin
import logging


class DataSplitterBase(abc.ABC):
    """
    Abstract base class for data splitters.
    """

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    @abc.abstractmethod
    def split(self, X: pd.Series, y: pd.Series):
        """
        Split the dataset into training and testing sets.
        """
        pass

    def get_cache_key(self):
        """
        Return a key representing the state of the splitter.
        """
        return f"{self.__class__.__name__}_{abs(hash(frozenset([self.test_size, self.random_state])))}"

    @staticmethod
    def _get_number_of_classes(labels):
        if not isinstance(labels, pd.Series):
            labels = pd.Series(labels)
        return len(labels.unique())


@gin.configurable
class RandomSplitter(DataSplitterBase):
    """
    Splits the dataset into training and testing sets using random sampling.
    The split can be stratified based on the target variable if specified.
    """

    def __init__(self, test_size=0.2, random_state=42, stratify=None):
        super().__init__(test_size=test_size, random_state=random_state)
        self.stratify = stratify

    def split(
        self, X: pd.Series, y: pd.Series
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:

        labels_to_stratify = None
        if self.stratify:
            if self._get_number_of_classes(y) == 2:
                labels_to_stratify = y
            else:
                logging.warning(
                    """The data splitter was configured to perform a stratified split, but the labels
                provided are not binary. The stratify=True argument is being ignored."""
                )
                labels_to_stratify = None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels_to_stratify,
        )
        return X_train, X_test, y_train, y_test


@gin.configurable
class ScaffoldSplitter(DataSplitterBase):
    """
    Splits the dataset into training and testing sets based on molecular (Murcko) scaffolds.
    This ensures that molecules with similar scaffolds are not split between training and testing sets,
    providing a more challenging and realistic task for model evaluation.
    """

    def __init__(self, test_size=0.2, random_state=42):
        super().__init__(test_size=test_size, random_state=random_state)

    def split(
        self, X: pd.Series, y: pd.Series
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        scaffolds = {}

        # Group molecules by their scaffolds, storing indices of molecules with the same scaffold
        # in a dictionary with scaffold SMILES as keys.
        for i, smi in enumerate(X):
            mol = Chem.MolFromSmiles(smi)
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold)
            if scaffold_smiles not in scaffolds:
                scaffolds[scaffold_smiles] = []
            scaffolds[scaffold_smiles].append(i)

        # Shuffle the scaffold sets to ensure randomness in the split
        scaffold_sets = list(scaffolds.values())
        np.random.seed(self.random_state)
        np.random.shuffle(scaffold_sets)

        # Split the scaffold sets into training and testing sets
        train_indices, test_indices = [], []
        n_test = int(len(X) * self.test_size)
        for scaffold_set in scaffold_sets:
            if len(test_indices) + len(scaffold_set) <= n_test:
                test_indices.extend(scaffold_set)
            else:
                train_indices.extend(scaffold_set)

        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        return X_train, X_test, y_train, y_test
