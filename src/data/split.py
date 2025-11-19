import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np
import abc
import gin
import logging

from src.data.filter import FilterBase


class DataSplitterBase(abc.ABC):
    """
    Abstract base class for data splitters.
    """

    def __init__(self, test_size=0.2, random_state=42, train_filter: FilterBase = None):
        self.test_size = test_size
        self.random_state = random_state
        self.train_filter = train_filter

    def get_filter(self):
        return self.train_filter

    def filter(
        self, to_filter_df: pd.DataFrame, filter_against_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Filter one dataset using another dataset, both passed to the filter object
        """
        if self.train_filter:
            return self.train_filter.get_filtered_df(to_filter_df, filter_against_df)
        else:
            return to_filter_df

    @abc.abstractmethod
    def split(self, X: pd.Series, y: pd.Series):
        """
        Split the dataset into training and testing sets.
        """
        pass

    @abc.abstractmethod
    def get_hashable_params_values(self) -> list:
        pass

    @property
    @abc.abstractmethod
    def name(self):
        pass

    @staticmethod
    def _get_number_of_classes(labels):
        if not isinstance(labels, pd.Series):
            labels = pd.Series(labels)
        return len(labels.unique())

    def get_friendly_name(
        self,
        multiple_friendly_names: list[str],
    ) -> str:

        generated_name_chunks: list[str] = [
            "_".join([part[:3] for part in name.split("_")[:3]])
            for name in multiple_friendly_names
        ]
        generated_name_chunks.append(self.get_cache_key())

        return "_".join(generated_name_chunks)

    def get_cache_key(self):
        """
        Generate a 5-character cache key based on the splitter's parameters.
        """
        return f"{self.name}_{abs(hash(frozenset(self.get_hashable_params_values()))) % (10 ** 5):05d}"


@gin.configurable
class RandomSplitter(DataSplitterBase):
    """
    Splits the dataset into training and testing sets using random sampling.
    The split can be stratified based on the target variable if specified.
    """

    def __init__(
        self,
        test_size=0.2,
        random_state=42,
        train_filter: FilterBase = None,
        stratify=None,
    ):
        super().__init__(
            test_size=test_size, random_state=random_state, train_filter=train_filter
        )
        self.stratify = stratify

    @property
    def name(self):
        return "random"

    def get_hashable_params_values(self):
        return [self.test_size, self.random_state, self.stratify]

    def split(
        self, X: pd.Series, y: pd.Series
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:

        if self.stratify:
            if self._get_number_of_classes(y) >= 5:
                logging.warning(
                    f"Detected {self._get_number_of_classes(y)} unique classes of {len(y)} samples. "
                    "Make sure you are not stratifying on a continuous target variable!"
                )
            labels_to_stratify = y
        else:
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

    def __init__(self, test_size=0.2, random_state=42, train_filter: FilterBase = None):
        super().__init__(
            test_size=test_size, random_state=random_state, train_filter=train_filter
        )

    @property
    def name(self):
        return "scaffold"

    def get_hashable_params_values(self):
        return [self.test_size, self.random_state]

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
