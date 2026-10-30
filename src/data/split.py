import abc
import hashlib
import logging
from collections import defaultdict

import gin
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split


class DataSplitterBase(abc.ABC):
    """Base class for train/test data splitters."""

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    @abc.abstractmethod
    def split(self, X: pd.Series, y: pd.Series):
        """Split the dataset into `X_train, X_test, y_train, y_test`."""
        pass

    @abc.abstractmethod
    def get_hashable_params_values(self) -> list:
        """Return parameters for hashing/caching purposes."""
        pass

    @property
    @abc.abstractmethod
    def name(self):
        """Short splitter name."""
        pass

    @staticmethod
    def _get_number_of_classes(labels):
        return pd.Series(labels).nunique()

    def get_cache_key(self) -> str:
        """Return `<name>_<5-char hash of params>`."""
        params = str(self.get_hashable_params_values()).encode("utf-8")
        return f"{self.name}_{hashlib.md5(params).hexdigest()[:5]}"


@gin.configurable
class RandomSplitter(DataSplitterBase):
    """Random train/test split, optionally stratified on the target."""

    def __init__(
        self,
        test_size=0.2,
        random_state=42,
        stratify=None,
    ):
        super().__init__(test_size=test_size, random_state=random_state)
        self.stratify = stratify

    @property
    def name(self):
        return "random"

    def get_hashable_params_values(self):
        return [self.test_size, self.random_state, self.stratify]

    def split(
        self, X: pd.Series, y: pd.Series
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        labels_to_stratify = None
        if self.stratify:
            n_classes = self._get_number_of_classes(y)
            if n_classes >= 5:
                logging.warning(
                    f"Detected {n_classes} unique classes of {len(y)} samples. "
                    "Make sure you are not stratifying on a continuous target variable!"
                )
            labels_to_stratify = y

        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels_to_stratify,
        )


@gin.configurable
class ScaffoldSplitter(DataSplitterBase):
    """Train/test split on Murcko scaffolds, so no scaffold spans both sets.

    Gives a harder, more realistic evaluation than a random split.
    """

    def __init__(self, test_size=0.2, random_state=42):
        super().__init__(test_size=test_size, random_state=random_state)

    @property
    def name(self):
        return "scaffold"

    def get_hashable_params_values(self):
        # "v2": stereo-agnostic scaffolds; keeps v1 cached splits from being reused.
        return [self.test_size, self.random_state, "v2"]

    @staticmethod
    def _scaffold(smi: str) -> str:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES, cannot compute scaffold: {smi!r}")
        # Ignore stereo, so stereoisomers of one ring system share a group.
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)

    def split(
        self, X: pd.Series, y: pd.Series
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        # Scaffold SMILES -> positional indices of its molecules.
        scaffolds = defaultdict(list)
        for i, smi in enumerate(X):
            scaffolds[self._scaffold(smi)].append(i)

        scaffold_sets = list(scaffolds.values())
        if len(scaffold_sets) < 2:
            raise ValueError("Scaffold split needs at least two distinct scaffolds.")
        rng = np.random.RandomState(self.random_state)
        rng.shuffle(scaffold_sets)

        # Greedily fill the test set with whole scaffold groups that still fit.
        train_indices, test_indices = [], []
        n_test = int(len(X) * self.test_size)
        for scaffold_set in scaffold_sets:
            if len(test_indices) + len(scaffold_set) <= n_test:
                test_indices.extend(scaffold_set)
            else:
                train_indices.extend(scaffold_set)

        # Every group exceeded `n_test`: fall back to the smallest one as test set.
        if not test_indices:
            smallest = min(scaffold_sets, key=len)
            test_indices = smallest
            train_indices = [
                i for group in scaffold_sets if group is not smallest for i in group
            ]
            logging.warning(
                f"No scaffold group fits test_size={self.test_size}; using the smallest "
                f"group as test set ({len(test_indices) / len(X):.1%} of samples)."
            )

        # Keep the input order within each set.
        train_indices, test_indices = sorted(train_indices), sorted(test_indices)
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        return X_train, X_test, y_train, y_test
