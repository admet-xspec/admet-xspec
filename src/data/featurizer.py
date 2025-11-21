import abc
import logging
from typing import List, Hashable

import gin
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.preprocessing import StandardScaler


class FeaturizerBase(abc.ABC):
    """Base class for molecular featurizers."""

    @abc.abstractmethod
    def featurize(self, smiles_list: List[str]) -> np.ndarray:
        """Convert SMILES strings to numerical feature arrays."""
        pass

    @property
    @abc.abstractmethod
    def feature_name(self) -> str:
        """Column name for storing features."""
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable featurizer name."""
        pass

    @abc.abstractmethod
    def get_hashable_params_values(self) -> List[Hashable]:
        """Return parameters for hashing/caching purposes."""
        pass

    def get_cache_key(self):
        """
        Generate a 5-character cache key.
        """
        return f"{self.name}_{abs(hash(frozenset(self.get_hashable_params_values()))) % (10 ** 5):05d}"


@gin.configurable
class EcfpFeaturizer(FeaturizerBase):
    """Extended Connectivity Fingerprint (Morgan/ECFP) featurizer."""

    def __init__(self, radius: int = 2, n_bits: int = 2048, count: bool = False):
        self.radius = radius
        self.n_bits = n_bits
        self.count = count
        self.generator = GetMorganGenerator(radius=radius, fpSize=n_bits)

    def featurize(self, smiles_list: List[str]) -> np.ndarray:
        """Generate ECFP fingerprints for given SMILES."""
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]

        # Log failed conversions
        for i, (mol, smi) in enumerate(zip(mols, smiles_list)):
            if mol is None:
                logging.debug(f"Failed to convert SMILES at index {i}: {smi}")

        # Generate fingerprints
        if self.count:
            fps = [self.generator.GetCountFingerprintAsNumPy(mol) for mol in mols]
        else:
            fps = [self.generator.GetFingerprintAsNumPy(mol) for mol in mols]

        return np.stack(fps)

    @property
    def feature_name(self) -> str:
        return "fp_ecfp"

    @property
    def name(self) -> str:
        return "ecfp_featurizer"

    def get_hashable_params_values(self) -> List[Hashable]:
        return [self.radius, self.n_bits, self.count]

    def __getstate__(self):
        """Pickle support: exclude non-serializable generator."""
        state = self.__dict__.copy()
        del state["generator"]
        return state

    def __setstate__(self, state):
        """Pickle support: reconstruct generator after unpickling."""
        self.__dict__.update(state)
        self.generator = GetMorganGenerator(radius=self.radius, fpSize=self.n_bits)


@gin.configurable
class PropertyFeaturizer(FeaturizerBase):
    """RDKit molecular property descriptor featurizer with normalization."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def featurize(self, smiles_list: List[str]) -> np.ndarray:
        """Generate normalized molecular descriptors."""
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]

        # Compute descriptors for each molecule
        descriptor_dicts = [self._compute_descriptors(mol) for mol in mols]

        # Convert to DataFrame and handle missing values
        desc_array = pd.DataFrame(descriptor_dicts).fillna(0).to_numpy()

        # Fit scaler on first call, then transform
        if not self.is_fitted:
            self.scaler.fit(desc_array)
            self.is_fitted = True

        return self.scaler.transform(desc_array)

    @staticmethod
    def _compute_descriptors(mol) -> dict:
        """Compute all RDKit descriptors for a molecule."""
        descriptors = {}
        for name, func in Descriptors._descList:
            try:
                descriptors[name] = func(mol)
            except Exception:
                descriptors[name] = np.nan
        return descriptors

    @property
    def feature_name(self) -> str:
        return "prop_desc"

    @property
    def name(self) -> str:
        return "prop_featurizer"

    def get_hashable_params_values(self) -> List[Hashable]:
        return [self.feature_name]


@gin.configurable
class PropertyEcfpFeaturizer(FeaturizerBase):
    """Combined property descriptor and ECFP fingerprint featurizer."""

    def __init__(self, radius: int = 2, n_bits: int = 2048, count: bool = False):
        self.ecfp = EcfpFeaturizer(radius=radius, n_bits=n_bits, count=count)
        self.properties = PropertyFeaturizer()

    def featurize(self, smiles_list: List[str]) -> np.ndarray:
        """Generate combined ECFP and property features."""
        ecfp_features = self.ecfp.featurize(smiles_list)
        property_features = self.properties.featurize(smiles_list)
        return np.hstack((ecfp_features, property_features))

    @property
    def feature_name(self) -> str:
        return "prop_ecfp"

    @property
    def name(self) -> str:
        return "prop_ecfp_featurizer"

    def get_hashable_params_values(self) -> List[Hashable]:
        return [
            self.feature_name,
            *self.ecfp.get_hashable_params_values(),
            *self.properties.get_hashable_params_values(),
        ]