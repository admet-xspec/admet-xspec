import abc
import logging
from typing import List, Hashable
import hashlib

import gin
import numpy as np
import pandas as pd
import pathlib
import map4
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
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
        params_values = self.get_hashable_params_values()
        params_values = str(params_values).encode("utf-8")
        hash_string = hashlib.md5(params_values).hexdigest()
        return f"{self.name}_{hash_string[:5]}"


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

    def __init__(self, scaler=StandardScaler()):
        self.scaler = scaler
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

        desc_array = self.scaler.transform(desc_array)
        # Check for nans and infs and replace them with zeros
        desc_array = np.array(desc_array, dtype=np.float32)
        if np.isnan(desc_array).sum() or np.isinf(desc_array).sum():
            logging.warning(
                f"PropertyFeaturizer: Found {np.isnan(desc_array).sum()} NaNs and "
                f"{np.isinf(desc_array).sum()} infinite values in descriptors. "
                "Replacing with zeros."
            )
            desc_array = np.nan_to_num(desc_array, nan=0.0, posinf=0.0, neginf=0.0)

        return np.array(desc_array, dtype=np.float32)

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
class MaccsFeaturizer(FeaturizerBase):
    """MACCS keys fingerprint featurizer."""

    def featurize(self, smiles_list: List[str]) -> np.ndarray:
        """Generate MACCS keys fingerprints for given SMILES."""
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]

        # Log failed conversions
        for i, (mol, smi) in enumerate(zip(mols, smiles_list)):
            if mol is None:
                logging.debug(f"Failed to convert SMILES at index {i}: {smi}")

        # Generate MACCS keys fingerprints
        fps = [MACCSkeys.GenMACCSKeys(mol) for mol in mols]
        fps_array = np.array([np.array(fp) for fp in fps])

        return fps_array

    @property
    def feature_name(self) -> str:
        return "fp_maccs"

    @property
    def name(self) -> str:
        return "maccs_featurizer"

    def get_hashable_params_values(self) -> List[Hashable]:
        return [self.feature_name]


@gin.configurable
class KlekotaRothFeaturizer(FeaturizerBase):
    """Klekota-Roth fingerprint featurizer."""

    def __init__(self, keys_path: str):
        super().__init__()
        self.keys_mols = self._read_krfp_keys(keys_path)

    def featurize(self, smiles_list: List[str]) -> np.ndarray:
        """Generate Klekota-Roth fingerprints for given SMILES."""
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]

        # Log failed conversions
        for i, (mol, smi) in enumerate(zip(mols, smiles_list)):
            if mol is None:
                logging.debug(f"Failed to convert SMILES at index {i}: {smi}")

        fps = [self._get_krfp_fingerprint(mol) for mol in mols]
        fps_array = np.array(fps)
        return fps_array

    @property
    def feature_name(self) -> str:
        return "fp_krfp"

    @property
    def name(self) -> str:
        return "krfp_featurizer"

    def get_hashable_params_values(self) -> List[Hashable]:
        return [self.feature_name]

    def _read_krfp_keys(self, keys_path: str) -> List[Chem.Mol]:
        """Read Klekota-Roth fingerprint keys from a file."""
        if not pathlib.Path(keys_path).exists():
            raise FileNotFoundError(f"Klekota-Roth keys file not found at: {keys_path}")
        klek_keys = [line.strip() for line in open(keys_path)]
        klek_keys_mols = list(map(Chem.MolFromSmarts, klek_keys))
        return klek_keys_mols

    def _get_krfp_fingerprint(self, mol: Chem.Mol) -> np.ndarray:
        """
        Generate Klekota-Roth fingerprint for a single molecule.
        Checks for the presence of 4860 predefined substructures corresponding to chemical
        motifs that were selected as relevant in medicinal chemistry.
        Original paper: https://doi.org/10.1093/bioinformatics/btn479
        """
        fp_list = []
        for key in self.keys_mols:
            if mol.HasSubstructMatch(key):
                fp_list.append(1)
            else:
                fp_list.append(0)
        return np.array(fp_list)


@gin.configurable
class PropertyEcfpFeaturizer(FeaturizerBase):
    """Combined property descriptor and ECFP fingerprint featurizer."""

    # TODO: Remove this class and implement an union featurizer that can combine any n featurizers

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


@gin.configurable
class Map4Featurizer(FeaturizerBase):
    """MAP4 fingerprint featurizer."""

    def __init__(
        self,
        size: int = 2048,
        radius: int = 2,
        include_duplicated_shingles: bool = False,
    ):
        super().__init__()
        self.map4_generator = map4.MAP4(
            dimensions=size,
            radius=radius,
            include_duplicated_shingles=include_duplicated_shingles,
        )

    def featurize(self, smiles_list: List[str]) -> np.ndarray:
        """Generate MAP4 fingerprints for given SMILES."""

        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]

        # Log failed conversions
        for i, (mol, smi) in enumerate(zip(mols, smiles_list)):
            if mol is None:
                logging.debug(f"Failed to convert SMILES at index {i}: {smi}")

        fps_array = self.map4_generator.calculate_many(
            mols, number_of_threads=2, verbose=False
        )

        return fps_array

    @property
    def feature_name(self) -> str:
        return "fp_map4"

    @property
    def name(self) -> str:
        return "map4_featurizer"

    def get_hashable_params_values(self) -> List[Hashable]:
        return [self.feature_name]
