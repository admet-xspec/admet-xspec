import abc
import logging
import tempfile
from typing import List, Hashable
import hashlib
import filelock
import joblib

import gin
import numpy as np
import pandas as pd
import pathlib
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.preprocessing import StandardScaler


class FeaturizerBase(abc.ABC):
    """Base class for molecular featurizers."""

    _DEFAULT_CACHE_DIR = pathlib.Path(tempfile.gettempdir()) / "admet_fp_cache"
    # False for featurizers with a fitted scaler, whose output varies between runs.
    _use_disk_cache: bool = True
    # Set False per instance to bypass caching entirely (e.g. in tests).
    cache_enabled: bool = True

    def _ensure_cache(self) -> None:
        """Lazily load the in-memory cache, seeded from disk if available."""
        if hasattr(self, "_cache_initialized"):
            return
        self._smiles_cache: dict[str, np.ndarray] = {}
        if self._use_disk_cache:
            cache_file = self._DEFAULT_CACHE_DIR / f"{self.get_cache_key()}.pkl"
            self._disk_cache_path = cache_file
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            if cache_file.exists():
                try:
                    self._smiles_cache = joblib.load(cache_file)
                except Exception:
                    logging.warning(
                        f"{self.name}: cache file corrupt or unreadable, starting fresh."
                    )
        self._cache_initialized = True

    def _persist_cache(self) -> None:
        """Merge the in-memory cache into the on-disk one under a file lock."""
        if not self._use_disk_cache:
            return
        cache_path = self._disk_cache_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = cache_path.with_suffix(".lock")

        with filelock.FileLock(lock_path, timeout=600):
            if cache_path.exists():
                on_disk = joblib.load(cache_path)
                on_disk.update(self._smiles_cache)
                self._smiles_cache = on_disk
            tmp_path = cache_path.with_suffix(".tmp")
            joblib.dump(self._smiles_cache, tmp_path)
            tmp_path.replace(cache_path)

    def featurize(self, smiles_list: List[str]) -> np.ndarray:
        """Return feature matrix, computing only SMILES absent from cache."""
        if not self.cache_enabled:
            return self._compute_features(smiles_list)
        self._ensure_cache()
        unique_uncached = list(
            dict.fromkeys(s for s in smiles_list if s not in self._smiles_cache)
        )
        if unique_uncached:
            new_features = self._compute_features(unique_uncached)
            self._smiles_cache.update(zip(unique_uncached, new_features))
            self._persist_cache()
        return np.stack([self._smiles_cache[smi] for smi in smiles_list])

    @abc.abstractmethod
    def _compute_features(self, smiles_list: List[str]) -> np.ndarray:
        """Compute features for the given SMILES, bypassing the cache."""
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

    def get_cache_key(self) -> str:
        """Return `<name>_<5-char hash of params>`."""
        params = str(self.get_hashable_params_values()).encode("utf-8")
        return f"{self.name}_{hashlib.md5(params).hexdigest()[:5]}"

    @staticmethod
    def _to_mols(smiles_list: List[str]) -> List[Chem.Mol | None]:
        """Parse SMILES, logging the ones RDKit fails to parse."""
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        for i, (mol, smi) in enumerate(zip(mols, smiles_list)):
            if mol is None:
                logging.debug(f"Failed to convert SMILES at index {i}: {smi}")
        return mols


@gin.configurable
class EcfpFeaturizer(FeaturizerBase):
    """Extended Connectivity Fingerprint (Morgan/ECFP) featurizer."""

    def __init__(self, radius: int = 2, n_bits: int = 2048, count: bool = False):
        self.radius = radius
        self.n_bits = n_bits
        self.count = count
        self.generator = GetMorganGenerator(radius=radius, fpSize=n_bits)

    def _compute_features(self, smiles_list: List[str]) -> np.ndarray:
        """Generate ECFP fingerprints for given SMILES."""
        mols = self._to_mols(smiles_list)
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
        """Drop the unpicklable generator."""
        state = self.__dict__.copy()
        del state["generator"]
        return state

    def __setstate__(self, state):
        """Rebuild the generator after unpickling."""
        self.__dict__.update(state)
        self.generator = GetMorganGenerator(radius=self.radius, fpSize=self.n_bits)


@gin.configurable
class PropertyFeaturizer(FeaturizerBase):
    """RDKit molecular property descriptor featurizer with normalization."""

    # Output depends on a fitted scaler, so disk cache would serve stale values.
    _use_disk_cache = False

    def __init__(self, scaler=None):
        # Fresh default per instance; a shared one would be re-fit across instances.
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.is_fitted = False

    def _compute_features(self, smiles_list: List[str]) -> np.ndarray:
        """Generate normalized molecular descriptors."""
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        descriptor_dicts = [self._compute_descriptors(mol) for mol in mols]
        desc_array = pd.DataFrame(descriptor_dicts).fillna(0).to_numpy()

        # float32 cast can overflow large descriptors to inf.
        desc_array = np.array(desc_array, dtype=np.float32)
        if np.isnan(desc_array).sum() or np.isinf(desc_array).sum():
            logging.warning(
                f"PropertyFeaturizer: Found {np.isnan(desc_array).sum()} NaNs and "
                f"{np.isinf(desc_array).sum()} infinite values in descriptors. "
                "Replacing with zeros."
            )
            desc_array = np.nan_to_num(desc_array, nan=0.0, posinf=0.0, neginf=0.0)

        # Fit on the first call only, so later calls reuse training statistics.
        if not self.is_fitted:
            self.scaler.fit(desc_array)
            self.is_fitted = True
        desc_array = self.scaler.transform(desc_array)
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

    def _compute_features(self, smiles_list: List[str]) -> np.ndarray:
        """Generate MACCS keys fingerprints for given SMILES."""
        mols = self._to_mols(smiles_list)
        fps = [MACCSkeys.GenMACCSKeys(mol) for mol in mols]
        return np.array([np.array(fp) for fp in fps])

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

    def _compute_features(self, smiles_list: List[str]) -> np.ndarray:
        """Generate Klekota-Roth fingerprints for given SMILES."""
        mols = self._to_mols(smiles_list)
        return np.array([self._get_krfp_fingerprint(mol) for mol in mols])

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
        with open(keys_path) as f:
            klek_keys = [line.strip() for line in f]
        return [Chem.MolFromSmarts(key) for key in klek_keys]

    def _get_krfp_fingerprint(self, mol: Chem.Mol) -> np.ndarray:
        """Return presence bits of the 4860 Klekota-Roth substructures in `mol`.

        Paper: https://doi.org/10.1093/bioinformatics/btn479
        """
        return np.array([int(mol.HasSubstructMatch(key)) for key in self.keys_mols])


@gin.configurable
class PropertyEcfpFeaturizer(FeaturizerBase):
    """Combined property descriptor and ECFP fingerprint featurizer."""

    # TODO: replace with a union featurizer combining any n featurizers.

    # Output depends on a fitted scaler, so disk cache would serve stale values.
    _use_disk_cache = False

    def __init__(self, radius: int = 2, n_bits: int = 2048, count: bool = False):
        self.ecfp = EcfpFeaturizer(radius=radius, n_bits=n_bits, count=count)
        self.properties = PropertyFeaturizer()

    def _compute_features(self, smiles_list: List[str]) -> np.ndarray:
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
