# Keeps `TransformerMixin | None` annotations unevaluated, so the docs build, which
# mocks sklearn, can import this module.
from __future__ import annotations

import abc
import logging
import os
import subprocess
import tempfile
from typing import List, Hashable
import hashlib
import filelock
import joblib

import gin
import numpy as np
import pandas as pd
import pathlib
from rdkit import Chem, DataStructs
from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP
from rdkit.Chem import Descriptors, MACCSkeys, rdReducedGraphs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import StandardScaler

_DEFAULT_CACHE_DIR = pathlib.Path(tempfile.gettempdir()) / "admet_fp_cache"


@gin.configurable
def fp_cache_dir(path: str | None = None) -> pathlib.Path:
    """Return the directory of the on-disk featurizer cache.

    Taken from gin (`fp_cache_dir.path`), then `$ADMET_FP_CACHE_DIR`, then the system
    temp dir. Point it at shared storage when the temp dir is node-local (e.g. SLURM).
    """
    path = path or os.environ.get("ADMET_FP_CACHE_DIR")
    return pathlib.Path(path) if path else _DEFAULT_CACHE_DIR


class FeaturizerBase(abc.ABC):
    """Base class for molecular featurizers."""

    # False for featurizers whose output must not be cached on disk.
    _use_disk_cache: bool = True
    # Set False per instance to bypass caching entirely (e.g. in tests).
    cache_enabled: bool = True
    # Rebuilt lazily, so never pickled: it would bloat every saved model.
    _cache_attrs = ("_smiles_cache", "_cache_initialized", "_disk_cache_path")

    def __getstate__(self):
        """Drop the cache."""
        state = self.__dict__.copy()
        for attr in self._cache_attrs:
            state.pop(attr, None)
        return state

    def __setstate__(self, state):
        """Restore state, dropping the cache that older pickles still carry."""
        for attr in self._cache_attrs:
            state.pop(attr, None)
        self.__dict__.update(state)

    def make_scaler(self) -> TransformerMixin | None:
        """Return an unfitted scaler for this featurizer's output, or None.

        Featurizers stay stateless, so their output can be cached and shared;
        the predictor fits the scaler on its training data.
        """
        return None

    def pop_legacy_scaler(self) -> TransformerMixin | None:
        """Return the fitted scaler of a model pickled when featurizers fitted it, or None."""
        return None

    def _ensure_cache(self) -> None:
        """Lazily load the in-memory cache, seeded from disk if available."""
        if hasattr(self, "_cache_initialized"):
            return
        self._smiles_cache: dict[str, np.ndarray] = {}
        if self._use_disk_cache:
            cache_file = fp_cache_dir() / f"{self.get_cache_key()}.pkl"
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
        state = super().__getstate__()
        del state["generator"]
        return state

    def __setstate__(self, state):
        """Rebuild the generator after unpickling."""
        super().__setstate__(state)
        self.generator = GetMorganGenerator(radius=self.radius, fpSize=self.n_bits)


class _TailScaler(TransformerMixin, BaseEstimator):
    """Scale the columns from `start` on, passing the leading ones through."""

    def __init__(self, scaler, start: int):
        self.scaler = scaler
        self.start = start

    def fit(self, X, y=None):
        self.scaler_ = clone(self.scaler).fit(X[:, self.start :])
        return self

    def transform(self, X):
        return np.hstack(
            [X[:, : self.start], self.scaler_.transform(X[:, self.start :])]
        )


@gin.configurable
class PropertyFeaturizer(FeaturizerBase):
    """RDKit molecular property descriptor featurizer.

    Returns raw descriptors, so they are disk-cached; `make_scaler` gives the
    scaler that predictors fit on their training data.
    """

    def __init__(self, scaler=None):
        # Unfitted template, cloned for every fit.
        self.scaler = scaler if scaler is not None else StandardScaler()

    def __setstate__(self, state):
        """Keep the fitted scaler of pickles from when this featurizer fitted it."""
        if state.pop("is_fitted", False):
            state["_legacy_scaler"] = state["scaler"]
        super().__setstate__(state)

    def make_scaler(self) -> TransformerMixin:
        return clone(self.scaler)

    def pop_legacy_scaler(self) -> TransformerMixin | None:
        return self.__dict__.pop("_legacy_scaler", None)

    def _compute_features(self, smiles_list: List[str]) -> np.ndarray:
        """Generate raw molecular descriptors."""
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
        return desc_array

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

    def __init__(self, radius: int = 2, n_bits: int = 2048, count: bool = False):
        self.ecfp = EcfpFeaturizer(radius=radius, n_bits=n_bits, count=count)
        self.properties = PropertyFeaturizer()

    def _compute_features(self, smiles_list: List[str]) -> np.ndarray:
        """Generate combined ECFP and property features."""
        ecfp_features = self.ecfp.featurize(smiles_list)
        property_features = self.properties.featurize(smiles_list)
        return np.hstack((ecfp_features, property_features))

    def make_scaler(self) -> TransformerMixin:
        """Scale only the descriptors, which follow the `n_bits` ECFP columns."""
        return _TailScaler(self.properties.make_scaler(), start=self.ecfp.n_bits)

    def pop_legacy_scaler(self) -> TransformerMixin | None:
        scaler = self.properties.pop_legacy_scaler()
        if scaler is None:
            return None
        legacy = _TailScaler(clone(scaler), start=self.ecfp.n_bits)
        legacy.scaler_ = scaler
        return legacy

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


# The 200 RDKit descriptors used by MapLight, fixed so new RDKit descriptors don't shift columns.
# fmt: off
_MAPLIGHT_DESCRIPTORS = (
    "BalabanJ", "BertzCT", "Chi0", "Chi0n", "Chi0v", "Chi1", "Chi1n", "Chi1v", "Chi2n",
    "Chi2v", "Chi3n", "Chi3v", "Chi4n", "Chi4v", "EState_VSA1", "EState_VSA10",
    "EState_VSA11", "EState_VSA2", "EState_VSA3", "EState_VSA4", "EState_VSA5",
    "EState_VSA6", "EState_VSA7", "EState_VSA8", "EState_VSA9", "ExactMolWt",
    "FpDensityMorgan1", "FpDensityMorgan2", "FpDensityMorgan3", "FractionCSP3",
    "HallKierAlpha", "HeavyAtomCount", "HeavyAtomMolWt", "Ipc", "Kappa1", "Kappa2",
    "Kappa3", "LabuteASA", "MaxAbsEStateIndex", "MaxAbsPartialCharge", "MaxEStateIndex",
    "MaxPartialCharge", "MinAbsEStateIndex", "MinAbsPartialCharge", "MinEStateIndex",
    "MinPartialCharge", "MolLogP", "MolMR", "MolWt", "NHOHCount", "NOCount",
    "NumAliphaticCarbocycles", "NumAliphaticHeterocycles", "NumAliphaticRings",
    "NumAromaticCarbocycles", "NumAromaticHeterocycles", "NumAromaticRings",
    "NumHAcceptors", "NumHDonors", "NumHeteroatoms", "NumRadicalElectrons",
    "NumRotatableBonds", "NumSaturatedCarbocycles", "NumSaturatedHeterocycles",
    "NumSaturatedRings", "NumValenceElectrons", "PEOE_VSA1", "PEOE_VSA10", "PEOE_VSA11",
    "PEOE_VSA12", "PEOE_VSA13", "PEOE_VSA14", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4",
    "PEOE_VSA5", "PEOE_VSA6", "PEOE_VSA7", "PEOE_VSA8", "PEOE_VSA9", "RingCount",
    "SMR_VSA1", "SMR_VSA10", "SMR_VSA2", "SMR_VSA3", "SMR_VSA4", "SMR_VSA5", "SMR_VSA6",
    "SMR_VSA7", "SMR_VSA8", "SMR_VSA9", "SlogP_VSA1", "SlogP_VSA10", "SlogP_VSA11",
    "SlogP_VSA12", "SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA6",
    "SlogP_VSA7", "SlogP_VSA8", "SlogP_VSA9", "TPSA", "VSA_EState1", "VSA_EState10",
    "VSA_EState2", "VSA_EState3", "VSA_EState4", "VSA_EState5", "VSA_EState6",
    "VSA_EState7", "VSA_EState8", "VSA_EState9", "fr_Al_COO", "fr_Al_OH",
    "fr_Al_OH_noTert", "fr_ArN", "fr_Ar_COO", "fr_Ar_N", "fr_Ar_NH", "fr_Ar_OH",
    "fr_COO", "fr_COO2", "fr_C_O", "fr_C_O_noCOO", "fr_C_S", "fr_HOCCN", "fr_Imine",
    "fr_NH0", "fr_NH1", "fr_NH2", "fr_N_O", "fr_Ndealkylation1", "fr_Ndealkylation2",
    "fr_Nhpyrrole", "fr_SH", "fr_aldehyde", "fr_alkyl_carbamate", "fr_alkyl_halide",
    "fr_allylic_oxid", "fr_amide", "fr_amidine", "fr_aniline", "fr_aryl_methyl",
    "fr_azide", "fr_azo", "fr_barbitur", "fr_benzene", "fr_benzodiazepine",
    "fr_bicyclic", "fr_diazo", "fr_dihydropyridine", "fr_epoxide", "fr_ester",
    "fr_ether", "fr_furan", "fr_guanido", "fr_halogen", "fr_hdrzine", "fr_hdrzone",
    "fr_imidazole", "fr_imide", "fr_isocyan", "fr_isothiocyan", "fr_ketone",
    "fr_ketone_Topliss", "fr_lactam", "fr_lactone", "fr_methoxy", "fr_morpholine",
    "fr_nitrile", "fr_nitro", "fr_nitro_arom", "fr_nitro_arom_nonortho", "fr_nitroso",
    "fr_oxazole", "fr_oxime", "fr_para_hydroxylation", "fr_phenol",
    "fr_phenol_noOrthoHbond", "fr_phos_acid", "fr_phos_ester", "fr_piperdine",
    "fr_piperzine", "fr_priamide", "fr_prisulfonamd", "fr_pyridine", "fr_quatN",
    "fr_sulfide", "fr_sulfonamd", "fr_sulfone", "fr_term_acetylene", "fr_tetrazole",
    "fr_thiazole", "fr_thiocyan", "fr_thiophene", "fr_unbrch_alkane", "fr_urea", "qed",
)
# fmt: on

_DESCRIPTOR_FUNCS = dict(Descriptors._descList)

# Length of the ErG fingerprint, fixed by RDKit.
_ERG_SIZE = 315


@gin.configurable
class MapLightFeaturizer(FeaturizerBase):
    """MapLight features: Morgan counts, Avalon counts, ErG and 200 RDKit descriptors.

    Columns are in that order, 2563 with the defaults. Values are raw, so they are
    disk-cached; `make_scaler` standardizes only the trailing descriptors.
    Paper: https://arxiv.org/abs/2310.00174
    Reference: https://github.com/maplightrx/MapLight-TDC/blob/main/maplight.py
    """

    def __init__(
        self,
        morgan_radius: int = 2,
        morgan_n_bits: int = 1024,
        avalon_n_bits: int = 1024,
        scaler=None,
    ):
        self.morgan_radius = morgan_radius
        self.morgan_n_bits = morgan_n_bits
        self.avalon_n_bits = avalon_n_bits
        # Unfitted template, cloned for every fit.
        self.scaler = scaler if scaler is not None else StandardScaler()

    @property
    def n_fingerprint_features(self) -> int:
        """Number of leading fingerprint columns, which precede the descriptors."""
        return self.morgan_n_bits + self.avalon_n_bits + _ERG_SIZE

    def make_scaler(self) -> TransformerMixin:
        """Scale only the descriptors, which follow the fingerprint columns."""
        return _TailScaler(clone(self.scaler), start=self.n_fingerprint_features)

    def _compute_features(self, smiles_list: List[str]) -> np.ndarray:
        """Generate MapLight features, raising if any SMILES fails to parse."""
        mols = self._to_mols(smiles_list)
        failed = [smi for smi, mol in zip(smiles_list, mols) if mol is None]
        if failed:
            raise ValueError(
                f"{self.name}: RDKit failed to parse {len(failed)} SMILES: {failed[:10]}"
            )

        morgan = GetMorganGenerator(
            radius=self.morgan_radius, fpSize=self.morgan_n_bits
        )
        # float32 cast can overflow large descriptors (e.g. Ipc) to inf.
        features = np.array(
            [self._compute_molecule_features(mol, morgan) for mol in mols],
            dtype=np.float32,
        )
        n_nan, n_inf = np.isnan(features).sum(), np.isinf(features).sum()
        if n_nan or n_inf:
            logging.warning(
                f"{self.name}: Found {n_nan} NaNs and {n_inf} infinite values in "
                "features. Replacing with zeros."
            )
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features

    def _compute_molecule_features(self, mol: Chem.Mol, morgan) -> np.ndarray:
        """Concatenate the four MapLight feature blocks of one molecule."""
        avalon = np.zeros((self.avalon_n_bits,))
        DataStructs.ConvertToNumpyArray(
            GetAvalonCountFP(mol, nBits=self.avalon_n_bits), avalon
        )
        return np.concatenate(
            [
                morgan.GetCountFingerprintAsNumPy(mol),
                avalon,
                rdReducedGraphs.GetErGFingerprint(mol),
                self._compute_descriptors(mol),
            ]
        )

    @staticmethod
    def _compute_descriptors(mol: Chem.Mol) -> np.ndarray:
        """Compute the MapLight descriptors, with NaN for any that fail."""
        values = []
        for name in _MAPLIGHT_DESCRIPTORS:
            try:
                values.append(_DESCRIPTOR_FUNCS[name](mol))
            except Exception:
                values.append(np.nan)
        return np.array(values, dtype=np.float64)

    @property
    def feature_name(self) -> str:
        return "fp_maplight"

    @property
    def name(self) -> str:
        return "maplight_featurizer"

    def get_hashable_params_values(self) -> List[Hashable]:
        # Bump the version if the descriptor list or the block order changes.
        return [
            "maplight_v1",
            self.morgan_radius,
            self.morgan_n_bits,
            self.avalon_n_bits,
        ]


@gin.configurable
class MinimolFeaturizer(FeaturizerBase):
    """MiniMol pre-trained GNN embedding featurizer (512 dimensions).

    MiniMol's dependencies clash with ours, so it runs in a subprocess inside the
    isolated venv in `env_dir`; see `external/minimol/README.md` for setup.
    Paper: https://arxiv.org/abs/2406.14986
    """

    def __init__(self, env_dir: str = "external/minimol", batch_size: int = 1000):
        # Kept as given and resolved per call, so pickled predictors stay portable.
        self.env_dir = env_dir
        self.batch_size = batch_size

    def _compute_features(self, smiles_list: List[str]) -> np.ndarray:
        """Embed SMILES with the MiniMol worker, raising if any molecule fails."""
        env_dir = pathlib.Path(self.env_dir)
        python = env_dir / ".venv" / "bin" / "python"
        if not python.exists():
            raise FileNotFoundError(
                f"MiniMol environment not found at {python}. "
                f"Run `{env_dir / 'setup.sh'}` to create it."
            )

        logging.info(f"{self.name}: embedding {len(smiles_list)} molecules.")
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = pathlib.Path(tmp_dir) / "smiles.txt"
            output_path = pathlib.Path(tmp_dir) / "embeddings.npy"
            # One terminated line per SMILES, so empty entries keep their position.
            input_path.write_text("".join(f"{smi}\n" for smi in smiles_list))
            # Our venv must not leak into the worker's interpreter.
            env = {
                k: v
                for k, v in os.environ.items()
                if k not in ("PYTHONPATH", "VIRTUAL_ENV")
            }
            result = subprocess.run(
                [
                    str(python),
                    "-I",
                    str(env_dir / "embed.py"),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--batch-size",
                    str(self.batch_size),
                ],
                capture_output=True,
                text=True,
                env=env,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"{self.name}: MiniMol worker failed:\n{result.stderr[-5000:]}"
                )
            embeddings = np.load(output_path)

        failed = [
            smi for smi, row in zip(smiles_list, embeddings) if np.isnan(row).any()
        ]
        if failed:
            raise ValueError(
                f"{self.name}: MiniMol failed to embed {len(failed)} SMILES: {failed[:10]}"
            )
        return embeddings

    @property
    def feature_name(self) -> str:
        return "emb_minimol"

    @property
    def name(self) -> str:
        return "minimol_featurizer"

    def get_hashable_params_values(self) -> List[Hashable]:
        # Batch size and device don't change the embeddings, only the checkpoint does.
        return ["minimol_v1"]
