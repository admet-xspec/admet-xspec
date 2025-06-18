import abc
import logging

import numpy as np
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import Chem
import pandas as pd
from typing import List
import numpy as np
import gin


class FeaturizerBase(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def featurize(self, smiles_list: List[str]) -> np.ndarray:
        """
        Featurize the given SMILES string and return a numpy array of descriptors.
        """
        pass


@gin.configurable
class EcfpFeaturizer(FeaturizerBase):
    def __init__(self, radius: int = 2, n_bits: int = 2048, count: bool = False):
        super(EcfpFeaturizer, self).__init__()
        self.radius = radius
        self.n_bits = n_bits
        self.count = count
        self.generator = GetMorganGenerator(radius=radius, fpSize=n_bits)

    def featurize(
        self,
        smiles_list: List[str],
    ) -> np.ndarray:
        """
        Featurize the given SMILES string
        """
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        if any(mol is None for mol in mols):
            raise ValueError(
                "One or more SMILES strings could not be converted to RDKit mol object"
            )
        if self.count:
            fp_list = [self.generator.GetCountFingerprintAsNumPy(mol) for mol in mols]
        else:
            fp_list = [self.generator.GetFingerprintAsNumPy(mol) for mol in mols]

        return np.stack(fp_list)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle the generator
        del state["generator"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.generator = GetMorganGenerator(radius=self.radius, fpSize=self.n_bits)


from mordred import Calculator, descriptors


@gin.configurable
class MordredFeaturizer(FeaturizerBase):
    def __init__(self, ignore_3D: bool = True):
        self.calc = Calculator(descriptors, ignore_3D=ignore_3D, version="1.0.0")
        self.invalid_descriptors = None
        super(MordredFeaturizer, self).__init__()

    def featurize(
        self,
        smiles_list: List[str],
    ) -> np.ndarray:
        """
        Featurize the given SMILES string using Mordred descriptors.
        :param smiles_list:
        :return:
        """

        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        if any(mol is None for mol in mols):
            raise ValueError(
                "One or more SMILES strings could not be converted to RDKit mol object"
            )

        # Calculate the descriptors
        descs = self.calc.pandas(mols)

        # Check for non-numeric values and replace them with NaN
        descs = descs.apply(pd.to_numeric, errors="coerce")

        # Check for invalid descriptors
        if self.invalid_descriptors is None:
            # Store the invalid descriptors for future use
            # Invalid descriptors are those that have non-numeric values
            self.invalid_descriptors = descs.select_dtypes(
                exclude=[np.number]
            ).columns.tolist()
            logging.warning(
                f"Found {len(self.invalid_descriptors)} invalid descriptors: {self.invalid_descriptors}"
            )

        # Drop invalid descriptors
        # descs = descs.drop(columns=self.invalid_descriptors, errors="ignore")

        descs.to_numpy()

        return descs


@gin.configurable
class MordredEcfpFeaturizer(FeaturizerBase):
    def __init__(
        self,
        radius: int = 2,
        n_bits: int = 2048,
        count: bool = False,
        ignore_3D: bool = False,
    ):
        self.ecfp = EcfpFeaturizer(radius=radius, n_bits=n_bits, count=count)
        self.mordred = MordredFeaturizer(ignore_3D=ignore_3D)
        super(MordredEcfpFeaturizer, self).__init__()

    def featurize(
        self,
        smiles_list: List[str],
    ) -> np.ndarray:

        ecfp_features = self.ecfp.featurize(smiles_list)
        mordred_features = self.mordred.featurize(smiles_list)

        return np.hstack((ecfp_features, mordred_features))
