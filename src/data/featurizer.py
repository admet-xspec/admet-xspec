import abc
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import Chem
import pandas as pd
from typing import List
import numpy as np
import gin
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler


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


@gin.configurable
class PropertyFeaturizer(FeaturizerBase):
    def __init__(self):
        super(PropertyFeaturizer, self).__init__()
        # Initialize a StandardScaler to normalize the descriptors
        self.scaler = StandardScaler()
        self.scaler_fitted = False

    def featurize(self, smiles_list: List[str]) -> np.ndarray:
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        descs = [self.get_descriptors(mol, missing=np.nan) for mol in mols]
        # Convert list of dicts to DataFrame
        desc_df = pd.DataFrame(descs)
        # Fill NaN values with 0
        desc_df.fillna(0, inplace=True)
        # Convert DataFrame to numpy array
        descs = desc_df.to_numpy()
        # Normalize the descriptors
        if not self.scaler_fitted:
            self.scaler.fit(descs)
            self.scaler_fitted = True
        descs = self.scaler.transform(descs)

        return descs

    @staticmethod
    def get_descriptors(mol, missing=None):
        desc_dict = {}
        for name, fn in Descriptors._descList:
            try:
                value = fn(mol)
            except:
                value = missing
            desc_dict[name] = value
        return desc_dict


@gin.configurable
class PropertyEcfpFeaturizer(FeaturizerBase):
    def __init__(self, radius: int = 2, n_bits: int = 2048, count: bool = False):
        super(PropertyEcfpFeaturizer, self).__init__()
        self.ecfp_featurizer = EcfpFeaturizer(radius=radius, n_bits=n_bits, count=count)
        self.property_featurizer = PropertyFeaturizer()

    def featurize(self, smiles_list: List[str]) -> np.ndarray:
        ecfp_features = self.ecfp_featurizer.featurize(smiles_list)
        property_features = self.property_featurizer.featurize(smiles_list)

        # Combine features
        combined_features = np.hstack((ecfp_features, property_features))
        return combined_features
