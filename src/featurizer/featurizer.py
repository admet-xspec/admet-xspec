import abc
import numpy as np
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import Chem
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
        self.generator = GetMorganGenerator(radius=radius, fpSize=n_bits)

        # If count is True, the fingerprint will be a count fingerprint
        self.count = count

    def featurize(
        self,
        smiles_list: List[str],
    ) -> np.ndarray:
        """
        Featurize the given SMILES string
        """

        fp_list = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(
                    f"SMILES string {smi} could not be converted to RDKit mol object"
                )

            # Generate the fingerprint
            if self.count:
                fp = self.generator.GetCountFingerprintAsNumPy(mol)
            else:
                fp = self.generator.GetFingerprintAsNumPy(mol)
            fp_list.append(fp)

        return np.stack(fp_list)


from mordred import Calculator, descriptors


@gin.configurable
class MordredFeaturizer(FeaturizerBase):
    def __init__(self, ignore_3D: bool = True):
        self.calc = Calculator(descriptors, ignore_3D=False)
        super(MordredFeaturizer, self).__init__()

    def featurize(
        self,
        smiles_list: List[str],
    ) -> np.ndarray:

        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        if any(mol is None for mol in mols):
            raise ValueError(
                "One or more SMILES strings could not be converted to RDKit mol object"
            )

        # Calculate the descriptors
        descs = np.array(self.calc.pandas(mols))

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
