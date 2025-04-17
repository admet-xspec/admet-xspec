from src.featurizer.FeaturizerBase import FeaturizerBase
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import Chem
from typing import List
import numpy as np
import gin


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
    ):
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

            if self.count:
                fp = self.generator.GetCountFingerprintAsNumPy(mol)
            else:
                fp = self.generator.GetFingerprintAsNumPy(mol)
            fp_list.append(fp)

        return np.stack(fp_list)

    def name(self):
        return "ecfp"
