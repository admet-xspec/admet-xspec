import rdkit.Chem as Chem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize.rdMolStandardize import Uncharger
from typing import List
from rdkit import RDLogger
import logging
from sklearn import metrics

# disable RDKit warnings
RDLogger.DisableLog("rdApp.*")


class SmilesCleaner:
    """
    Strips salts and removes charges from molecules.
    Its purpose is to be the standard procedure for cleaning SMILES strings before using them in our
    training and inference pipelines. Any modifications to this class should be done with caution.
    """

    def __init__(self):
        self.un = Uncharger()
        self.salt_remover = SaltRemover()

    def clean(self, smiles: str) -> str | None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Remove salts
            mol = self.salt_remover.StripMol(mol)
            # Uncharge the molecule
            mol = self.un.uncharge(mol)
            # Convert back to SMILES
            return Chem.MolToSmiles(mol)
        else:
            return None

# koziarskilab/BindingAffinityPipeline/blob/main/src/data/utils.py :)
def get_canonical_smiles(smiles: str, remove_salt: bool = True) -> str | None:
    """Convert a SMILES string to its canonical form."""
    if isinstance(smiles, str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        if remove_salt:
            mol = SaltRemover().StripMol(mol)
            # leave only the largest fragment
            mol_fragments = Chem.GetMolFrags(mol, asMols=True)
            if len(mol_fragments) > 0:
                mol = max(
                    Chem.GetMolFrags(mol, asMols=True), key=lambda x: x.GetNumAtoms()
                )
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    else:
        return None

def get_nice_class_name(obj):
    """
    Takes an object of any class and returns a clean name of the class.
    :param x: object (any)
    :return: str
    """
    return type(obj).__name__


def get_metric_callable(metric_name: str):
    metrics_dict = {
        "accuracy": metrics.accuracy_score,
        "roc_auc": metrics.roc_auc_score,
        "f1": metrics.f1_score,
        "precision": metrics.precision_score,
        "recall": metrics.recall_score,
        "mse": metrics.mean_squared_error,
        "mae": metrics.mean_absolute_error,
        "r2": metrics.r2_score,
        "rmse": metrics.root_mean_squared_error,
    }
    if metric_name not in metrics_dict.keys():
        raise ValueError
    return metrics_dict[metric_name]
