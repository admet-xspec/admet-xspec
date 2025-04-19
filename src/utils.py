import rdkit.Chem as Chem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize.rdMolStandardize import Uncharger
from typing import List
from rdkit import RDLogger
import logging

# disable RDKit warnings
RDLogger.DisableLog("rdApp.*")


def clean_smiles(smiles_list: List[str]) -> List[str | None]:
    """Remove invalid SMILES from a list of SMILES strings, strip salts, and remove duplicates."""
    un = Uncharger()
    salt_remover = SaltRemover()
    cleaned_smiles = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Remove salts
            mol = salt_remover.StripMol(mol)
            # Uncharge the molecule
            mol = un.uncharge(mol)
            # Convert back to SMILES
            cleaned_smiles.append(Chem.MolToSmiles(mol))
        else:
            logging.debug(f"Invalid SMILES in the dataset: {smiles}")
            cleaned_smiles.append(None)

    return cleaned_smiles


def get_nice_class_name(obj):
    """
    Takes an object of any class and returns a clean name of the class.
    :param x: object (any)
    :return: str
    """
    return type(obj).__name__
