import logging

import rdkit.Chem as Chem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize.rdMolStandardize import Uncharger
from rdkit import RDLogger
from sklearn import metrics

# disable RDKit warnings
RDLogger.DisableLog("rdApp.*")

_RdkitUncharger = Uncharger()
_RdkitSaltRemover = SaltRemover()

def get_clean_smiles(smiles: str, remove_salt: bool = True) -> str | None:
    """
    Strips salts and removes charges from a molecule. Returns SMILES in canonical form.
    Its purpose is to be the standard procedure for cleaning SMILES strings before using them in our
    training and inference pipelines. Any modifications to this function should be done with caution.
    """
    if isinstance(smiles, str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        if remove_salt:
            mol = SaltRemover().StripMol(mol)
            mol = _RdkitUncharger.uncharge(mol) # (*) except for this
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

def get_converted_unit(
        val: float,
        from_unit: str,
        to_unit: str,
        mol_weight: float | None = None,
        tissue_density: float | None = 1,
    ):
    """Convert a measurment from one unit to another."""
    to_unit_implemented = {"uM"}
    assert to_unit in to_unit_implemented, f"Conversion to unit {to_unit} not implemented yet."

    if to_unit == "ug ml-1":
        to_unit = "ug/ml"
    if to_unit == "umol/L":
        to_unit = "uM"

    if to_unit == "uM":
        assert mol_weight is not None, (
            f"Molecular weight not provided "
            f"for {from_unit}->{to_unit} conversion."
        )

        match from_unit:
            case "ng/g":
                return (val * tissue_density / mol_weight) / 1000
            case "ug/g":
                return val / mol_weight
            case "ng/ml":
                return (val / mol_weight) / 1000
            case "ug/ml":
                return val / mol_weight
            case "ug/L":
                return (val / mol_weight) / 1000
            case "mg%":
                mg_per_liter = val * 10
                return (mg_per_liter * 1000) / mol_weight
            case "nM":
                return val / 1000
            case "uM":
                return val
            case "nmol/g":
                return val * tissue_density / 1000
            case "pmol/g":
                return val * tissue_density / 1_000_000
            case "umol/Kg":
                return val * tissue_density / 1000
            case _:
                logging.info(f"Do not support conversion from {from_unit}, returning None.")
                return None