import rdkit.Chem as Chem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize.rdMolStandardize import Uncharger
from rdkit import RDLogger
from sklearn import metrics
import logging

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
            mol = _RdkitUncharger.un.uncharge(mol) # (*) except for this
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

def log_markdown_table(dictionary: dict):
    """
    Logs a dictionary as a horizonatal markdown table.
    :param dictionary: dict
    :return: None
    """
    if not isinstance(dictionary, dict):
        raise ValueError("Input must be a dictionary")
    if len(dictionary) == 0:
        logging.info("Empty dictionary - nothing to log")
        return

    header = "| " + " | ".join(dictionary.keys()) + " |"
    separator = "| " + " | ".join(["---"] * len(dictionary)) + " |"
    values = "| " + " | ".join([f"{v:.3f}" if isinstance(v, float) else str(v) for v in dictionary.values()]) + " |"

    table = "\n".join([header, separator, values])
    logging.info("\n" + table)