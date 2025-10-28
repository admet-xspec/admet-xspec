import rdkit.Chem as Chem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize.rdMolStandardize import Uncharger
from rdkit import RDLogger
from sklearn import metrics
import pandas as pd
import logging
from pathlib import Path
import csv

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
    values = (
        "| "
        + " | ".join(
            [
                f"{v:.3f}" if isinstance(v, float) else str(v)
                for v in dictionary.values()
            ]
        )
        + " |"
    )

    table = "\n".join([header, separator, values])
    logging.info("\n" + table)


def parse_smiles_from_messy_csv(path: str | Path) -> pd.Series:
    """
    Parses a SMILES column from a .csv, which may have different possible names.
    :param path: str | Path, path to the .csv file
    :return: tuple of (smiles: pd.Series, target: pd.Series)
    """

    # Different possible names for the smiles column
    SMILES_COLS = ["smiles", "SMILES", "Smiles", "molecule", "Molecule", "MOL"]
    df = pd.read_csv(path)

    # Check how many possible SMILES columns are present
    smiles_cols_present = [col for col in SMILES_COLS if col in df.columns]
    if len(smiles_cols_present) == 0:
        raise ValueError(f"No SMILES column found. Expected one of: {SMILES_COLS}")
    elif len(smiles_cols_present) > 1:
        raise ValueError(
            f"Multiple SMILES columns found: {smiles_cols_present}. Expected only one of {SMILES_COLS}."
        )
    return df[smiles_cols_present[0]]


def parse_targets_from_messy_csv(
    path: str | Path, target_col_name: str = None
) -> pd.Series | None:
    """
    Parses a target column from a .csv, which may have different possible names.
    Sometimes there is no target column at all (e.g. in inference datasets), or you may
    want to specify a custom target column name, which this function allows.
    :param path: str | Path, path to the .csv file
    :param target_col_name: str, custom name of the target column to look for
    :return: pd.Series or None if no target column is found
    """

    # Different possible names for the target column
    TARGET_COLS = ["y", "Y", "target", "Target", "label", "Label"]
    df = pd.read_csv(path)

    if target_col_name is not None:
        # Look for the custom target column name first
        if target_col_name in df.columns:
            return df[target_col_name]
        else:
            raise ValueError(
                f"Specified target column '{target_col_name}' not found in the data."
            )

    # Check how many possible target columns are present
    target_cols_present = [col for col in TARGET_COLS if col in df.columns]
    if len(target_cols_present) == 0:
        logging.debug(
            f"No target column found in the dataset. Expected one of: {TARGET_COLS}"
        )
        return None
    elif len(target_cols_present) > 1:
        raise ValueError(
            f"Multiple target columns found: {target_cols_present}. Expected only one of {TARGET_COLS}."
        )
    return df[target_cols_present[0]]

def detect_csv_delimiter(path: str | Path) -> str:
    """
    Detects the delimiter used in a CSV file.
    :param path: str | Path, path to the .csv file
    :return: str, detected delimiter
    """
    with open(path, 'r') as file:
        sample = file.read(1024)
    sniffer = csv.Sniffer()
    dialect = sniffer.sniff(sample)
    return dialect.delimiter