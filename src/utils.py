import rdkit.Chem as Chem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize.rdMolStandardize import Uncharger
from rdkit import RDLogger
from sklearn import metrics
import scipy.stats as stats
import pandas as pd
import logging
from pathlib import Path
import optuna
from typing import Any, Dict, Tuple, Mapping
import numpy as np
import csv
import os
import toml

RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)

_RdkitUncharger = Uncharger()
_RdkitSaltRemover = SaltRemover()


def get_clean_smiles(smiles: str, remove_salt: bool = True) -> str | None:
    """Strip salts, neutralize charges and return canonical SMILES.

    Standard SMILES cleaning for training and inference; change with caution.

    Returns:
        Canonical SMILES with at least one atom, or None if the input is not a
        string, cannot be parsed, has no atoms (before or after salt stripping),
        or RDKit fails unexpectedly.
    """
    if not isinstance(smiles, str):
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        if remove_salt:
            stripped = _RdkitSaltRemover.StripMol(mol)
            if stripped is not None:
                mol = stripped
            mol = _RdkitUncharger.uncharge(mol)
            # Keep the largest fragment.
            mol_fragments = Chem.GetMolFrags(mol, asMols=True)
            if len(mol_fragments) > 0:
                mol = max(mol_fragments, key=lambda x: x.GetNumAtoms())

        # "" parses to a zero-atom Mol, so this is needed even without salt removal.
        if mol.GetNumAtoms() == 0:
            return None

        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True) or None

    except Exception:
        logger.warning("get_clean_smiles failed for input %r", smiles, exc_info=True)
        return None


def log_markdown_table(dictionary: dict):
    """Log a dictionary as a horizontal markdown table."""
    if not isinstance(dictionary, dict):
        raise ValueError("Input must be a dictionary")
    if len(dictionary) == 0:
        logging.info("Empty dictionary - nothing to log")
        return

    header = "| " + " | ".join(dictionary.keys()) + " |"
    separator = "| " + " | ".join(["---"] * len(dictionary)) + " |"
    cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in dictionary.values()]
    values = "| " + " | ".join(cells) + " |"

    table = "\n".join([header, separator, values])
    logging.info("\n" + table + "\n")


def detect_csv_delimiter(path: str | Path) -> str:
    """Detect the delimiter of a CSV file from its first 1024 characters."""
    with open(path, "r") as file:
        sample = file.read(1024)
    return csv.Sniffer().sniff(sample).delimiter


def sample_optuna_params(
    trial: optuna.Trial, params_distribution: Dict[str, Any]
) -> Dict[str, Any]:
    """Sample parameters from `params_distribution` via Optuna `suggest_*` calls.

    Supported specs:
        ('int',          low, high)          -> suggest_int
        ('int_log',      low, high)          -> suggest_int(..., log=True)
        ('float',        low, high)          -> suggest_float
        ('float_log',    low, high)          -> suggest_float(..., log=True)
        ('categorical',  [v1, v2, ...])      -> suggest_categorical
    """
    sampled: Dict[str, Any] = {}
    for name, (kind, *args) in params_distribution.items():
        if kind == "int":
            sampled[name] = trial.suggest_int(name, *args)
        elif kind == "int_log":
            sampled[name] = trial.suggest_int(name, *args, log=True)
        elif kind == "float":
            sampled[name] = trial.suggest_float(name, *args)
        elif kind == "float_log":
            sampled[name] = trial.suggest_float(name, *args, log=True)
        elif kind == "categorical":
            sampled[name] = trial.suggest_categorical(name, args[0])
        else:
            raise ValueError(
                f"Unknown distribution kind '{kind}' for param '{name}'. "
                "Supported: 'int', 'int_log', 'float', 'float_log', 'categorical'."
            )
    return sampled


def pr_auc_score(y_true, y_pred):
    """Return the area under the precision-recall curve."""
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred)
    return metrics.auc(recall, precision)


def get_metric_callable(metric_name: str):
    """Return the `metric(y_true, y_pred)` callable for `metric_name`."""
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
        "spearman": lambda y_true, y_pred: stats.spearmanr(y_true, y_pred).correlation,
        "pr_auc": pr_auc_score,
    }
    if metric_name not in metrics_dict:
        raise ValueError(
            f"Invalid metric name: '{metric_name}'. Supported metrics: {list(metrics_dict.keys())}"
        )
    return metrics_dict[metric_name]


def get_available_cpus() -> int:
    """Return the number of CPUs this process may actually use.

    `os.cpu_count()` reports every core of the node, not the SLURM allocation, so
    take the minimum of `SLURM_CPUS_PER_TASK`, the CPU affinity mask and
    `os.cpu_count()`.
    """
    candidates = []
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK", "")
    if slurm_cpus.isdigit() and int(slurm_cpus) > 0:
        candidates.append(int(slurm_cpus))
    try:
        candidates.append(len(os.sched_getaffinity(0)))
    except AttributeError:  # not available on macOS/Windows
        pass
    candidates.append(os.cpu_count() or 1)
    return max(1, min(candidates))


def plan_parallel_trials(
    n_cpus: int,
    n_trials: int,
    n_workers: int | None = -1,
    threads_per_trial: int | None = None,
    model_is_threaded: bool = True,
    default_threads_per_trial: int = 4,
) -> Tuple[int, int]:
    """Split a CPU budget between concurrent Optuna trials and threads per model.

    Args:
        n_cpus: Total CPUs available to the process.
        n_trials: Number of trials in the study; caps the number of workers.
        n_workers: Concurrent trials; -1, 0 or None chooses automatically.
        threads_per_trial: Native threads per model; None chooses automatically.
        model_is_threaded: Whether the model can use more than one thread.
        default_threads_per_trial: Threads per trial when both are automatic.

    Returns:
        `(n_workers, threads_per_trial)`, with their product <= `n_cpus` unless the
        caller explicitly asked for more.
    """
    n_cpus = max(1, n_cpus)
    n_trials = max(1, n_trials)
    if not model_is_threaded:
        threads_per_trial = 1

    if n_workers is None or n_workers < 1:
        per_trial = threads_per_trial or min(default_threads_per_trial, n_cpus)
        n_workers = max(1, n_cpus // per_trial)
    n_workers = min(n_workers, n_trials)

    if threads_per_trial is None:
        # Equal share, including cores freed by the `n_trials` cap.
        threads_per_trial = max(1, n_cpus // n_workers)

    if n_workers * threads_per_trial > n_cpus:
        logging.warning(
            f"{n_workers} parallel trials x {threads_per_trial} thread(s) each exceeds the "
            f"{n_cpus} available CPU(s); expect oversubscription."
        )
    return n_workers, threads_per_trial


_BLOCK_EIGHTHS = " ▏▎▍▌▋▊▉█"


def _render_bar(fraction: float, width: int, ascii_only: bool = False) -> str:
    """Render a left-aligned bar filling `fraction` of `width` characters.

    Non-zero fractions draw at least the smallest mark, so rare bins stay visible.
    """
    fraction = min(max(float(fraction), 0.0), 1.0)
    if ascii_only:
        filled = int(round(fraction * width))
        if fraction > 0 and filled == 0:
            filled = 1
        return ("#" * filled).ljust(width)
    eighths = int(round(fraction * width * 8))
    if fraction > 0 and eighths == 0:
        eighths = 1
    full, rem = divmod(eighths, 8)
    bar = "█" * full + (_BLOCK_EIGHTHS[rem] if rem else "")
    return bar.ljust(width)


def format_cli_histogram(
    values, n_bins: int = 10, width: int = 40, ascii_only: bool = False
) -> str:
    """Return a horizontal text histogram of the finite numeric `values`.

    Non-finite and non-numeric entries are ignored. One row per bin:
    `[low, high) |bar| count (percent)`; the last bin is closed.
    """
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return "  (no finite values to plot)"

    # Identical values would otherwise spread over mostly empty bins.
    n_bins = 1 if finite.min() == finite.max() else max(1, int(n_bins))
    counts, edges = np.histogram(finite, bins=n_bins)
    total = int(counts.sum())
    max_count = int(counts.max())
    lows = [f"{e:.4g}" for e in edges[:-1]]
    highs = [f"{e:.4g}" for e in edges[1:]]
    low_w = max(len(s) for s in lows)
    high_w = max(len(s) for s in highs)
    count_w = len(str(max_count))
    sep = "|" if ascii_only else "│"

    lines = []
    for i, count in enumerate(counts):
        close = "]" if i == len(counts) - 1 else ")"
        label = f"[{lows[i]:>{low_w}}, {highs[i]:>{high_w}}{close}"
        bar = _render_bar(count / max_count, width, ascii_only)
        pct = 100.0 * count / total
        lines.append(f"  {label} {sep}{bar}{sep} {int(count):>{count_w}} ({pct:5.1f}%)")
    return "\n".join(lines)


def format_cli_bar_chart(
    counts: Mapping[Any, int], width: int = 40, ascii_only: bool = False
) -> str:
    """Return a horizontal text bar chart, one row per key: `key |bar| count (percent)`."""
    if not counts:
        return "  (nothing to plot)"
    total = sum(counts.values())
    max_count = max(counts.values())
    label_w = max(len(str(k)) for k in counts)
    count_w = len(str(max_count))
    sep = "|" if ascii_only else "│"

    lines = []
    for key, count in counts.items():
        frac = count / max_count if max_count else 0.0
        bar = _render_bar(frac, width, ascii_only)
        pct = 100.0 * count / total if total else 0.0
        lines.append(f"  {str(key):>{label_w}} {sep}{bar}{sep} {count:>{count_w}} ({pct:5.1f}%)")
    return "\n".join(lines)


def parse_project_version() -> str:
    version = "unknown"
    # src/utils.py -> repository root
    pyproject_toml_file = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if pyproject_toml_file.exists() and pyproject_toml_file.is_file():
        data = toml.load(pyproject_toml_file)
        # check project.version
        if "project" in data and "version" in data["project"]:
            version = data["project"]["version"]
    return version