from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, TypedDict


TASK_TO_PREDICTOR_DIR = {
    "binary_classification": "classifiers",
    "regression": "regressors",
}


@dataclass(frozen=True)
class ExperimentSelection:
    """Container describing one training experiment gin configuration."""

    processing_plan: str
    featurizer: str
    splitter: str
    sim_filter: str
    predictor: str
    datasets: list[str]
    test_origin_dataset: str
    task_setting: str


class GinOptions(TypedDict):
    processing_plans: list[str]
    featurizers: list[str]
    splitters: list[str]
    sim_filters: list[str]
    predictors: dict[str, list[str]]


def _list_gin_files(
    directory: Path, root_dir: Path, recursive: bool = False
) -> list[str]:
    if not directory.exists():
        return []
    pattern = "**/*.gin" if recursive else "*.gin"
    files = sorted(path for path in directory.glob(pattern) if path.is_file())
    return [f"configs/{path.relative_to(root_dir).as_posix()}" for path in files]


def _filter_out_internal_files(paths: Iterable[str]) -> list[str]:
    return [
        p
        for p in paths
        if not Path(p).name.startswith("_") and "/optimization/" not in p
    ]


def discover_gin_options(configs_dir: Path) -> GinOptions:
    """Discover selectable gin fragments under the provided `configs` directory."""

    processing_plans = _filter_out_internal_files(
        _list_gin_files(configs_dir / "processing_plans", root_dir=configs_dir)
    )
    featurizers = _filter_out_internal_files(
        _list_gin_files(
            configs_dir / "featurizers", root_dir=configs_dir, recursive=True
        )
    )
    splitters = _filter_out_internal_files(
        _list_gin_files(configs_dir / "splitters", root_dir=configs_dir)
    )
    sim_filters = _filter_out_internal_files(
        _list_gin_files(configs_dir / "sim_filters", root_dir=configs_dir)
    )

    predictors_by_task: dict[str, list[str]] = {}
    for task_setting, subdir in TASK_TO_PREDICTOR_DIR.items():
        predictor_paths = _list_gin_files(
            configs_dir / "predictors" / subdir,
            root_dir=configs_dir,
            recursive=True,
        )
        predictors_by_task[task_setting] = _filter_out_internal_files(predictor_paths)

    return {
        "processing_plans": processing_plans,
        "featurizers": featurizers,
        "splitters": splitters,
        "sim_filters": sim_filters,
        "predictors": predictors_by_task,
    }


def load_dataset_names(registry_file: Path) -> list[str]:
    """Load dataset names from a registry file while preserving input order."""

    if not registry_file.exists():
        return []

    seen: set[str] = set()
    names: list[str] = []
    for raw_line in registry_file.read_text(encoding="utf-8").splitlines():
        name = raw_line.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def validate_selection(selection: ExperimentSelection) -> None:
    """Raise a ValueError when an experiment selection is invalid."""

    if not selection.datasets:
        raise ValueError("At least one training dataset must be selected.")

    if selection.test_origin_dataset not in selection.datasets:
        raise ValueError(
            "`test_origin_dataset` must be one of the selected training datasets."
        )

    if selection.task_setting not in TASK_TO_PREDICTOR_DIR:
        raise ValueError(
            f"Unsupported task setting '{selection.task_setting}'. "
            f"Expected one of {list(TASK_TO_PREDICTOR_DIR)}."
        )

    expected_dir = TASK_TO_PREDICTOR_DIR[selection.task_setting]
    if f"predictors/{expected_dir}/" not in selection.predictor:
        raise ValueError(
            "Predictor path does not match the selected task setting. "
            f"Expected predictor from '{expected_dir}'."
        )


def render_gin_config(selection: ExperimentSelection) -> str:
    """Render one training configuration file in gin format."""

    validate_selection(selection)

    quoted_datasets = ",\n    ".join(f"'{name}'" for name in selection.datasets)

    return (
        f"include '{selection.processing_plan}'\n"
        "include 'configs/data_interface/data_interface.gin'\n"
        f"include '{selection.splitter}'\n"
        f"include '{selection.featurizer}'\n"
        f"include '{selection.sim_filter}'\n"
        f"include '{selection.predictor}'\n\n"
        "ProcessingPipeline.data_interface = %data_interface\n"
        "ProcessingPipeline.splitter = %splitter\n"
        "ProcessingPipeline.featurizer = %featurizer\n"
        "ProcessingPipeline.sim_filter = %sim_filter\n"
        "ProcessingPipeline.predictor = %predictor\n\n"
        "#========================================================#\n\n"
        "ProcessingPipeline.datasets = [\n"
        f"    {quoted_datasets}\n"
        "]\n\n"
        f"ProcessingPipeline.test_origin_dataset = '{selection.test_origin_dataset}'\n"
        f"ProcessingPipeline.task_setting = '{selection.task_setting}'\n\n"
        "ProcessingPipeline.hyperparams_source_sim_filter = None\n"
    )


def build_experiment_filename(selection: ExperimentSelection, index: int) -> str:
    """Create a deterministic and filesystem-safe filename for one experiment."""

    def stem(path: str) -> str:
        return Path(path).stem.replace("_", "-")

    dataset_token = selection.test_origin_dataset.replace("_", "-")
    return (
        f"exp_{index:03d}_{stem(selection.processing_plan)}_"
        f"{stem(selection.predictor)}_{stem(selection.featurizer)}_"
        f"{stem(selection.splitter)}_{stem(selection.sim_filter)}_"
        f"{dataset_token}.gin"
    )


def write_batch_configs(
    selections: Sequence[ExperimentSelection], output_dir: Path, overwrite: bool = False
) -> list[Path]:
    """Write multiple rendered gin configs and return written file paths."""

    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for idx, selection in enumerate(selections, start=1):
        filename = build_experiment_filename(selection, idx)
        target = output_dir / filename
        if target.exists() and not overwrite:
            raise FileExistsError(
                f"{target} already exists. Enable overwrite to replace files."
            )
        target.write_text(render_gin_config(selection), encoding="utf-8")
        written.append(target)

    return written
