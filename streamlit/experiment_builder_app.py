from __future__ import annotations

from datetime import datetime
from itertools import product
from pathlib import Path
import sys
from typing import cast

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gin_config.gin_config_builder import (
    ExperimentSelection,
    GinOptions,
    build_experiment_filename,
    discover_gin_options,
    load_dataset_names,
    render_gin_config,
    write_batch_configs,
)

CONFIGS_DIR = PROJECT_ROOT / "configs"
DATASET_REGISTRY = PROJECT_ROOT / "data" / "datasets" / "registry.txt"
DEFAULT_OUTPUT_ROOT_DIR = PROJECT_ROOT / "generated_configs"
MAX_BATCH_WARNING = 250


def _build_selections(
    processing_plans: list[str],
    featurizers: list[str],
    splitters: list[str],
    sim_filters: list[str],
    predictors: list[str],
    datasets: list[str],
    test_origin_dataset: str,
    task_setting: str,
) -> list[ExperimentSelection]:
    selections: list[ExperimentSelection] = []
    for plan, feat, split, sim, pred in product(
        processing_plans, featurizers, splitters, sim_filters, predictors
    ):
        selections.append(
            ExperimentSelection(
                processing_plan=plan,
                featurizer=feat,
                splitter=split,
                sim_filter=sim,
                predictor=pred,
                datasets=datasets,
                test_origin_dataset=test_origin_dataset,
                task_setting=task_setting,
            )
        )
    return selections


def main() -> None:
    st.set_page_config(page_title="ADMET-XSpec Experiment Builder", layout="wide")
    st.title("ADMET-XSpec")
    st.caption("Experiment config builder")

    options = cast(GinOptions, discover_gin_options(CONFIGS_DIR))
    dataset_names = load_dataset_names(DATASET_REGISTRY)

    if "output_dir_name" not in st.session_state:
        st.session_state["output_dir_name"] = datetime.now().strftime(
            "run_%Y%m%d_%H%M%S"
        )

    task_setting = st.selectbox(
        "Task setting",
        options=["binary_classification", "regression"],
        index=0,
        help="Controls predictor family and ProcessingPipeline.task_setting.",
    )

    left_col, right_col = st.columns(2)

    with left_col:
        selected_plans = st.multiselect(
            "Processing plans",
            options=options["processing_plans"],
            key="selected_processing_plans",
        )
        selected_featurizers = st.multiselect(
            "Featurizers",
            options=options["featurizers"],
            key="selected_featurizers",
        )
        selected_splitters = st.multiselect(
            "Splitters",
            options=options["splitters"],
            key="selected_splitters",
        )

    with right_col:
        selected_sim_filters = st.multiselect(
            "Similarity filters",
            options=options["sim_filters"],
            key="selected_sim_filters",
        )
        predictor_options = options["predictors"][task_setting]
        selected_predictors = st.multiselect(
            "Predictors",
            options=predictor_options,
            key="selected_predictors",
        )

        # Streamlit can keep stale values when task setting changes between reruns.
        # Keep only predictors valid for the current task to avoid preview crashes.
        allowed_predictors = set(predictor_options)
        selected_predictors = [
            path for path in selected_predictors if path in allowed_predictors
        ]

    selected_datasets = st.multiselect(
        "Training datasets",
        options=dataset_names,
        key="selected_datasets",
        help="These values become ProcessingPipeline.datasets.",
    )

    if selected_datasets:
        test_origin_dataset = st.selectbox(
            "Test origin dataset",
            options=selected_datasets,
            index=0,
            help="Must be one of the selected training datasets.",
        )
    else:
        st.info("Select at least one training dataset to choose a test origin dataset.")
        test_origin_dataset = ""

    output_root_input = st.text_input(
        "Output root directory",
        value=str(DEFAULT_OUTPUT_ROOT_DIR),
        help="Root directory where the selected output directory name will be created.",
    )

    output_dir_name = st.text_input(
        "Output directory name",
        value=st.session_state["output_dir_name"],
        help="Name of a subdirectory created under the output root directory.",
    )

    invalid_output_name = (
        not output_dir_name.strip() or "/" in output_dir_name or "\\" in output_dir_name
    )
    output_dir = Path(output_root_input) / output_dir_name.strip()
    st.caption(f"Final output directory: `{output_dir}`")
    if invalid_output_name:
        st.error(
            "Output directory name must be non-empty and cannot contain path separators."
        )

    overwrite = st.checkbox("Overwrite existing files", value=False)

    selections = _build_selections(
        processing_plans=selected_plans,
        featurizers=selected_featurizers,
        splitters=selected_splitters,
        sim_filters=selected_sim_filters,
        predictors=selected_predictors,
        datasets=selected_datasets,
        test_origin_dataset=test_origin_dataset,
        task_setting=task_setting,
    )

    st.markdown(f"**Planned files:** {len(selections)}")
    if len(selections) > MAX_BATCH_WARNING:
        st.warning(
            f"This will create {len(selections)} files. "
            "Consider narrowing selections before generating."
        )

    if selections:
        with st.expander("Preview first generated config"):
            try:
                st.code(render_gin_config(selections[0]), language="ini")
            except ValueError as exc:
                st.warning(f"Preview unavailable: {exc}")

        with st.expander("Preview output filenames"):
            preview_count = min(5, len(selections))
            preview_names = [
                build_experiment_filename(selections[idx], idx + 1)
                for idx in range(preview_count)
            ]
            st.code("\n".join(preview_names), language="text")

    missing_inputs: list[str] = []
    if not selected_plans:
        missing_inputs.append("processing plans")
    if not selected_featurizers:
        missing_inputs.append("featurizers")
    if not selected_splitters:
        missing_inputs.append("splitters")
    if not selected_sim_filters:
        missing_inputs.append("similarity filters")
    if not selected_predictors:
        missing_inputs.append("predictors")
    if not selected_datasets:
        missing_inputs.append("training datasets")

    if missing_inputs:
        st.warning("Complete required selections: " + ", ".join(missing_inputs) + ".")

    generate_disabled = not selections or not selected_datasets or invalid_output_name
    if st.button("Generate configs", disabled=generate_disabled, type="primary"):
        try:
            written = write_batch_configs(
                selections=selections,
                output_dir=output_dir,
                overwrite=overwrite,
            )
        except Exception as exc:  # pragma: no cover - UI-level error surface
            st.error(str(exc))
        else:
            st.success(f"Generated {len(written)} config files in {output_dir}")
            st.dataframe(
                {"file": [str(path) for path in written]},
                use_container_width=True,
                hide_index=True,
            )


if __name__ == "__main__":
    main()
