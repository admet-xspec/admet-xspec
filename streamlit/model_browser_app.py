"""
Model Browser and Analysis App
Streamlit application for browsing, filtering, visualizing, and comparing trained models.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import io
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from ruamel.yaml import YAML
from streamlit_sortables import sort_items

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

MODELS_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "models"
YAML_HANDLER = YAML()
YAML_HANDLER.preserve_quotes = True
YAML_HANDLER.default_flow_style = False


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================


def load_model_metadata(model_run_path: Path) -> dict[str, Any]:
    """Load model metadata from a model run directory.

    Args:
        model_run_path: Path to a specific model run (contains model_metadata.yaml)

    Returns:
        Dictionary with metadata, or empty dict if file not found
    """
    metadata_file = model_run_path / "model_metadata.yaml"
    if not metadata_file.exists():
        return {}

    try:
        with open(metadata_file, "r") as f:
            return dict(YAML_HANDLER.load(f)) or {}
    except Exception as e:
        st.warning(f"Error loading metadata from {model_run_path}: {e}")
        return {}


def load_metrics(model_run_path: Path) -> dict[str, Any]:
    """Load evaluation metrics from a model run directory.

    Args:
        model_run_path: Path to a specific model run (contains metrics.yaml)

    Returns:
        Dictionary with metrics and confidence intervals
    """
    metrics_file = model_run_path / "metrics.yaml"
    if not metrics_file.exists():
        return {}

    try:
        with open(metrics_file, "r") as f:
            return dict(YAML_HANDLER.load(f)) or {}
    except Exception as e:
        st.warning(f"Error loading metrics from {model_run_path}: {e}")
        return {}


def scan_all_models() -> list[dict[str, Any]]:
    """Scan all models in the cache directory and load their metadata/metrics.

    Returns:
        List of model data dictionaries with structure:
        {
            'model_name': str,
            'run_name': str,
            'model_path': Path,
            'metadata': dict,
            'metrics': dict,
        }
    """
    if not MODELS_CACHE_DIR.exists():
        return []

    models = []
    for model_dir in sorted(MODELS_CACHE_DIR.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "__pycache__":
            continue

        # Scan for model runs (subdirectories)
        for run_dir in sorted(model_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            metadata = load_model_metadata(run_dir)
            if not metadata:  # Skip if no metadata found
                continue

            metrics = load_metrics(run_dir)

            models.append(
                {
                    "model_name": model_dir.name,
                    "run_name": run_dir.name,
                    "model_path": run_dir,
                    "metadata": metadata,
                    "metrics": metrics,
                }
            )

    return models


def get_unique_filter_values(models: list[dict[str, Any]], field: str) -> list[str]:
    """Extract unique values for a metadata field across all models.

    Args:
        models: List of model dictionaries
        field: Metadata field name (e.g., 'Featurizer', 'Predictor')

    Returns:
        Sorted list of unique values, excluding None/empty values
    """
    values = set()
    for model in models:
        value = model["metadata"].get(field)

        # Handle list fields (like Datasets)
        if isinstance(value, list):
            values.update(v for v in value if v)
        # Handle single values
        elif value is not None and str(value).lower() != "none":
            values.add(str(value))

    return sorted(list(values))


def apply_filters(
    models: list[dict[str, Any]], filters: dict[str, Any]
) -> list[dict[str, Any]]:
    """Apply metadata filters to model list.

    Args:
        models: List of model dictionaries
        filters: Dictionary with selected filter values

    Returns:
        Filtered list of models
    """
    filtered = models

    for field, selected_values in filters.items():
        if not selected_values:  # Skip empty filters
            continue

        filtered_models = []
        for model in filtered:
            value = model["metadata"].get(field)

            # Handle list fields
            if isinstance(value, list):
                if any(v in selected_values for v in value):
                    filtered_models.append(model)
            # Handle single values
            elif str(value) in selected_values or str(value).lower() == "none":
                filtered_models.append(model)

        filtered = filtered_models

    return filtered


# ============================================================================
# STREAMLIT UI - MAIN PAGE
# ============================================================================


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "all_models" not in st.session_state:
        st.session_state.all_models = None
    if "selected_models_comparison" not in st.session_state:
        st.session_state.selected_models_comparison = []


def get_models():
    """Scan and return all models (cached in session state)."""
    if "cached_models" not in st.session_state:
        st.session_state.cached_models = scan_all_models()
    return st.session_state.cached_models


def render_model_browser():
    """Render the main model browser tab."""
    st.header("📊 Model Browser")

    # Load all models
    all_models = get_models()
    if not all_models:
        st.error("No models found in cache directory!")
        return

    unique_models = len(set(m["model_name"] for m in all_models))
    st.info(f"Found **{len(all_models)}** model runs across **{unique_models}** models")

    # Build filter sidebar
    st.sidebar.header("🔍 Filters")

    filters = {}
    filter_fields = [
        "Datasets",
        "Featurizer",
        "Predictor",
        "Splitter",
        "Task Setting",
        "Similarity Filter",
        "Optimized Hyperparameters",
    ]

    for field in filter_fields:
        unique_values = get_unique_filter_values(all_models, field)
        if unique_values:
            filters[field] = st.sidebar.multiselect(
                field, options=unique_values, key=f"filter_{field}"
            )

    # Add multiend filter
    has_multiend = any(m["metadata"].get("is_multiend") for m in all_models)
    if has_multiend:
        is_multiend = st.sidebar.checkbox(
            "Multiend only", value=False, key="filter_multiend"
        )
        if is_multiend:
            filters["is_multiend"] = [True]

    # Apply filters
    filtered_models = apply_filters(all_models, filters)
    remaining = len(filtered_models)
    st.sidebar.success(f"**{remaining}** models match filters")

    # Display filtered models
    st.subheader(f"Filtered Results ({remaining} models)")

    # Create dataframe for display
    display_data = []
    for model in filtered_models:
        meta = model["metadata"]
        metrics = model["metrics"].get("metrics", {})

        display_data.append(
            {
                "Model Name": model["model_name"],
                "Run Name": model["run_name"],
                "Datasets": (
                    ", ".join(meta.get("Datasets", []))
                    if isinstance(meta.get("Datasets"), list)
                    else str(meta.get("Datasets", ""))
                ),
                "Featurizer": meta.get("Featurizer", "N/A"),
                "Predictor": meta.get("Predictor", "N/A"),
                "Splitter": meta.get("Splitter", "N/A"),
                "Task Setting": meta.get("Task Setting", "N/A"),
                "Test Origin": meta.get("Test Origin Dataset", "N/A"),
                "Is Multiend": meta.get("is_multiend", False),
                "Training Size": meta.get("Training set size", "N/A"),
            }
        )

    df_display = pd.DataFrame(display_data)

    # Column selection for table display
    default_cols = ["Model Name", "Run Name", "Featurizer", "Predictor", "Task Setting"]
    selected_cols = st.multiselect(
        "Columns to display",
        options=df_display.columns.tolist(),
        default=default_cols,
        key="display_columns",
    )

    st.dataframe(
        df_display[selected_cols],
        width="stretch",
        hide_index=True,
    )

    # Model selection for plotting
    st.subheader("📈 Select Models for Plotting")

    selected_plot_indices = st.multiselect(
        "Select models to plot",
        options=range(len(filtered_models)),
        format_func=lambda i: f"{filtered_models[i]['model_name']} / {filtered_models[i]['run_name']}",
        key="browser_selected_for_plot",
    )

    if selected_plot_indices:
        render_plot_config_and_export(
            [filtered_models[i] for i in selected_plot_indices]
        )

    # Detailed view of single model
    st.subheader("🔍 View Model Details")
    selected_run_idx = st.selectbox(
        "Select a model to view details",
        options=range(len(filtered_models)),
        format_func=lambda i: f"{filtered_models[i]['model_name']} / {filtered_models[i]['run_name']}",
        key="selected_model_idx",
    )

    if selected_run_idx is not None and selected_run_idx < len(filtered_models):
        render_model_details(filtered_models[selected_run_idx])


def render_model_details(model: dict[str, Any]):
    """Render detailed view of a single model."""
    st.markdown("---")
    col1, col2 = st.columns(2)

    meta = model["metadata"]
    metrics = model["metrics"].get("metrics", {})
    ci = model["metrics"].get("percentile_95_ci", {})

    # Left column - Metadata
    with col1:
        st.subheader("Model Metadata")

        metadata_to_display = {
            "Model Name": model["model_name"],
            "Run Name": model["run_name"],
            "Datasets": ", ".join(meta.get("Datasets", [])),
            "Featurizer": meta.get("Featurizer", "N/A"),
            "Predictor": meta.get("Predictor", "N/A"),
            "Splitter": meta.get("Splitter", "N/A"),
            "Similarity Filter": meta.get("Similarity Filter", "None"),
            "Task Setting": meta.get("Task Setting", "N/A"),
            "Test Origin Dataset": meta.get("Test Origin Dataset", "N/A"),
            "Optimized HP": meta.get("Optimized Hyperparameters", False),
            "Multiend": meta.get("is_multiend", False),
            "Training Set Size": meta.get("Training set size", "N/A"),
        }

        for key, value in metadata_to_display.items():
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            st.metric(key, str(value))

    # Right column - Key Metrics
    with col2:
        st.subheader("Evaluation Metrics")

        if metrics:
            # Display main metrics with confidence intervals
            for metric_name in sorted(metrics.keys()):
                metric_value = metrics[metric_name]

                # Get CI bounds if available
                ci_bounds = ci.get(metric_name, [None, None]) if ci else [None, None]

                if ci_bounds[0] is not None and ci_bounds[1] is not None:
                    ci_text = f"CI95: [{ci_bounds[0]:.4f}, {ci_bounds[1]:.4f}]"
                    st.metric(
                        metric_name.replace("_", " ").title(),
                        f"{metric_value:.4f}",
                        help=ci_text,
                    )
                else:
                    st.metric(
                        metric_name.replace("_", " ").title(), f"{metric_value:.4f}"
                    )
        else:
            st.info("No metrics available for this model")

    # Training set sources breakdown
    if "Training set sources" in meta:
        st.subheader("Training Set Sources")
        sources = meta["Training set sources"]
        if isinstance(sources, dict):
            source_df = pd.DataFrame(
                [{"Dataset": k, "Count": v} for k, v in sources.items()]
            )
            st.bar_chart(source_df.set_index("Dataset"))


def render_advanced_analysis():
    """Render advanced analysis tab with custom queries."""
    st.header("🔬 Advanced Analysis")

    all_models = get_models()
    if not all_models:
        st.error("No models found!")
        return

    # Create analysis dataframe
    analysis_data = []
    for model in all_models:
        meta = model["metadata"]
        metrics = model["metrics"].get("metrics", {})

        row = {
            "Model": model["model_name"],
            "Run": model["run_name"],
            "Featurizer": meta.get("Featurizer", "N/A"),
            "Predictor": meta.get("Predictor", "N/A"),
            "Splitter": meta.get("Splitter", "N/A"),
            "Similarity Filter": meta.get("Similarity Filter", "None"),
            "Task": meta.get("Task Setting", "N/A"),
            "Test Origin": meta.get("Test Origin Dataset", "N/A"),
            "Training Size": meta.get("Training set size", 0),
            "Multiend": meta.get("is_multiend", False),
        }

        # Add metrics
        if metrics:
            row.update({f"metric_{k}": v for k, v in metrics.items()})

        analysis_data.append(row)

    analysis_df = pd.DataFrame(analysis_data)

    # Grouping analysis
    st.subheader("📊 Metrics Grouped by Metadata")

    group_by = st.selectbox(
        "Group results by",
        options=["Featurizer", "Predictor", "Splitter", "Task"],
        key="group_by_select",
    )

    metric_cols = [col for col in analysis_df.columns if col.startswith("metric_")]

    if metric_cols:
        selected_metric = st.selectbox(
            "Select metric to analyze",
            options=[col.replace("metric_", "") for col in metric_cols],
            key="analysis_metric_select",
        )

        metric_col = f"metric_{selected_metric}"

        # Group and aggregate
        grouped = analysis_df.groupby(group_by)[metric_col].agg(
            ["mean", "std", "min", "max", "count"]
        )
        grouped = grouped.round(4)

        st.dataframe(grouped, width="stretch")

        # Visualization
        fig, ax = plt.subplots(figsize=(12, 6))

        sns.boxplot(data=analysis_df, x=group_by, y=metric_col, ax=ax)
        ax.set_ylabel(selected_metric.replace("_", " ").title())
        ax.set_xlabel(group_by)
        ax.set_title(
            f"{selected_metric.replace('_', ' ').title()} Distribution by {group_by}"
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    # Export full analysis
    st.subheader("💾 Export Data")
    if st.button("🔽 Download Full Analysis"):
        csv = analysis_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="full_model_analysis.csv",
            mime="text/csv",
        )


def _auto_plot_title(models: list[dict[str, Any]]) -> str:
    """Build an auto title from shared metadata across models."""

    def common(field: str) -> str:
        vals = {str(m["metadata"].get(field, "N/A")) for m in models}
        return next(iter(vals)) if len(vals) == 1 else "Mixed"

    task_raw = common("Task Setting")
    if "classif" in task_raw.lower():
        task = "Classification"
    elif "regress" in task_raw.lower():
        task = "Regression"
    else:
        task = task_raw

    return f"{task} | {common('Featurizer')} | {common('Predictor')} | {common('Splitter')}"


def _unique_display_labels(models: list[dict[str, Any]]) -> list[str]:
    """Return display labels matching plot labels, deduplicated with a counter suffix."""
    raw = [_model_plot_label(m).replace("\n", " | ") for m in models]
    counts: dict[str, int] = {}
    for lbl in raw:
        counts[lbl] = counts.get(lbl, 0) + 1
    seen: dict[str, int] = {}
    result = []
    for lbl in raw:
        if counts[lbl] > 1:
            seen[lbl] = seen.get(lbl, 0) + 1
            result.append(f"{lbl} ({seen[lbl]})")
        else:
            result.append(lbl)
    return result


_COLOR_PALETTES = [
    "husl",
    "tab10",
    "Set1",
    "Set2",
    "Set3",
    "Dark2",
    "Paired",
    "Accent",
    "colorblind",
    "muted",
]


def _model_plot_label(model: dict[str, Any]) -> str:
    meta = model["metadata"]
    datasets = meta.get("Datasets", [])
    if isinstance(datasets, list):
        datasets_str = ", ".join(datasets[:2])
        if len(datasets) > 2:
            datasets_str += f" (+{len(datasets)-2})"
    else:
        datasets_str = str(datasets)
    multiend_str = "Multiend" if meta.get("is_multiend", False) else "Single"
    return f"{datasets_str}\n{multiend_str}"


def render_plot_config_and_export(selected_models: list[dict[str, Any]]):
    """Render inline plot configuration and export for the given models."""
    with st.expander("📈 Plot & Export", expanded=True):
        # --- Auto title (resets when models change) ---
        auto_title = _auto_plot_title(selected_models)
        if st.session_state.get("_plot_last_auto_title") != auto_title:
            st.session_state["plot_title"] = auto_title
            st.session_state["_plot_last_auto_title"] = auto_title
        plot_title = st.text_input("Plot title", key="plot_title")

        # --- Size / resolution ---
        col1, col2, col3 = st.columns(3)
        with col1:
            plot_width = st.slider(
                "Width (inches)", min_value=6, max_value=20, value=14, key="plot_width"
            )
            plot_height = st.slider(
                "Height (inches)", min_value=4, max_value=16, value=8, key="plot_height"
            )
        with col2:
            dpi = st.slider(
                "DPI", min_value=72, max_value=300, value=300, step=10, key="plot_dpi"
            )
            palette = st.selectbox(
                "Color palette", options=_COLOR_PALETTES, index=0, key="plot_palette"
            )
        with col3:
            auto_yrange = st.checkbox(
                "Auto y-axis range", value=True, key="plot_auto_yrange"
            )
            if auto_yrange:
                y_min: Optional[float] = None
                y_max: Optional[float] = None
            else:
                y_min = st.number_input(
                    "Y min", value=0.0, step=0.05, format="%.3f", key="plot_ymin"
                )
                y_max = st.number_input(
                    "Y max", value=1.0, step=0.05, format="%.3f", key="plot_ymax"
                )

        # --- Metrics ---
        all_metrics: set[str] = set()
        for model in selected_models:
            all_metrics.update(model["metrics"].get("metrics", {}).keys())

        if not all_metrics:
            st.warning("No metrics found in selected models")
            return

        selected_metrics = st.multiselect(
            "Select metrics to plot",
            options=sorted(all_metrics),
            default=sorted(all_metrics)[:5],
            key="plot_selected_metrics",
        )

        if not selected_metrics:
            st.warning("Select at least one metric to plot")
            return

        # --- Model order (drag-and-drop) ---
        st.write("**Model display order** — drag to reorder bars")
        display_labels = _unique_display_labels(selected_models)
        label_to_model = dict(zip(display_labels, selected_models))
        sorted_labels = sort_items(display_labels, key="plot_model_order")
        ordered_models = [label_to_model[lbl] for lbl in sorted_labels]

        # --- Summary ---
        col1, col2 = st.columns(2)
        with col1:
            st.write(
                f"**Models:** {len(ordered_models)}  |  **Metrics:** {len(selected_metrics)}"
            )
        with col2:
            st.write(
                f'**Size:** {plot_width}" × {plot_height}", {dpi} DPI  |  **Palette:** {palette}'
            )

        if not st.button(
            "🎨 Generate Plot", type="primary", width="stretch", key="generate_plot"
        ):
            return

        with st.spinner("🎨 Generating plot..."):
            try:
                plot_data = []
                for model in ordered_models:
                    metrics = model["metrics"].get("metrics", {})
                    ci = model["metrics"].get("percentile_95_ci", {})
                    model_label = _model_plot_label(model)

                    for metric_name in selected_metrics:
                        if metric_name not in metrics:
                            continue
                        value = metrics[metric_name]
                        ci_bounds = (ci or {}).get(metric_name, [None, None])
                        plot_data.append(
                            {
                                "Metric": metric_name.replace("_", " ").title(),
                                "Model": model_label,
                                "Value": value,
                                "CI_Lower": (
                                    ci_bounds[0] if ci_bounds[0] is not None else value
                                ),
                                "CI_Upper": (
                                    ci_bounds[1] if ci_bounds[1] is not None else value
                                ),
                            }
                        )

                if not plot_data:
                    st.error("No valid data found for selected models and metrics")
                    return

                plot_df = pd.DataFrame(plot_data)

                FONT_LABEL = 20
                FONT_TITLE = 24
                FONT_TICK = 17
                FONT_LEGEND = 16

                fig, ax = plt.subplots(figsize=(plot_width, plot_height), dpi=dpi)

                metrics_list = sorted(plot_df["Metric"].unique())
                # Preserve user-defined order (ordered_models determines sequence)
                seen: set[str] = set()
                models_list = []
                for m in ordered_models:
                    lbl = _model_plot_label(m)
                    if lbl not in seen:
                        seen.add(lbl)
                        models_list.append(lbl)

                bar_width = 0.8 / len(models_list)
                colors = sns.color_palette(palette, len(models_list))

                for model_idx, model_label in enumerate(models_list):
                    x_positions, y_values, y_errors_lower, y_errors_upper = (
                        [],
                        [],
                        [],
                        [],
                    )
                    for metric_idx, metric in enumerate(metrics_list):
                        mask = (plot_df["Metric"] == metric) & (
                            plot_df["Model"] == model_label
                        )
                        if mask.any():
                            row = plot_df[mask].iloc[0]
                            x_pos = (
                                metric_idx
                                + (model_idx - len(models_list) / 2 + 0.5) * bar_width
                            )
                            x_positions.append(x_pos)
                            y_values.append(row["Value"])
                            y_errors_lower.append(row["Value"] - row["CI_Lower"])
                            y_errors_upper.append(row["CI_Upper"] - row["Value"])

                    ax.bar(
                        x_positions,
                        y_values,
                        width=bar_width,
                        label=model_label,
                        color=colors[model_idx],
                        alpha=0.85,
                        edgecolor="white",
                        linewidth=0.5,
                    )
                    ax.errorbar(
                        x_positions,
                        y_values,
                        yerr=[y_errors_lower, y_errors_upper],
                        fmt="none",
                        ecolor="black",
                        capsize=6,
                        linewidth=2.5,
                        alpha=0.7,
                        label="_nolegend_",
                    )

                ax.set_xlabel(
                    "Metric", fontsize=FONT_LABEL, fontweight="bold", labelpad=8
                )
                ax.set_ylabel(
                    "Value (with 95% CI)",
                    fontsize=FONT_LABEL,
                    fontweight="bold",
                    labelpad=8,
                )
                ax.set_title(plot_title, fontsize=FONT_TITLE, fontweight="bold", pad=12)
                ax.set_xticks(range(len(metrics_list)))
                ax.set_xticklabels(metrics_list, fontsize=FONT_TICK)
                ax.tick_params(axis="y", labelsize=FONT_TICK)
                legend = ax.legend(
                    title="Dataset | Type",
                    bbox_to_anchor=(1.02, 1),
                    loc="upper left",
                    fontsize=FONT_LEGEND,
                    borderpad=0.6,
                    labelspacing=0.4,
                )
                legend.get_title().set_fontsize(FONT_LEGEND)
                legend.get_title().set_fontweight("bold")
                ax.grid(axis="y", alpha=0.3, linewidth=1.2)
                ax.set_axisbelow(True)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                if y_min is not None and y_max is not None:
                    ax.set_ylim(y_min, y_max)

                plt.tight_layout(pad=0.8)

                st.pyplot(fig)

                output_dir = PROJECT_ROOT / "data" / "cache" / "plots"
                output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"model_metrics_plot_{timestamp}.png"
                filepath = output_dir / filename
                fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
                st.success(f"✅ Plot saved to: `{filepath}`")
                st.info(f"File size: {filepath.stat().st_size / 1024:.1f} KB")

                with open(filepath, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Plot",
                        data=f.read(),
                        file_name=filename,
                        mime="image/png",
                    )

                plt.close(fig)

            except Exception as e:
                st.error(f"Error generating plot: {e}")
                import traceback

                traceback.print_exc()


def main() -> None:
    st.set_page_config(
        page_title="ADMET Model Browser",
        page_icon="📊",
        layout="wide",
    )
    st.title("📊 ADMET Model Browser & Analysis")
    st.caption(
        "Browse, filter, compare, and visualize trained ADMET prediction models."
    )

    initialize_session_state()

    tab_browser, tab_advanced = st.tabs(
        [
            "🔍 Model Browser",
            "🔬 Advanced Analysis",
        ]
    )

    with tab_browser:
        render_model_browser()
    with tab_advanced:
        render_advanced_analysis()


if __name__ == "__main__":
    main()
