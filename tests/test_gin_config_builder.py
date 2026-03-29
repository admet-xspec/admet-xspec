from pathlib import Path

from src.gin_config_builder import (
    ExperimentSelection,
    discover_gin_options,
    load_dataset_names,
    render_gin_config,
    write_batch_configs,
)


def test_load_dataset_names_deduplicates_preserving_order(tmp_path: Path) -> None:
    registry = tmp_path / "registry.txt"
    registry.write_text("A\nB\nA\n\nC\n", encoding="utf-8")

    assert load_dataset_names(registry) == ["A", "B", "C"]


def test_render_gin_config_contains_expected_blocks() -> None:
    selection = ExperimentSelection(
        processing_plan="configs/processing_plans/train.gin",
        featurizer="configs/featurizers/ecfp.gin",
        splitter="configs/splitters/scaffold.gin",
        sim_filter="configs/sim_filters/tanimoto_5p_to_whole.gin",
        predictor="configs/predictors/classifiers/lgbm.gin",
        datasets=["AChE_human_pIC50", "AChE_rat_pIC50"],
        test_origin_dataset="AChE_human_pIC50",
        task_setting="binary_classification",
    )

    config = render_gin_config(selection)

    assert "include 'configs/processing_plans/train.gin'" in config
    assert "ProcessingPipeline.datasets = [" in config
    assert "'AChE_human_pIC50'" in config
    assert "ProcessingPipeline.test_origin_dataset = 'AChE_human_pIC50'" in config
    assert "ProcessingPipeline.task_setting = 'binary_classification'" in config


def test_write_batch_configs_creates_output_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "generated"
    selection = ExperimentSelection(
        processing_plan="configs/processing_plans/train.gin",
        featurizer="configs/featurizers/ecfp.gin",
        splitter="configs/splitters/random.gin",
        sim_filter="configs/sim_filters/none.gin",
        predictor="configs/predictors/regressors/lgbm.gin",
        datasets=["logD"],
        test_origin_dataset="logD",
        task_setting="regression",
    )

    written = write_batch_configs([selection], output_dir=output_dir, overwrite=False)

    assert len(written) == 1
    assert written[0].exists()


def test_discover_gin_options_excludes_internal_and_optimization(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    (configs / "processing_plans").mkdir(parents=True)
    (configs / "processing_plans" / "train.gin").write_text("", encoding="utf-8")
    (configs / "processing_plans" / "_possible_plans.gin").write_text(
        "", encoding="utf-8"
    )

    (configs / "featurizers").mkdir(parents=True)
    (configs / "featurizers" / "ecfp.gin").write_text("", encoding="utf-8")

    (configs / "splitters").mkdir(parents=True)
    (configs / "splitters" / "scaffold.gin").write_text("", encoding="utf-8")

    (configs / "sim_filters").mkdir(parents=True)
    (configs / "sim_filters" / "none.gin").write_text("", encoding="utf-8")

    (configs / "predictors" / "classifiers" / "optimization").mkdir(parents=True)
    (configs / "predictors" / "classifiers" / "lgbm.gin").write_text(
        "", encoding="utf-8"
    )
    (configs / "predictors" / "classifiers" / "optimization" / "lgbm_h.gin").write_text(
        "", encoding="utf-8"
    )

    (configs / "predictors" / "regressors").mkdir(parents=True)
    (configs / "predictors" / "regressors" / "rf.gin").write_text(
        "", encoding="utf-8"
    )

    options = discover_gin_options(configs)

    assert "configs/processing_plans/train.gin" in options["processing_plans"]
    assert "configs/processing_plans/_possible_plans.gin" not in options["processing_plans"]
    assert "configs/predictors/classifiers/lgbm.gin" in options["predictors"]["binary_classification"]
    assert all(
        "/optimization/" not in p
        for p in options["predictors"]["binary_classification"]
    )

