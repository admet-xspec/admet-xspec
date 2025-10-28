import gin
import pytest
import os
from pathlib import Path

from src.mgmt_pipeline import ManagementPipeline

@pytest.fixture()
def config_dir():
    return Path("configs")

@pytest.fixture()
def mock_raw_input_dir():
    return Path("tests/mocks/data/admet_transfer")

@pytest.fixture()
def mock_normalized_input_dir():
    return Path("tests/mocks/data/preprocessing")

@pytest.fixture()
def mock_output_dir():
    return Path("tests/mocks/data/mgmt_output")

@pytest.fixture()
def expected_normalized_paths(mock_raw_input_dir):
    br_mouse_upt_raw = f"{mock_raw_input_dir}/brain/mouse_uptake/DOWNLOAD-MOUSE-LONGNAME.csv"
    maoa_rat_prop_raw = f"{mock_raw_input_dir}/MAO-A/rat/property_inhibition/DOWNLOAD-RAT-LONGNAME.csv"

    return {
        br_mouse_upt_raw: "brain_mouse_uptake",
        maoa_rat_prop_raw: "maoa_rat_property_inhibition"
    }

@pytest.fixture()
def normalize_config(
    mock_raw_input_dir,
    mock_normalized_input_dir,
    mock_output_dir,
):
    normalize_config = (
        "include 'configs/_global.gin'",
        "ManagementPipeline.root_categories = %root_categories",
        "ManagementPipeline.mode = 'normalize'",
        "ManagementPipeline.force_normalize_all = False",
        f"ManagementPipeline.raw_input_dir = '{str(mock_raw_input_dir)}'",
        f"ManagementPipeline.normalized_input_dir = '{str(mock_normalized_input_dir)}'",
        f"ManagementPipeline.output_dir = '{str(mock_output_dir)}'",
    )

    return "\n".join(normalize_config)

@pytest.fixture()
def pca_config(
    mock_raw_input_dir,
    mock_normalized_input_dir,
    mock_output_dir,
):
    pca_config = (
        "include 'configs/_global.gin'",
        "ManagementPipeline.reducer = @reducer/gin.singleton()",
        "ManagementPipeline.featurizer = @featurizer/gin.singleton()",
        "ManagementPipeline.mode = 'visualize'",
        "ManagementPipeline.explore_datasets_list = [",
        "    'brain_mouse_uptake'",
        "    'maoa_rat_property_inhibition'",
        "]",
        f"ManagementPipeline.normalized_input_dir = '{str(mock_normalized_input_dir)}'",
        f"ManagementPipeline.output_dir = '{str(mock_output_dir)}'",
        "include 'configs/reducers/pca2.gin'",
    )

    return "\n".join(pca_config)

@pytest.fixture()
def mgmt_pipeline_normalize_mode(
    normalize_config,
    config_dir
):
    temp_config_path = config_dir / "_mock_normalize_config.gin"
    with open(temp_config_path, "w") as fp:
        fp.write(normalize_config)

    gin.parse_config_file(str(temp_config_path))
    pipeline = ManagementPipeline()

    yield pipeline

    os.remove(temp_config_path)

@pytest.fixture()
def mgmt_pipeline_pca_mode(
    normalize_config,
    config_dir
):
    temp_config_path = config_dir / "_mock_pca_config.gin"
    with open(temp_config_path, "w") as fp:
        fp.write(normalize_config)

    gin.parse_config_file(str(temp_config_path))
    pipeline = ManagementPipeline()

    yield pipeline

    os.remove(temp_config_path)