from pathlib import Path
import pytest

@pytest.fixture()
def mock_raw_datasets_dir():
    return Path("tests/mocks/data/admet_transfer")

@pytest.fixture()
def mock_normalized_datasets_dir():
    return Path("tests/mocks/data/preprocesssing")

@pytest.fixture()
def expected_normalized_paths(mock_raw_datasets_dir):
    br_mouse_upt_raw = f"{mock_raw_datasets_dir}/brain/mouse_uptake/DOWNLOAD-MOUSE-LONGNAME.csv"
    maoa_rat_prop_raw = f"{mock_raw_datasets_dir}/MAO-A/rat/property_inhibition/DOWNLOAD-RAT-LONGNAME.csv"

    return {
        br_mouse_upt_raw: "brain_mouse_uptake",
        maoa_rat_prop_raw: "maoa_rat_property_inhibition"
    }

@pytest.fixture()
def normalize_config(
    mock_raw_datasets_dir,
    mock_normalized_datasets_dir,
):
    normalize_config_str = (
        "ManagementPipeline.mode = 'normalize'"
        "ManagementPipeline.force_normalize_all = False"
        f"ManagementPipeline.raw_dataset_dir = '{str(mock_raw_datasets_dir)}'"
        f"ManagementPipeline.normalized_dataset_dir = '{str(mock_normalized_datasets_dir)}'"
    )

    return normalize_config_str