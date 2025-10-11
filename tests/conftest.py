from pathlib import Path
import pytest

@pytest.fixture()
def mock_raw_datasets_dir():
    return Path("tests/mocks/data/admet_transfer")

@pytest.fixture()
def mock_normalized_datasets_dir():
    return Path("tests/mocks/data/preprocesssing")

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