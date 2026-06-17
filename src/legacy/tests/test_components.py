import pytest
import pandas as pd
import gin
import glob
import tempfile
from pathlib import Path

def test_pca(
    mock_normalized_input_dir,
    mgmt_pipeline_pca_mode
):
    pipeline = mgmt_pipeline_pca_mode
    ecfp_featurized_df_dict = {
        ds_basename: pipeline.get_featurized_dataset_df(f"tests/mocks/preprocessing/{ds_basename}")
        for ds_basename in pipeline.explore_datasets_list
    }

    pca_reduced_df_dict = {
        ds_basename: pipeline.get_reduced_dataset(ecfp_featurized_df_dict)
        for ds_basename, ecfp_df in ecfp_featurized_df_dict.items()
    }

    # ASSERT
    assert all(correct_pca_reduction)