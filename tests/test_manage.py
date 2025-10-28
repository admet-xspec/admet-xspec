import pytest
import pandas as pd
import gin
import glob
import tempfile
from pathlib import Path

def test_normalize(
    mock_raw_input_dir,
    mock_normalized_input_dir,
    expected_normalized_paths,
    mgmt_pipeline_normalize_mode
):
    pipeline = mgmt_pipeline_normalize_mode
    pipeline.run()

    globbed_datasets = glob.glob(
        str(mock_normalized_input_dir / "**/*LONGNAME.csv"),
        recursive=True
    )

    # FINISHED ARRANGING

    # ACT #1
    before_and_after_paths = []
    for dataset in globbed_datasets:
        before_and_after_paths.append(
            (dataset, pipeline.get_normalized_filename(Path(dataset)))
        )

    # ACT #2
    raw_and_normalized: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for dataset in globbed_datasets:
        raw_df = pd.read_csv(dataset, delimiter=";")
        normalized_df = pipeline.get_normalized_df(
            Path(dataset), delimiter=";"
        )

        smiles_colname = pipeline.get_smiles_col_in_raw(raw_df)

        raw_clean_df = pipeline.get_clean_smiles_df(
            raw_df, smiles_colname
        )
        normalized_clean_df = pipeline.get_clean_smiles_df(
            normalized_df, "smiles"
        )

        raw_and_normalized.append((raw_clean_df, normalized_clean_df))

    # FOR ASSERT #1
    correct_file_names = [
        a_filename == expected_normalized_paths[b_filename]
        for b_filename, a_filename in before_and_after_paths
    ]

    # FOR ASSERT #2
    correct_delimiter = [
        r_df.equals(n_df) for r_df, n_df in raw_and_normalized
    ]

    # FOR ASSERT #3
    correct_column_names = [
        "smiles" in n_df.columns and
        pipeline.get_smiles_col_in_raw(r_df) not in n_df.columns
        for r_df, n_df in raw_and_normalized
    ]

    # ASSERT
    assert all(correct_column_names)
    assert all(correct_file_names)
    assert all(correct_delimiter)