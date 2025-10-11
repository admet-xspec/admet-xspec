"""
test ideas:

python -m manage --cfg normalize.gin
=> normalizes all of the datasets placed in a directory in a predictable way, say:
   ./data/admet_transfer/brain/mouse_uptake --> brain_mouse_uptake.csv

python -m manage --cfg pca_brain.gin
=> peforms pca on only the 'brain' data, the gin config file should look something like:
...
dataset_list = None
dataset_categories = ["brain"]
...

contrast this with:
python -m manage --cfg umap_mine.gin
...
dataset_list = ["brain_mouse_uptake.csv", ..., "maoa_rat_property_inhibition.csv"]
dataset_categories = None
...

"""
import pytest
import pandas as pd
import gin
import glob
import tempfile

from src.mgmt_pipeline import ManagementPipeline

def test_normalize(
        normalize_config,
        mock_raw_datasets_dir,
        mock_normalized_datasets_dir
):
    with tempfile.NamedTemporaryFile(mode="w") as fp:
        fp.write(normalize_config)
        gin.parse_config(fp.name)

    pipeline = ManagementPipeline()
    pipeline.run()

    correct_file_names = [
        "brain_mouse_uptake.csv" in out_filenames,
        "maoa_rat_property_inhibition.csv" in out_filenames
    ]

    globbed_datasets = glob.glob(str(mock_normalized_datasets_dir / "**/*LONGNAME.csv"))

    # FINISHED ARRANGING

    # ACT #1
    before_and_after_paths = []
    for dataset in globbed_datasets:
        before_and_after_paths.append(
            (dataset, pipeline.get_dataset_output_basename(dataset))
        )

    # ACT #2
    raw_and_normalized: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for dataset in globbed_datasets:
        raw_df = pd.read_csv(dataset, delimiter=";")
        normalized_df = pipeline.normalize_df(raw_df)

        raw_clean_df = pipeline.get_clean_smiles_df(raw_df)
        normalized_clean_df = pipeline.get_clean_smiles_df(normalized_df)

        raw_and_normalized.append((raw_df, normalized_df))

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
        pipeline._get_smiles_col_in_raw(r_df) not in n_df.columns
        for r_df, n_df in raw_and_normalized
    ]

    # ASSERT
    assert all(correct_column_names)
    assert all(correct_file_names)
    assert all(correct_delimiter)