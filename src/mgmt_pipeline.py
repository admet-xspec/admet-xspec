import glob
import pandas as pd
from PIL import Image
from src.utils import (
    clean_smiles,
    get_canonical_smiles,
    get_nice_class_name
)
import logging
from src.predictor.predictor_base import PredictorBase
from src.data.featurizer import (
    FeaturizerBase,
)
from src.data.explorer import ExplorerBase
from src.data.split import DataSplitterBase
from src.training_pipeline import TrainingPipeline
import json
import gin
import numpy as np
from pathlib import Path


@gin.configurable()
class ManagementPipeline:
    """ 'meta-Pipeline' (sounds cool, eh?) for handling temporary/one-off work """

    def __init__(
        self,
        dataset_dir: Path | str,
        model_name: str,
        out_dir: Path | str,
        explorer: ExplorerBase = None,
        splitter: DataSplitterBase = None,
        predictor: PredictorBase = None,
        featurizer: FeaturizerBase = None,
        stratify: bool = True,
    ):

        self.dataset_dir = Path(dataset_dir)
        self.explorer = explorer
        self.splitter = splitter
        self.splitter = splitter
        self.predictor = predictor
        self.featurizer = featurizer
        self.model_name = model_name
        self.out_dir = Path(out_dir)
        self.stratify = stratify

        self.train_path = None
        self.test_path = None

    @staticmethod
    def _get_smiles_col_in_raw(raw_df) -> str:
        possible_variants = ["smiles", "SMILES", "Smiles", "molecule"]
        for smiles_col_variant in possible_variants:
            if smiles_col_variant in raw_df.columns:
                return smiles_col_variant

        # Failed
        raise ValueError(
            "Failed to find one of SMILES column name variants:",
            str(possible_variants),
            "in dataframe:",
            str(raw_df),
        )

    def run(self):
        if self.mode == "normalize":
            assert self.force_normalize_all is not None, (
                "Ran ManagementPipeline in 'normalize' mode without specifying .force_normalize_all attribute"
            )
            self.normalize_datasets()
        if self.mode == "explorer":
            assert self.explorer is not None, (
                "Ran ManagementPipeline in 'explorer' mode without specifying .explorer: ExplorerBase attribute"
            )
            self.dump_exploratory_visualization()

    def normalize_datasets(self, force_normalize_all: bool = False):
        datasets = glob.glob(f"{str(self.dataset_dir)}/**/*.csv", recursive=True)

        # pairs of "before, after" paths
        datasets_paths: list[tuple[Path, Path]] = [
            (
                Path(ds_glob),
                self.get_df_output_path(
                    self.get_normalized_basename(Path(ds_glob))
                )
            )
            for ds_glob in datasets
        ]

        # filter already normalized
        if not force_normalize_all:
            datasets_paths = [
                (ds_glob, ds_path) 
                for ds_glob, ds_path in datasets_paths 
                if not ds_path.exists()
            ]
        
        for ds_glob, ds_path in datasets_paths:
            normalized_dataset_df = self.get_normalized_df(
                ds_glob
            )

            self.save_dataframe(
                normalized_dataset_df, 
                ds_path
            )

    def get_normalized_df(self, ds_globbed_path: Path, delimiter: str = ";") -> pd.DataFrame:
        """Get ready-to-save df without NaNs and with canonical SMILES"""
        df_to_normalize = pd.read_csv(ds_globbed_path, delimiter=delimiter)
        df_to_normalize.rename(
            columns={
                self._get_smiles_col_in_raw(df_to_normalize): "smiles"
            }, inplace=True
        )

        df_to_normalize = self.get_clean_smiles_df(
            df_to_normalize,
            smiles_col="smiles"
        )

        df_to_normalize = self.get_canon_smiles_df(
            df_to_normalize,
            smiles_col="smiles"
        )

        return df_to_normalize

    def get_clean_smiles_df(self, df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
        """Consolidate pandas, our-internal NaN-dropping into one function"""
        pre_dropna_len = len(df)
        df = df.dropna(subset=smiles_col)

        pre_cleaning_len = len(df)
        df[smiles_col] = clean_smiles(df[smiles_col].to_list())

        df = df.dropna(subset=[smiles_col]).reset_index(drop=True)
        
        if pre_dropna_len != pre_cleaning_len:
            logging.info(f"Dropped {pre_dropna_len - pre_cleaning_len} 'nan' SMILES after pd.read_csv")
        if pre_cleaning_len != len(df):
            logging.info(f"Dropped {pre_cleaning_len - len(df)} invalid SMILES")
        logging.info(f"Dataset size: {len(df)}")

        return df

    def get_canon_smiles_df(self, df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
        pre_canonicalization_len = len(df)
        df[smiles_col].apply(
            lambda smiles: get_canonical_smiles(smiles)
        )

        df = df.dropna(subset=[smiles_col]).reset_index(drop=True)

        if pre_canonicalization_len != len(df):
            logging.info(
                f"Canonicalization resulted in {pre_canonicalization_len - len(df)} "
                "'None' SMILES, all were dropped."
            )

        return df

    def get_normalized_basename(self, ds_globbed_path: Path) -> str:
        root_domains = ["brain", "liver", "MAO-A"]

        dataset_dir_parts = ds_globbed_path.parent.parts

        begin_index = None
        for root_domain in root_domains:
            if root_domain in dataset_dir_parts:
                begin_index = dataset_dir_parts.index(root_domain)
                break

        if begin_index is None:
            raise ValueError(
                f"Unable to match any part of Path: {str(ds_globbed_path)} "
                f"to one of root domains (of interest): {str(root_domains)}"
            )

        basename_parts = dataset_dir_parts[begin_index:]
        normalized_basename = str(Path(*basename_parts)).replace("/", "_")
        normalized_basename = normalized_basename.replace("-", "")
        normalized_basename = normalized_basename.lower()

        return normalized_basename

    def get_df_output_path(
            self,
            normalized_basename: str,
            prefix: str = "",
            suffix: str = "",
            extension: str = ".csv",
    ) -> Path:
        output_path_str = (
            f"{str(self.normalized_dataset_dir)}/"
            f"{prefix}_{normalized_basename}_{suffix}.{extension}"
        )
        return Path(output_path_str)



    # LEGACY



    def get_dataset_output_basename(self, globbed_dataset_path: str) -> str:
        """Return filename without extension"""
        dataset_name = str(Path(*Path(globbed_dataset_path).parent.parts[2:])).replace("/", "_")
        dataset_name = f"{dataset_name}_{type(self.featurizer).__name__}"
        return dataset_name

    def featurize_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Featurizes the entire training dataset"""
        df_to_featurize = pd.read_csv(dataset_path, delimiter=";")
        df_to_featurize.columns = df_to_featurize.columns.str.lower()

        smiles_to_featurize: list = self.get_clean_smiles_from_dataframe(df_to_featurize)

        descriptors = self.featurizer.featurize(smiles_to_featurize)
        descriptors = ["".join([str(bit) for bit in np_array]) for np_array in descriptors]

        df_featurized = pd.DataFrame({
            "smiles": smiles_to_featurize,
            "fp_ecfp": descriptors 
        })
        
        return df_featurized

    def save_featurized_dataset(self, dataset_path: str, df_featurized: pd.DataFrame):
        output_basename = self.get_dataset_output_basename(dataset_path)
        df_featurized.to_csv(self.out_dir / f"ecfp/{output_basename}.csv")

    def load_featurized_dataset(self, dataset_path) -> pd.DataFrame:
        output_basename = self.get_dataset_output_basename(dataset_path)
        dataset_path = self.out_dir / f"ecfp/{output_basename}.csv"
        
        df_featurized = pd.read_csv(dataset_path, delimiter=",")

        return df_featurized
    
    def get_ecfp_bitcolumn_dataframe(self, df_featurized: pd.DataFrame) -> pd.DataFrame:
        """
        Take in a dataframe containing SMILES with their <n_bits>-long ECFP string
        Output a dataframe with SMILES and each ECFP bit in separate col, i.e. bit_0, ..., bit_2047 (if n_bits=2048)
        """

        bit_columns = df_featurized["fp_ecfp"].apply(
            lambda s: pd.Series([int(s[i]) for i in range(len(s))], index=[f"bit_{i}" for i in range(len(s))])
        )

        ecfp_dataframe = pd.concat([df_featurized[["smiles"]], bit_columns], axis=1)
        
        return ecfp_dataframe

    def dump_pca_visualization(self, dataset_df_dict: dict[str, pd.DataFrame]):
        dataset_smiles_dict = {
            ds_basename: df["smiles"] for ds_basename, df in dataset_df_dict.items() 
        }

        dataset_pca_form_dict = {
            ds_basename: df.drop(columns=["smiles"]) for ds_basename, df in dataset_df_dict.items()
        }

        dataset_after_pca_ndarray_dict = {
            ds_basename: self.explorer.get_pca(df) for ds_basename, df in dataset_pca_form_dict.items()
        }

        dataset_after_pca_df_dict = {}
        for ds_basename, ndarray in dataset_after_pca_ndarray_dict.items():
            dataset_after_pca_df_dict[ds_basename] = pd.DataFrame(
                {
                    f"dim_{i + 1}": ndarray[:, i] for i in range(ndarray.shape[1])
                }
            )
        
        image = self.explorer.visualize(dataset_after_pca_df_dict)

        image.save("test.png")
        