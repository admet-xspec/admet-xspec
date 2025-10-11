import glob
import pandas as pd
from PIL import Image
from src.utils import clean_smiles, get_nice_class_name
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
        self.test_size = test_size
        self.stratify = stratify

        self.train_path = None
        self.test_path = None

    def normalize_datasets(self, force_normalize_all: bool = False):
        datasets = glob.glob(f"{str(self.dataset_dir)}/**/*.csv", recursive=True)
        
        datasets_paths: list[tuple[str, Path]] = [
            (ds_glob, self.get_normalized_path(ds_glob)) for ds_glob in datasets
        ]

        if not force_normalize_all:
            datasets_paths = [
                (ds_glob, ds_path) 
                for ds_glob, ds_path in datasets_paths 
                if not ds_path.exists()
            ]
        
        for ds_glob, ds_path in datasets_paths:
            normalized_dataset_df = self.get_normalized_dataset(
                ds_glob
            )

            self.save_dataframe(
                normalized_dataset_df, 
                ds_path
            )

    def get_clean_smiles_from_dataframe(self, df) -> list[str]:
        """Consolidate NaN-dropping, ";-separated" data loading into one function"""

        self.no

        pre_dropna_length = len(df)
        df = df.dropna(subset="smiles")
        pre_cleaning_length = len(df)
        df["smiles"] = clean_smiles(df["smiles"].to_list())
        df = df.dropna(subset=["smiles"]).reset_index(drop=True)
        
        if pre_dropna_length != pre_cleaning_length:
            logging.info(f"Dropped {pre_dropna_length - pre_cleaning_length} 'nan' SMILES after pd.read_csv")
        if pre_cleaning_length != len(df):
            logging.info(f"Dropped {pre_cleaning_length - len(df)} invalid SMILES")
        logging.info(f"Dataset size: {len(df)}")

        return df["smiles"].tolist()

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
        