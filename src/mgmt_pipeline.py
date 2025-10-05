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
        explorer: ExplorerBase,
        splitter: DataSplitterBase,
        predictor: PredictorBase,
        featurizer: FeaturizerBase,
        model_name: str,
        out_dir: Path | str,
        test_size: float = 0.2,
        stratify: bool = True,
    ):

        self.dataset_dir = Path(dataset_dir)
        self.explorer = explorer
        self.splitter = splitter
        self.predictor = predictor
        self.featurizer = featurizer
        self.model_name = model_name
        self.out_dir = Path(out_dir)
        self.test_size = test_size
        self.stratify = stratify

        self.train_path = None
        self.test_path = None

    def get_clean_smiles_from_dataframe(self, df) -> list[str]:
        """Consolidate NaN-dropping, ";-separated" data loading into one function"""

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

        dataset_name = str(Path(globbed_dataset_path).parent).replace("/", "_")
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
        
        self.save_featurized_dataset(dataset_path, df_featurized)

    def save_featurized_dataset(self, dataset_path: str, df_featurized: pd.DataFrame):
        output_basename = self.get_dataset_output_basename(dataset_path)
        df_featurized.to_csv(self.out_dir / f"ecfp/{output_basename}.csv")

    def load_featurized_dataset(self, dataset_path) -> dict[str, np.ndarray]:
        output_basename = self.get_dataset_output_basename(dataset_path)
        dataset_path = self.out_dir / f"ecfp/{output_basename}.csv"
        
        df_featurized = pd.read_csv(dataset_path, delimiter=",")

        descriptors = df_featurized["fp_ecfp"].to_list()
        descriptors = [np.array([bit for bit in fp_string]) for fp_string in descriptors]

        featurized_dataset = {
            smiles: descriptor for smiles, descriptor in zip(
                df_featurized["smiles"].to_list(),
                descriptors
            )
        }

        return featurized_dataset
    
    def dump_visualization(self, df_featurized: pd.DataFrame):
        visualization_name: str
        image: Image.Image
        
        visualization_name, image = self.explorer.visualize(df_featurized)
