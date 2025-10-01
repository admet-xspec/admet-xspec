import pandas as pd
from src.utils import clean_smiles, get_nice_class_name
import logging
from src.predictor.predictor_base import PredictorBase
from src.data.featurizer import (
    FeaturizerBase,
)
from src.data.split import DataSplitterBase
from src.training_pipeline import TrainingPipeline
import json
import gin
from pathlib import Path


@gin.configurable()
class ManagementPipeline:
    """ 'meta-Pipeline' (sounds cool, eh?) for handling temporary/one-off work """

    def __init__(
        self,
        data_path: Path | str,
        splitter: DataSplitterBase,
        predictor: PredictorBase,
        featurizer: FeaturizerBase,
        model_name: str,
        out_dir: Path | str = "preprocessing",
        test_size: float = 0.2,
        stratify: bool = True,
    ):

        self.data_path = Path(data_path)
        self.splitter = splitter
        self.predictor = predictor
        self.featurizer = featurizer
        self.model_name = model_name
        self.out_dir = out_dir
        self.test_size = test_size
        self.stratify = stratify

        self.train_path = None
        self.test_path = None

    def featurize_dataset(self):
        """
        Featurizes the entire training dataset and outputs it
        """

        # Get it to do the first step for us
        pseudo_training_pipeline = TrainingPipeline(
            data_path=self.data_path,
            splitter=self.splitter,
            predictor=self.predictor,
            featurizer=self.featurizer, 
            model_name=self.model_name,
            out_dir=self.out_dir,
            test_size=self.test_size,
            stratify=self.stratify
        )

        pseudo_training_pipeline.prepare_data()
        data_to_featurize_path = pseudo_training_pipeline.get_training_data_path()

        # df_train = pd.DataFrame([X_train, y_train], columns=["smiles", "y"])
        # df_train.to_csv(save_path_train)
        df_to_featurize = pd.read_csv(data_to_featurize_path)
        smiles_to_featurize: list = df_to_featurize["smiles"].to_list()

        descriptors = self.featurizer.featurize(smiles_to_featurize)

        df_featurized = pd.DataFrame([smiles_to_featurize, descriptors], columns=["smiles", "descriptor"])
        
        dataset = self.data_path.name
        df_featurized.to_csv(self.out_dir / f"{dataset}_{type(self.featurizer).__name__}")