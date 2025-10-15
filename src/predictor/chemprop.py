import logging
from src.predictor.predictor_base import PredictorBase
import chemprop as chp
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
import ray
from typing import List
from ray.train import ScalingConfig
from ray import tune
from ray.train.torch import TorchTrainer
from src.gin_config.distributions import Uniform, LogUniform, QUniform, QLogUniform
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import FIFOScheduler
import numpy as np
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import torch
import gin


class ChempropPredictor(PredictorBase):
    def __init__(
        self,
        featurizer: chp.featurizers.Featurizer,
        n_workers: int,
        optimize_hyperparameters: bool,
        params_distribution: dict,
        optimization_iterations: int,
        params: dict,
        epochs: int,
        use_gpu: bool,
    ):
        """
        Represents a ChemProp message passing neural network model.
        :param featurizer: featurizer object
        :param n_workers: number of workers
        :param optimize_hyperparameters: whether to optimize hyperparameters using Ray Tune
        :param params_distribution: dictionary of hyperparameter distributions for optimization
        :param optimization_iterations: number of iterations for hyperparameter optimization
        :param params: dictionary of hyperparameters for the model (ignored if optimize_hyperparameters is True)
        :param epochs: number of epochs for training
        :param use_gpu: whether to use GPU for training
        """
        self.params = params
        self.featurizer = featurizer
        self.num_workers = n_workers
        self.optimize_hyperparameters = optimize_hyperparameters
        self.params_distribution = self.process_param_distribution_dict(
            params_distribution
        )
        self.n_tries = optimization_iterations
        self.epochs = epochs
        self.use_gpu = use_gpu
        super(ChempropPredictor, self).__init__()

    def _init_ffn(self, num_layers: int, hidden_dim: int):
        """
        Initialize the feed forward network (FFN) for the model, as defined in the ChemProp library.
        Will be handled by the ChemPropRegressor and ChemPropBinaryClassifier classes, as the
        FFN is different for regression and classification tasks.
        :return: FFN object, as defined in the ChemProp library
        """
        raise NotImplementedError()

    def train(self, smiles_list: List[str], target_list: List[float]):
        """
        Train the model with the given smiles and target list.
        This method should also raise the ready_flag by calling _ready() method
        :param smiles_list: List of SMILES strings
        :param target_list: List of target values
        """
        if self.optimize_hyperparameters:
            # If hyperparameter optimization is enabled, call the train_optimize method
            self._train_optimize(smiles_list, target_list)
        else:
            # Otherwise, call the regular train method
            self._train_once(smiles_list, target_list)

        # Raise the ready flag
        self._ready()

    def _train_once(
        self, smiles_list: List[str], target_list: List[float], config=None
    ):

        if config is not None:
            # Use a dictionary of parameters to initialize the model
            self.model = self._init_model(config=config)
            logging.info(f"Using the following parameters for training: {config}")
        else:
            # Use the parameters provided in the config file (default)
            self.model = self._init_model()

        train_loader, val_loader = self.prepare_dataloaders(smiles_list, target_list)

        checkpointing = ModelCheckpoint(
            "checkpoints",  # Directory where model checkpoints will be saved
            "best-{epoch}-{val_loss:.2f}",  # Filename format for checkpoints, including epoch and validation loss
            "val_loss",  # Metric used to select the best checkpoint (based on validation loss)
            mode="min",  # Save the checkpoint with the lowest validation loss (minimization objective)
            save_last=True,  # Always save the most recent checkpoint, even if it's not the best
        )

        wandb_logger = WandbLogger(project="admet")
        trainer = pl.Trainer(
            logger=wandb_logger,
            enable_checkpointing=True,
            enable_progress_bar=True,
            accelerator="auto",
            devices=1,
            max_epochs=self.epochs,  # number of epochs to train for
            callbacks=[checkpointing],  # Use the configured checkpoint callback
        )

        # Train the model
        trainer.fit(self.model, train_loader, val_loader)

    def _train_optimize(self, smiles_list: List[str], target_list: List[float]):

        ray.init()
        scheduler = FIFOScheduler()

        # Scaling config controls the resources used by Ray
        scaling_config = ScalingConfig(
            num_workers=1 if self.use_gpu else num_workers,  # number of workers to use
            use_gpu=self.use_gpu,  # change to True if you want to use GPU
        )

        # Checkpoint config controls the checkpointing behavior of Ray
        checkpoint_config = ray.tune.CheckpointConfig(
            num_to_keep=1,  # number of checkpoints to keep
            checkpoint_score_attribute="val_loss",  # Save the checkpoint based on this metric
            checkpoint_score_order="min",  # Save the checkpoint with the lowest metric value
        )

        run_config = ray.tune.RunConfig(
            checkpoint_config=checkpoint_config,
        )

        ray_trainer = TorchTrainer(
            lambda config: self._train_once(smiles_list, target_list, config),
            scaling_config=scaling_config,
            run_config=run_config,
        )

        search_alg = HyperOptSearch(
            n_initial_points=1,
            random_state_seed=42,
        )

        tune_config = tune.TuneConfig(
            metric="val_loss",
            mode="min",
            num_samples=self.n_tries,  # number of trials to run
            scheduler=scheduler,
            search_alg=search_alg,
            trial_dirname_creator=lambda trial: str(
                trial.trial_id
            ),  # shorten filepaths
        )

        tuner = tune.Tuner(
            ray_trainer,
            param_space={
                "train_loop_config": self.params_distribution,
            },
            tune_config=tune_config,
        )

        # Start the hyperparameter search
        results = tuner.fit()

        # Log hyperparameters of the best model
        best_result = results.get_best_result()
        best_config = best_result.config
        logging.info(
            "Hyperparameter search finished successfully. Best model's hyperparameters:"
        )
        logging.info(best_config["train_loop_config"])

        # Raise the ready flag
        self._ready()

    def predict(self, smiles_list: List[str]) -> np.array:
        datapoints = [chp.data.MoleculeDatapoint.from_smi(smi) for smi in smiles_list]
        dataset = chp.data.MoleculeDataset(datapoints, self.featurizer)
        loader = chp.data.build_dataloader(
            dataset, num_workers=self.num_workers, shuffle=False
        )

        with torch.inference_mode():
            trainer = pl.Trainer(logger=None, enable_progress_bar=True, devices=1)
            preds = trainer.predict(self.model, loader)
            preds = torch.cat(preds, dim=0).cpu().numpy()
            return np.array(preds).reshape(-1, 1)

    @staticmethod
    def _init_mp(mp_type: str, d_h: int, depth: int):
        if mp_type.lower() == "atom":
            return chp.nn.AtomMessagePassing(
                d_h=d_h,
                depth=depth,
            )
        elif mp_type.lower() == "bond":
            return chp.nn.BondMessagePassing(
                d_h=d_h,
                depth=depth,
            )
        else:
            raise ValueError(
                f"Unsupported message passing type: {mp_type}. Can be 'atom' or 'bond'."
            )

    @staticmethod
    def _init_agg(agg_type="mean"):
        if agg_type.lower() == "mean":
            return chp.nn.MeanAggregation()
        elif agg_type.lower() == "sum":
            return chp.nn.SumAggregation()
        elif agg_type.lower() == "norm":
            return chp.nn.NormAggregation()
        else:
            raise ValueError(
                f"Unsupported aggregation type: {agg_type}. Can be 'mean', 'sum' or 'norm'."
            )

    def _init_model(self, config: dict = None):
        """
        Initialize the ChemProp model consisting of the given message passing network (MPNN) with
        the parameters provided in the config file, or in a dictionary passed to the function.
        """
        if config is None:
            # Use the parameters provided in the config file
            config = self.params

        # Check if the config file is valid
        for param in [
            "mp_type",
            "mp_hidden_dim",
            "mp_num_layers",
            "agg_type",
            "ffn_hidden_dim",
            "ffn_num_layers",
            "batch_norm",
        ]:
            if param not in config.keys():
                raise ValueError(f"Missing parameter {param} in the passed config dict")

        logging.info("Initializing ChemProp model with the following parameters:")
        logging.info(config)

        return chp.models.MPNN(
            self._init_mp(
                config["mp_type"], config["mp_hidden_dim"], config["mp_num_layers"]
            ),
            self._init_agg(config["agg_type"]),
            self._init_ffn(config["ffn_hidden_dim"], config["ffn_num_layers"]),
            config["batch_norm"],
        )

    def prepare_dataloaders(self, smiles_list, target_list):
        """
        Build the dataloaders for the given molecules and target values.
        :param smiles_list: List of SMILES strings
        :param target_list: List of target values
        :return: Tuple of train and validation dataloaders
        """
        target_list = np.array(target_list).reshape(-1, 1)

        # get molecule datapoint
        all_data = [
            chp.data.MoleculeDatapoint.from_smi(smi, y)
            for smi, y in zip(smiles_list, target_list)
        ]

        mols = [
            d.mol for d in all_data
        ]  # RDkit Mol objects are use for structure-based splits
        train_indices, _, val_indices = chp.data.make_split_indices(
            mols=mols, sizes=(0.8, 0, 0.2), seed=42, split="random"
        )

        # Split the data into train and validation sets
        train_data, val_data, _ = chp.data.split_data_by_indices(
            data=all_data,
            train_indices=train_indices,
            val_indices=val_indices,
        )

        # Create the datasets
        train_dset = chp.data.MoleculeDataset(train_data[0], self.featurizer)
        val_dset = chp.data.MoleculeDataset(val_data[0], self.featurizer)

        # Create the dataloaders
        train_loader = chp.data.build_dataloader(
            train_dset, num_workers=self.num_workers
        )
        val_loader = chp.data.build_dataloader(
            val_dset, num_workers=self.num_workers, shuffle=False
        )

        return train_loader, val_loader

    @staticmethod
    def process_param_distribution_dict(input_dict: dict) -> dict:
        """
        Get Ray library distribution objects for the parameters in the params_distribution dictionary,
        which are defined as subclasses of scipy.stats.rv_continuous or rv_discrete.
        :return: Dictionary of parameter distributions (instances of classes from ray.tune)
        """
        ray_dict = {}
        for key, value in input_dict.items():
            if isinstance(value, list):
                # If the value is a list, use tune.choice
                ray_dict[key] = tune.choice(value)
            elif isinstance(value, (Uniform, LogUniform, QUniform, QLogUniform)):
                # If the value is a distribution, use the get_ray_distrib method
                ray_dict[key] = value.get_ray_distrib()
            else:
                raise ValueError(f"Unsupported distribution type: {type(value)}")
        return ray_dict


@gin.configurable
class ChempropRegressor(ChempropPredictor):
    def __init__(
        self,
        featurizer: (
            chp.featurizers.Featurizer | None
        ) = SimpleMoleculeMolGraphFeaturizer(),
        n_workers: int = 1,
        optimize_hyperparameters: bool = False,
        params_distribution: dict | None = None,
        optimization_iterations: int = 10,
        params: dict | None = None,
        epochs: int = 100,
        use_gpu: bool = True,
    ):
        super(ChempropRegressor, self).__init__(
            featurizer=featurizer,
            n_workers=n_workers,
            optimize_hyperparameters=optimize_hyperparameters,
            params_distribution=params_distribution,
            optimization_iterations=optimization_iterations,
            params=params,
            epochs=epochs,
            use_gpu=use_gpu,
        )

    def _init_ffn(self, num_layers: int, hidden_dim: int):
        return chp.nn.RegressionFFN(hidden_dim=hidden_dim, n_layers=num_layers)


@gin.configurable
class ChempropBinaryClassifier(ChempropPredictor):
    def __init__(
        self,
        featurizer: (
            chp.featurizers.Featurizer | None
        ) = SimpleMoleculeMolGraphFeaturizer(),
        n_workers: int = 1,
        optimize_hyperparameters: bool = False,
        params_distribution: dict | None = None,
        optimization_iterations: int = 10,
        params: dict | None = None,
        epochs: int = 100,
        use_gpu: bool = True,
    ):
        super(ChempropBinaryClassifier, self).__init__(
            featurizer=featurizer,
            n_workers=n_workers,
            optimize_hyperparameters=optimize_hyperparameters,
            params_distribution=params_distribution,
            optimization_iterations=optimization_iterations,
            params=params,
            epochs=epochs,
            use_gpu=use_gpu,
        )

    def _init_ffn(self, num_layers: int, hidden_dim: int):
        return chp.nn.BinaryClassificationFFN(
            hidden_dim=hidden_dim, n_layers=num_layers
        )
