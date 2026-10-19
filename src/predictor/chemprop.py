import logging
from src.predictor.predictor_base import PredictorBase
import chemprop as chp
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
import ray
from typing import List
from ray.train import ScalingConfig
from ray import tune
from ray.train.torch import TorchTrainer
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
        """Wrap a ChemProp message passing neural network.

        Args:
            n_workers: Dataloader / Ray worker count.
            optimize_hyperparameters: Tune hyperparameters with Ray Tune.
            params_distribution: Search space used when optimizing.
            optimization_iterations: Number of tuning trials.
            params: Model hyperparameters; ignored when optimizing.
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
        super(ChempropPredictor, self).__init__(task="classification")

    def _init_ffn(self, num_layers: int, hidden_dim: int):
        """Return the task-specific ChemProp FFN head."""
        raise NotImplementedError()

    def train(self, smiles_list: List[str], target_list: List[float]):
        """Train the model, optionally tuning hyperparameters first."""
        if self.optimize_hyperparameters:
            self._train_optimize(smiles_list, target_list)
        else:
            self._train_once(smiles_list, target_list)
        self._ready()

    def _train_once(
        self, smiles_list: List[str], target_list: List[float], config=None
    ):
        """Train a single model with `config`, or with `self.params` if None."""
        if config is not None:
            self.model = self._init_model(config=config)
            logging.info(f"Using the following parameters for training: {config}")
        else:
            self.model = self._init_model()

        train_loader, val_loader = self.prepare_dataloaders(smiles_list, target_list)

        checkpointing = ModelCheckpoint(
            "checkpoints",
            "best-{epoch}-{val_loss:.2f}",
            "val_loss",
            mode="min",
            save_last=True,
        )

        wandb_logger = WandbLogger(project="admet")
        trainer = pl.Trainer(
            logger=wandb_logger,
            enable_checkpointing=True,
            enable_progress_bar=True,
            accelerator="auto",
            devices=1,
            max_epochs=self.epochs,
            callbacks=[checkpointing],
        )
        trainer.fit(self.model, train_loader, val_loader)

    def _train_optimize(self, smiles_list: List[str], target_list: List[float]):
        """Tune hyperparameters with Ray Tune, minimizing validation loss."""
        ray.init()
        scheduler = FIFOScheduler()

        scaling_config = ScalingConfig(
            num_workers=1 if self.use_gpu else num_workers,
            use_gpu=self.use_gpu,
        )
        checkpoint_config = ray.tune.CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute="val_loss",
            checkpoint_score_order="min",
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
            num_samples=self.n_tries,
            scheduler=scheduler,
            search_alg=search_alg,
            # Short trial dirs keep paths under OS limits.
            trial_dirname_creator=lambda trial: str(trial.trial_id),
        )

        tuner = tune.Tuner(
            ray_trainer,
            param_space={
                "train_loop_config": self.params_distribution,
            },
            tune_config=tune_config,
        )

        results = tuner.fit()
        best_result = results.get_best_result()
        best_config = best_result.config
        logging.info(
            "Hyperparameter search finished successfully. Best model's hyperparameters:"
        )
        logging.info(best_config["train_loop_config"])
        self._ready()

    def predict(self, smiles_list: List[str]) -> np.array:
        """Return predictions as an (n, 1) array."""
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
        """Return an atom- or bond-based message passing block."""
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
        """Return a mean, sum or norm aggregation block."""
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
        """Build an MPNN from `config`, defaulting to `self.params`."""
        if config is None:
            config = self.params

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
        """Return (train, val) dataloaders from a random 80/20 split."""
        target_list = np.array(target_list).reshape(-1, 1)

        all_data = [
            chp.data.MoleculeDatapoint.from_smi(smi, y)
            for smi, y in zip(smiles_list, target_list)
        ]

        mols = [d.mol for d in all_data]
        train_indices, _, val_indices = chp.data.make_split_indices(
            mols=mols, sizes=(0.8, 0, 0.2), seed=42, split="random"
        )

        train_data, val_data, _ = chp.data.split_data_by_indices(
            data=all_data,
            train_indices=train_indices,
            val_indices=val_indices,
        )

        train_dset = chp.data.MoleculeDataset(train_data[0], self.featurizer)
        val_dset = chp.data.MoleculeDataset(val_data[0], self.featurizer)

        train_loader = chp.data.build_dataloader(
            train_dset, num_workers=self.num_workers
        )
        val_loader = chp.data.build_dataloader(
            val_dset, num_workers=self.num_workers, shuffle=False
        )

        return train_loader, val_loader

    @staticmethod
    def process_param_distribution_dict(input_dict: dict) -> dict:
        """Convert a search-space dict into Ray Tune distributions.

        Lists become `tune.choice`; distribution objects use `get_ray_distrib`.
        """
        ray_dict = {}
        for key, value in input_dict.items():
            if isinstance(value, list):
                ray_dict[key] = tune.choice(value)
            elif isinstance(value, (Uniform, LogUniform, QUniform, QLogUniform)):
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
