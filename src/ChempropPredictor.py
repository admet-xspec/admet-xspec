from PredictorBase import PredictorBase
from chemprop import data, featurizers, models, nn
import abc
import numpy as np
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import torch


class ChempropPredictor(PredictorBase):
    def __init__(self, mp, agg, metric_list, batch_norm):
        """
        Represents a ChemProp message passing neural network model
        :param mp: ChemProp message passing neural network model
        :param agg: ChemProp aggregation
        :param batch_norm: ChemProp batch normalization
        """
        self.mp = mp
        self.agg = agg
        self.batch_norm = batch_norm
        self.metric_list = metric_list
        self.ffn = self.init_ffn()
        self.metrics = self.init_metrics()
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

        self.model = models.MPNN(
            self.mp, self.agg, self.ffn, self.batch_norm, self.metric_list
        )

        super(ChempropPredictor, self).__init__()

    @abc.abstractmethod
    def init_ffn(self):
        """
        Initialize the feed forward network for the model
        """
        pass

    @abc.abstractmethod
    def init_metrics(self):
        """
        Initialize the metrics for the model
        """
        pass

    def name(self):
        return "chemprop"

    def train(self, smiles_list, target_list, num_workers=4):

        # get molecule datapoint
        all_data = [
            data.MoleculeDatapoint.from_smi(smi, y)
            for smi, y in zip(smiles_list, target_list)
        ]

        mols = [
            d.mol for d in all_data
        ]  # RDkit Mol objects are use for structure based splits
        train_indices, val_indices, test_indices = data.make_split_indices(
            mols, "random", (0.8, 0.1, 0.1)
        )
        # unpack the tuple into three separate lists
        train_data, val_data, test_data = data.split_data_by_indices(
            all_data, train_indices, val_indices, test_indices
        )

        # initialize a featurizer
        train_dset = data.MoleculeDataset(train_data[0], self.featurizer)
        scaler = train_dset.normalize_targets()

        val_dset = data.MoleculeDataset(val_data[0], self.featurizer)
        val_dset.normalize_targets(scaler)

        test_dset = data.MoleculeDataset(test_data[0], self.featurizer)

        checkpointing = ModelCheckpoint(
            "checkpoints",  # Directory where model checkpoints will be saved
            "best-{epoch}-{val_loss:.2f}",  # Filename format for checkpoints, including epoch and validation loss
            "val_loss",  # Metric used to select the best checkpoint (based on validation loss)
            mode="min",  # Save the checkpoint with the lowest validation loss (minimization objective)
            save_last=True,  # Always save the most recent checkpoint, even if it's not the best
        )

        trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=True,
            # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
            enable_progress_bar=True,
            accelerator="auto",
            devices=1,
            max_epochs=20,  # number of epochs to train for
            callbacks=[checkpointing],  # Use the configured checkpoint callback
        )

        train_loader = data.build_dataloader(train_dset, num_workers=num_workers)
        val_loader = data.build_dataloader(
            val_dset, num_workers=num_workers, shuffle=False
        )
        test_loader = data.build_dataloader(
            test_dset, num_workers=num_workers, shuffle=False
        )

        trainer.fit(self.model, train_loader, val_loader)
        results = trainer.test(dataloaders=test_loader)
        return results

    def predict(self, smiles_list):
        datapoints = [data.MoleculeDatapoint.from_smi(smi) for smi in smiles_list]
        dataset = data.MoleculeDataset(datapoints, self.featurizer)
        loader = data.build_dataloader(dataset, num_workers=4, shuffle=False)

        with torch.inference_mode():
            trainer = pl.Trainer(
                logger=None, enable_progress_bar=True, devices=1
            )
            preds = trainer.predict(self.model, loader)

        return np.concatenate(preds, axis=0)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


class ChempropRegressor(ChempropPredictor):
    def __init__(self):
        super(ChempropRegressor, self).__init__(
            mp=nn.AtomMessagePassing(),
            agg=nn.MeanAggregation(),
            metric_list=[models.RMSE(), models.MAE()],
            batch_norm=True,
        )

    def init_ffn(self):
        return nn.RegressionFFN()

    def init_metrics(self):
        return [models.RMSE(), models.MAE()]


class ChempropBinaryClassifier(ChempropPredictor):
    def __init__(self):
        super(ChempropBinaryClassifier, self).__init__(
            mp=nn.AtomMessagePassing(),
            agg=nn.MeanAggregation(),
            metric_list=[nn.metrics.BinaryAUROC(), nn.metrics.BinaryAccuracy()],
            batch_norm=True,
        )

    def init_ffn(self):
        return nn.BinaryClassificationFFN()

    def init_metrics(self):
        return [nn.metrics.BinaryAUROC(), nn.metrics.BinaryAccuracy()]
