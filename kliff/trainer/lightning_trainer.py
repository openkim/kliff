# Mostly to be used by GNNs for now
import importlib
import importlib.metadata
import json
import os.path
from copy import deepcopy
from typing import List, Union

import dill
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from loguru import logger
from torch import multiprocessing
from torch_scatter import scatter_add

from .base_trainer import Trainer, TrainerError

# check torch version, if <= 1.13, use torch_geometric.data.lightning_module
# This is temporary fix till torch 1 -> 2 migration is complete
if importlib.metadata.version("torch") <= "1.13":
    from torch_geometric.data.lightning_datamodule import LightningDataset
else:
    from torch_geometric.data.lightning import LightningDataset

from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from kliff.dataset import Dataset
from kliff.utils import get_n_configs_in_xyz

from .torch_trainer_utils.dataloaders import GraphDataset

try:
    from torch_ema import ExponentialMovingAverage

    is_torch_ema_present = True
except:
    is_torch_ema_present = False

import hashlib

from pytorch_lightning.callbacks import EarlyStopping

from .torch_trainer_utils.lightning_checkpoints import (
    LossTrajectoryCallback,
    SaveModelCallback,
)


class LightningTrainerWrapper(pl.LightningModule):
    """
    This class extends the base Trainer class for training GNNs using PyTorch Lightning.
    input_args = ["x", "coords", "edge_index0", "edge_index1" ...,"batch"]
    """

    def __init__(
        self,
        model,
        input_args: List,
        batch_size=5,
        ckpt_dir=None,
        device="cpu",
        ema=True,
        ema_decay=0.99,
        optimizer_name="Adam",
        lr=0.001,
        n_workers=1,
        energy_weight=1.0,
        forces_weight=1.0,
        lr_scheduler=None,
        lr_scheduler_args=None,
    ):

        super().__init__()
        self.model = model
        self.input_args = input_args
        self.batch_size = batch_size
        self.ckpt_dir = ckpt_dir
        self.ema = ema
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_workers = n_workers
        self.energy_weight = energy_weight
        self.forces_weight = forces_weight

        if is_torch_ema_present and ema:
            ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
            ema.to(device)
            self.ema = ema

        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_args = lr_scheduler_args

    def forward(self, batch):
        batch["coords"].requires_grad_(True)
        model_inputs = {k: batch[k] for k in self.input_args}
        predicted_energy = self.model(**model_inputs)
        (predicted_forces,) = torch.autograd.grad(
            [predicted_energy],
            batch["coords"],
            create_graph=True,  # TODO: grad against arbitrary param name
            retain_graph=True,
            grad_outputs=torch.ones_like(predicted_energy),
        )
        predicted_forces = scatter_add(predicted_forces, batch["images"], dim=0)
        return predicted_energy, -predicted_forces

    def training_step(self, batch, batch_idx):
        target_energy = batch.energy
        target_forces = batch.forces

        energy_weight = self.energy_weight
        forces_weight = self.forces_weight

        predicted_energy, predicted_forces = self.forward(batch)

        loss = energy_weight * F.mse_loss(
            predicted_energy.squeeze(), target_energy.squeeze()
        ) + forces_weight * F.mse_loss(
            predicted_forces.squeeze(), target_forces.squeeze()
        )
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer_name)(
            self.model.parameters(), lr=self.lr
        )

        if self.lr_scheduler:
            self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler)(
                optimizer, **self.lr_scheduler_args
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": self.lr_scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": "val_loss",
                },
            }
        else:
            return optimizer

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)

        target_energy = batch.energy
        target_forces = batch.forces

        energy_weight = self.energy_weight
        forces_weight = self.forces_weight

        predicted_energy, predicted_forces = self.forward(batch)

        per_atom_force_loss = torch.sum(
            (predicted_forces.squeeze() - target_forces.squeeze()) ** 2, dim=1
        )

        loss = (
            energy_weight
            * F.mse_loss(predicted_energy.squeeze(), target_energy.squeeze())
            + forces_weight * torch.mean(per_atom_force_loss) / 3
        )  # divide by 3 to get correct MSE

        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return {"val_loss": loss, "per_atom_force_loss": per_atom_force_loss}

    # def test_step(self, batch, batch_idx):
    #     pass
    #
    # def setup_model(self):
    #     pass
    #
    # def train(self):
    #     pass
    #
    # def validate(self):
    #     pass
    #
    # def test(self):
    #     pass
    #
    # def save_model(self):
    #     pass


class GNNLightningTrainer(Trainer):
    def __init__(self, manifest, model):
        self.pl_model: LightningTrainerWrapper = None
        self.data_module = None

        super().__init__(manifest, model)

        # loggers and callbacks
        self.tb_logger = self._tb_logger()
        self.csv_logger = self._csv_logger()
        self.setup_dataloaders()
        self.callbacks = self._get_callbacks()

        # setup lightning trainer
        self.pl_trainer = self._get_pl_trainer()

    def setup_model(self):
        # if dict has key ema, then set ema to True, decay to the dict value, else set ema false
        ema = True if self.optimizer_manifest.get("ema", False) else False
        if ema:
            ema_decay = self.optimizer_manifest.get("ema_decay", 0.99)
        else:
            ema_decay = None

        scheduler = self.optimizer_manifest.get("lr_scheduler", {})

        self.pl_model = LightningTrainerWrapper(
            model=self.model,
            input_args=self.model_manifest["input_args"],
            batch_size=self.optimizer_manifest["batch_size"],
            ckpt_dir=self.current["run_dir"],
            device=self.current["device"],
            ema=ema,
            ema_decay=ema_decay,
            optimizer_name=self.optimizer_manifest["name"],
            lr=self.optimizer_manifest["learning_rate"],
            energy_weight=self.loss_manifest["weights"]["energy"],
            forces_weight=self.loss_manifest["weights"]["forces"],
            lr_scheduler=scheduler.get("name", None),
            lr_scheduler_args=scheduler.get("args", None),
        )

    def train(self):
        self.pl_trainer.fit(self.pl_model, self.data_module)

    def setup_dataloaders(self):
        if self.dataset_manifest["dynamic_loading"]:
            transform = self.configuration_transform
        else:
            transform = None

        self.train_dataset = GraphDataset(self.train_dataset, transform)
        if self.val_dataset:
            self.val_dataset = GraphDataset(self.val_dataset, transform)

        if not transform:
            for config in self.train_dataset:
                config.fingerprint = self.configuration_transform(config)
            if self.val_dataset:
                for config in self.val_dataset:
                    config.fingerprint = self.configuration_transform(config)

        self.data_module = LightningDataset(
            self.train_dataset,
            self.val_dataset,
            batch_size=self.optimizer_manifest["batch_size"],
            num_workers=self.optimizer_manifest["num_workers"],
        )
        logger.info("Data modules setup complete.")

    def _tb_logger(self):
        return TensorBoardLogger(
            f"{self.current['run_dir']}/logs", name="lightning_logs"
        )

    def _csv_logger(self):
        return CSVLogger(f"{self.current['run_dir']}/logs", name="csv_logs")

    def _get_pl_trainer(self):
        return pl.Trainer(
            logger=[self.tb_logger, self.csv_logger],
            max_epochs=self.optimizer_manifest["epochs"],
            accelerator="auto",
            strategy="ddp",
            callbacks=self.callbacks,
        )

    def _get_callbacks(self):
        callbacks = []

        ckpt_dir = f"{self.current['run_dir']}/checkpoints"
        ckpt_interval = self.training_manifest.get("ckpt_interval", 50)
        save_model_callback = SaveModelCallback(ckpt_dir, ckpt_interval)
        callbacks.append(save_model_callback)
        logger.info("Checkpointing setup complete.")

        if self.training_manifest.get("early_stopping", False):
            patience = self.training_manifest["early_stopping"].get("patience", 10)
            if not isinstance(patience, int):
                raise TrainerError(
                    f"Early stopping should be an integer, got {patience}"
                )
            delta = self.training_manifest["early_stopping"].get("delta", 1e-3)
            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=patience,
                mode="min",
                min_delta=delta,
            )
            callbacks.append(early_stopping)
            logger.info("Early stopping setup complete.")

        if self.loss_manifest.get("loss_traj", False):
            loss_traj_folder = f"{self.current['run_dir']}/loss_trajectory"
            loss_idxs = self.dataset_sample_manifest["val_indices"]
            ckpt_interval = self.training_manifest.get("ckpt_interval", 10)
            loss_trajectory_callback = LossTrajectoryCallback(
                loss_traj_folder, loss_idxs, ckpt_interval
            )
            callbacks.append(loss_trajectory_callback)
            logger.info("Loss trajectory setup complete.")

        return callbacks

    def setup_optimizer(self):
        # Not needed as Pytorch Lightning handles the optimizer
        pass
