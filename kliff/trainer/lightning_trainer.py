# Mostly to be used by GNNs for now
from typing import List

from .base_trainer import Trainer, TrainerError
import lightning as L
import torch
import torch.nn.functional as F

import importlib

try:
    from torch_ema import ExponentialMovingAverage
    is_torch_ema_present= True
except:
    is_torch_ema_present = False


class LightningTrainerWrapper(L.LightningModule):
    """
    This class extends the base Trainer class for training GNNs using PyTorch Lightning.
    input_keys = ["x", "coords", "edge_index0", "edge_index1" ...,"batch"]
    """
    def __init__(self, model, input_dict:List , batch_size=5, ckpt_dir=None, device="cpu", ema=True, decay=0.99, optimizer_name="Adam", lr=0.001):

        super().__init__()
        self.model = model
        self.input_dict = input_dict
        self.batch_size = batch_size
        self.ckpt_dir = ckpt_dir
        self.ema = ema
        self.optimizer_name = optimizer_name
        self.lr = lr

        if is_torch_ema_present and ema:
            ema = ExponentialMovingAverage(self.model.parameters(), decay=decay)
            ema.to(device)
            self.ema = ema

    def forward(self, batch):
        predicted_energy = self.model(**batch)
        predicted_forces, = torch.autograd.grad([predicted_energy], batch["coords"], create_graph=True, retain_graph=True,grad_outputs=torch.ones_like(predicted_energy))
        return predicted_energy, -predicted_forces

    def training_step(self, batch, batch_idx):
        target_energy = batch.energy
        target_forces = batch.forces
        energy_weight = batch.energy_weight
        forces_weight = batch.forces_weight

        eval_batch = {key: batch[key] for key in self.input_dict if key in batch}
        predicted_energy, predicted_forces = self.forward(eval_batch)
        loss = (energy_weight * F.mse_loss(predicted_energy, target_energy) +
                forces_weight * F.mse_loss(predicted_forces, target_forces))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer_name)(self.model.parameters(), lr=self.lr)
        return optimizer

    def validation_step(self, batch, batch_idx):
        target_energy = batch.energy
        target_forces = batch.forces
        energy_weight = batch.energy_weight
        forces_weight = batch.forces_weight

        eval_batch = {key: batch[key] for key in self.input_dict if key in batch}
        predicted_energy, predicted_forces = self.forward(eval_batch)
        loss = (energy_weight * F.mse_loss(predicted_energy, target_energy) +
                forces_weight * F.mse_loss(predicted_forces, target_forces))
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # def test_step(self, batch, batch_idx):
    #     pass
    #
    # def configure_optimizers(self):
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

    class GNNTrainer(Trainer):
        def __init__(self, manifest):
            super().__init__(manifest)
            self.model = None
            self.trainer = None

        def setup_model(self):
            self.model = self.model
            self.trainer = LightningTrainerWrapper(model=self.model, **self.trainer)
            self.trainer.setup_model()

        def train(self):
            self.trainer.train()
