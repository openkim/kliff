import lightning as L
from .stochastic_ml_trainer import TorchTrainer


class LightningTorchTrainer(TorchTrainer,L.LightningModule):
    def __init__(self, model, optimizer, scheduler, loss_fn, device, config):
        super().__init__(model, optimizer, scheduler, loss_fn, device, config)

    def forward(self, **model_input):
        E, F = self.model(**model_input)
        return E, F

    def training_step(self, batch, batch_idx):
        model_input = self._get_gnn_model_inputs(batch)
        E_target = batch.energy
        F_target = batch.forces
        E, F = self.forward(**model_input)
        loss = self.loss_fn(E, F, E_target, F_target)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        model_input = self._get_gnn_model_inputs(batch)
        E_target = batch.energy
        F_target = batch.forces
        E, F = self.forward(**model_input)
        loss = self.loss_fn(E, F, E_target, F_target)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler }

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update(self.model.parameters())
