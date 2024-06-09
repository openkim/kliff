import os

import dill
import pytorch_lightning as pl
import torch

from kliff.dataset import Dataset


class SaveModelCallback(pl.Callback):
    """
    Callback to save the model at the end of each epoch. The model is saved in the ckpt_dir with the name
    "last_model.pth". The best model is saved with the name "best_model.pth". The model is saved every
    ckpt_interval epochs with the name "epoch_{epoch}.pth".
    """

    def __init__(self, ckpt_dir, ckpt_interval=100):
        super().__init__()
        self.ckpt_dir = ckpt_dir
        self.best_val_loss = float("inf")
        self.ckpt_interval = ckpt_interval
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        # Save the last model
        last_save_path = os.path.join(self.ckpt_dir, "last_model.pth")
        torch.save(pl_module.state_dict(), last_save_path)

        # Save the best model
        if trainer.callback_metrics.get("val_loss") < self.best_val_loss:
            self.best_val_loss = trainer.callback_metrics["val_loss"]
            best_save_path = os.path.join(self.ckpt_dir, "best_model.pth")
            torch.save(pl_module.state_dict(), best_save_path)

        # Save the model every ckpt_interval epochs
        if pl_module.current_epoch % self.ckpt_interval == 0:
            epoch_save_path = os.path.join(
                self.ckpt_dir, f"epoch_{pl_module.current_epoch}.pth"
            )
            torch.save(pl_module.state_dict(), epoch_save_path)


class LossTrajectoryCallback(pl.Callback):
    """
    Callback to save the loss trajectory of the model during validation. The loss trajectory is saved in the
    loss_traj_file. The loss trajectory is saved every ckpt_interval epochs. Currently, it only logs per atom force loss.
    """

    def __init__(self, loss_traj_folder, val_dataset: Dataset, ckpt_interval=10):
        super().__init__()
        self.loss_traj_folder = loss_traj_folder
        self.ckpt_interval = ckpt_interval

        os.makedirs(self.loss_traj_folder, exist_ok=True)
        with open(os.path.join(self.loss_traj_folder, "loss_traj_idx.csv"), "w") as f:
            f.write("epoch,loss\n")

        dill.dump(
            val_dataset,
            open(os.path.join(self.loss_traj_folder, "val_dataset.pkl"), "wb"),
        )
        self.val_losses = []

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if trainer.current_epoch % self.ckpt_interval == 0:
            val_force_loss = outputs["per_atom_force_loss"].detach().cpu().numpy()
            self.val_losses.extend(val_force_loss)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.ckpt_interval == 0:
            with open(
                os.path.join(self.loss_traj_folder, "loss_traj_idx.csv"), "a"
            ) as f:
                loss_str = ",".join(
                    [str(trainer.current_epoch)]
                    + [f"{loss:.5f}" for loss in self.val_losses]
                )
                f.write(f"{loss_str}\n")
            self.val_losses = []
