import os
import pickle as pkl
from typing import Any, Union

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

        # save the trainer checkpoint as well
        trainer.save_checkpoint(os.path.join(self.ckpt_dir, "trainer_checkpoint.ckpt"))


class SavePerAtomPredictions(pl.Callback):
    """
    Callback to save the per atom predictions of the model during validation. The per
    atom predictions are saved in the supplied lmdb file. Usually it is named
    `per_atom_pred_database.lmdb` in the run dir
    """

    def __init__(self, lmdb_file_handle, ckpt_interval):
        super().__init__()
        self.lmdb_file_handle = lmdb_file_handle
        self.ckpt_interval = ckpt_interval

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
            epoch = trainer.current_epoch
            predicted_forces = outputs["per_atom_pred"]
            self._log_per_atom_outputs(epoch, batch, predicted_forces)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.current_epoch % self.ckpt_interval == 0:
            epoch = trainer.current_epoch
            predicted_forces = outputs["per_atom_pred"]
            self._log_per_atom_outputs(epoch, batch, predicted_forces)

    def _log_per_atom_outputs(
        self,
        epoch: Union[int, torch.Tensor],
        batch: Any,
        predicted_forces: torch.Tensor,
    ):
        """
        This function is duplicate of ~:class:`kliff.trainer.Trainer.log_per_atom_outputs`.

        Args:
            epoch: Current epoch
            idxs: Index of the configurations
            predicted_forces: Predicted forces
        """
        with self.lmdb_file_handle.begin(write=True) as txn:
            idxs = batch["idx"]
            n_configs = len(idxs)

            from_ = 0
            to_ = -1

            for i in range(n_configs):
                # get the prediction pointer, every even index is contributing
                n_atoms = (
                    batch["contributions"][batch["contributions"] == (2 * i)]
                ).shape[0]
                to_ = from_ + n_atoms
                pred = predicted_forces[from_:to_].detach().cpu().numpy()
                from_ = to_

                # save the predictions
                txn.put(
                    f"epoch_{epoch}|index_{idxs[i]}".encode(),
                    pkl.dumps({"pred_0": pred, "n_atoms": n_atoms}),
                )


# TODO: LTAU callback?
