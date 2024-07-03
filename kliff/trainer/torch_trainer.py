import os
import tarfile
from copy import deepcopy
from typing import TYPE_CHECKING

import libdescriptor as lds
import numpy as np
import torch
from loguru import logger
from monty.dev import requires
from torch.utils.data import DataLoader as TorchDataLoader
from torch_scatter import scatter_add

from .base_trainer import Trainer, TrainerError
from .torch_trainer_utils.dataloaders import DescriptorDataset

if TYPE_CHECKING:
    from kliff.transforms.configuration_transforms import Descriptor


class DNNTrainer(Trainer):
    """
    This class extends the base KLIFF trainer class to implement and train MLIPs compatible
    with the TorchML model driver. This is an example of simple, customizable training
    for a TorchML model. For large scale, highly optimized training, consider using
    Lightning Based GNN Trainer.

    Args:
        configuration (dict): A dictionary containing the configuration for the trainer.
        model (torch.nn.Module): A torch model to be trained. If not provided, the model
            will be loaded from the model manifest. For model manifest based loading
            the model must be a torchscript model, or valid TorchML model (tar or dir).
    """

    def __init__(self, configuration: dict, model=None):
        self.configuration_transform: "Descriptor" = (
            None  # for type checking the functions
        )
        self.torchscript_file = None
        self.train_dataloader = None
        self.validation_dataloader = None
        self.lr_scheduler = None
        self.early_stopping = None
        self.dtype = torch.float64

        super().__init__(configuration, model)

        self.loss_function = self._get_loss_function()
        self.setup_dataloaders()
        self.early_stopping = self.get_early_stopping()
        self.lr_scheduler = self.get_scheduler()

    # Optim Loss and Checkpoint #######################################################
    def setup_optimizer(self):
        self.optimizer = self.get_optimizer()

    def loss(self, x: torch.Tensor, y: torch.Tensor):
        return self.loss_function(x, y)

    def checkpoint(self):
        """
        Checkpoint the model and optimizer state to disk. Also append training and validation
        loss to the log file.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "current_step": self.current["step"],
            "current_loss": self.current["loss"],
        }

        if self.lr_scheduler:
            checkpoint["lr_scheduler"] = self.lr_scheduler.state_dict()

        if self.early_stopping:
            checkpoint["early_stopping"] = {
                "counter": self.early_stopping.counter,
                "best_loss": self.early_stopping.best_loss,
            }

        torch.save(
            checkpoint,
            f"{self.current['run_dir']}/checkpoint_{self.current['step']}.pkl",
        )

        # save best and last model
        if self.current["loss"]["val"] < self.current["best_loss"]:
            self.current["best_loss"] = self.current["loss"]["val"]
            torch.save(
                self.model.state_dict(),
                f"{self.current['run_dir']}/best_model.pth",
            )

        torch.save(
            self.model.state_dict(),
            f"{self.current['run_dir']}/last_model.pth",
        )

        if os.path.exists(f"{self.current['run_dir']}/loss.log"):
            with open(f"{self.current['run_dir']}/loss.log", "a") as f:
                f.write(
                    f"{self.current['step']},{self.current['loss']['train']},{self.current['loss']['val']}\n"
                )
        else:
            with open(f"{self.current['run_dir']}/loss.log", "w") as f:
                f.write("step,train_loss,val_loss\n")
                f.write(
                    f"{self.current['step']},{self.current['loss']['train']},{self.current['loss']['val']}\n"
                )

    def load_checkpoint(self, path: str):
        """
        Load the model and optimizer state from a checkpoint file.

        Args:
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current["step"] = checkpoint["current_step"]
        self.current["loss"] = checkpoint["current_loss"]

        if self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        if self.early_stopping:
            self.early_stopping.counter = checkpoint["early_stopping"]["counter"]
            self.early_stopping.best_loss = checkpoint["early_stopping"]["best_loss"]

    def get_last_checkpoint(self):
        """
        Get the last checkpoint file in the run directory.

        Returns:
            str: Path to the last checkpoint file.
        """
        checkpoints = [
            f
            for f in os.listdir(f"{self.current['run_dir']}")
            if f.startswith("checkpoint")
        ]
        max_step = max(
            list(map(lambda x: int(x.split("_")[1].split(".")[0]), checkpoints))
        )
        return f"{self.current['run_dir']}/checkpoint_{max_step}.pkl"

    def get_optimizer(self):
        optimizer = getattr(torch.optim, self.optimizer_manifest["name"])
        if self.optimizer_manifest["kwargs"]:
            return optimizer(
                self.model.parameters(),
                lr=self.optimizer_manifest["learning_rate"],
                **self.optimizer_manifest["kwargs"],
            )
        else:
            return optimizer(
                self.model.parameters(), lr=self.optimizer_manifest["learning_rate"]
            )

    def get_scheduler(self):
        """
        Get the scheduler for the optimizer. The scheduler is defined in the optimizer manifest.

        Returns:
            torch.optim.lr_scheduler: The scheduler object.
        """
        if self.optimizer_manifest.get("lr_scheduler", None):
            scheduler = getattr(
                torch.optim.lr_scheduler, self.optimizer_manifest["scheduler"]
            )
            return scheduler(
                self.optimizer, **self.optimizer_manifest["scheduler_kwargs"]
            )
        return None

    def get_early_stopping(self):
        """
        Get the early stopping callback. The early stopping callback is defined in the optimizer manifest.

        Returns:
            EarlyStopping: The early stopping callback object.
        """
        if self.optimizer_manifest.get("early_stopping", None):
            return _EarlyStopping(
                self.optimizer_manifest["early_stopping"]["patience"],
                self.optimizer_manifest["early_stopping"]["min_delta"],
            )
        return None

    def _get_loss_function(self):
        if self.loss_manifest["function"].lower() == "mse":
            return torch.nn.MSELoss()
        else:
            # try torchmetrics for more loss functions
            try:
                import torchmetrics

                return getattr(torchmetrics, self.loss_manifest["function"])()
            except ImportError:
                logger.info(
                    "torchmetrics not found. If you want to use its loss functions, please install it."
                )
            raise TrainerError(
                f"Loss function {self.loss_manifest['function']} not supported."
            )
        # TODO: Custom loss functions

    # train steps #####################################################################
    def train_step(self, batch):
        if self.transform_manifest["configuration"]["name"].lower() == "descriptor":
            return self._descriptor_train_step(batch)
        else:
            raise TrainerError(
                f"Configuration transformation type {self.transform_manifest['configuration']['name']} not supported."
            )

    def _descriptor_train_step(self, batch):
        self.optimizer.zero_grad()
        loss = self._descriptor_eval_batch(batch)
        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()
        return loss

    def validation_step(self, batch):
        if self.transform_manifest["configuration"]["name"].lower() == "descriptor":
            return self._descriptor_validation_step(batch)
        else:
            raise TrainerError(
                f"Configuration transformation type {self.transform_manifest['configuration']} not supported."
            )

    def _descriptor_validation_step(self, batch):
        return self._descriptor_eval_batch(batch)

    # eval batches #####################################################################
    @requires(lds, "libdescriptor is needed for descriptor training.")
    def _descriptor_eval_batch(self, batch):
        n_atoms = batch["n_atoms"]
        species = batch["species"]
        neigh_list = batch["neigh_list"]
        num_neigh = batch["num_neigh"]
        image = batch["image"]
        coords = batch["coords"]
        descriptors = batch["descriptors"]
        properties = batch["property_dict"]
        contribution = batch["contribution"]
        ptr = batch["ptr"]

        descriptor_tensor = torch.tensor(
            descriptors,
            dtype=self.dtype,
            device=self.current["device"],
            requires_grad=True,
        )
        predictions = self.model(descriptor_tensor)
        predictions = scatter_add(
            predictions,
            torch.tensor(
                contribution, device=self.current["device"], dtype=torch.int64
            ),
            dim=0,
        )

        loss = self.loss(
            predictions,
            torch.as_tensor(
                properties["energy"], dtype=self.dtype, device=self.current["device"]
            ),
        )  # energy will always be present for conservative models
        # TODO: Add per component loss extraction
        # TODO: Add per configuration loss extraction

        if self.loss_manifest["weights"]["energy"]:
            loss = loss * self.loss_manifest["weights"]["energy"]

        if (
            self.loss_manifest["weights"]["forces"]
            or self.loss_manifest["weights"]["stress"]
        ):
            dE_dzeta = torch.autograd.grad(
                predictions.sum(),
                descriptor_tensor,
                retain_graph=True,
            )[0]

            forces = lds.gradient_batch(
                self.configuration_transform._cdesc,
                n_atoms,
                ptr,
                species,
                neigh_list,
                num_neigh,
                coords,
                descriptors,
                dE_dzeta.double().detach().cpu().numpy(),
            )

            forces_predicted = torch.zeros(
                properties["forces"].shape,
                device=self.current["device"],
                dtype=self.dtype,
            )
            forces = torch.tensor(
                forces, device=self.current["device"], dtype=self.dtype
            )
            force_summed = scatter_add(
                forces,
                torch.tensor(image, device=self.current["device"], dtype=torch.int64),
                dim=0,
            )
            n_atoms_tensor = torch.tensor(
                n_atoms, device=self.current["device"], dtype=torch.int64
            )
            ptr_tensor = torch.tensor(
                ptr, device=self.current["device"], dtype=torch.int64
            )
            for i in range(len(ptr_tensor) - 1):
                from_ = torch.sum(n_atoms_tensor[:i])
                to_ = from_ + n_atoms_tensor[i]
                forces_predicted[from_:to_] = force_summed[
                    ptr_tensor[i] : ptr_tensor[i] + n_atoms_tensor[i]
                ]

            loss_forces = self.loss(
                forces_predicted,
                torch.tensor(
                    properties["forces"],
                    device=self.current["device"],
                    dtype=self.dtype,
                ),
            )
            # TODO: Add force loss dump feature for Josh's UQ
            loss = loss + loss_forces * self.loss_manifest["weights"]["forces"]
            # TODO: Discuss and check if this is correct
            # F = - ∂E/∂r, ℒ = f(E, F)
            # F = - ∂E/∂ζ * ∂ζ/∂r <- ∂ζ/∂r is a jacobian, vjp is computed by enzyme
            # ∂ℒ/∂θ = ∂ℒ/∂E * ∂E/∂θ + ∂ℒ/∂F * ∂F/∂θ
            # tricky part is ∂F/∂θ, as F is computed using Enzyme
            # ∂F/∂θ = ∂^2E/∂ζ∂θ * ∂ζ/∂r + ∂E/∂ζ * ∂^2ζ/∂r∂θ
            #       = ∂^2E/∂ζ∂θ * ∂ζ/∂r + 0 (∂ζ/∂r independent of θ)
            # So we do not need second derivative wrt to ζ,
            # and ∂ζ/∂r is provided by the descriptor module. So autograd should be able
            # to handle this. But need to confirm, else we need to explicitly compute
            # ∂^2E/∂ζ∂θ and then call lds.gradient again.
            # ask pytorch forum. Or use custom gradient optimization.
        # if self.loss_manifest["weights"]["stress"]:
        #     # stress = \sum_i (f_i \otimes r_i)
        #     stress = torch.zeros(len(ptr), 6)  # voigt notation
        #     for i in range(len(ptr) - 1):
        #         from_ = torch.sum(n_atoms[:i])
        #         to_ = from_ + n_atoms[i]
        #         full_stress = torch.einsum(
        #             "ij,ik->ijk", forces_predicted[from_:to_], coords[from_:to_]
        #         )
        #         summed_stress = torch.sum(full_stress, dim=0)
        #         stress[i, 0] = summed_stress[0, 0]
        #         stress[i, 1] = summed_stress[1, 1]
        #         stress[i, 2] = summed_stress[2, 2]
        #         stress[i, 3] = summed_stress[1, 2]
        #         stress[i, 4] = summed_stress[0, 2]
        #         stress[i, 5] = summed_stress[0, 1]
        #
        #     loss_stress = self.loss(stress, properties["stress"])
        #     loss = loss + loss_stress * self.loss_manifest["weights"]["stress"]

        return loss

    def train(self):
        # TODO: granularity of train: train, train_step, train_epoch?
        # currently it is train -> train_step, so train is wrapper for train_epoch
        if self.current["appending_to_previous_run"]:
            self.load_checkpoint(self.get_last_checkpoint())

        for epoch in range(self.optimizer_manifest["epochs"]):
            epoch_train_loss = 0.0
            self.model.train()
            for i, batch in enumerate(self.train_dataloader):
                loss = self.train_step(batch)
                epoch_train_loss += loss.detach().cpu().numpy()

            if epoch % self.current["ckpt_interval"] == 0:
                if self.val_dataset:
                    epoch_val_loss = 0.0
                    self.model.eval()
                    for batch in self.val_dataloader:
                        loss = self.validation_step(batch)
                        epoch_val_loss += loss.detach().cpu().numpy()

                    self.current["loss"] = {
                        "train": epoch_train_loss,
                        "val": epoch_val_loss,
                    }
                else:
                    self.current["loss"] = {"train": epoch_train_loss, "val": None}
                logger.info(f"Epoch {epoch} completed. val loss: {epoch_val_loss}")
                self.checkpoint()
                if self.early_stopping and self.early_stopping(epoch_val_loss):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            self.current["step"] += 1
            logger.info(f"Epoch {epoch} completed. Train loss: {epoch_train_loss}")

            # create .finished file to indicate that training is done
            with open(f"{self.current['run_dir']}/.finished", "w") as f:
                f.write("")

    # model io #####################################################################
    def setup_model(self):
        """
        Load the torchscript model from the model manifest. If model is provided, ignore the manifest.
        """
        torchscript_path = self.model_manifest["path"]
        model = torch.jit.load(torchscript_path)

        # change precision of model
        # if self.training_manifest["precision"] == "single":
        #     self.model = self.model.float()
        # elif self.training_manifest["precision"] == "double":
        #     self.model = self.model.double()
        # else:
        #     raise TrainerError(
        #         f"Precision {self.training_manifest['precision']} not supported."
        #     )
        self.model = model.to(device=self.current["device"], dtype=self.dtype)

    def save_kim_model(self, path: str = "kim-model"):
        """
        Save the KIM model to the given path. The KIM model is saved as a portable
        TorchML model.

        Args:
            path: Path to save the model
        """
        # create folder if not already present
        if self.export_manifest["model_path"]:
            path = self.export_manifest["model_path"]

        os.makedirs(path, exist_ok=True)

        best_model = self.model.load_state_dict(
            torch.load(f"{self.current['run_dir']}/checkpoints/best_model.pth")
        )

        model = torch.jit.script(best_model)

        model = model.cpu()
        torch.jit.save(model, f"{path}/model.pt")

        # save the configuration transform
        self.configuration_transform.export_kim_model(
            path, "model.pt"
        )  # kim_model.param
        self.configuration_transform.save_descriptor_state(path)  # descriptor.dat

        # CMakeLists.txt
        if not self.export_manifest["model_name"]:
            qualified_model_name = f"{self.current['run_title']}_MO_000000000000_000"
        else:
            qualified_model_name = self.export_manifest["model_name"]

        cmakefile = self._generate_kim_cmake(
            qualified_model_name,
            "TorchML__MD_173118614730_000",
            ["model.pt", "descriptor.dat", "kim_model.param"],
        )

        with open(f"{path}/CMakeLists.txt", "w") as f:
            f.write(cmakefile)

        # write training environment
        self.write_training_env_edn(path)

        if self.export_manifest["generate_tarball"]:
            tarball_name = f"{path}.tar.gz"
            with tarfile.open(tarball_name, "w:gz") as tar:
                tar.add(path, arcname=os.path.basename(path))
            logger.info(f"Model tarball saved: {tarball_name}")
        logger.info(f"KIM model saved at {path}")

    def get_torchscript_model_from_tar(self):
        tarfile_path = self.model_manifest["path"]
        with tarfile.open(tarfile_path, "r") as tar:
            torchscript_file = None
            for member in tar.getmembers():
                if member.name.endswith(".pt") or member.name.endswith(".pth"):
                    torchscript_file = member
                    break
            if torchscript_file is None:
                raise TrainerError(f"No torchscript file found in {tarfile_path}.")
            else:
                tar.extract(torchscript_file, path=self.current["run_dir"])
                self.torchscript_file = (
                    f"{self.current['run_dir']}/{torchscript_file.name}"
                )

    # Data loaders #####################################################################
    def setup_dataloaders(self):
        if self.transform_manifest["configuration"]["name"].lower() == "descriptor":
            self._setup_descriptor_dataloaders()
        else:
            raise TrainerError(
                f"Configuration transformation type {self.transform_manifest['configuration']['name']} not supported."
            )

    def _setup_descriptor_dataloaders(self):
        if self.dataset_manifest["dynamic_loading"]:
            self.train_dataset = DescriptorDataset(self.train_dataset)

            if self.val_dataset:
                self.val_dataset = DescriptorDataset(self.val_dataset)

            self.train_dataset.add_transform(self.configuration_transform)
            if self.val_dataset:
                self.val_dataset.add_transform(self.configuration_transform)
        else:
            for config in self.train_dataset:
                config.fingerprint = self.configuration_transform(
                    config, return_extended_state=True
                )

            self.train_dataset = DescriptorDataset(self.train_dataset)

            if self.val_dataset:
                for config in self.val_dataset:
                    config.fingerprint = self.configuration_transform(
                        config, return_extended_state=True
                    )

                self.val_dataset = DescriptorDataset(self.val_dataset)

        self.train_dataloader = TorchDataLoader(
            self.train_dataset,
            batch_size=self.optimizer_manifest["batch_size"],
            shuffle=True,
            collate_fn=self.train_dataset.collate,
        )
        if self.val_dataset:
            self.val_dataloader = TorchDataLoader(
                self.val_dataset,
                batch_size=self.optimizer_manifest["batch_size"],
                shuffle=False,
                collate_fn=self.val_dataset.collate,
            )
        else:
            logger.warning("No validation dataset loaded.")

    # Auxilliary #####################################################################
    def setup_parameter_transforms(self):
        # no parameter transforms for torch models, yet.
        pass


class _EarlyStopping:
    def __init__(self, patience: int, delta: float):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float("inf")

    def __call__(self, loss):
        if loss < self.best_loss - self.delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            return True
        return False

    # TODO:
    # - Add device management
    # - Add model export
    # - Add restart capabilities
    # - Add per component loss extraction
    # - Add custom options for torchmetrics
    # - Precision management
    # - DDP?
