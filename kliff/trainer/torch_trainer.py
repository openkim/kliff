import os
import tarfile

import torch
from monty.dev import requires
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader

from torch_scatter import scatter_add

from loguru import logger

from .base_trainer import Trainer, TrainerError

from .torch_trainer_utils.dataloaders import (
    DescriptorDataset,
    GraphDataset,
    NeighborListDataset,
)

try:
    import libdescriptor as lds
except ImportError:
    lds = None


class TorchTrainer(Trainer):
    """
    This class extends the base KLIFF trainer class to implement and train MLIPs compatible
    with the TorchML model driver.
    """

    def __init__(self, configuration: dict, model=None):
        self.torchscript_file = None
        self.train_dataloader = None
        self.validation_dataloader = None
        self.model = model
        # self.base_model = None
        super().__init__(configuration)

        self.loss_function = self._get_loss_function()
        self.setup_dataloaders()

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
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "current_step": self.current["step"],
                "current_loss": self.current["loss"],
            },
            f"{self.current['run_dir']}/checkpoint_{self.current['step']}.pkl",
        )
        with open(f"{self.current['run_dir']}/log.txt", "a") as f:
            f.write(
                f"Step: {self.current['step']}, Train Loss: {self.current['loss']['train']}, Val Loss: {self.current['loss']['val']}\n"
            )

    def get_optimizer(self, *args, **kwargs):
        if self.optimizer_manifest["provider"] == "torch":
            raise TrainerError(
                f"Optimizer provider {self.optimizer_manifest['provider']} not supported."
            )
        optimizer = getattr(torch.optim, self.optimizer["name"])
        if self.optimizer["kwargs"]:
            return optimizer(
                self.model.parameters(), lr=self.optimizer_manifest["learning_rate"], **self.optimizer["kwargs"]
            )
        else:
            return optimizer(self.model.parameters(), lr=self.optimizer_manifest["learning_rate"])
        # TODO: Scheduler and ema

    def _get_loss_function(self):
        if self.loss_manifest["function"].lower() == "mseloss" or self.loss_manifest["function"].lower() == "mse":
            return torch.nn.MSELoss()
        else:
            raise TrainerError(
                f"Loss function {self.loss_manifest['function']} not supported."
            )
        # TODO: Custom loss functions

    # train steps #####################################################################
    def train_step(self, batch):
        if self.transform_manifest["configuration"].lower() == "graph":
            return self._graph_train_step(batch)
        elif self.transform_manifest["configuration"].lower() == "descriptor":
            return self._descriptor_train_step(batch)
        elif self.transform_manifest["configuration"].lower() == "neighbors":
            return self._general_train_step(batch)
        else:
            raise TrainerError(
                f"Configuration transformation type {self.transform_manifest['configuration']} not supported."
            )

    def _descriptor_train_step(self, batch):
        self.optimizer.zero_grad()
        loss = self._descriptor_eval_batch(batch)
        loss.backward()
        return loss

    def _graph_train_step(self, batch):
        pass

    def _general_train_step(self, batch):
        pass

    def validation_step(self, batch):
        if self.transform_manifest["configuration"].lower() == "graph":
            return self._graph_validation_step(batch)
        elif self.transform_manifest["configuration"].lower() == "descriptor":
            return self._descriptor_validation_step(batch)
        elif self.transform_manifest["configuration"].lower() == "neighbors":
            return self._general_validation_step(batch)
        else:
            raise TrainerError(
                f"Configuration transformation type {self.transform_manifest['configuration']} not supported."
            )

    def _descriptor_validation_step(self, batch):
        return self._descriptor_eval_batch(batch)

    def _graph_validation_step(self, batch):
        pass

    def _general_validation_step(self, batch):
        pass

    # eval batches #####################################################################
    @requires(lds, "libdescriptor is needed for descriptor training.")
    def _descriptor_eval_batch(self, batch):
        batch = batch.to(self.current["device"])
        n_atoms = batch["n_atoms"]
        species = batch["species"]
        neigh_list = batch["neigh_list"]
        num_neigh = batch["num_neigh"]
        image = batch["image"]
        coords = batch["coords"]
        descriptors = batch["descriptors"]
        config = batch["config"]
        properties = batch["property_dict"]
        contribution = batch["contribution"]
        ptr = batch["ptr"]

        descriptors.requires_grad_(True)
        predictions = self.model(descriptors)

        loss = self.loss(predictions, properties["energy"]) # energy will always be present for conservative models

        # TODO: Add per component loss extraction
        # TODO: Add per configuration loss extraction
        if self.loss_manifest["weights"]["energy"]:
            loss = loss * self.loss_manifest["weights"]["energy"]

        if self.loss_manifest["weights"]["forces"]:
            dE_dzeta = torch.autograd.grad(
                predictions, descriptors, grad_outputs=torch.ones_like(predictions)
            )[0]

            forces = lds.gradient(self.configuration_transform._cdesc,
                              torch.sum(n_atoms).detach().cpu().numpy(),
                              species.detach().cpu().numpy(),
                              neigh_list.detach().cpu().numpy(),
                              num_neigh.detach().cpu().numpy(),
                              coords.detach().cpu().numpy(),
                              descriptors.detach().cpu().numpy(),
                              dE_dzeta.detach().cpu().numpy()
            )
            forces_predicted = torch.zeros_like(properties["forces"])
            force_summed = scatter_add(forces, image, dim=0)
            for i in range(len(ptr) - 1):
                from_ = torch.sum(n_atoms[:i])
                to_ = from_ + n_atoms[i]
                forces_predicted[from_, to_] = force_summed[ptr[i]: ptr[i] + n_atoms[i]]

            loss_forces = self.loss(forces_predicted, properties["forces"]) # TODO: Add force loss dump feature for Josh's UQ
            loss = loss + loss_forces * self.loss_manifest["weights"]["forces"]
        if self.loss_manifest["weights"]["stress"]:
            pass # TODO: Add stress loss dump feature for Josh's UQ

        return loss

    def _graph_eval_batch(self, batch):
        pass

    def _general_eval_batch(self, batch):
        pass

    def train(self, *args, **kwargs):
        # TODO: granularity of train: train, train_step, train_epoch?
        # currently it is train -> train_step, so train is wrapper for train_epoch
        for epoch in range(self.current["epochs"]):
            epoch_train_loss = 0.0
            self.model.train()
            for batch in self.train_dataloader:
                loss = self.train_step(batch)
                epoch_train_loss += loss.detach().cpu().numpy()

            if epoch % self.current["ckpt_interval"] == 0:
                epoch_val_loss = 0.0
                self.model.eval()
                for batch in self.val_dataloader:
                    loss = self.validation_step(batch)
                    epoch_val_loss += loss.detach().cpu().numpy()

                self.current["loss"] = {"train": epoch_train_loss, "val": epoch_val_loss}
                self.checkpoint()
            self.current["step"] += 1
            logger.info(f"Epoch {epoch} completed.")

    # model io #####################################################################
    def setup_model(self):
        if not self.model:
            if self.model_manifest["type"].lower() == "tar":
                self.get_torchscript_model_from_tar()
            elif self.model_manifest["type"].lower() == "torch":
                self.torchscript_file = self.model_manifest["path"]
            else:
                raise TrainerError(f"Model type {self.model_manifest['type']} not supported.")
            self.model = torch.jit.load(self.torchscript_file)
        else:
            logger.warning("Model already provided. Ignoring model manifest.")
            self.model_manifest["type"] = "pytorch"

        self.model.to(self.current["device"])

    def save_kim_model(self):
        if self.model_manifest["type"].loweR() == "tar":
            # save the torchscript model in tar file again
            self.current["best_model"].save(self.torchscript_file)
            with tarfile.open(self.model_manifest["path"], "w") as tar:
                tar.add(self.torchscript_file, arcname="model.pt")
        elif self.model_manifest["type"] == "torch" or self.model_manifest["type"] == "pytorch":
            model_pt = torch.jit.script(self.current["best_model"])
            model_path = f"{self.export_manifest['model_path'] }/{self.export_manifest['model_name']}"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model_pt.save(f"{model_path}/model.pt")
            # TODO: add remaining files to make it complete model
        else:
            raise TrainerError(f"Model type {self.model_manifest['type']} not supported.")

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
        if self.transform_manifest["configuration"]["name"].lower() == "graph":
            self._setup_graph_dataloaders()
        elif self.transform_manifest["configuration"]["name"].lower() == "descriptor":
            self._setup_descriptor_dataloaders()
        elif self.transform_manifest["configuration"]["name"].lower() == "neighbors":
            self._setup_general_dataloaders()
        else:
            raise TrainerError(
                f"Configuration transformation type {self.transform_manifest['configuration']['name']} not supported."
            )

    def _setup_graph_dataloaders(self):
        self.train_dataset = GraphDataset(self.train_dataset)
        self.val_dataset = GraphDataset(self.val_dataset)

        self.train_dataloader = GeometricDataLoader(
            self.train_dataset, batch_size=self.optimizer_manifest["batch_size"], shuffle=self.dataset_manifest["shuffle"]
        )
        self.val_dataloader = GeometricDataLoader(
            self.val_dataset, batch_size=self.optimizer_manifest["batch_size"], shuffle=False
        )

    def _setup_descriptor_dataloaders(self):
        self.train_dataset = DescriptorDataset(self.train_dataset)
        self.val_dataset = DescriptorDataset(self.val_dataset)

        self.train_dataloader = TorchDataLoader(
            self.train_dataset,
            batch_size=self.optimizer_manifest["batch_size"],
            shuffle=self.dataset_manifest["shuffle"],
            collate_fn=self.train_dataset.collate,
        )
        if self.val_dataset:
            print("here:", self.val_dataset)
            self.val_dataloader = TorchDataLoader(
                self.val_dataset,
                batch_size=self.optimizer_manifest["batch_size"],
                shuffle=False,
                collate_fn=self.val_dataset.collate,
            )
        else:
            logger.warning("No validation dataset loaded.")

    def _setup_general_dataloaders(self):
        pass

    # Auxilliary #####################################################################
    def setup_parameter_transforms(self):
        # no parameter transforms for torch models, yet.
        pass

    # TODO:
    # - Add device management
    # - Add model export
    # - Add GNN support
    # - Add restart capabilities
    # - Add per component loss extraction
