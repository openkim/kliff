import os
import sys
import tarfile

import numpy as np
import torch
import torch_geometric as pyg
from multiprocessing import Pool
from kliff.neighbor import NeighborList
from functools import partial
from operator import itemgetter
from torch_scatter import scatter_add

from loguru import logger
from kliff.dataset import Dataset, Configuration

from .kliff_trainer import *
from kliff.transforms.configuration_transforms import KLIFFTorchGraphGenerator, Descriptor, KLIFFTorchGraph

try:
    from torch_ema import ExponentialMovingAverage
except ImportError:
    ExponentialMovingAverage = None


# def NeighborListTransform(cutoff: float):
#     return partial(NeighborList, cutoff=cutoff)


class KLIFFTorchGeometricDataset(pyg.data.InMemoryDataset):
    def __init__(self, transform=None, pre_transform=None, path=None, data_list=None):
        super().__init__(None, transform, pre_transform)
        if data_list:
            self.data, self.slices = self.collate(data_list)
        elif path:
            self.data, self.slices = torch.load(path)
        else:
            raise TrainerError("Invalid dataset initialization.")


class TorchTrainer(Trainer):
    def __init__(self, configuration: dict):
        super().__init__(configuration)
        self.start()
        self.get_model()
        self.current_epoch = 0
        self.current_val_loss = 0
        self.current_train_loss = 0
        self.current_val_predictions = None
        self.lr_scheduler = None
        self.ema = None
        self.get_optimizer()
        # save configuration
        self.history = {"train_loss": [], "val_loss": []}
        self.train_batch_count = 0
        self.val_batch_count = 0
        self.best_epoch = 0
        self.epoch_without_improvement = {"val": 0, "train": 0}
        self.use_torch_geometric_dataloader = False
        self.configuration_transform = self.get_configuration_transform()
        self.train_dataset = None
        self.val_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.input_keys_list = None
        self.get_dataset()

    def start(self):
        os.makedirs(self.current_run_dir, exist_ok=True)
        logger.info(f"Made training directory: {self.current_run_dir}")
        self.to_file(f"{self.current_run_dir}/configuration.yaml")

    def get_configuration_transform(self):
        try:
            transform = self.configuration["configuration_transformation"]["transform"]
            if transform:
                transform_type = ConfigurationTransformationTypes.get_config_transformation_type(transform)
            else:
                return None
        except KeyError:
            return None
        if transform_type == ConfigurationTransformationTypes.GRAPH:
            if self.configuration["configuration_transformation"]["use_torch_geometric_dataloader"]:
                self.use_torch_geometric_dataloader = True
            logger.info("Initializing Configuration Transforms")
            return KLIFFTorchGraphGenerator(**self.configuration["configuration_transformation"]["kwargs"])
        elif transform_type == ConfigurationTransformationTypes.DESCRIPTORS:
            logger.info("Initializing Configuration Transforms")
            return Descriptor(**self.configuration["configuration_transformation"]["kwargs"])
        elif transform_type == ConfigurationTransformationTypes.NEIGHBORS:
            logger.info("Initializing Configuration Transforms")
            return None
            # NeighborListTransform(**self.configuration["configuration_transformation"]["kwargs"])
        else:
            raise TrainerError("Invalid configuration transformation.")

    def loss(self, energy_prediction, energy_target, forces_prediction, forces_target):
        loss = (
            torch.sum((energy_prediction - energy_target) ** 2)
            * self.energy_loss_weight
        )
        loss += (
            torch.sum((forces_prediction - forces_target) ** 2)
            * self.forces_loss_weight
        )
        return loss

    def _checkpoint(self, checkpoint_name, overwrite=False):
        current_chkpt_path = f"{self.current_run_dir}/checkpoints/{checkpoint_name}"
        try:
            os.makedirs(current_chkpt_path)
        except FileExistsError:
            if overwrite:
                os.remove(f"{current_chkpt_path}/model.pt")
                os.remove(f"{current_chkpt_path}/optimizer.pt")
            else:
                raise TrainerError("Checkpoint already exists.")
        torch.save(self.model.state_dict(), f"{current_chkpt_path}/model.pt")
        torch.save(self.optimizer.state_dict(), f"{current_chkpt_path}/optimizer.pt")

    def checkpoint(self):
        self._checkpoint(f"last_model", overwrite=True)
        best_epoch = np.argmin(self.history["val_loss"])
        if best_epoch > self.best_epoch:
            self.best_epoch = best_epoch
            self._checkpoint(f"best_model", overwrite=True)
        if self.current_epoch % self.checkpoint_freq == 0:
            self._checkpoint(f"epoch_{self.current_epoch:06d}", overwrite=False)

    def log_errors(self):
        np.savetxt(
            f"{self.current_run_dir}/val_predictions.txt", self.current_val_predictions
        )
        with open(f"{self.current_run_dir}/loss.txt", "a+") as f:
            f.write(
                f"{self.current_epoch}\t{self.current_train_loss}\t{self.current_val_loss}\n"
            )

    def get_optimizer(self):
        if self.configuration["optimizer_provider"] == OptimizerProvider.TORCH:
            # initialize optimizer from the name
            optimizer_name = self.configuration["optimizer"]
            optimizer = getattr(torch.optim, optimizer_name)
            self.optimizer = optimizer(
                self.model.parameters(), **self.configuration["optimizer_kwargs"]
            )
            logger.info("Initialized optimizer")
            if self.configuration["lr_scheduler"]:
                self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, **self.configuration["lr_scheduler"]
                )
                logger.info("Initialized lr_scheduler")
            if self.configuration["use_ema"]:
                if ExponentialMovingAverage is None:
                    raise TrainerError(
                        "torch_ema is not installed but ema is requested."
                    )
                self.ema = ExponentialMovingAverage(
                    self.model.parameters(), decay=self.configuration["ema_decay"]
                )
                logger.info("Initialized ema")
        else:
            raise TrainerError("Optimizer provider not implemented.")

    def optimizer_step(self):
        self.optimizer.step()
        if self.ema is not None:
            self.ema.update(self.model.parameters())

    def get_model(self):
        if self.configuration["model_source"] == ModelTypes.TORCH:
            # initialize model from the name
            model_name = self.configuration["model_name"]
            model = torch.jit.load(model_name + ".pt", map_location="cpu")
            self.model = model
            logger.info(f"Loaded TORCH model to train")
        elif self.configuration["model_source"] == ModelTypes.TAR:
            # initialize model from the name
            model_name = self.configuration["model_name"]
            tarfile.open(model_name, "r:gz").extractall(f"{self.current_run_dir}/model")
            model = torch.jit.load(
                f"{self.current_run_dir}/model/{model_name}.pt", map_location="cpu"
            )
            self.model = model
            logger.info(f"Loaded TAR model to train")
        else:
            raise TrainerError("Model provider not implemented.")

    def get_dataset(self):  # Specific to trainer
        if self.configuration["dataset_type"] == DataTypes.TORCH_GEOMETRIC:
            dataset = KLIFFTorchGeometricDataset(
                path=self.configuration["dataset_path"]
            )
            logger.info(f"Loaded TORCH GEOMETRIC type dataset.")
        # elif self.configuration["dataset_type"] == DataTypes.TORCH:
        #     dataset = torch.load(self.configuration["dataset_path"])
        elif self.configuration["dataset_type"] == DataTypes.ASE:
            dataset = Dataset(
                self.configuration["dataset_path"],
                parser="ase",
                energy_key=self.configuration["energy_key"],
                forces_key=self.configuration["forces_key"],
            )
            logger.info(f"Loaded ASE type dataset.")
        elif self.configuration["dataset_type"] == DataTypes.KLIFF:
            dataset = Dataset(self.configuration["dataset_path"])
            logger.info(f"Loaded KLIFF type dataset.")
        elif self.configuration["dataset_type"] == DataTypes.COLABFIT:
            dataset = Dataset(
                colabfit_dataset=self.configuration["dataset_name"],
                colabfit_database=self.configuration["database_name"],
            )
            logger.info(f"Loaded COLABFIT type dataset.")
        else:
            raise TrainerError("Dataset provider not implemented.")

        dataset_size = len(dataset)
        self.get_indices(dataset_size)

        # if self.configuration["configuration_transformation"] == ConfigurationTransformationTypes.GRAPH:
        #     if self.configuration["configuration_transformation"]["as_torch_geometric_data"]:
        #         dataloader_provider = pyg.data.DataLoader
        #     else:
        #         dataloader_provider = torch.utils.data.DataLoader
        # else:
        #     dataloader_provider = torch.utils.data.DataLoader

        if isinstance(dataset, Dataset) and self.use_torch_geometric_dataloader:
            # if we need to use torch geometric dataloader then map the dataset to torch
            # geometric in-memory dataset first.

            # smp_pool = Pool(self.configuration["cpu_workers"])
            # dataset = smp_pool.map(self.configuration_transform, dataset.get_configs())
            # smp_pool.close()
            # smp_pool.join()

            dataset = list(map(self.configuration_transform, dataset.get_configs()))
            dataset = KLIFFTorchGeometricDataset(data_list=dataset)
            logger.info("Converted dataset to torch geometric in-memory dataset.")

        if isinstance(dataset, Dataset):
            dataset_list = dataset.get_configs()
            dataset_train = [dataset_list[i] for i in self.train_indices]
            dataset_val = [dataset_list[i] for i in self.val_indices]
            dataloader_train = torch.utils.data.DataLoader(
                dataset_train,
                batch_size=self.configuration["batch_size"],
                shuffle=True,
                num_workers=self.configuration["cpu_workers"],
                collate_fn=[self.configuration_transform.collate_fn if self.configuration_transform else lambda x: x][0],
            )
            dataloader_val = torch.utils.data.DataLoader(
                dataset_val,
                batch_size=self.configuration["batch_size"],
                shuffle=True,
                num_workers=self.configuration["cpu_workers"],
                collate_fn=[self.configuration_transform.collate_fn if self.configuration_transform else lambda x: x][0],
            )
            logger.info("Loaded dataset into torch dataloader.")
        else:
            dataset_train = dataset[self.train_indices]
            dataset_val = dataset[self.val_indices]
            dataloader_train = pyg.loader.DataLoader(
                dataset_train,
                batch_size=self.configuration["batch_size"],
                shuffle=True,
                num_workers=self.configuration["cpu_workers"],
            )
            dataloader_val = pyg.loader.DataLoader(
                dataset_val,
                batch_size=self.configuration["batch_size"],
                shuffle=False,
                num_workers=self.configuration["cpu_workers"],
            )
            logger.info("Loaded dataset into torch geometric dataloader.")

        self.train_dataset = dataset_train
        self.val_dataset = dataset_val

        self.train_dataloader = dataloader_train
        self.val_dataloader = dataloader_val
        # total number of batches
        self.train_batch_count = len(self.train_dataloader)
        self.val_batch_count = len(self.val_dataloader)

    def stop_now(self):
        # hard stops conditions
        now = datetime.now()
        if now > self.end_time or self.current_epoch > self.configuration["max_epoch"]:
            return True

        # early stopping conditions
        if (
            self.history["val_loss"][self.best_epoch] - self.history["val_loss"][-1]
        ) > float(self.configuration["early_stopping_minimum_delta"]["validation"]):
            self.epoch_without_improvement["val"] = 0
        else:
            self.epoch_without_improvement["val"] += 1

        if (
            self.history["train_loss"][self.best_epoch] - self.history["train_loss"][-1]
        ) > float(self.configuration["early_stopping_minimum_delta"]["training"]):
            self.epoch_without_improvement["train"] = 0
        else:
            self.epoch_without_improvement["train"] += 1

        if (
            self.epoch_without_improvement["val"]
            > self.configuration["early_stopping_patience"]["validation"]
            or self.epoch_without_improvement["train"]
            > self.configuration["early_stopping_patience"]["training"]
        ):
            finish_time = datetime.now()
            with open(f"{self.current_run_dir}/.finished", "w") as f:
                f.write(f"Start time {self.start_time.strftime('%Y-%m-%d-%H-%M-%S')}\n")
                f.write(f"Finish time {finish_time.strftime('%Y-%m-%d-%H-%M-%S')}\n")
                f.write(
                    f"Total time {(finish_time - self.start_time).strftime('%Y-%m-%d-%H-%M-%S')}\n"
                )
            return True

    def get_model_inputs(self, batch):
        """ As per supported models, the model input can be
        1. KLIFFTorchGraphBatch: GNN, model(species, coords, graph1, graph2,..., contributing)
        2. Torch Tensor: Descriptor based models, model(descriptor)
        3. NeighborList: Free-reign models, model(species, coords, n_neigh, nlist, contributing)

        """
        batch_type = type(batch).__name__ # This was the simplest way,
        # Pytorch Geometric graphs are bit weired, they are of type "GlobalStorage" and "KLIFFTorchGraphBatch"
        # depending on how you get the batch name. So best to convert it to string and check.
        # also save additional dependency at import time
        if batch_type == "KLIFFTorchGraphBatch":
            n_layers = self.configuration["configuration_transformation"]["kwargs"]["n_layers"]
            layer_names = [f"edge_index{i}" for i in range(n_layers)]
            input_fields = ["species", "coords", *layer_names, "batch"]
            input_dict = {key: batch[key] for key in input_fields}
            input_dict["coords"].requires_grad_(True)
            return input_dict
        elif batch_type == "Tensor":
            return batch
        elif batch_type == "list":
            batch_list = []
            for config in batch:
                nl = NeighborList(config, infl_dist=self.configuration["configuration_transformation"]["kwargs"]["cutoff"])
                n_neigh, neigh_list = nl.get_numneigh_and_neighlist_1D()
                coords = nl.get_coords()
                species = nl.get_species_indexes()
                contributing = np.ones(coords.shape[0])
                contributing[0:config.get_num_atoms()] = 0
                batch_list.append({
                    "species": torch.tensor(species),
                    "coords": torch.tensor(coords, requires_grad=True),
                    "n_neigh": torch.tensor(n_neigh),
                    "neigh_list": torch.tensor(neigh_list),
                    "contributing": torch.tensor(contributing, dtype=torch.int)
                })
            return batch_list
        else:
            raise TrainerError("Invalid model input type.")

    def train_step(self, batch):
        model_inputs = self.get_model_inputs(batch)
        self.model.train()
        self.optimizer.zero_grad()
        energy_prediction, forces_prediction = self.model(**model_inputs)
        forces_prediction = self.sum_forces(forces_prediction, batch)
        loss = self.loss(
            energy_prediction, batch.energy, forces_prediction, batch.forces
        )
        loss.backward()
        self.optimizer_step()
        self.current_train_loss += loss.item()

    def validation_step(self, batch):
        model_inputs = self.get_model_inputs(batch)
        # self.model.eval()
        # with torch.no_grad():
        energy_prediction, forces_prediction = self.model(**model_inputs)
        forces_prediction = self.sum_forces(forces_prediction, batch)
        self.current_val_loss += self.loss(
            energy_prediction, batch.energy, forces_prediction, batch.forces
        ).item()
        self.current_val_predictions = np.append(
            self.current_val_predictions,
            energy_prediction.squeeze().detach().numpy(),
        )

    def sum_forces(self, forces, batch):
        if self.configuration["configuration_transformation"]["transform"].lower() == "graph":
            forces_summed = scatter_add(forces, batch.images, 0)
            return forces_summed
        else:
            pass
        # elif self.configuration_transform == ConfigurationTransformationTypes.NEIGHBORS:
        #     return 0

    def train(self):
        self.current_train_loss = 0
        for batch in self.train_dataloader:
            self.train_step(batch)
            self.optimizer_step()
        self.history["train_loss"].append(
            self.current_train_loss / self.train_batch_count
        )

        self.current_val_predictions = np.array([])
        self.current_val_loss = 0
        for batch in self.val_dataloader:
            self.validation_step(batch)
        self.history["val_loss"].append(self.current_val_loss / self.val_batch_count)

        # clear out accumulated gradients in validation
        self.optimizer.zero_grad()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(self.history["val_loss"][-1])

        self.checkpoint()

        self.current_epoch += 1
        if self.stop_now():
            logger.info("Exiting training loop, one of the exit conditions reached.")
            return True
        else:
            logger.info(f"Epoch {self.current_epoch} completed.")
            self.train()
