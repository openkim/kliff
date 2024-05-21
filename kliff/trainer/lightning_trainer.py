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

from pytorch_lightning.loggers import TensorBoardLogger

from kliff.dataset import Dataset
from kliff.utils import get_n_configs_in_xyz

from .torch_trainer_utils.dataloaders import GraphDataset

try:
    from torch_ema import ExponentialMovingAverage

    is_torch_ema_present = True
except:
    is_torch_ema_present = False

import hashlib


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
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer_name)(
            self.model.parameters(), lr=self.lr
        )
        return optimizer

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)

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
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
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


class GNNLightningTrainer(Trainer):
    def __init__(self, manifest, model):
        self.pl_model: LightningTrainerWrapper = None
        self.data_module = None

        super().__init__(manifest, model)

        self.tb_logger = self._tb_logger()
        self.setup_dataloaders()
        self.pl_trainer = self._get_pl_trainer()

    def setup_model(self):
        # if dict has key ema, then set ema to True, decay to the dict value, else set ema false
        ema = True if self.optimizer_manifest.get("ema", False) else False
        if ema:
            ema_decay = self.optimizer_manifest.get("ema_decay", 0.99)
        else:
            ema_decay = None

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
        )

    def setup_dataset(self):
        """
        Set up the dataset based on the provided information. Override this method to
        provide parallel data loading for the dataset.
        TODO: Integrate parallel processing in `get_dataset_from_manifest` method.
        TODO: It will check the {workspace}/{dataset_name} directory to see if dataset hash
        is already there. The dataset hash is determined but hashing the full path to
        the dataset + transforms names + configuration transform properties.
        TODO: reload hashed dataset if it exists.
        """
        dataset_type = self.dataset_manifest.get("type").lower()
        dataset_immutable_str = json.dumps(
            self.dataset_manifest | self.transform_manifest, sort_keys=True
        )
        dataset_hash = hashlib.md5(dataset_immutable_str.encode()).hexdigest()

        self.current["dataset_hash"] = dataset_hash  # TODO: move it in base trainer

        # check if dataset hash named dill file exists in the workspace
        if os.path.exists(f"{self.current['data_dir']}/{dataset_hash}.dill"):
            logger.info(
                f"Dataset hash {dataset_hash} found. Loading dataset from cache."
            )
            self.dataset = dill.load(
                open(f"{self.current['data_dir']}/{dataset_hash}.dill", "rb")
            )

        else:
            if dataset_type != "ase":
                super().setup_dataset()
                return
            else:
                logger.warning(
                    "Using parallel processing for dataset loading. Indices order may not be preserved."
                )
                # num_chunks = n_workers?
                dataset_chunks = _parallel_read_and_transform(
                    self.dataset_manifest["path"],
                    self.optimizer_manifest["num_workers"],  # num_chunks
                    energy_key=self.dataset_manifest.get("keys", {}).get(
                        "energy", "energy"
                    ),
                    forces_key=self.dataset_manifest.get("keys", {}).get(
                        "forces", "forces"
                    ),
                    transform_manifest=self.transform_manifest,
                )
                self.dataset = deepcopy(dataset_chunks[0])
                for ds in dataset_chunks[1:]:
                    self.dataset.configs.extend(ds)

                # save the dataset to the data_dir, it should exist at base class init
                dill.dump(
                    self.dataset,
                    open(f"{self.current['data_dir']}/{dataset_hash}.dill", "wb"),
                )
                # del dataset_chunks

    def train(self):
        self.pl_trainer.fit(self.pl_model, self.data_module)

    def setup_dataloaders(self):
        self.train_dataset = GraphDataset(self.train_dataset)
        self.val_dataset = GraphDataset(self.val_dataset)
        self.data_module = LightningDataset(
            self.train_dataset,
            self.val_dataset,
            batch_size=self.optimizer_manifest["batch_size"],
            num_workers=self.optimizer_manifest["num_workers"],
        )
        logger.info("Data modules setup complete.")

    def _tb_logger(self):
        return TensorBoardLogger(self.current["run_dir"], name="lightning_logs")

    def _get_pl_trainer(self):
        return pl.Trainer(
            logger=self.tb_logger,
            max_epochs=self.optimizer_manifest["epochs"],
            accelerator="auto",
            strategy="ddp",
        )

    def setup_optimizer(self):
        # Not needed as Pytorch Lightning handles the optimizer
        pass


# Parallel processing for dataset loading #############################################
def _parallel_read_and_transform(
    file_path,
    num_chunks=None,
    energy_key="Energy",
    forces_key="forces",
    transform_manifest=None,
) -> List[Dataset]:
    """
    Read and transform frames in parallel. Returns n_chunks of datasets.
    Args:
        file_path:
        num_chunks:
        energy_key:
        forces_key:
        transform:

    Returns:
        List of datasets processed by each node

    """
    if not num_chunks:
        num_chunks = multiprocessing.cpu_count()
    total_frames = get_n_configs_in_xyz(file_path)
    frames_per_chunk = total_frames // num_chunks

    # Generate slices for each chunk
    chunks = []
    for i in range(num_chunks):
        start = i * frames_per_chunk
        end = (i + 1) * frames_per_chunk if i < num_chunks - 1 else total_frames
        chunks.append((start, end))

    with multiprocessing.Pool(processes=num_chunks) as pool:
        ds = pool.starmap(
            _read_and_transform_frames,
            [
                (file_path, start, end, energy_key, forces_key, transform_manifest)
                for start, end in chunks
            ],
        )

    return ds


def _read_and_transform_frames(
    file_path, start, end, energy_key, forces_key, transform_manifest=None
):
    ds = Dataset.from_ase(
        path=file_path,
        energy_key=energy_key,
        forces_key=forces_key,
        slices=slice(start, end),
    )
    if transform_manifest:
        configuration_transform: Union[dict, None] = transform_manifest.get(
            "configuration", None
        )
        # property_transform: Union[list, None] = transform_manifest.get(
        #     "property", None
        # ) # Figure out how dataset wide properties can be parallelized
        # TODO: Property transform. Simply add one more transform to the function
        if configuration_transform:
            configuration_module_name: Union[str, None] = configuration_transform.get(
                "name", None
            )
            if configuration_module_name == "Graph":
                configuration_module_name = "KIMDriverGraph"
            if not configuration_module_name:
                logger.warning(
                    "Configuration transform module name not provided."
                    "Skipping configuration transform."
                )
                configuration_module = None
            else:
                configuration_transform_module = importlib.import_module(
                    f"kliff.transforms.configuration_transforms"
                )
                configuration_module = getattr(
                    configuration_transform_module, configuration_module_name
                )
                kwargs: Union[dict, None] = configuration_transform.get("kwargs", None)
                if not kwargs:
                    raise TrainerError(
                        "Configuration transform module options not provided."
                    )
                configuration_module = configuration_module(
                    **kwargs, copy_to_config=True
                )
        else:
            configuration_module = None
        if configuration_module:
            for config in ds.get_configs():
                _ = configuration_module(config, return_extended_state=True)
        del configuration_module
    return ds
