# Mostly to be used by GNNs for now
# check torch version, if <= 1.13, use torch_geometric.data.lightning_module
# This is temporary fix till torch 1 -> 2 migration is complete
import importlib.metadata
import os
import tarfile
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from loguru import logger
from torch_scatter import scatter_add

from .base_trainer import Trainer, TrainerError

if importlib.metadata.version("torch") <= "1.13":
    from torch_geometric.data.lightning_datamodule import LightningDataset
else:
    from torch_geometric.data.lightning import LightningDataset

from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from .utils.dataloaders import GraphDataset

try:
    from torch_ema import ExponentialMovingAverage

    is_torch_ema_present = True
except:
    is_torch_ema_present = False

from pytorch_lightning.callbacks import EarlyStopping

from .utils.lightning_utils import SaveModelCallback, SavePerAtomPredictions


class LightningTrainer(pl.LightningModule):
    """
    Wrapper class for Pytorch Lightning Module. This class is used to wrap the model and
    the training loop in a Pytorch Lightning Module. It returns the energy, and forces
    predicted by the model. Stress computations need partial-derivatives of the
    coordinates, and hence are done in the `training_step` functions.

    Args:
        model: Pytorch model to be trained
        input_args: List of input arguments to the model. They are passed as dictionary
            to the model. Example: ["x", "coords", "edge_index0", "edge_index1" ...,"batch"]
        ckpt_dir: Directory to save the checkpoints
        device: Device to run the model on. Default is "cpu"
        ema: Whether to use Exponential Moving Average. Default is True
        ema_decay: Decay rate for Exponential Moving Average. Default is 0.99
        optimizer_name: Name of the optimizer to use. Default is "Adam"
        lr: Learning rate for the optimizer. Default is 0.001
        energy_weight: Weight for the energy loss. Default is 1.0
        forces_weight: Weight for the forces loss. Default is 1.0
        lr_scheduler: Name of the learning rate scheduler. Default is None
        lr_scheduler_args: Arguments for the learning rate scheduler. Default is None
    """

    def __init__(
        self,
        model,
        input_args: List,
        ckpt_dir=None,
        device="cpu",
        ema=True,
        ema_decay=0.99,
        optimizer_name="Adam",
        lr=0.001,
        energy_weight=1.0,
        forces_weight=1.0,
        lr_scheduler=None,
        lr_scheduler_args=None,
    ):

        super().__init__()
        self.model = model
        self.input_args = input_args
        self.ckpt_dir = ckpt_dir
        self.ema = ema
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.energy_weight = energy_weight
        self.forces_weight = forces_weight

        if is_torch_ema_present and ema:
            ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
            ema.to(device)
            self.ema = ema

        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_args = lr_scheduler_args

    def forward(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model. Returns the energy, and forces predicted by the model.
        Args:
            batch: dict containing torch tensors. It should at least have same keys as
                defined in `self.input_keys`

        Returns:
            Energy and forces of the supplied batch. Please note that the forces are
            partial forces, and hence needs to be summed to the contributing particles.

        """
        batch["coords"].requires_grad_(True)
        model_inputs = {k: batch[k] for k in self.input_args}
        predicted_energy = self.model(**model_inputs)
        (predicted_forces,) = torch.autograd.grad(
            predicted_energy.sum(),
            batch["coords"],
            create_graph=True,  # TODO: grad against arbitrary param name
        )
        predicted_forces = -scatter_add(predicted_forces, batch["images"], dim=0)
        return predicted_energy, predicted_forces

    def training_step(self, batch, batch_idx):
        """
        Single training step for lightning.

        TODO:
            Support for Stresses

        Args:
            batch: batch to take step over
            batch_idx: batch index, input from lightning

        Returns:
            loss: loss for the batch
        """
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
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return {"loss": loss, "per_atom_pred": predicted_forces}

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Dict]:
        """
        Configure and return optimizer and learning rate scheduler

        Returns:
            A dict of optimizer and lr_scheduler if scheduler is requested, else it
            returns just the optimizer.
        """
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

    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """
        Single validation step for lightning.

        Args:
            batch: batch to take step over
            batch_idx: batch index, input from lightning

        Returns:
            loss: loss for the batch, and per atom force loss for loss trajectory
        """
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
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return {"val_loss": loss, "per_atom_pred": predicted_forces}

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
    """
    Trainer class for GNN models. This class is used to train GNN models using Pytorch
    Lightning. It uses the `LightningTrainer` to wrap the model and the training
    loop in a Pytorch Lightning Module. It also handles the dataloaders, loggers, and
    callbacks.
    """

    def __init__(self, manifest, model=None):
        """
        Initialize the GNNLightningTrainer.

        Args:
            manifest: Dictionary containing the manifest for the trainer.
            model: Pytorch model to be trained.
        """
        self.pl_model: LightningTrainer = None
        self.data_module = None

        super().__init__(manifest, model)

        # loggers and callbacks
        self.tb_logger = self._tb_logger()
        self.csv_logger = self._csv_logger()
        self.setup_dataloaders()
        self.callbacks = self._get_callbacks()

        # setup lightning trainer
        self.setup_model()  # call setup_model explicitly as it converty torch -> lightning
        self.pl_trainer = self._get_pl_trainer()

    def setup_model(self):
        """
        Set up the model for training. This function initializes the `LightningTrainer`
        with the model, and the training parameters.
        """
        # if dict has key ema, then set ema to True, decay to the dict value, else set ema false
        if not self.model:
            try:
                self.model = torch.jit.load(self.model_manifest["model_path"])
            except ValueError:
                raise TrainerError(
                    "No model was provided, and model_path is not a valid TorchScript model."
                )

        ema = True if self.optimizer_manifest.get("ema", False) else False
        if ema:
            ema_decay = self.optimizer_manifest.get("ema_decay", 0.99)
            logger.info(f"Using Exponential Moving Average with decay rate {ema_decay}")
        else:
            ema_decay = None

        scheduler = self.optimizer_manifest.get("lr_scheduler", {})

        self.pl_model = LightningTrainer(
            model=self.model,
            input_args=self.model_manifest["input_args"],
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
        logger.info("Lightning Model setup complete.")

    def train(self):
        """
        Call the `fit` method of the Pytorch Lightning Trainer to train the model.
        """
        if self.current["appending_to_previous_run"]:
            logger.warning("Resuming training from checkpoint ...")
            self.pl_trainer.fit(
                self.pl_model,
                self.data_module,
                ckpt_path=f"{self.current['run_dir']}/checkpoints/trainer_checkpoint.ckpt",
            )
        else:
            logger.warning("Starting training from scratch ...")
            self.pl_trainer.fit(self.pl_model, self.data_module)

        # currently getting warnings. https://github.com/pytorch/pytorch/issues/89064

        # training finished, 'touch' a .finished file
        if self.pl_trainer.state.finished:
            with open(f"{self.current['run_dir']}/.finished", "w") as f:
                f.write("")
            logger.info("Training complete.")
        else:
            logger.error("Training incomplete. Check logs for errors.")

    def setup_dataloaders(self):
        """
        Set up the dataloaders for the training and validation datasets. If `dynamic_loading`
        is set to True in the dataset manifest, the dataloaders are set up with the
        configuration transform. Otherwise, the dataloaders are set up without the
        configuration transform. Number of workers for the dataloaders is set to the number
        of CPUs available in the system. If a SLURM job is running, the number of workers is
        set to the number of CPUs per task.
        """
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

        if self.optimizer_manifest["num_workers"]:
            num_workers = self.optimizer_manifest["num_workers"]
        else:
            num_workers = os.getenv("SLURM_CPUS_PER_TASK", 1)

        self.data_module = LightningDataset(
            self.train_dataset,
            self.val_dataset,
            batch_size=self.optimizer_manifest["batch_size"],
            num_workers=num_workers,
        )
        logger.info("Data modules setup complete.")

    def _tb_logger(self) -> TensorBoardLogger:
        """
        Set up the TensorBoard logger for the training.

        Returns: TensorBoardLogger
        """
        return TensorBoardLogger(
            f"{self.current['run_dir']}/logs", name="lightning_logs"
        )

    def _csv_logger(self) -> CSVLogger:
        """
        Set up the CSV logger for the training. Just for logging the losses in plain text
        """
        return CSVLogger(f"{self.current['run_dir']}/logs", name="csv_logs")

    def _get_pl_trainer(self) -> pl.Trainer:
        """
        Set up the Pytorch Lightning Trainer with the required parameters. The trainer is
        set up with the TensorBoard and CSV loggers, and the callbacks. Support for single
        precision is not added yet. The number of nodes is set to the number of nodes in the
        SLURM job, or 1 if not running on SLURM. Other job management systems are not
        supported yet.

        Returns: Pytorch Lightning Trainer
        """
        num_nodes = int(os.getenv("SLURM_JOB_NUM_NODES", 1))
        # precision = 32 if self.model_manifest["precision"] == "single" else 64
        precision = self.model_manifest.get("precision", "double")
        if precision == "single":
            logger.warning(
                "Single precision is not supported yet. Using double precision."
            )
            # TODO: Add support for single precision
        return pl.Trainer(
            logger=[self.tb_logger, self.csv_logger],
            max_epochs=self.optimizer_manifest["epochs"],
            accelerator="auto",
            strategy="ddp",
            callbacks=self.callbacks,
            num_nodes=num_nodes,
            # precision=32
        )

    def _get_callbacks(self):
        """
        Set up the model checkpoints, early stopping, and loss trajectory callbacks.
        """
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

        if self.current["log_per_atom_pred"]:
            per_atom_pred_callback = SavePerAtomPredictions(
                self.current["per_atom_pred_database"], ckpt_interval
            )
            callbacks.append(per_atom_pred_callback)
            logger.info("Per atom pred dumping setup complete.")
        else:
            logger.info("Per atom pred dumping not enabled.")

        return callbacks

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

        # save the best pl_model
        pl_module = deepcopy(self.pl_model)
        pl_module.load_state_dict(
            torch.load(f"{self.current['run_dir']}/checkpoints/best_model.pth")
        )
        try:
            model = torch.jit.script(pl_module.model)
        except RuntimeError:
            from e3nn.util import jit  # model might be an e3nn model

            model = jit.script(pl_module.model)
        model = model.cpu()
        torch.jit.save(model, f"{path}/model.pt")

        # save the configuration transform
        self.configuration_transform.export_kim_model(path, "model.pt")

        # CMakeLists.txt
        if not self.export_manifest["model_name"]:
            # current_model_iter = glob.glob(f"{self.export_manifest['model_path']}/*MO_000000000*")
            # current_model_iter.sort()
            # if current_model_iter:
            #     model_iter = int(current_model_iter[-1].split("_")[-1]) + 1
            # else:
            #     model_iter = 0
            # TODO: get the model iter from the model name

            qualified_model_name = f"{self.current['run_title']}_MO_000000000000_000"
        else:
            qualified_model_name = self.export_manifest["model_name"]

        cmakefile = self._generate_kim_cmake(
            qualified_model_name,
            "TorchML__MD_173118614730_000",
            ["model.pt", "kliff_graph.param"],
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

    def setup_optimizer(self):
        # Not needed as Pytorch Lightning handles the optimizer
        pass

    def seed_all(self):
        super().seed_all()
        pl.seed_everything(self.workspace["seed"])


# TODO: Custom loss (via torchmetrics)?
# TODO: switch str everywhere to Path
