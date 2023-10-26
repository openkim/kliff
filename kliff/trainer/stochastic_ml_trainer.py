from .kliff_trainer import *
import os
import torch
from kliff.dataset import Dataset
import numpy as np
import tarfile

import torch_geometric as pyg


try:
    from torch_ema import ExponentialMovingAverage
except ImportError:
    ExponentialMovingAverage = None


class KLIFFTorchGeometricDataset(pyg.data.InMemoryDataset):
    def __init__(self,transform=None, pre_transform=None, path="./data.pt"):
        super().__init__(None, transform, pre_transform)
        self.data, self.slices = torch.load(path)


class TorchTrainer(Trainer):
    def __init__(self, configuration:dict):
        super().__init__(configuration)
        self.current_epoch = 0
        self.current_val_loss = 0
        self.current_train_loss = 0
        self.current_val_predictions = None
        self.lr_scheduler = None
        self.ema = None
        # save configuration
        self.to_file(f"{self.current_run_dir}/configuration.yaml")
        self.history = {"train_loss": [], "val_loss": []}
        self.train_batch_count = 0
        self.val_batch_count = 0
        self.best_epoch = 0
        self.epoch_without_improvement = {"val": 0, "train": 0}

    def loss(self, energy_prediction, energy_target, forces_prediction, forces_target):
        loss = torch.sum((energy_prediction - energy_target) ** 2) * self.energy_loss_weight
        loss += torch.sum((forces_prediction - forces_target) ** 2) * self.forces_loss_weight
        return loss

    def _checkpoint(self, checkpoint_name, overwrite=False):
        current_chkpt_path = f"{self.current_run_dir}/checkpoints/{checkpoint_name}"
        try:
            os.mkdir(current_chkpt_path)
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
        np.savetxt(f"{self.current_run_dir}/val_predictions.txt", self.current_val_predictions)
        with open(f"{self.current_run_dir}/loss.txt", "a+") as f:
            f.write(f"{self.current_epoch}\t{self.current_train_loss}\t{self.current_val_loss}\n")

    def get_optimizer(self):
        if self.configuration["optimizer_provider"] == OptimizerProvider.TORCH:
            # initialize optimizer from the name
            optimizer_name = self.configuration["optimizer_name"]
            optimizer = getattr(torch.optim, optimizer_name)
            self.optimizer = optimizer(self.model.parameters(), **self.configuration["optimizer_kwargs"])
            if self.configuration["lr_scheduler"]:
                self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **self.configuration["lr_scheduler"])
            if self.configuration["use_ema"]:
                if ExponentialMovingAverage is None:
                    raise TrainerError("torch_ema is not installed but ema is requested.")
                self.ema = ExponentialMovingAverage(self.model.parameters(), decay=self.configuration["ema_decay"])
        else:
            raise TrainerError("Optimizer provider not implemented.")

    def optimizer_step(self):
        self.optimizer.step()
        if self.ema is not None:
            self.ema.update(self.model.parameters())
        self.lr_scheduler.step()

    def get_model(self):
        if self.configuration["model_provider"] == ModelTypes.TORCH:
            # initialize model from the name
            model_name = self.configuration["model_name"]
            model = torch.jit.load(model_name + ".pt", map_location="cpu")
            self.model = model
        elif self.configuration["model_provider"] == ModelTypes.TAR:
            # initialize model from the name
            model_name = self.configuration["model_name"]
            tarfile.open(model_name, "r:gz").extractall(f"{self.current_run_dir}/model")
            model = torch.jit.load(f"{self.current_run_dir}/model/{model_name}.pt", map_location="cpu")
            self.model = model
        else:
            raise TrainerError("Model provider not implemented.")

    def get_dataset(self): # Specific to trainer
        if self.configuration["dataset_provider"] == DataTypes.TORCH_GEOMETRIC:
            dataset = KLIFFTorchGeometricDataset(path=self.configuration["dataset_path"])
        # elif self.configuration["dataset_provider"] == DataTypes.TORCH:
        #     dataset = torch.load(self.configuration["dataset_path"])
        elif self.configuration["dataset_provider"] == DataTypes.ASE:
            dataset = Dataset(self.configuration["dataset_path"], parser="ase", energy_key=self.configuration["energy_key"], forces_key=self.configuration["forces_key"])
        elif self.configuration["dataset_provider"] == DataTypes.KLIFF:
            dataset = Dataset(self.configuration["dataset_path"])
        elif self.configuration["dataset_provider"] == DataTypes.COLABFIT:
            dataset = Dataset(colabfit_dataset=self.configuration["dataset_name"], colabfit_database=self.configuration["database_name"])
        else:
            raise TrainerError("Dataset provider not implemented.")

        dataset_size = len(dataset)
        self.get_indices(dataset_size)

        if isinstance(dataset, Dataset):
            dataset_train = dataset.get_configs()[self.train_indices]
            dataset_val = dataset.get_configs()[self.val_indices]
            dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=self.configuration["batch_size"], shuffle=True, num_workers=self.configuration["cpu_workers"])
            dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=self.configuration["batch_size"], shuffle=True, num_workers=self.configuration["cpu_workers"])
        else:
            dataset_train = dataset[self.train_indices]
            dataset_val = dataset[self.val_indices]
            dataloader_train = pyg.data.DataLoader(dataset_train, batch_size=self.configuration["batch_size"], shuffle=True, num_workers=self.configuration["cpu_workers"])
            dataloader_val = pyg.data.DataLoader(dataset_val, batch_size=self.configuration["batch_size"], shuffle=False, num_workers=self.configuration["cpu_workers"])

        self.train_dataset = dataloader_train
        self.val_dataset = dataloader_val
        # total number of batches
        self.train_batch_count = len(self.train_dataset)
        self.val_batch_count = len(self.val_dataset)


    def stop_now(self):
        # hard stops conditions
        now = datetime.now()
        if now > self.end_time or self.current_epoch > self.configuration["max_epochs"]:
            return True

        # early stopping conditions
        if (self.history["val_loss"][self.best_epoch] - self.history["val_loss"][-1]) > self.configuration["early_stopping_minimum_delta"]["validation"]:
            self.epoch_without_improvement["val"] = 0
        else:
            self.epoch_without_improvement["val"] += 1

        if (self.history["train_loss"][self.best_epoch] - self.history["train_loss"][-1]) > self.configuration["early_stopping_minimum_delta"]["train"]:
            self.epoch_without_improvement["train"] = 0
        else:
            self.epoch_without_improvement["train"] += 1

        if self.epoch_without_improvement["val"] > self.configuration["early_stopping"]["early_stopping_patience"] or self.epoch_without_improvement["train"] > self.configuration["early_stopping"]["early_stopping_patience"]:
            finish_time = datetime.now()
            with open(f"{self.current_run_dir}/.finished", "w") as f:
                f.write(f"Start time {self.start_time.strftime('%Y-%m-%d-%H-%M-%S')}\n")
                f.write(f"Finish time {finish_time.strftime('%Y-%m-%d-%H-%M-%S')}\n")
                f.write(f"Total time {(finish_time - self.start_time).strftime('%Y-%m-%d-%H-%M-%S')}\n")
            return True

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        energy_prediction, forces_prediction = self.model(**batch)
        loss = self.loss(energy_prediction, batch["energy"], forces_prediction, batch["forces"])
        loss.backward()
        self.optimizer_step()
        self.current_train_loss += loss.item()

    def validation_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            energy_prediction, forces_prediction = self.model(**batch)
            self.current_val_loss += self.loss(energy_prediction, batch["energy"], forces_prediction, batch["forces"]).item()
            self.current_val_predictions = np.append(self.current_val_predictions, energy_prediction.squeeze().detach().numpy())

    def train(self):
        self.current_train_loss = 0
        for batch in self.train_dataset:
            self.train_step(batch)
            self.optimizer_step()
        self.history["train_loss"].append(self.current_train_loss/self.train_batch_count)

        self.current_val_predictions = np.array([])
        self.current_val_loss = 0
        for batch in self.val_dataset:
            self.validation_step(batch)
        self.history["val_loss"].append(self.current_val_loss/self.val_batch_count)

        self.checkpoint()

        self.current_epoch += 1
        if self.stop_now():
            exit()
        else:
            self.train()

