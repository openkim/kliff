import multiprocessing as mp
import os
import subprocess
import tarfile
from pathlib import Path

from loguru import logger
import numpy as np
from scipy.optimize import minimize

from kliff.dataset import Dataset

from ..models import KIMModel
from .kliff_trainer import *


class PhysicsBasedTrainer(Trainer):
    def __init__(self, configuration):
        super().__init__(configuration)
        self.model = self.get_model()
        self.dataset = self.get_dataset()
        self.params_to_optimize = self.configuration["params_to_optimize"]
        if self.configuration["seed"] is not None:
            np.random.seed(self.configuration["seed"])

    def loss(self, energy_prediction, energy_target, forces_prediction, forces_target):
        loss = (
            np.sum((energy_prediction - energy_target) ** 2) * self.energy_loss_weight
        )
        loss += (
            np.sum((forces_prediction - forces_target) ** 2) * self.forces_loss_weight
        )
        return loss

    def checkpoint(self):
        chkpt_folder = f"{self.current_run_dir}/trained_model"
        try:
            os.makedirs(chkpt_folder, exist_ok=True)
        except FileExistsError:
            pass
        self.model.write_kim_model(Path(f"{chkpt_folder}"))
        self.model.save(Path(f"{chkpt_folder}/{self.model_name}.yaml"))
        self.to_file(f"{self.current_run_dir}/configuration.yaml")
    # def train_step(self):
    #     TrainerError("train_step not implemented.")

    # def validation_step(self):
    #     TrainerError("validation_step not implemented.")

    # def get_optimizer(self):
    #     TrainerError("get_optimizer not implemented.")

    def get_dataset(self):  # Specific to trainer
        if (
            self.dataset_type != DataTypes.COLABFIT
            and self.dataset_type != DataTypes.KLIFF
            and self.dataset_type != DataTypes.ASE
        ):
            raise TrainerError("Invalid dataset type.")
        else:
            if self.dataset_type == DataTypes.COLABFIT:
                return Dataset(
                    colabfit_dataset=self.dataset_name,
                    colabfit_database=self.database_name,
                )
            elif self.dataset_type == DataTypes.KLIFF:
                return Dataset(self.dataset_path)
            elif self.dataset_type == DataTypes.ASE:
                return Dataset(
                    self.dataset_path,
                    parser="ase",
                    energy_key=self.configuration["energy_key"],
                    forces_key=self.configuration["forces_key"],
                )
            else:
                raise TrainerError("Invalid dataset type.")  # Should never happen

    def get_model(self):
        if self.model_source == ModelTypes.KIM:
            return KIMModel(self.model_name)
        elif self.model_source == ModelTypes.TAR:
            tarfile.open(self.model_name, "r:gz").extractall(
                f"{self.current_run_dir}/model"
            )
            # execute shell command kim-api-collections-management install <path-to-model>
            os.chdir(f"{self.current_run_dir}/model")
            subprocess.run(
                [
                    "kim-api-collections-management",
                    "install",
                    "CWD",
                    f"{self.model_name}",
                ]
            )
            os.chdir("../../")
            return KIMModel(self.model_name)

    def set_parameters_to_optimize(self):
        self.model.set_opt_params(**self.params_to_optimize)

    def train(self):
        self.set_parameters_to_optimize()
        # pool = mp.Pool(self.configuration["cpu_workers"])

        def map_loss(configuration):
            predictions = self.model(configuration)
            return self.loss(
                predictions["energy"],
                configuration.energy,
                predictions["forces"],
                configuration.forces,
            )

        def loss_wrapper(params):
            self.model.update_model_params(params)
            # losses = pool.map(map_loss, self.dataset.get_configs())
            losses = list(map(map_loss, self.dataset.get_configs()))
            # pool.close()
            # pool.join()
            # does minimize have own parallelization?
            return np.sum(losses)

        result = minimize(
            loss_wrapper,
            self.model.get_opt_params(),
            bounds=self.model.get_formatted_param_bounds(),
            method=self.configuration["optimizer"],
            options=self.configuration["optimizer_kwargs"],
        )
        self.model.update_model_params(result.x)
        self.checkpoint()
