import importlib
import multiprocessing
import subprocess
import tarfile
from pathlib import Path
from typing import Callable, Tuple, Union

import kimpy
import numpy as np
from loguru import logger

import kliff.models
from kliff._exceptions import TrainerError
from kliff.dataset import Configuration
from kliff.models import KIMModel
from kliff.utils import install_kim_model

from .kim_residuals import MSE_residuals
from .kliff_trainer import Trainer
from .option_enumerations import ModelTypes, OptimizerProvider


SCIPY_MINIMIZE_METHODS = [
    "Nelder-Mead",
    "Powell",
    "CG",
    "BFGS",
    "Newton-CG",
    "L-BFGS-B",
    "TNC",
    "COBYLA",
    "SLSQP",
    "trust-constr",
    "dogleg",
    "trust-ncg",
    "trust-exact",
    "trust-krylov",
]


class KIMTrainer(Trainer):
    """
    This class extends the base Trainer class for training OpenKIM physics based models.
    It will use the scipy optimizers. It will perform a check to exclude TorchML model
    driver based models, as they would be handled by TorchTrainer.
    """

    def __init__(self, configuration: dict, collection: str = "user"):
        self.collection = collection

        self.model_driver_name = None
        self.parameters = None
        self.mutable_parameters_list = []
        self.use_energy = True
        self.use_forces = False

        super().__init__(configuration)

        self.loss_function = self._get_loss_fn()

    def setup_model(self):
        """
        Load either the KIM model, or install it from the tarball. If the model driver
        required is TorchML* family, then it will raise an error, as it should be handled
        by the TorchTrainer.
        """
        # check for unsupported model drivers
        # 1. get the model driver name
        if self.model_type == ModelTypes.KIM:
            self.model_driver_name = self.get_model_driver_name_for_kim(self.model_name)
        elif self.model_type == ModelTypes.TAR:
            self.model_driver_name = self.get_model_driver_name_for_tarball(
                self.model_name
            )
        else:
            raise TrainerError(f"Model type {self.model_type} not supported.")

        # 2. check if the model driver is supported
        if self.model_driver_name in UNSUPPORTED_MODEL_DRIVERS:
            raise TrainerError(
                f"Model driver {self.model_driver_name} not supported by KIMTrainer."
            )
        elif self.model_driver_name is None:
            logger.warning(
                f"Could not determine model-driver name for {self.model_name}. Please be careful and check if the model is supported."
            )
        else:
            logger.info(f"Model driver name: {self.model_driver_name}")

        # 3. load the model
        if self.model_type == ModelTypes.KIM:
            self.ensure_kim_model_installation(self.model_name, self.collection)
        elif self.model_type == ModelTypes.TAR:
            # reinstall model to be sure
            self.ensure_tarball_model_installation(self.model_name, self.collection)

        self.model = KIMModel(self.model_name)
        self.parameters = self.model.get_model_params()

    def setup_optimizer(self):
        """
        Set up the optimizer based on the provided information. If the optimizer is not
        provided, it will be loaded from the optimizer_name. If the optimizer_name is not
        provided, it will raise an error. If the optimizer_provider is scipy, it will be
        loaded from the scipy.optimize. If the optimizer_provider is torch, it will be
        loaded from the torch.optim. Left for the derived classes to implement.
        """
        if self.optimizer_provider is not OptimizerProvider.SCIPY:
            raise TrainerError(
                f"Optimizer provider {self.optimizer_provider} not supported by KIMTrainer."
            )

        if self.optimizer_name not in SCIPY_MINIMIZE_METHODS:
            raise TrainerError(f"Optimizer not supported: {self.optimizer_name}.")
        optimizer_lib = importlib.import_module(f"scipy.optimize")
        self.optimizer = getattr(optimizer_lib, "minimize")

    def loss(self, x):
        """
        Compute the loss function for the given parameters. It will compute the loss
        function. It seems like MPI might be only way to make it parallel as the
        multiprocessing does not work with the KIM models. KIMPY models are not yet
        pickelable. TODO: include MPI support.
        """
        # set the parameters
        self.model.update_model_params(x)
        # compute the loss
        loss = 0.0
        for configuration in self.train_dataset:
            compute_energy = True if configuration.weight.energy_weight else False
            compute_forces = True if configuration.weight.forces_weight else False
            compute_stress = True if configuration.weight.stress_weight else False

            prediction = self.model(
                configuration,
                compute_energy=compute_energy,
                compute_forces=compute_forces,
                compute_stress=compute_stress,
            )

            if configuration.weight.energy_weight:
                loss += configuration.weight.energy_weight * self.loss_function(
                    prediction["energy"], configuration.energy
                )
            if configuration.weight.forces_weight:
                loss += configuration.weight.forces_weight * self.loss_function(
                    prediction["forces"], configuration.forces
                )
            if configuration.weight.stress_weight:
                loss += configuration.weight.stress_weight * self.loss_function(
                    prediction["stress"], configuration.stress
                )
            loss *= configuration.weight.config_weight

        return loss

    def checkpoint(self, *args, **kwargs):
        TrainerError("checkpoint not implemented.")

    def train_step(self, *args, **kwargs):
        TrainerError("train_step not implemented.")

    def validation_step(self, *args, **kwargs):
        TrainerError("validation_step not implemented.")

    def get_optimizer(self, *args, **kwargs):
        TrainerError("get_optimizer not implemented.")

    def train(self, *args, **kwargs):
        def _wrapper_func(x):
            return self.loss(x)

        x = self.model.get_opt_params()
        options = self.optimizer_kwargs
        options["options"] = {"maxiter": self.max_epochs, "disp": self.verbose}
        result = self.optimizer(
            _wrapper_func, x, method=self.optimizer_name, **self.optimizer_kwargs
        )

        if result.success:
            logger.info(f"Optimization successful: {result.message}")
            self.model.update_model_params(result.x)
        else:
            logger.error(f"Optimization failed: {result.message}")

    def set_parameters_as_mutable(self):
        if self.parameter_transform_options is not None:
            for param_to_transform in self.parameter_transform_options[
                "parameter_list"
            ]:
                if isinstance(param_to_transform, dict):
                    parameter_name = list(param_to_transform.keys())[0]
                elif isinstance(param_to_transform, str):
                    parameter_name = param_to_transform
                else:
                    raise TrainerError(
                        f"Optimizable parameters must be string or value dict. Got {param_to_transform} instead."
                    )
                self.mutable_parameters_list.append(parameter_name)
        else:
            for param in self.parameters:
                self.mutable_parameters_list.append(param)

        self.model.set_params_mutable(self.mutable_parameters_list)
        logger.info(f"Mutable parameters: {self.mutable_parameters_list}")

    def _get_loss_fn(self):
        if self.loss_function_name == "MSE":
            return MSE_residuals

    def save_kim_model(self):
        if self.export_model_type is ModelTypes.KIM:
            path = Path(self.export_model_path) / self.export_model_name
            self.model.write_kim_model(path)
        elif self.export_model_type is ModelTypes.TAR:
            path = Path(self.export_model_path) / self.export_model_name
            self.model.write_kim_model(path)
            tarfile_path = path.with_suffix(".tar.gz")
            with tarfile.open(tarfile_path, "w:gz") as tar:
                tar.add(path, arcname=path.name)
