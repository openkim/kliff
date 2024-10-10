import importlib
import tarfile
from pathlib import Path

import numpy as np
from loguru import logger

from kliff.models import KIMModel

from .base_trainer import Trainer, TrainerError
from .utils.losses import MAE_loss, MSE_loss

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
    driver based models, as they would be handled by Torch based trainers. It can read model tarballs
    as well as export the models as tarballs for ease of use. It will use the KIMModel
    class to load the model and set the parameters. It also provides explicit interface
    for parameters transformation.

    Args:
        configuration (dict): The configuration dictionary.
    """

    def __init__(self, configuration: dict):
        self.model_driver_name = None
        self.parameters = None
        self.mutable_parameters_list = []
        self.use_energy = True
        self.use_forces = False
        self.use_stress = False
        self.is_model_tarfile = False

        super().__init__(configuration)

        self.loss_function = self._get_loss_fn()
        self.result = None

    def setup_model(self):
        """
        Load either the installed KIM model, or install it from the source. If the model
        driver required is TorchML* family, then it will raise an error, as it should be
        handled by the DNNTrainer, or GNNLightningTrainer.

        Path can be a folder containing the model, or a tar file. The model name is the KIM
        model name.
        """
        if self.model_manifest["path"]:
            try:
                self.is_model_tarfile = tarfile.is_tarfile(self.model_manifest["path"])
            except (IsADirectoryError, TypeError) as e:
                self.is_model_tarfile = False
                logger.debug(f"Model path is not a tarfile: {e}")

        # check for unsupported model drivers
        self.model = KIMModel.get_model_from_manifest(
            self.model_manifest, self.transform_manifest, self.is_model_tarfile
        )

        self.parameters = self.model.get_model_params()

    def setup_optimizer(self):
        """
        Set up the optimizer based on the provided information. If the optimizer name is
        not provided, it will raise an error. It will use the ~:class:~scipy.optimize
        class for optimizers. It will raise an error if the optimizer is not supported.
        """
        if self.optimizer_manifest["name"] not in SCIPY_MINIMIZE_METHODS:
            raise TrainerError(
                f"Optimizer not supported: {self.optimizer_manifest['name']}."
            )
        optimizer_lib = importlib.import_module(f"scipy.optimize")
        self.optimizer = getattr(optimizer_lib, "minimize")
        # TODO: LM-Geodesic optimizer

    def loss(self, x: np.ndarray) -> float:
        """
        Compute the loss function for the given parameters. It sets the KIM model
        parameters, compute the desired loss function doe all trainable properties
        and return the total loss after scaling losses with ~:class:~kliff.configuration.Weight.

        TODO:
            Include MPI support.

        Args:
            x (np.ndarray): The model parameters.

        Returns:
            float: The total loss.
        """
        # set the parameters
        self.model.update_model_params(x)
        # compute the loss
        loss = 0.0
        for configuration in self.train_dataset:
            compute_energy = (
                True if configuration.weight.energy_weight is not None else False
            )
            compute_forces = (
                True if configuration.weight.forces_weight is not None else False
            )
            compute_stress = (
                True if configuration.weight.stress_weight is not None else False
            )

            prediction = self.model(
                configuration,
                compute_energy=compute_energy,
                compute_forces=compute_forces,
                compute_stress=compute_stress,
            )

            if self.current["log_per_atom_pred"]:
                self.log_per_atom_outputs(
                    self.current["epoch"],
                    [configuration.metadata.get("index")],
                    [prediction["forces"]],
                )

            if configuration.weight.energy_weight is not None:
                loss += self.loss_function(
                    prediction["energy"],
                    configuration.energy,
                    configuration.weight.energy_weight,
                )
            if configuration.weight.forces_weight is not None:
                loss += self.loss_function(
                    prediction["forces"],
                    configuration.forces,
                    configuration.weight.forces_weight,
                )
            if configuration.weight.stress_weight is not None:
                loss += self.loss_function(
                    prediction["stress"],
                    configuration.stress,
                    configuration.weight.stress_weight,
                )
            if configuration.weight.config_weight is not None:
                loss *= configuration.weight.config_weight

        self.current["epoch"] += 1
        return loss

    def checkpoint(self, *args, **kwargs):
        TrainerError("checkpoint not implemented.")

    def train_step(self, *args, **kwargs):
        TrainerError("train_step not implemented.")

    def validation_step(self, *args, **kwargs):
        TrainerError("validation_step not implemented.")

    def get_optimizer(self, *args, **kwargs):
        TrainerError("get_optimizer not implemented.")

    def train(self):
        """
        Train the model using the provided optimizer. It will set the model parameters
        to the optimal values found by the optimizer. It will log the optimization
        status and the message. It will raise an error if the optimization fails.

        TODO:
            Include MPI support.
            Log loss trajectory for KIM models.
        """

        def _wrapper_func(x):
            return self.loss(x)

        x = self.model.get_opt_params()
        options = self.optimizer_manifest.get("kwargs", {})
        options["options"] = {
            "maxiter": self.optimizer_manifest["epochs"],
            "disp": self.current["verbose"],
        }
        self.result = self.optimizer(
            _wrapper_func, x, method=self.optimizer_manifest["name"], **options
        )

        if self.result.success:
            logger.info(f"Optimization successful: {self.result.message}")
            self.model.update_model_params(self.result.x)
        else:
            logger.error(f"Optimization failed: {self.result.message}")

    def _get_loss_fn(self) -> callable:
        """
        Get the loss function based on the provided loss manifest. It will raise an error
        if the loss function is not supported.

        Returns:
            function: The loss function.
        """
        if self.loss_manifest["function"].lower() == "mse":
            return MSE_loss
        if self.loss_manifest["function"].lower() == "mae":
            return MAE_loss
        else:
            raise TrainerError(
                f"Loss function {self.loss_manifest['function']} not supported."
            )

    def save_kim_model(self):
        """
        Save the KIM model to the provided path. It will also generate a tarball if
        specified in the export manifest.
        """
        path = (
            Path(self.export_manifest["model_path"])
            / self.export_manifest["model_name"]
        )
        self.model.write_kim_model(path)
        self.write_training_env_edn(path)
        if self.export_manifest["generate_tarball"]:
            tarfile_path = path.with_suffix(".tar.gz")
            with tarfile.open(tarfile_path, "w:gz") as tar:
                tar.add(path, arcname=path.name)
            logger.info(f"Model tarball saved: {tarfile_path}")
        logger.info(f"KIM model saved at {path}")


# TODO: Support for lst_sq in optimizer
