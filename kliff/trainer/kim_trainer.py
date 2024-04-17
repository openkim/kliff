import importlib
import tarfile
from pathlib import Path

from loguru import logger

from kliff.models import KIMModel

from .base_trainer import Trainer, TrainerError
from .kim_residuals import MSE_residuals

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
        if (
            self.model_manifest["type"].lower() == "kim"
            or self.model_manifest["type"].lower() == "tar"
        ):
            self.model = KIMModel.get_model_from_manifest(
                self.model_manifest, self.transform_manifest
            )
        else:
            raise TrainerError(
                f"Model type {self.model_manifest['type']} not supported."
            )

        self.parameters = self.model.get_model_params()

    def setup_optimizer(self):
        """
        Set up the optimizer based on the provided information. If the optimizer is not
        provided, it will be loaded from the optimizer_name. If the optimizer_name is not
        provided, it will raise an error. If the optimizer_provider is scipy, it will be
        loaded from the scipy.optimize. If the optimizer_provider is torch, it will be
        loaded from the torch.optim. Left for the derived classes to implement.
        """
        if self.optimizer_manifest["provider"] != "scipy":
            raise TrainerError(
                f"Optimizer provider {self.optimizer_manifest['provider']} not supported by KIMTrainer."
            )

        if self.optimizer_manifest["name"] not in SCIPY_MINIMIZE_METHODS:
            raise TrainerError(
                f"Optimizer not supported: {self.optimizer_manifest['name']}."
            )
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
        options = self.optimizer_manifest.get("kwargs", {})
        options["options"] = {
            "maxiter": self.optimizer_manifest["epochs"],
            "disp": self.current["verbose"],
        }
        result = self.optimizer(
            _wrapper_func, x, method=self.optimizer_manifest["name"], **options
        )

        if result.success:
            logger.info(f"Optimization successful: {result.message}")
            self.model.update_model_params(result.x)
        else:
            logger.error(f"Optimization failed: {result.message}")

    def _get_loss_fn(self):
        if self.loss_manifest["function"].lower() == "mse":
            return MSE_residuals

    def save_kim_model(self):
        if self.export_manifest["model_type"].lower() == "kim":
            path = (
                Path(self.export_manifest["model_path"])
                / self.export_manifest["model_name"]
            )
            self.model.write_kim_model(path)
        elif self.export_manifest["model_type"] == "tar":
            path = (
                Path(self.export_manifest["model_path"])
                / self.export_manifest["model_name"]
            )
            self.model.write_kim_model(path)
            tarfile_path = path.with_suffix(".tar.gz")
            with tarfile.open(tarfile_path, "w:gz") as tar:
                tar.add(path, arcname=path.name)
