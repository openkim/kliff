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

# list of model drivers that are not supported by this trainer.
# example quip, torchml, etc.
# TODO: Get the complete list of unsupported model drivers.
UNSUPPORTED_MODEL_DRIVERS = [
    "TorchML",
]
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

    def setup_parameter_transforms(self):
        """
        This method set up the transformed parameter space for models. It can be used
        for any model type in general, but as there exists a significant difference
        between how models handles their parameters, it is left for the subclass to
        implement. Although to ensure that `initialize` function remains consistent
        this method will not raise NotImplemented error, rather it will quietly pass.
        So be aware.
        """
        self.set_parameters_as_mutable()
        mutable_params = self.model.parameters()
        parameter_transforms_input = self.parameter_transform_options["parameter_list"]
        if parameter_transforms_input is not None:
            for model_params, input_params in zip(
                mutable_params, parameter_transforms_input
            ):
                if isinstance(input_params, dict):
                    param_name = list(input_params.keys())[0]
                    if param_name != model_params.name:
                        raise TrainerError(
                            f"Parameter name mismatch. Expected {model_params.name}, got {param_name}."
                        )

                    param_value_dict = input_params[param_name]
                    transform_name = param_value_dict.get("transform_name", None)
                    params_value = param_value_dict.get("value", None)
                    bounds = param_value_dict.get("bounds", None)

                    if transform_name is not None:
                        transform_module = getattr(
                            importlib.import_module(
                                f"kliff.transforms.parameter_transforms"
                            ),
                            transform_name,
                        )
                        transform_module = transform_module()
                        model_params.add_transform(transform_module)

                    if params_value is not None:
                        model_params.copy_from_model_space(params_value)

                    if bounds is not None:
                        model_params.add_bounds_model_space(np.array(bounds))

                elif isinstance(input_params, str):
                    if input_params != model_params.name:
                        raise TrainerError(
                            f"Parameter name mismatch. Expected {model_params.name}, got {input_params}."
                        )
                else:
                    raise TrainerError(
                        f"Optimizable parameters must be string or value dict. Got {input_params} instead."
                    )

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

    @staticmethod
    def get_model_driver_name_for_kim(model_name: str) -> Union[str, None]:
        """
        Get the model driver from the model name. It will return the model driver
        string from the installed KIM API model. If the model is not installed, and the
        model name is a tarball, it will extract the model driver name from the CMakeLists.txt.
        This is needed to ensure that it excludes the model drivers that it cannot handle.
        Example: TorchML driver based models. These models are to be trained using the
        TorchTrainer.

        TODO: This is not a clean solution. I think KIMPY must have a better way to handle this.
              Ask Mingjian/Yaser for comment.

        Args:
            model_name: name of the model.

        Returns:
            Model driver name.
        """
        collections = kimpy.collections.create()
        try:
            shared_obj_path, collection = (
                collections.get_item_library_file_name_and_collection(
                    kimpy.collection_item_type.portableModel, model_name
                )
            )
        except RuntimeError:  # not a portable model
            return None
        shared_obj_content = open(shared_obj_path, "rb").read()
        md_start_idx = shared_obj_content.find(b"model-driver")

        if md_start_idx == -1:
            return None
        else:
            md_start_idx += 15  # length of 'model-driver" "'
            md_end_idx = shared_obj_content.find(b'"', md_start_idx)
            return shared_obj_content[md_start_idx:md_end_idx].decode("utf-8")

    @staticmethod
    def get_model_driver_name_for_tarball(tarball: str) -> Union[str, None]:
        """
        Get the model driver name from the tarball. It will extract the model driver
        name from the CMakeLists.txt file in the tarball. This is needed to ensure that
        it excludes the model drivers that it cannot handle. Example: TorchML driver based
        models. These models are to be trained using the TorchTrainer.

        Args:
            tarball: path to the tarball.

        Returns:
            Model driver name.
        """
        archive_content = tarfile.open(tarball)
        cmake_file_path = archive_content.getnames()[0] + "/CMakeLists.txt"
        cmake_file = archive_content.extractfile(cmake_file_path)
        cmake_file_content = cmake_file.read().decode("utf-8")

        md_start_idx = cmake_file_content.find("DRIVER_NAME")
        if md_start_idx == -1:
            return None
        else:
            # name strats at "
            md_start_idx = cmake_file_content.find('"', md_start_idx) + 1
            if md_start_idx == -1:
                return None
            md_end_idx = cmake_file_content.find('"', md_start_idx)
            return cmake_file_content[md_start_idx:md_end_idx]

    @staticmethod
    def ensure_kim_model_installation(model_name: str, collection: str = "user"):
        """
        Ensure that the KIM model is installed. If the model is not installed, it will
        install the model in the user collection. If the model is already installed, it
        will not do anything.

        Args:
            model_name: name of the model.
            collection: collection to install the model in.
        """
        is_model_installed = install_kim_model(model_name)
        if not install_kim_model(model_name):
            logger.error(
                f"Mode: {model_name} neither installed nor available in the KIM API collections. Please check the model name and try again."
            )
            raise TrainerError(f"Model {model_name} not found.")
        else:
            logger.info(f"Model {model_name} is present in {collection} collection.")

    def ensure_tarball_model_installation(self, tarball: str, collection: str = "user"):
        """
        Ensure that the model is installed from the tarball. If the model is not installed,
        it will install the model in the user collection. If the model is already installed,
        it will reinstall the model.

        Args:
            tarball: path to the tarball.
            collection: collection to install the model in.
        """
        scratch_dir = f"{self.current_run_dir}/.scratch"
        archive_content = tarfile.open(tarball)
        model = archive_content.getnames()[0]
        archive_content.extractall(scratch_dir)
        subprocess.run(
            [
                "kim-api-collections-management",
                "install",
                "--force",
                collection,
                scratch_dir + "/" + model,
            ],
            check=True,
        )
        logger.info(f"Tarball Model {model} installed in {collection} collection.")

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
