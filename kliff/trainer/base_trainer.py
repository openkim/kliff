import hashlib
import importlib
import json
import os
import pickle as pkl
import random
import sys
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Union

import numpy as np
import yaml
from loguru import logger

from kliff.dataset import Dataset
from kliff.dataset.weight import Weight

if TYPE_CHECKING:
    from kliff.transforms.configuration_transforms import ConfigurationTransform


class Trainer:
    """Base class for all trainers.

    This class is the base class for all trainers. It provides the basic structure for
    training a model. The derived classes should implement the required methods. This
    class will provide the basic functionality for training, such as setting up the
    work directory, saving the configuration, and setting up the indices for training
    and validation datasets. It will save hashes of the configuration fingerprints and
    training configuration to the work directory. This would ensure reproducibility of
    the training process, and easy restarting.

    The core trainer class will provide the following functionality:
    - Set up the work directory
    - Set up the dataset
    - Set up the test train split
    Model, parameter transform and optimizer setup are left for the derived classes to
    implement.
    """

    def __init__(self, training_manifest: dict, model=None):
        """
        Initialize the trainer with the provided manifest and model. Model can be initialized
        from the manifest itself for simpler, KIM-API based models. For more complex models,
        the model should be provided as an argument.

        The model manifest should contain the following sections, or keys:
        - workspace: Workspace configuration, folders, name of run etc
            - name: Name of the workspace
            - seed: Seed for random number generators
            - resume: Resume from the previous run if possible

        - dataset: Dataset to train on, usually an XYZ file, or ColabFit dataset. The
                   content of this block will be given {func}`kliff.dataset.Dataset.from_manifest`
                   function. Consult the documentation of the function for more information.

        - model: Model to train, usually a KIM-API model. The content of this block will be given to the
                    {func}`kliff.models.Model.get_model_from_manifest` function. Consult the documentation of the
                    function for more information. The above function usually works for
                    physics based KIM-API models. For more complex ML models, you need to
                    initialize the model and provide it as an argument. Tarfiles can be used
                    for TorchML driver based model drivers.
                    Additional keys might be needed on model specific basis.
        - transforms: Transformations to be applied to the dataset. This includes
                    parameter transforms (for KIM models), configuration transforms
                    (descriptors, and graphs), and property transforms (energy, forces,
                    stress).
        - training: Training configuration, including loss function, optimizer, batch size,
                    number of epochs, early stopping, dataset test-train split etc.
                    TODO: Add more details about the training block.

        - export: Export the trained model. This includes the model name and the path to
                    save the model.

        Args:
            training_manifest: Dictionary containing the training configuration
            model: Model instance or None, usually a pytorch model
        """
        # workspace variables
        self.workspace: dict = {
            "name": "kliff_workspace",
            "seed": 12345,
            "resume": False,
        }

        # dataset variables
        self.dataset_manifest: dict = {
            "type": "path",
            "path": "./",
            "save": False,
            "keys": {"energy": "energy", "forces": "forces"},
            "dynamic_loading": False,
            "colabfit_dataset": {
                "dataset_name": None,
                "database_name": None,
                "database_url": None,
            },
        }
        self.dataset = None

        # model variables
        self.model_manifest: dict = {
            "name": None,
            "path": None,
            "collection": "user",
        }
        self.model: Callable = model

        # transform variables
        self.transform_manifest: dict = {
            "property": [
                {
                    "name": None,
                    "property_key": None,
                }
            ],
            "parameter": [],
            "configuration": {
                "name": None,
                "kwargs": None,
            },
        }
        self.property_transforms = []
        self.configuration_transform: "ConfigurationTransform" = None

        # training variables
        # this is too complicated to put it in singe dict, therefore the training
        # block is divided into loss, optimizer, dataset_sample
        self.training_manifest: dict = {}
        self.loss_manifest: dict = {
            "function": "mse",
            "weights": {
                "energy": 1.0,
                "forces": None,
                "stress": None,
                "config": 1.0,
            },
            "normalize_per_atom": False,
        }

        self.optimizer_manifest: dict = {
            "name": None,
            "learning_rate": None,
            "kwargs": None,
            "epochs": 10000,
            "num_workers": None,
            "batch_size": 1,
        }
        self.optimizer = None

        # part of current?
        self.dataset_sample_manifest: dict = {
            "train_size": None,
            "val_size": None,
            "indices_files": {"train": None, "val": None},
            "val_indices": None,
            "train_indices": None,
        }
        self.train_dataset = None
        self.val_dataset = None

        # export trained model
        self.export_manifest: dict = {
            "model_name": None,
            "model_path": None,
            "generate_tarball": False,
        }

        # state variables
        self.current: dict = {
            "run_title": None,
            "run_dir": None,
            "run_hash": None,
            "start_time": None,
            "best_loss": np.inf,
            "best_model": None,
            "loss": None,
            "epoch": 0,
            "step": 0,
            "device": "cpu",
            "warned_once": False,
            "dataset_hash": None,
            "data_dir": None,
            "appending_to_previous_run": False,
            "verbose": False,
            "ckpt_interval": 100,
            "log_per_atom_pred": False,
            # "per_atom_pred": {"train": None, "val": None},
            "per_atom_pred_database": None,
        }
        self.parse_manifest(training_manifest)
        self.initialize()

    def parse_manifest(self, manifest: dict):
        """
        It accepts the raw manifest dictionary, and processes it to the formatted
        manifest. This includes mapping the string fields to enums, and setting sane
        defaults for missing fields.

        Args:
            manifest: raw incoming configuration

        Returns:
            Processed manifest
        """
        _date_time_format = "%Y-%m-%d-%H-%M-%S"
        start_time = datetime.now()
        date_time_str = start_time.strftime(_date_time_format)
        self.current["start_time"] = start_time

        # Workspace variables ################################################
        workspace_block: Union[None, dict] = manifest.get("workspace", None)
        if workspace_block is None:
            logger.warning(
                "Workspace block not found in the configuration. Using default values."
            )
        else:
            self.workspace |= workspace_block

        # Dataset manifest #################################################
        dataset_manifest: Union[None, dict] = manifest.get("dataset", None)
        if dataset_manifest is None:
            raise TrainerError("Dataset block not found in the configuration. Exiting.")

        self.dataset_manifest |= dataset_manifest

        # model variables ####################################################
        model_manifest: Union[None, dict] = manifest.get("model", None)
        if model_manifest is None:
            raise TrainerError("Model block not found in the configuration. Exiting.")
        self.model_manifest |= model_manifest

        if self.model_manifest.get("name", None) is None:
            self.current["run_title"] = (
                f"{self.model_manifest.get('type')}_{date_time_str}"
            )
        else:
            self.current["run_title"] = (
                f"{self.model_manifest.get('name')}_{date_time_str}"
            )

        # transform variables ####################################################
        transform_manifest: Union[None, dict] = manifest.get("transforms", None)
        if transform_manifest is None:
            logger.warning(
                "Transform block not found in the configuration. This is bit unusual."
            )
        else:
            self.transform_manifest |= transform_manifest

        # training variables ########################################################
        training_manifest: Union[None, dict] = manifest.get("training", None)
        if training_manifest is None:
            logger.warning(
                "Training block not found in the configuration."
                "Will try and resume the previous run if possible."
            )
        self.training_manifest |= training_manifest

        if self.training_manifest.get("loss", None) is None:
            logger.warning(
                "Loss block not found in the configuration. Using default values."
            )

        self.loss_manifest |= self.training_manifest.get("loss")

        if self.training_manifest.get("optimizer", None) is None:
            logger.warning(
                "Optimizer block not found in the configuration."
                "Will resume the previous run if possible."
            )

        self.optimizer_manifest |= self.training_manifest.get("optimizer")
        self.optimizer_manifest["epochs"] = self.training_manifest.get("epochs", 10000)
        self.optimizer_manifest["num_workers"] = self.training_manifest.get(
            "num_workers", None
        )
        self.optimizer_manifest["batch_size"] = self.training_manifest.get(
            "batch_size", 1
        )

        self.current["ckpt_interval"] = self.training_manifest.get("ckpt_interval", 100)
        self.current["verbose"] = self.training_manifest.get("verbose", False)
        self.current["device"] = self.training_manifest.get("device", "cpu")

        # dataset sample variables will be processed in the setup_dataset method
        self.export_manifest |= manifest.get("export", {})

        # per save atom prediction?
        self.current["log_per_atom_pred"] = training_manifest.get(
            "log_per_atom_pred", False
        )

    def config_to_dict(self):
        """
        Convert the configuration to a dictionary.
        """
        config = {
            "workspace": self.workspace,
            "dataset": self.dataset_manifest,
            "model": self.model_manifest,
            "transforms": self.transform_manifest,
            "training": self.training_manifest,
        }
        return config

    @classmethod
    def from_file(cls, filename: Path):
        """
        Load the manifest from a YAML file.

        Args:
            filename: name of the yaml file

        Returns:
            Trainer instance

        """
        manifest = yaml.safe_load(open(filename, "r"))
        return cls(manifest)

    def get_trainer_hash(self):
        """
        Get the hash of the current configuration. It will be used to create a unique
        directory for the current run. It will be the hash of the configuration dictionary
        string.
        """
        config = self.config_to_dict()
        config_immut_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_immut_str.encode()).hexdigest()

    def initialize(self):
        """
        Initialize the trainer. Assigns the configuration objects, and
        call setup methods.
        """
        # Step 1 - Assign the processed configuration objects to the class variables
        # This has been done in the __init__ method
        # Step 2 - Initialize all seeds
        self.seed_all()
        logger.info(f"Seed set to {self.workspace['seed']}.")
        # Step 3 - Set up the workspace folder
        self.setup_workspace()
        logger.info(f"Workspace set to {self.current['run_dir']}.")
        # Step 4 - Read or load the dataset, initialize the property/configuration transforms
        self.setup_dataset()
        logger.info(f"Dataset loaded.")
        # Step 4.5 - Set up the dataset transforms
        self.setup_dataset_transforms()
        # Step 5 - Set up the test and train datasets, based on the provided indices
        self.setup_dataset_split()
        logger.info(f"Train and validation datasets set up.")
        # Step 6 - Set up the model, if not provided
        if not self.model:
            self.setup_model()
        logger.info(f"Model loaded.")
        # Step 6.5 - Setup parameter transform
        self.setup_parameter_transforms()
        # Step 7 - Set up the optimizer
        self.setup_optimizer()
        logger.info(f"Optimizer loaded.")
        # Step 8 - Save the configuration for future
        self.save_config()

    def seed_all(self):
        """
        Seed all the random number generators.
        """
        np.random.seed(self.workspace["seed"])
        random.seed(self.workspace["seed"])
        # torch.manual_seed(self.seed) # enable torch seed in respective children
        # torch.cuda.manual_seed_all(self.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    def setup_workspace(self):
        """
        Check all the existing runs in the root directory and see if it finished the run
        or not. If it is finished, it will start a new run. If it is not finished, it will
        resume the training. If the resume is not requested, it will start a new run.
        """
        dir_list = sorted(
            glob(f"{self.workspace['name']}/{self.model_manifest['name']}*")
        )
        dir_list = [p for p in dir_list if os.path.isdir(p)]

        if len(dir_list) == 0 or not self.workspace["resume"]:
            logger.info(
                "Either a fresh run or resume is not requested. Starting a new run."
            )
            self.current["appending_to_previous_run"] = False
            self.current["run_dir"] = (
                f"{self.workspace['name']}/{self.current['run_title']}"
            )
            os.makedirs(self.current["run_dir"], exist_ok=True)
        else:
            last_dir = dir_list[-1]
            was_it_finished = os.path.exists(f"{last_dir}/.finished")
            if was_it_finished:  # start new run
                logger.warning(
                    "Resuming from last training was requested, but it was completed. Exiting."
                )
                # current_run_dir = (
                #     f"{self.workspace['name']}/{self.current['run_title']}"
                # )
                # os.makedirs(current_run_dir, exist_ok=True)
                # self.current["appending_to_previous_run"] = False
                sys.exit()
            else:
                logger.info("Last trainer was not finished. Resuming the training.")
                self.current["appending_to_previous_run"] = True
                self.current["run_dir"] = dir_list[-1]

        # make dataset directory
        self.current["data_dir"] = f"{self.workspace['name']}/datasets"
        os.makedirs(self.current["data_dir"], exist_ok=True)

    def setup_dataset(self):
        """
        Set up the dataset based on the provided information. If the per atom prediction
        logging is requested, it will also assign a sequential index to each configuration
        for logging.
        TODO: ColabFit integration for extreme scale datasets.
        """

        self.dataset = Dataset.get_dataset_from_manifest(self.dataset_manifest)

        weights = self.loss_manifest.get("weights", None)

        if weights is not None:
            if isinstance(weights, str):
                weights = Path(weights)
                Dataset.add_weights(self.dataset, weights)
            elif isinstance(weights, dict):
                weights = Weight(
                    config_weight=weights.get("config", None),
                    energy_weight=weights.get("energy", None),
                    forces_weight=weights.get("forces", None),
                    stress_weight=weights.get("stress", None),
                )
                if isinstance(weights.forces_weight, list):
                    weights.forces_weight = np.array(weights.forces_weight)
                if isinstance(weights.config_weight, list):
                    weights.config_weight = np.array(weights.config_weight)

                Dataset.add_weights(self.dataset, weights)
            else:
                raise TrainerError("Weights must be a path or a dictionary.")

        # index the dataset
        # TODO: Can use identifier from the configuration?
        if self.current["log_per_atom_pred"]:
            for idx, config in enumerate(self.dataset):
                config.metadata |= {"index": idx}

            # TODO: add lmdb to the requirements
            import lmdb  # conditional import, only needed for per-atom predictions

            self.current["per_atom_pred_database"] = lmdb.open(
                f"{self.current['run_dir']}/per_atom_pred_database.lmdb",
                map_size=1e12,
                subdir=False,
            )

    def save_config(self):
        """
        Hash and save the configuration to the current run directory.
        """
        config_hash = self.get_trainer_hash()
        config_file = f"{self.current['run_dir']}/{config_hash}.yaml"
        with open(config_file, "w") as f:
            yaml.dump(self.config_to_dict(), f, default_flow_style=False)
        logger.info(f"Configuration saved in {config_file}.")

    def setup_dataset_transforms(self):
        """
        Set up the dataset transforms based on the provided information. If the
        transforms are not provided, it will raise an error. If the transform is of type
        ASE, it will be loaded from the ASE library. If the transform is of type
        KLIFF, it will be loaded from the KLIFF library. Left for the derived classes to
        implement.
        """
        # transforms?
        if self.transform_manifest:
            configuration_transform: Union[dict, None] = self.transform_manifest.get(
                "configuration", None
            )
            property_transform: Union[list, None] = self.transform_manifest.get(
                "property", None
            )

            if property_transform:
                for property_to_transform in property_transform:
                    property_name = property_to_transform.get("name", None)
                    if not property_name:
                        continue  # it is probably an empty property
                    transform_class_name = property_to_transform[property_name].get(
                        "name", None
                    )
                    if not transform_class_name:
                        raise TrainerError(
                            "Property transform module name not provided."
                        )
                    property_transform_module = importlib.import_module(
                        f"kliff.transforms.property_transforms"
                    )
                    TransformClass = getattr(
                        property_transform_module, transform_class_name
                    )
                    TransformClass = TransformClass(
                        proprty_key=property_name,
                        **property_to_transform[property_name].get("kwargs", {}),
                    )
                    self.dataset = TransformClass(self.dataset)
                    self.property_transforms.append(TransformClass)

            if configuration_transform:
                configuration_class_name: Union[str, None] = (
                    configuration_transform.get("name", None)
                )
                if not configuration_class_name:
                    logger.warning(
                        "Configuration transform module name not provided."
                        "Skipping configuration transform."
                    )
                else:
                    configuration_transform_module = importlib.import_module(
                        f"kliff.transforms.configuration_transforms"
                    )
                    ConfigurationClass = getattr(
                        configuration_transform_module, configuration_class_name
                    )
                    kwargs: Union[dict, None] = configuration_transform.get(
                        "kwargs", None
                    )
                    if not kwargs:
                        raise TrainerError(
                            "Configuration transform module options not provided."
                        )
                    ConfigurationClass = ConfigurationClass(
                        **kwargs, copy_to_config=False
                    )
                    self.configuration_transform = ConfigurationClass

    def setup_model(self):
        """
        Set up the model based on the provided information. If the model is not provided,
        it will be loaded from the model_path. If the model_path is not provided, it will
        raise an error. If the model_type is KIM, it will be loaded from the KIM model
        repository. If KIM type model is installed in CWD, it will be loaded from there, and
        model_path will be set to the KIM CWD. If model is of type TAR, it will be untarred
        and model_path will be set to the untarred directory. Left for the derived classes
        to implement.
        """
        raise TrainerError("setup_model not implemented.")

    def setup_parameter_transforms(self):
        """
        This method set up the transformed parameter space for models. It can be used
        for any model type in general, but as there exists a significant difference
        between how models handles their parameters, it is left for the subclass to
        implement. Although to ensure that `initialize` function remains consistent
        this method will not raise NotImplemented error, rather it will quietly pass.
        So be aware.
        """
        pass

    def setup_optimizer(self):
        """
        Set up the optimizer based on the provided information. If the optimizer is not
        provided, it will be loaded from the optimizer_name. If the optimizer_name is not
        provided, it will raise an error. If the optimizer_provider is scipy, it will be
        loaded from the scipy.optimize. If the optimizer_provider is torch, it will be
        loaded from the torch.optim. Left for the derived classes to implement.
        """
        raise TrainerError("setup_optimizer not implemented.")

    def setup_dataset_split(self):
        """
        Simple test train split for now, will have more options like stratification
         in the future.

        """
        # test train splits
        train_size = self.training_manifest.get("training_dataset", {}).get(
            "train_size", len(self.dataset)
        )
        val_size = self.training_manifest.get("validation_dataset", {}).get(
            "val_size", 0
        )

        # sanity checks
        if not isinstance(train_size, int) or train_size < 1:
            logger.warning(
                "Train size is not provided or is less than 1. Using full dataset for training."
            )
            train_size = len(self.dataset)
        else:
            logger.info(f"Training dataset size: {train_size}")

        if not isinstance(val_size, int) or val_size < 0:
            logger.warning(
                "Val size is not provided or is less than 0. Using 0 for validation."
            )
            val_size = 0
        else:
            logger.info(f"Validation dataset size: {val_size}")

        if train_size + val_size > len(self.dataset):
            raise TrainerError(
                "Sum of train, val, and test sizes is greater than the dataset size."
            )

        # check if indices are provided
        train_indices = self.dataset_sample_manifest.get("train_indices")
        val_indices = None
        if isinstance(train_indices, str):
            train_indices = np.genfromtxt(train_indices, dtype=int)
            if val_size > 0:
                val_indices = np.genfromtxt(
                    self.dataset_sample_manifest.get("val_indices"), dtype=int
                )
        else:
            TrainerError(f"Could not load indices from {train_indices}.")

        if train_indices is None:
            indices = np.random.permutation(len(self.dataset))
            train_indices = indices[:train_size]
            if val_size > 0:
                val_indices = indices[-val_size:]

        self.dataset_sample_manifest["train_size"] = train_size
        self.dataset_sample_manifest["val_size"] = val_size
        self.dataset_sample_manifest["train_indices"] = train_indices
        self.dataset_sample_manifest["val_indices"] = val_indices

        train_dataset = self.dataset[train_indices]
        self.train_dataset = train_dataset

        if val_size > 0:
            val_dataset = self.dataset[val_indices]
            self.val_dataset = val_dataset
        else:
            self.val_dataset = None

        # save the indices if generated
        if isinstance(train_indices, str):
            self.dataset_sample_manifest["indices_files"]["train"] = train_indices
        else:
            self.dataset_sample_manifest["indices_files"][
                "train"
            ] = f"{self.current['run_dir']}/train_indices.txt"
            np.savetxt(
                self.dataset_sample_manifest["indices_files"]["train"],
                train_indices,
                fmt="%d",
            )

        if val_size > 0:
            if isinstance(val_indices, str):
                self.dataset_sample_manifest["indices_files"]["val"] = val_indices
            else:
                self.dataset_sample_manifest["indices_files"][
                    "val"
                ] = f"{self.current['run_dir']}/val_indices.txt"
                np.savetxt(
                    self.dataset_sample_manifest["indices_files"]["val"],
                    val_indices,
                    fmt="%d",
                )

    def log_per_atom_outputs(
        self,
        epoch: int,
        idx: Union[List[int], np.ndarray],
        predictions: List[np.ndarray],
    ):
        """
        Log the per atom outputs to the database. It saves dictionary of predictions and
        n_atoms for each configuration. The key for predictions is pred_{n}, where n is
        the index of the prediction. For more than one prediction, it will save pred_0,
        pred_1, pred_2, etc. The key for the indices is idx

        Args:
            epoch: Current epoch
            idx: List of indices of the configurations
            predictions: List of predictions for the configurations

        """
        if self.current["per_atom_pred_database"] is None:
            return

        with self.current["per_atom_pred_database"].begin(write=True) as txn:
            for ids, pred in zip(idx, predictions):
                if pred is not None:
                    txn.put(
                        f"epoch_{epoch}|index_{ids}".encode(),
                        pkl.dumps({"pred_0": pred, "n_atoms": pred.shape[0]}),
                    )
                else:
                    logger.warning(f"Prediction for index {ids} is None. Skipping.")

    def loss(self, *args, **kwargs):
        raise TrainerError("loss not implemented.")

    def checkpoint(self, *args, **kwargs):
        raise TrainerError("checkpoint not implemented.")

    def train_step(self, *args, **kwargs):
        raise TrainerError("train_step not implemented.")

    def validation_step(self, *args, **kwargs):
        raise TrainerError("validation_step not implemented.")

    def get_optimizer(self, *args, **kwargs):
        raise TrainerError("get_optimizer not implemented.")

    def train(self, *args, **kwargs):
        raise TrainerError("train not implemented.")

    def save_kim_model(self, *args, **kwargs):
        raise TrainerError("save_kim_model not implemented.")

    @staticmethod
    def _generate_kim_cmake(model_name: str, driver_name: str, file_list: List) -> str:
        """
        Generate the CMakeLists.txt file for KIM API. This will be used to compile the
        driver with the KIM API. The driver name is the name of the driver, and the file
        list is the list of files to be included in the CMakeLists.txt file.
        Private method.
        Args:
            driver_name: Name of the driver
            file_list: List of files to be included in the CMakeLists.txt file
        Returns:
            CMakeLists.txt file as a string
        """
        model_name = model_name.replace("-", "_")
        cmake = f"""cmake_minimum_required(VERSION 3.10)

                    list(APPEND CMAKE_PREFIX_PATH $ENV{{KIM_API_CMAKE_PREFIX_DIR}})
                    find_package(KIM-API-ITEMS 2.2 REQUIRED CONFIG)

                    kim_api_items_setup_before_project(ITEM_TYPE "portableModel")
                    project({model_name})
                    kim_api_items_setup_after_project(ITEM_TYPE "portableModel")

                    add_kim_api_model_library(
                    NAME            ${{PROJECT_NAME}}
                    DRIVER_NAME     "{driver_name}"
                    PARAMETER_FILES {" ".join(file_list)}
                    )
                """
        return cmake

    def write_training_env_edn(self, path: str):
        """
        Generate the training_env.edn file for the KIM API. This file will be used to
        accurately determine the training environment . The file will be saved in the current run directory.
        It saves the hash of the configuration, and list of all python dependencies from
        pip freeze.
        """
        env_file = f"{path}/training_env.edn"
        hash = self.get_trainer_hash()
        with open(env_file, "w") as f:
            try:
                from pip._internal.operations.freeze import freeze

                from kliff import __version__
            except ImportError:
                logger.warning(
                    "Could not import kliff version or pip freeze. Skipping."
                )
                return
            python_env = []
            for module in list(freeze()):
                if "@" in module:
                    module = module.split("@")[0]
                python_env.append(module)

            f.write("{\n")
            f.write(f'"kliff-version" "{__version__}"\n')
            f.write(f'"trainer-used" "{type(self).__name__}"\n')
            f.write(f'"manifest-hash" "{hash}"\n')
            f.write(f'"python-dependencies" [\n')
            for module in python_env:
                f.write(f'    "{module}"\n')
            f.write(f"]\n")
            f.write("}\n")


class TrainerError(Exception):
    """
    Exceptions to be raised in Trainer and associated classes.
    """

    def __init__(self, message):
        super().__init__(message)


# TODO:
# 1. Test dataset
# 2. Stress
# 3. Get top k models
