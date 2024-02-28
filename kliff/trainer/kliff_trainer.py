import json
import os
import random
from copy import deepcopy
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path
from typing import Callable, Union

import numpy as np
import yaml
from loguru import logger

import kliff.transforms.configuration_transforms
from kliff._exceptions import TrainerError

from .option_enumerations import DataSource, ModelTypes, OptimizerProvider
import importlib

import dill  # TODO: include dill in requirements.txt
import hashlib

from kliff.dataset import Dataset
from kliff.transforms.configuration_transforms import ConfigurationTransform
from kliff.transforms.parameter_transforms import ParameterTransform
from kliff.transforms.property_transforms import PropertyTransform


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
    Model and optimizer setup are left for the derived classes to implement.

    Args:
        configuration: configuration dictionary
    """

    def __init__(self, configuration: dict):
        # workspace variables
        self.workspace_name = None  # name of default directory, root
        self.workspace_name = (
            None  # where to save everything from current run (inside workspace)
        )
        self.current_run_title = (
            None  # title of current run, usually model name + date and time
        )
        self.export_kim_model = False  # whether to export the model to KIM model
        self.seed = 12345  # random seed
        self.resume = False  # whether to resume from previous run (conditions apply)
        self.walltime = None  # maximum walltime for the run

        # dataset variables
        self.dataset_type: DataSource = DataSource.UNDEFINED
        self.dataset_path = None
        self.dataset_save = None
        self.dataset_shuffle = None
        self.dataset = None
        self.train_size = None
        self.val_size = None
        self.indices_files: dict = {"train": None, "val": None}
        self.ase_keys = {"energy_key": "energy", "forces_key": "forces"}
        self.val_indices = None
        self.train_indices = None
        self.train_dataset = None
        self.val_dataset = None
        self.colabfit_dataset: dict = {
            "dataset_name": None,
            "database_name": None,
            "database_url": None,
        }

        # model variables
        self.model_type: ModelTypes = ModelTypes.UNDEFINED
        self.model: Callable = None
        self.model_name = None  # KIM string or name of pt/pth file
        self.model_path = None  # path to the model file

        # transform variables
        self.property_transform: PropertyTransform = None
        self.property_transform_options = None
        self.parameter_transform: ParameterTransform = None
        self.parameter_transform_options = None
        self.configuration_transform: ConfigurationTransform = None
        self.configuration_transform_options = None

        # training variables
        self.loss_function: Callable = None
        self.energy_loss_weight = 1.0
        self.forces_loss_weight = 0.0

        self.optimizer_provider: OptimizerProvider = OptimizerProvider.UNDEFINED
        self.optimizer = None  # instance of optimizer, "scipy" for scipy torch.optim instance for torch
        self.optimizer_name = (
            None  # name of optimizer, e.g. "l-bfgs-b" for scipy, "adam" for torch
        )
        self.learning_rate = None  # learning rate for torch

        self.max_epochs = 10000  # maximum number of epochs
        self.device = "cpu"
        self.batch_size = 1
        self.chkpt_interval = 100
        self.stop_condition = None  # function to check if training should stop

        self.configuration = self.config_from_dict(configuration)

        # state variables
        self.current_epoch = 0
        self.current_step = 0
        self.current_best_loss = None
        self.current_best_model = None
        self.current_loss = None
        self.current_run_dir = None
        self.appending_to_previous_run = False
        self.current_dataset_hash = None
        self.start_current_run_title = None  # start time of the current run
        self.expected_end_time = None

        self._initialize()

    def config_from_dict(self, configuration: dict):
        """
        It accepts the raw configuration dictionary, and processes it to the formatted
        configuration. This includes mapping the string fields to enums, and setting sane
        defaults for missing fields.

        Args:
            configuration: raw incoming dictionary

        Returns:
            Processed configuration dictionary
        """
        start_time = datetime.now()
        date_time_str = start_time.strftime("%Y-%m-%d-%H-%M-%S")
        processed_configuration = {}

        # Workspace variables
        workspace_block = configuration.get("workspace", None)
        if workspace_block is not None:
            processed_configuration["start_time"] = start_time
            processed_configuration["workspace_name"] = workspace_block.get(
                "name", f"kliff_{date_time_str}"
            )
            processed_configuration["current_run_title"] = (
                None  # will be assigned in the model block
            )
            processed_configuration["export_kim_model"] = workspace_block.get(
                "export", False
            )
            processed_configuration["seed"] = workspace_block.get("seed", 12345)
            processed_configuration["resume"] = workspace_block.get("resume", False)
            walltime: Union[str,int] = workspace_block.get("walltime", "2:00:00:00")
            if type(walltime) is int: # yaml parsed the time
                processed_configuration["walltime"] = timedelta(seconds=walltime)
            elif type(walltime) is str:
                processed_configuration["walltime"] = timedelta(
                    days=int(walltime.split(":")[0]),
                    hours=int(walltime.split(":")[1]),
                    minutes=int(walltime.split(":")[2]),
                    seconds=int(walltime.split(":")[3]),
                )
            else:
                raise TrainerError("Walltime not in correct format. dd:hh:mm:ss expected.")
            processed_configuration["expected_end_time"] = (
                start_time + processed_configuration["walltime"]
            )
        else:
            raise TrainerError("Workspace block not found in the configuration.")

        # Dataset variables
        dataset_block = configuration.get("dataset", None)
        if dataset_block is not None:
            processed_configuration["dataset_type"] = DataSource.get_data_enum(
                dataset_block.get("type", "kliff")
            )
            processed_configuration["dataset_path"] = dataset_block.get(
                "path", None
            )
            processed_configuration["dataset_save"] = dataset_block.get("save", False)
            processed_configuration["dataset_shuffle"] = dataset_block.get(
                "shuffle", False
            )
            ase_keys = dataset_block.get("keys", {})
            processed_configuration["ase_keys"] = {
                "energy_key": ase_keys.get("energy", "energy"),
                "forces_key": ase_keys.get("forces", "forces"),
            }
            train_dataset_info = dataset_block.get("training_dataset", None)
            if train_dataset_info is not None:
                # none values will be tackled during dataset loading
                processed_configuration["train_size"] = train_dataset_info.get(
                    "train_size", None
                )
                processed_configuration["train_indices"] = train_dataset_info.get(
                    "train_indices", None
                )
            else:
                processed_configuration["train_size"] = None
                processed_configuration["train_indices"] = None

            val_dataset_info = dataset_block.get("validation_dataset", None)
            if val_dataset_info is not None:
                processed_configuration["val_size"] = val_dataset_info.get(
                    "val_size", None
                )
                processed_configuration["val_indices"] = val_dataset_info.get(
                    "val_indices", None
                )
            else:
                processed_configuration["val_size"] = None
                processed_configuration["val_indices"] = None
            processed_configuration["indices_file"] = {"train": None, "val": None}
            if type(processed_configuration["train_indices"]) is str:
                processed_configuration["indices_file"] = processed_configuration[
                    "train_indices"
                ]
            if type(processed_configuration["val_indices"]) is str:
                processed_configuration["indices_file"] = processed_configuration[
                    "val_indices"
                ]

            processed_configuration["train_dataset"] = None  # To be assigned
            processed_configuration["val_dataset"] = None  # To be assigned
            processed_configuration["dataset"] = None  # To be assigned

            colabfit_dict = dataset_block.get("colabfit_dataset", None)
            if colabfit_dict is not None:
                processed_configuration["colabfit_dataset"] = {
                    "dataset_name": colabfit_dict.get("dataset_name", None),
                    "database_name": colabfit_dict.get("database_name", None),
                    "database_url": colabfit_dict.get("database_url", None),
                }
        else:
            raise TrainerError("Dataset block not found in the configuration.")

        # model variables
        model_block = configuration.get("model", {})
        processed_configuration["model_type"] = ModelTypes.get_model_enum(
            model_block.get("model_type", "kim")
        )
        processed_configuration["model_name"] = model_block.get("model_name", None)
        processed_configuration["model_path"] = model_block.get("model_path", None)
        processed_configuration["model"] = None  # To be assigned
        if processed_configuration["model_name"] is None:
            processed_configuration["current_run_title"] = (
                f"{processed_configuration['model_type']}_{date_time_str}"
            )
        else:
            processed_configuration["current_run_title"] = (
                f"{processed_configuration['model_name']}_{date_time_str}"
            )

        # transform variables
        transform_block = configuration.get("transforms", {})
        property_transform_sub_block = transform_block.get("property", {})
        parameter_transform_sub_block = transform_block.get("parameter", {})
        configuration_transform_sub_block = transform_block.get("configuration", {})

        processed_configuration["property_transform_options"] = {
            "name": property_transform_sub_block.get("name", None),
            "property_key": property_transform_sub_block.get("property_key", None)
        }
        processed_configuration["property_transform"] = (
            property_transform_sub_block.get("instance", None)
        )  # no executable given. initialize on own

        processed_configuration["parameter_transform_options"] = {
            "name": parameter_transform_sub_block.get("name", None),
        }
        processed_configuration["parameter_transform"] = (
            parameter_transform_sub_block.get("instance", None)
        )  # no executable given. initialize on own

        # map default hyperparameters
        configuration_transform_kwargs = configuration_transform_sub_block.get("kwargs", {})
        hyperparams = configuration_transform_kwargs.get("hyperparameters", None)
        if hyperparams == "default":
            configuration_transform_kwargs["hyperparameters"] = \
                kliff.transforms.configuration_transforms.get_default_hyperparams()

        processed_configuration["configuration_transform_options"] = (
            configuration_transform_sub_block  # this might contain lot of variables
        )
        processed_configuration["configuration_transform"] = (
            configuration_transform_sub_block.get("instance", None)
        )  # no executable given. initialize on own

        # training variables
        training_block = configuration.get("training", {})
        loss_block = training_block.get("loss", {})
        processed_configuration["loss_function"] = loss_block.get("loss_function", None)
        processed_configuration["energy_loss_weight"] = loss_block.get(
            "energy_loss_weight", 1.0
        )
        processed_configuration["forces_loss_weight"] = loss_block.get(
            "forces_loss_weight", 0.0
        )

        optimizer_block = training_block.get("optimizer", {})
        processed_configuration["optimizer_provider"] = (
            OptimizerProvider.get_optimizer_enum(
                optimizer_block.get("provider", "scipy")
            )
        )
        processed_configuration["optimizer"] = None  # To be assigned
        processed_configuration["optimizer_name"] = optimizer_block.get("name", None)
        processed_configuration["learning_rate"] = optimizer_block.get(
            "learning_rate", None
        )

        processed_configuration["max_epochs"] = training_block.get("max_epochs", 10000)
        processed_configuration["device"] = training_block.get("device", "cpu")
        processed_configuration["batch_size"] = training_block.get("batch_size", 1)
        processed_configuration["chkpt_interval"] = training_block.get(
            "chkpt_interval", 100
        )
        processed_configuration["stop_condition"] = training_block.get(
            "stop_condition", None
        )

        return processed_configuration

    def config_to_dict(self):
        """
        Convert the configuration to a dictionary.
        """
        config = {}
        config["workspace"] = {
            "name": self.workspace_name,
            "export": self.export_kim_model,
            "seed": self.seed,
            "resume": self.resume,
            "walltime": self.walltime.total_seconds(),
        }

        config["dataset"] = {
            "type": DataSource.get_data_str(self.dataset_type),
            "path": self.dataset_path,
            "save": self.dataset_save,
            "shuffle": self.dataset_shuffle,
            "training_dataset": {
                "train_size": self.train_size,
                "train_indices": self.indices_files["train"],
            },
            "validation_dataset": {
                "val_size": self.val_size,
                "val_indices": self.indices_files["val"],
            },
            "colabfit_dataset": {
                "dataset_name": self.colabfit_dataset["dataset_name"],
                "database_name": self.colabfit_dataset["database_name"],
                "database_url": self.colabfit_dataset["database_url"],
            }
        }
        if self.ase_keys is not None:
            config["dataset"]["keys"] = {
                "energy": self.ase_keys["energy_key"],
                "forces": self.ase_keys["forces_key"],
            }

        config["model"] = {
            "model_type": ModelTypes.get_model_str(self.model_type),
            "model_name": self.model_name,
            "model_path": self.model_path,
        }

        config["transforms"] = {
            "property": {
                "name": self.property_transform_options["name"],
                "property_key": self.property_transform_options["property_key"],
            },
            "parameter": {
                "name": self.parameter_transform_options["name"],
            },
            "configuration": {
                "name": self.configuration_transform_options["name"],
                "kwargs": self.configuration_transform_options,
            }
        }

        config["training"] = {
            "loss": {
                "loss_function": self.loss_function,
                "weight": {
                    "energy": self.energy_loss_weight,
                    "forces": self.forces_loss_weight,
                },
            },
            "optimizer": {
                "provider": OptimizerProvider.get_optimizer_str(self.optimizer_provider),
                "name": self.optimizer_name,
                "learning_rate": self.learning_rate,
            },
            "epochs": self.max_epochs,
            "device": self.device,
            "batch_size": self.batch_size,
            "chkpt_interval": self.chkpt_interval,
            "stop_condition": self.stop_condition,
        }

        return config

    @classmethod
    def from_file(cls, filename: Path):
        """
        Load the configuration from a YAML file.

        Args:
            filename: name of the yaml file

        Returns:
            Trainer instance

        """
        with open(filename, "r") as f:
            configuration = yaml.safe_load(f)
        configuration["filename"] = str(filename)
        return cls(configuration)

    def get_trainer_hash(self):
        """
        Get the hash of the current configuration. It will be used to create a unique
        directory for the current run. It will be the hash of the configuration dictionary
        string.
        """
        config = self.config_to_dict()
        config_immut_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_immut_str.encode()).hexdigest()

    def _initialize(self):
        """
        Initialize the trainer. Assigns the configuration objects, and
        call setup methods.
        """
        # Step 1 - Assign the processed configuration objects to the class variables
        for key, value in self.configuration.items():
            setattr(self, key, value)
        # Step 2 - Initialize all seeds
        self.seed_all()
        # Step 3 - Set up the workspace folder
        self.setup_workspace()
        # Step 4 - Read or load the dataset, initialize the property/configuration transforms
        self.setup_dataset()
        # Step 5 - Set up the test and train datasets, based on the provided indices
        self.setup_test_train_datasets()
        # Step 6 - Set up the model
        self.setup_model()
        # Step 7 - Set up the optimizer
        self.setup_optimizer()
        # Step 8 - Save the configuration for future
        self.save_config()

    def seed_all(self):
        """
        Seed all the random number generators.
        """
        np.random.seed(self.seed)
        random.seed(self.seed)
        # torch.manual_seed(self.seed) # enable torch seed in respective children
        # torch.cuda.manual_seed_all(self.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    def setup_workspace(self):
        """
        Check all the existing runs in the root directory and see if it finished the run
        """
        dir_list = sorted(glob(f"{self.workspace_name}*"))
        if len(dir_list) == 0 or not self.resume:
            self.appending_to_previous_run = False
            self.current_run_dir = f"{self.workspace_name}/{self.current_run_title}"
            os.makedirs(self.current_run_dir, exist_ok=True)
        else:
            last_dir = dir_list[-1]
            was_it_finished = os.path.exists(f"{last_dir}/.finished")
            if was_it_finished:  # start new run
                current_run_dir = f"{self.workspace_name}/{self.current_run_title}"
                os.makedirs(current_run_dir, exist_ok=True)
                self.appending_to_previous_run = False
            else:
                self.appending_to_previous_run = True
                self.current_run_dir = dir_list[-1]

    def setup_dataset(self):
        """
        Set up the dataset based on the provided information. It will check the
        {workspace}/{dataset_name} directory to see if dataset hash is already there.
        The dataset hash is determined but hashing the full path to the dataset +
        transforms names + configuration transform properties.
        """
        dataset_path = ""
        dataset_transforms = ""
        dataset_hash = ""
        if self.dataset_type == DataSource.KLIFF or self.dataset_type == DataSource.ASE:
            dataset_path = os.path.abspath(self.dataset_path)
        elif self.dataset_type == DataSource.COLABFIT:
            dataset_path = (
                self.colabfit_dataset["database_url"]
                + "/"
                + self.colabfit_dataset["database_name"]
            )
        else:
            raise TrainerError(f"Dataset type {self.dataset_type} not supported.")

        if self.property_transform_options is not None:
            dataset_transforms += self.property_transform_options["name"]

        dataset_transforms += "_"

        if self.configuration_transform_options is not None:
            dataset_transforms += self.configuration_transform_options["name"] + "_"
            dataset_transforms += str(self.configuration_transform_options["kwargs"]["cutoff"])

        dataset_hash_str = dataset_path + "_" + dataset_transforms
        dataset_hash = hashlib.md5(dataset_hash_str.encode()).hexdigest()
        self.current_dataset_hash = dataset_hash
        dataset_dir = f"{self.workspace_name}/{dataset_hash}"
        os.makedirs(dataset_dir, exist_ok=True)
        try:
            self.dataset = dill.load(open(f"{dataset_dir}/dataset.dill", "rb"))
            logger.info(f"Dataset found in {dataset_dir}.")
            return
        except FileNotFoundError:
            logger.info(
                f"Dataset not found in {self.workspace_name} directory. Creating dataset."
            )

        if self.dataset_type == DataSource.KLIFF:
            dataset = Dataset.from_path(dataset_path)
        elif self.dataset_type == DataSource.ASE:
            dataset = Dataset.from_ase(dataset_path, **self.ase_keys)
        elif self.dataset_type == DataSource.COLABFIT:
            dataset = Dataset.from_colabfit(
                self.colabfit_dataset["dataset_name"],
                self.colabfit_dataset["database_name"],
                self.colabfit_dataset["database_url"],
            )
        else:
            raise TrainerError(f"Dataset type {self.dataset_type} not supported.")

        if self.property_transform is not None:
            if not isinstance(self.property_transform, PropertyTransform):
                raise TrainerError(
                    "Property transform is not none and not an instance of PropertyTransform."
                )
        else:
            # check if property_instance_options have "instance"
            if self.property_transform_options.get("instance") is not None:
                self.property_transform = self.property_transform_options["instance"]
            else:
                try:
                    # try getting class "name" from kliff.transforms.property_transforms
                    module = importlib.import_module(
                        "kliff.transforms.property_transforms"
                    )
                    class_ = getattr(module, self.property_transform_options["name"])
                    self.property_transform = class_(
                        property_key=self.property_transform_options["property_key"],
                    )
                except AttributeError:
                    raise TrainerError(
                        f"Property transform {self.property_transform_options['name']} not found."
                        "If it is a custom transform, please provide the instance."
                    )

        self.property_transform(dataset)

        if self.configuration_transform is not None:
            if not isinstance(self.configuration_transform, ConfigurationTransform):
                raise TrainerError(
                    "Configuration transform is not none and not an instance of ConfigurationTransform."
                )
        else:
            # check if configuration_instance_options have "instance"
            if "instance" in self.configuration_transform_options \
                    and self.configuration_transform_options["instance"] is not None:
                self.configuration_transform = self.configuration_transform_options[
                    "instance"
                ]
            else:
                try:
                    # try getting class "name" from kliff.transforms.configuration_transforms
                    module = importlib.import_module(
                        "kliff.transforms.configuration_transforms"
                    )
                    class_ = getattr(
                        module, self.configuration_transform_options["name"]
                    )
                    self.configuration_transform = class_(
                        **self.configuration_transform_options["kwargs"],
                        copy_to_config=True,
                    )
                except AttributeError:
                    raise TrainerError(
                        f"Configuration transform {self.configuration_transform_options['name']} not found."
                        "If it is a custom transform, please provide the instance."
                    )
        for configuration in dataset:
            self.configuration_transform(configuration)

        dill.dump(dataset, open(f"{dataset_dir}/dataset.dill", "wb"))
        logger.info(f"Dataset saved in {dataset_dir}.")
        if self.dataset_shuffle:
            random.shuffle(dataset.configs)
        self.dataset = dataset

    def setup_test_train_datasets(self):
        """
        Set up the test and train datasets based on the provided indices. If the indices
        are not provided, shuffled serial indices will be used. If val_indices are not
        provided, the train_indices no validation dataset will be used.
        """

        # training indices
        if self.indices_files["train"] is not None:
            self.train_indices = np.load(self.indices_files["train"])
        else:
            if self.train_size is not None:
                self.train_indices = np.arange(self.train_size)
            else:
                self.train_indices = np.arange(len(self.dataset))

        # validation indices
        if self.indices_files["val"] is not None:
            self.val_indices = np.load(self.indices_files["val"])
        else:
            if self.val_size is not None:
                self.val_indices = np.arange(self.val_size)
            else:
                self.val_indices = None

        self.train_dataset = self.dataset[self.train_indices]
        self.indices_files["train"] = f"{self.current_run_dir}/train_indices.npy"
        self.train_indices.dump(self.indices_files["train"])

        if self.val_indices:
            self.val_dataset = self.dataset[self.val_indices]
            self.indices_files["val"] = f"{self.current_run_dir}/val_indices.npy"
            self.val_indices.dump(self.indices_files["val"])

    def save_config(self):
        """
        Hash and save the configuration to the current run directory.
        """
        config_hash = self.get_trainer_hash()
        config_file = f"{self.current_run_dir}/{config_hash}.yaml"
        with open(config_file, "w") as f:
            yaml.dump(self.configuration, f, default_flow_style=False)
        logger.info(f"Configuration saved in {config_file}.")

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

    def setup_optimizer(self):
        """
        Set up the optimizer based on the provided information. If the optimizer is not
        provided, it will be loaded from the optimizer_name. If the optimizer_name is not
        provided, it will raise an error. If the optimizer_provider is scipy, it will be
        loaded from the scipy.optimize. If the optimizer_provider is torch, it will be
        loaded from the torch.optim. Left for the derived classes to implement.
        """
        raise TrainerError("setup_optimizer not implemented.")

    def loss(self, *args, **kwargs):
        TrainerError("loss not implemented.")

    def checkpoint(self, *args, **kwargs):
        TrainerError("checkpoint not implemented.")

    def train_step(self, *args, **kwargs):
        TrainerError("train_step not implemented.")

    def validation_step(self, *args, **kwargs):
        TrainerError("validation_step not implemented.")

    def get_optimizer(self, *args, **kwargs):
        TrainerError("get_optimizer not implemented.")

    def train(self, *args, **kwargs):
        TrainerError("train not implemented.")
