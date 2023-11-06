import os
from copy import deepcopy
from datetime import datetime, timedelta
from enum import Enum
from glob import glob
from pathlib import Path

import numpy as np
import yaml
from loguru import logger


class ModelTypes(Enum):
    KIM = 0
    TORCH = 1
    TAR = 2

    @staticmethod
    def get_model_type(input_str: str):
        if input_str.lower() == "kim":
            return ModelTypes.KIM
        elif (
            input_str.lower() == "torch"
            or input_str.lower() == "pt"
            or input_str.lower() == "pth"
        ):
            return ModelTypes.TORCH
        elif input_str.lower() == "tar":
            return ModelTypes.TAR
        else:
            raise TrainerError(f"Model type {input_str} not supported.")

    @staticmethod
    def get_model_config(input_type):
        if input_type == ModelTypes.KIM:
            return "KIM"
        elif input_type == ModelTypes.TORCH:
            return "TORCH"
        elif input_type == ModelTypes.TAR:
            return "TAR"
        else:
            raise TrainerError(f"Model type {input_type} not supported.")


class DataTypes(Enum):
    ASE = 0
    COLABFIT = 1
    KLIFF = 2
    TORCH_GEOMETRIC = 3

    @staticmethod
    def get_data_type(input_str: str):
        if input_str.lower() == "ase":
            return DataTypes.ASE
        elif input_str.lower() == "colabfit":
            return DataTypes.COLABFIT
        elif input_str.lower() == "kliff":
            return DataTypes.KLIFF
        elif input_str.lower() == "torch_geometric":
            return DataTypes.TORCH_GEOMETRIC
        else:
            raise TrainerError(f"Data type {input_str} not supported.")

    @staticmethod
    def get_data_config(input_type):
        if input_type == DataTypes.ASE:
            return "ASE"
        elif input_type == DataTypes.COLABFIT:
            return "COLABFIT"
        elif input_type == DataTypes.KLIFF:
            return "KLIFF"
        elif input_type == DataTypes.TORCH_GEOMETRIC:
            return "TORCH_GEOMETRIC"
        else:
            raise TrainerError(f"Data type {input_type} not supported.")


class ConfigurationTransformationTypes(Enum):
    GRAPH = 0
    DESCRIPTORS = 1
    NEIGHBORS = 2

    @staticmethod
    def get_config_transformation_type(input_str: str):
        if input_str.lower() == "graph":
            return ConfigurationTransformationTypes.GRAPH
        elif input_str.lower() == "descriptors":
            return ConfigurationTransformationTypes.DESCRIPTORS
        elif input_str.lower() == "neighbors" or input_str.lower() == "none":
            return ConfigurationTransformationTypes.NEIGHBORS
        else:
            raise TrainerError(f"Configuration transformation type {input_str} not supported.")

    @staticmethod
    def get_config_transformation_config(input_type):
        if input_type == ConfigurationTransformationTypes.GRAPH:
            return "GRAPH"
        elif input_type == ConfigurationTransformationTypes.DESCRIPTORS:
            return "DESCRIPTORS"
        else:
            raise TrainerError(f"Configuration transformation type {input_type} not supported.")

class OptimizerProvider(Enum):
    TORCH = 0
    SCIPY = 1

    @staticmethod
    def get_optimizer_provider(input_str: str):
        if input_str.lower() == "torch":
            return OptimizerProvider.TORCH
        elif input_str.lower() == "scipy":
            return OptimizerProvider.SCIPY
        else:
            raise TrainerError(f"Optimizer provider {input_str} not supported.")

    @staticmethod
    def get_optimizer_config(input_type):
        if input_type == OptimizerProvider.TORCH:
            return "TORCH"
        elif input_type == OptimizerProvider.SCIPY:
            return "SCIPY"
        else:
            raise TrainerError(f"Optimizer provider {input_type} not supported.")


class Trainer:
    def __init__(self, configuration: dict):
        self.start_time = datetime.now()
        self.indices_file = None
        self.val_indices = None
        self.train_indices = None
        self.train_dataset = None
        self.val_dataset = None
        self.dataset = None
        self.model_source = None
        self.model = None
        self.optimizer = None
        logger.info(
            f"Starting training. Time: {self.start_time.strftime('%Y-%m-%d-%H-%M-%S')}"
        )

        self.configuration = self.parse_dict(configuration)

        # set computation limits
        logger.info(f"Starting trainer. {self.configuration['optimizer_provider']}")
        if self.configuration["optimizer_provider"] == OptimizerProvider.TORCH:
            # Cant interject SCIPY optimizer with walltime
            max_walltime = timedelta(seconds=configuration["max_walltime"])
            self.end_time = self.start_time + max_walltime

        self.root_dir = configuration["root_dir"]
        self.current_run_title = configuration["run_title"]
        self.append = configuration["append"]
        self.resume = configuration["resume"]
        self.current_run_dir = configuration["current_run_dir"]
        self.optimizer_provider = configuration["optimizer_provider"]
        self.device = configuration["device"]
        self.model_name = configuration["model_name"]
        self.model_source = configuration["model_source"]
        self.dataset_type = configuration["dataset_type"]
        self.dataset_path = configuration["dataset_path"]
        if self.dataset_type == DataTypes.COLABFIT:
            self.dataset_name = configuration["dataset_name"]
            self.database_name = configuration["database_name"]
        self.seed = configuration["seed"]

        # set up indices and dataset
        self.indices_file = configuration["indices_file"]
        self.energy_loss_weight = configuration["loss_weights"]["energy"]
        self.forces_loss_weight = configuration["loss_weights"]["forces"]

        self.checkpoint_freq = configuration["checkpoint_freq"]
        self.max_epochs = configuration["max_epoch"]

    def parse_dict(self, configuration: dict):
        if "run_title" not in configuration:
            logger.error("run_title not provided.")
            raise ValueError("run_title not provided.")

        if "root_dir" not in configuration:
            logger.warning("root_dir not provided.")
            configuration["root_dir"] = "root_dir"

        if "append" not in configuration:
            configuration["append"] = False

        resume, current_run_dir = self.workdir(
            f"{configuration['root_dir']}/{configuration['run_title']}",
            configuration["append"],
        )
        configuration["current_run_dir"] = current_run_dir
        configuration["resume"] = resume

        if "seed" not in configuration:
            configuration["seed"] = None

        if "model_name" not in configuration:
            configuration["model_name"] = "model"

        if "model_source" not in configuration:
            TrainerError("model_source not provided.")
        else:
            configuration["model_source"] = ModelTypes.get_model_type(
                configuration["model_source"]
            )

        if "dataset_type" not in configuration:
            TrainerError("dataset_type not provided.")
        else:
            configuration["dataset_type"] = DataTypes.get_data_type(
                configuration["dataset_type"]
            )

        if configuration["dataset_type"] == DataTypes.COLABFIT:
            if (
                "dataset_name" not in configuration
                or "database_name" not in configuration
            ):
                raise TrainerError("colabfit_name not provided.")
        elif (
            configuration["dataset_type"] == DataTypes.ASE
            or configuration["dataset_type"] == DataTypes.KLIFF
        ):
            if "dataset_path" not in configuration:
                raise TrainerError("dataset_name not provided.")

        # optimizer parameters
        configuration["optimizer_provider"] = OptimizerProvider.get_optimizer_provider(
            configuration["optimizer_provider"]
        )

        if configuration["optimizer_provider"] == OptimizerProvider.TORCH:
            if "n_train" not in configuration:
                raise TrainerError("n_train not provided.")
            if "n_val" not in configuration:
                raise TrainerError("n_val not provided.")
            if "batch_size" not in configuration:
                raise TrainerError("batch_size not provided.")
            if "max_epoch" not in configuration:
                configuration["max_epoch"] = 10000
            if "max_walltime" not in configuration:
                configuration["max_walltime"] = 48 * 60 * 60  # max in NYU Greene

        if "optimizer" not in configuration:
            configuration["optimizer"] = (
                "adam"
                if configuration["optimizer_provider"] == OptimizerProvider.TORCH
                else "l-bfgs-b"
            )

        # defaults

        if "optimizer_kwargs" not in configuration:
            configuration["optimizer_kwargs"] = {}

        if "indices_file" not in configuration:
            # to be populated later
            configuration["indices_file"] = {"train": None, "val": None}

        if "checkpoint_freq" not in configuration:
            configuration["checkpoint_freq"] = 100

        if "device" not in configuration:
            configuration["device"] = "cpu"

        if "loss_weights" not in configuration:
            configuration["loss_weights"] = {"energy": 1.0, "forces": 1.0}

        if "cpu_workers" not in configuration:
            configuration["cpu_workers"] = 1

        if "max_epoch" not in configuration:
            configuration["max_epoch"] = None

        if "max_walltime" not in configuration:
            configuration["max_walltime"] = None

        return configuration

    def get_dict(self):
        configuration_dict = deepcopy(self.configuration)
        configuration_dict["model_source"] = ModelTypes.get_model_config(
            configuration_dict["model_source"]
        )
        configuration_dict["dataset_type"] = DataTypes.get_data_config(
            configuration_dict["dataset_type"]
        )
        configuration_dict[
            "optimizer_provider"
        ] = OptimizerProvider.get_optimizer_config(
            configuration_dict["optimizer_provider"]
        )
        return configuration_dict

    def get_indices(self, size_of_dataset: int):
        if self.configuration["indices_file"]["train"] is None:
            all_indices = np.arange(size_of_dataset)
            np.random.shuffle(all_indices)
            self.train_indices = all_indices[: self.configuration["n_train"]]
            self.val_indices = all_indices[-self.configuration["n_val"] :]
        else:
            self.train_indices = np.load(self.configuration["indices_file"]["train"])
            self.val_indices = np.load(self.configuration["indices_file"]["val"])

    def workdir(
        self,
        current_run_dir,
        append,
    ):
        """
        Check all the existing runs in the root directory and see if it finished the run
        :param current_run_dir:
        :return:
        """
        dir_list = sorted(glob(f"{current_run_dir}*"))
        if len(dir_list) == 0:
            resume = False
            current_run_dir = current_run_dir
            return resume, current_run_dir
        elif not append:
            resume = False
            current_run_dir = (
                f"{current_run_dir}_{self.start_time.strftime('%Y-%m-%d-%H-%M-%S')}"
            )
            return resume, current_run_dir
        else:
            last_dir = dir_list[-1]
            was_it_finished = os.path.exists(f"{last_dir}/.finished")
            if was_it_finished:
                resume = False
                current_run_dir = (
                    f"{current_run_dir}_{self.start_time.strftime('%Y-%m-%d-%H-%M-%S')}"
                )
                return resume, current_run_dir

        # incomplete run encountered
        # config_file = f"{dir_list[-1]}/config.yaml"
        # try:
        #     with open(config_file, "r") as f:
        #         last_config = yaml.safe_load(f)
        # except FileNotFoundError:
        #     raise FileNotFoundError(f"Previous config file not found, most likely corrupted data.")

        # check if anything changed from the last time
        # dataset
        # when can we resume vs new run?
        return True, dir_list[-1]

    @classmethod
    def from_file(cls, filename: Path):
        with open(filename, "r") as f:
            configuration = yaml.safe_load(f)
        configuration["filename"] = str(filename)
        return cls(configuration)

    def to_file(self, filename):
        configuration = self.get_dict()
        try:
            if self.indices_file is None:
                configuration["indices_file"]["train"] = (
                    filename.split("/")[-1] + "train_indices.txt"
                )
                configuration["indices_file"]["val"] = (
                    filename.split("/")[-1] + "val_indices.txt"
                )
                np.savetxt(configuration["indices_file"]["train"], self.train_indices)
                np.savetxt(configuration["indices_file"]["val"], self.val_indices)
        except ValueError:
            logger.warning("Indices file not saved. It is normal for KIM models.")

        with open(filename, "w") as f:
            yaml.dump(configuration, f, default_flow_style=False)

    def loss(self, energy_prediction, energy_target, forces_prediction, forces_target):
        TrainerError("loss not implemented.")

    def checkpoint(self):
        TrainerError("checkpoint not implemented.")

    def train_step(self):
        TrainerError("train_step not implemented.")

    def validation_step(self):
        TrainerError("validation_step not implemented.")

    def get_optimizer(self):
        TrainerError("get_optimizer not implemented.")

    def get_dataset(self):  # Specific to trainer
        TrainerError("get_dataset not implemented.")

    def train(self):
        TrainerError("train not implemented.")


class TrainerError(Exception):
    def __init__(self, message):
        super().__init__(message)
