import hashlib
import json
import os
import random
from copy import deepcopy
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path
from typing import Callable, Union

import dill  # TODO: include dill in requirements.txt
import numpy as np
import yaml
from loguru import logger

from kliff.dataset import Dataset


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

    Args:
        training_manifest: training manifest
    """

    def __init__(self, training_manifest: dict):
        # workspace variables
        self.workspace: dict = {
            "name": "kliff_workspace",
            "seed": 12345,
            "resume": False,
            "walltime": "2:00:00:00",
        }

        # dataset variables
        self.dataset_manifest: dict = {
            "type": "kliff",
            "path": "./",
            "save": False,
            "shuffle": False,
            "ase_keys": {"energy": "energy", "forces": "forces"},
            "colabfit_dataset": {
                "dataset_name": None,
                "database_name": None,
                "database_url": None,
            },
        }
        self.dataset = None

        # model variables
        self.model_manifest: dict = {
            "type": "kim",
            "name": None,
            "path": None,
        }
        self.model: Callable = None

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
            "loss_traj": False,
        }

        self.optimizer_manifest: dict = {
            "provider": "scipy",
            "name": None,
            "learning_rate": None,
            "kwargs": None,
            "epochs": 10000,
            "stop_condition": None,
            "num_workers": 1,
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
            "model_type": None,
            "model_name": None,
            "model_path": None,
        }

        # state variables
        self.current: dict = {
            "run_title": None,
            "run_dir": None,
            "run_hash": None,
            "start_time": None,
            "end_time": None,
            "best_loss": None,
            "best_model": None,
            "loss": None,
            "epoch": 0,
            "step": 0,
            "device": "cpu",
            "expected_end_time": None,
            "warned_once": False,
            "dataset_hash": None,
            "appending_to_previous_run": False,
            "verbose": False,
            "ckpt_interval": 100,
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

        if isinstance(self.workspace["walltime"], int):
            expected_end_time = datetime.now() + timedelta(
                seconds=self.workspace["walltime"]
            )
        else:
            expected_end_time = datetime.now() + timedelta(
                days=int(self.workspace["walltime"].split(":")[0]),
                hours=int(self.workspace["walltime"].split(":")[1]),
                minutes=int(self.workspace["walltime"].split(":")[2]),
                seconds=int(self.workspace["walltime"].split(":")[3]),
            )
        self.current["expected_end_time"] = expected_end_time

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
            # TODO: implement resume
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
            # TODO: implement resume

        self.optimizer_manifest |= self.training_manifest.get("optimizer")
        self.optimizer_manifest["epochs"] = self.training_manifest.get("epochs", 10000)
        self.optimizer_manifest["stop_condition"] = self.training_manifest.get(
            "stop_condition", None
        )
        self.optimizer_manifest["num_workers"] = self.training_manifest.get(
            "num_workers", 1
        )
        self.optimizer_manifest["batch_size"] = self.training_manifest.get(
            "batch_size", 1
        )

        self.current["ckpt_interval"] = self.training_manifest.get("ckpt_interval", 100)
        self.current["verbose"] = self.training_manifest.get("verbose", False)
        self.current["device"] = self.training_manifest.get("device", "cpu")

        # dataset sample variables will be processed in the setup_dataset method
        self.export_manifest |= manifest.get("export", {})

    def config_to_dict(self):
        """
        Convert the configuration to a dictionary.
        """
        config = {}
        config |= self.workspace
        config |= self.dataset_manifest
        config |= self.model_manifest
        config |= self.transform_manifest
        config |= self.training_manifest
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
        # Step 3 - Set up the workspace folder
        self.setup_workspace()
        # Step 4 - Read or load the dataset, initialize the property/configuration transforms
        self.setup_dataset()
        # Step 5 - Set up the test and train datasets, based on the provided indices
        self.setup_test_train_datasets()
        # Step 6 - Set up the model
        self.setup_model()
        # Step 6.5 - Setup parameter transform
        self.setup_parameter_transforms()
        # Step 7 - Set up the optimizer
        self.setup_optimizer()
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
        """
        dir_list = sorted(glob(f"{self.workspace['name']}*"))
        if len(dir_list) == 0 or not self.workspace["resume"]:
            self.current["appending_to_previous_run"] = False
            self.current["run_dir"] = (
                f"{self.workspace['name']}/{self.current['run_title']}"
            )
            os.makedirs(self.current["run_dir"], exist_ok=True)
        else:
            last_dir = dir_list[-1]
            was_it_finished = os.path.exists(f"{last_dir}/.finished")
            if was_it_finished:  # start new run
                current_run_dir = (
                    f"{self.workspace['name']}/{self.current['run_title']}"
                )
                os.makedirs(current_run_dir, exist_ok=True)
                self.current["appending_to_previous_run"] = False
            else:
                self.current["appending_to_previous_run"] = True
                self.current["run_dir"] = dir_list[-1]

    def setup_dataset(self):
        """
        Set up the dataset based on the provided information.

        TODO: It will check the {workspace}/{dataset_name} directory to see if dataset hash
        is already there. The dataset hash is determined but hashing the full path to
        the dataset + transforms names + configuration transform properties.
        TODO: reload hashed dataset if it exists.
        """
        dataset_module_manifest = deepcopy(self.dataset_manifest)
        dataset_module_manifest["weights"] = self.loss_manifest["weights"]
        self.dataset = Dataset.get_dataset_from_manifest(
            dataset_module_manifest, self.transform_manifest
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

    def setup_test_train_datasets(self):
        """
        Simple test train split for now, will have more options like stratification
         in the future.

        """
        # test train splits
        train_size = self.dataset_sample_manifest.get("train_size", len(self.dataset))
        val_size = self.dataset_sample_manifest.get("val_size", 0)

        # sanity checks
        if not isinstance(train_size, int) or train_size < 1:
            logger.warning(
                "Train size is not provided or is less than 1. Using full dataset for training."
            )
            train_size = len(self.dataset)

        if not isinstance(val_size, int) or val_size < 0:
            logger.warning(
                "Val size is not provided or is less than 0. Using 0 for validation."
            )
            val_size = 0

        if train_size + val_size > len(self.dataset):
            raise TrainerError(
                "Sum of train, val, and test sizes is greater than the dataset size."
            )

        # check if indices are provided
        train_indices = self.dataset_sample_manifest.get("train_indices")
        if train_indices is None:
            train_indices = np.arange(train_size, dtype=int)
        elif isinstance(train_indices, str):
            train_indices = np.genfromtxt(train_indices, dtype=int)
        else:
            TrainerError("train_indices should be a numpy array or a path to a file.")

        val_indices = self.dataset_sample_manifest.get("val_indices")
        if val_indices is None:
            val_indices = np.arange(train_size, train_size + val_size, dtype=int)
        elif isinstance(val_indices, str):
            val_indices = np.genfromtxt(val_indices, dtype=int)
        else:
            TrainerError("val_indices should be a numpy array or a path to a file.")

        if self.dataset_manifest.get("shuffle", False):
            # instead of shuffling the main dataset, validation/train indices are shuffled
            # this gives better control over future active learning scenarios
            np.random.shuffle(train_indices)
            np.random.shuffle(val_indices)

        train_dataset = self.dataset[train_indices]

        if val_size > 0:
            val_dataset = self.dataset[val_indices]
        else:
            val_dataset = None

        self.dataset_sample_manifest["train_size"] = train_size
        self.dataset_sample_manifest["val_size"] = val_size
        self.dataset_sample_manifest["train_indices"] = train_indices
        self.dataset_sample_manifest["val_indices"] = val_indices

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

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


class TrainerError(Exception):
    """
    Exceptions to be raised in Trainer and associated classes.
    """

    def __init__(self, message):
        super().__init__(message)
