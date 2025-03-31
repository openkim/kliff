import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from kliff.trainer.torch_trainer import DNNTrainer


def test_descriptor_trainer():
    """
    Basic tests for proper initialization of the Trainer module
    """
    manifest_file_template = (
        Path(__file__)
        .parents[1]
        .joinpath("test_data/trainer_data/training_manifest_ase_dnn.yaml.tpl")
    )
    data_file = Path(__file__).parents[1].joinpath("test_data/configs/Si_4.xyz")
    model_file = (
        Path(__file__).parents[1].joinpath("test_data/trainer_data/model_dnn.pt")
    )

    manifest_data = yaml.safe_load(open(manifest_file_template, "r"))

    manifest_data["dataset"]["path"] = str(data_file)
    manifest_data["model"]["path"] = str(model_file)

    manifest_file = (
        Path(__file__)
        .parents[1]
        .joinpath("test_data/trainer_data/manifest_ase_dnn.yaml")
    )
    with open(manifest_file, "w") as f:
        yaml.dump(manifest_data, f)

    model = torch.jit.load(model_file)

    model = model.double()

    manifest = yaml.safe_load(open(manifest_file, "r"))

    trainer = DNNTrainer(manifest)

    loaded_model_dict = trainer.model.state_dict()
    trainer_model_dict = model.state_dict()

    for key in loaded_model_dict.keys():
        assert torch.allclose(loaded_model_dict[key], trainer_model_dict[key])

    assert trainer.current["loss"] == None
    assert trainer.current["epoch"] == 0
    assert trainer.current["step"] == 0
    assert trainer.current["device"] == "cpu"
    assert trainer.current["warned_once"] == False
    assert trainer.current["dataset_hash"] == None
    assert trainer.current["data_dir"] == "test_run/datasets"
    assert trainer.current["appending_to_previous_run"] == False
    assert trainer.current["verbose"] == False

    # check dataset manifest
    expected_dataset_manifest = {
        "type": "ase",
        "path": str(data_file),
        "save": False,
        "keys": {"energy": "Energy", "forces": "force"},
        "dynamic_loading": False,
        "colabfit_dataset": {
            "dataset_name": None,
            "database_name": None,
            "database_url": None,
        },
    }
    assert trainer.dataset_manifest == expected_dataset_manifest

    # check descriptor settings
    config_transform = {
        "name": "Descriptor",
        "kwargs": {
            "cutoff": 4.0,
            "species": ["Si"],
            "descriptor": "SymmetryFunctions",
            "hyperparameters": "set51",
        },
    }
    assert trainer.transform_manifest["configuration"] == config_transform

    expected_loss_manifest = {
        "function": "MSE",
        "weights": {"config": 1.0, "energy": 1.0, "forces": 1.0},
        "normalize_per_atom": True,
    }

    assert trainer.loss_manifest == expected_loss_manifest

    # dataset samples
    assert trainer.dataset_sample_manifest["train_size"] == 3
    assert trainer.dataset_sample_manifest["val_size"] == 1
    assert isinstance(trainer.dataset_sample_manifest["val_indices"], np.ndarray)
    assert isinstance(trainer.dataset_sample_manifest["train_indices"], np.ndarray)

    # check optimizer settings
    expected_optimizer_manifest = {
        "name": "Adam",
        "learning_rate": 1.0e-3,
        "kwargs": {},
        "epochs": 2,
        "num_workers": None,
        "batch_size": 2,
        "lr_scheduler": {
            "name": "ReduceLROnPlateau",
            "args": {"factor": 0.5, "patience": 5, "min_lr": 1.0e-6},
        },
    }
    assert trainer.optimizer_manifest == expected_optimizer_manifest

    # dummy training
    trainer.train()
    # check if the trainer exited without any errors, check if .finished file is created
    assert os.path.exists(f'{trainer.current["run_dir"]}/.finished')

    # check if the kim model is saved, default folder is kim-model
    trainer.save_kim_model()
    assert os.path.exists("./kim-model/CMakeLists.txt")

    # check if checkpoints are properly saved
    ckpt = f'{trainer.current["run_dir"]}/checkpoints/checkpoint_0.pkl'
    assert os.path.exists(ckpt)
    ckpt_dict = torch.load(ckpt)
    assert ckpt_dict["model_state_dict"].keys() == model.state_dict().keys()
    assert (
        ckpt_dict["optimizer_state_dict"].keys()
        == trainer.optimizer.state_dict().keys()
    )
    assert ckpt_dict["current_step"] == 0
    assert ckpt_dict["lr_scheduler"].keys() == trainer.lr_scheduler.state_dict().keys()
    assert ckpt_dict["early_stopping"] == {"counter": 0, "best_loss": np.inf}
