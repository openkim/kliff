from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from kliff.trainer import GNNLightningTrainer


def test_trainer():
    """
    Basic tests for proper initialization of the Trainer module
    """
    torch.set_default_tensor_type(torch.DoubleTensor)
    manifest_file = filename = (
        Path(__file__)
        .parents[1]
        .joinpath("test_data/trainer_data/example_config_ase_lightning_gnn.yaml")
    )
    model_file = (
        Path(__file__).parents[1].joinpath("test_data/trainer_data/dummy_model.pt")
    )

    manifest = yaml.safe_load(open(manifest_file, "r"))
    model = torch.jit.load(model_file)

    trainer = GNNLightningTrainer(manifest, model)

    # check basic initialization
    assert trainer.model is model
    assert trainer.current["best_loss"] == np.inf
    assert trainer.current["best_model"] == None
    assert trainer.current["loss"] == None
    assert trainer.current["epoch"] == 0
    assert trainer.current["step"] == 0
    assert trainer.current["device"] == "cpu"
    assert trainer.current["warned_once"] == False
    assert trainer.current["dataset_hash"] == None
    assert trainer.current["data_dir"] == "test_run/datasets"
    assert trainer.current["appending_to_previous_run"] == False
    assert trainer.current["verbose"] == False
    assert trainer.current["ckpt_interval"] == 3
    assert trainer.current["per_atom_loss"] == {"train": None, "val": None}

    # check dataset manifest
    expected_dataset_manifest = {
        "type": "ase",
        "path": "../test_data/configs/Si_4.xyz",
        "save": False,
        "keys": {"energy": "Energy", "forces": "force"},
        "dynamic_loading": True,
        "colabfit_dataset": {
            "dataset_name": None,
            "database_name": None,
            "database_url": None,
        },
    }
    assert trainer.dataset_manifest == expected_dataset_manifest

    assert trainer.model_manifest["type"] == "torch"

    # check graph settings
    config_transform = {
        "name": "Graph",
        "kwargs": {"cutoff": 3.77, "species": ["Si"], "n_layers": 1},
    }
    assert trainer.transform_manifest["configuration"] == config_transform

    expected_loss_manifest = {
        "function": "MSE",
        "weights": {"config": 1.0, "energy": 1.0, "forces": 10.0, "stress": None},
        "normalize_per_atom": False,
        "loss_traj": False,
    }
    assert trainer.loss_manifest == expected_loss_manifest

    # dataset samples
    assert trainer.dataset_sample_manifest["train_size"] == 3
    assert trainer.dataset_sample_manifest["val_size"] == 1
    assert isinstance(trainer.dataset_sample_manifest["val_indices"], np.ndarray)
    assert isinstance(trainer.dataset_sample_manifest["train_indices"], np.ndarray)

    # check optimizer settings
    expected_optimizer_manifest = {
        "provider": "torch",
        "name": "Adam",
        "learning_rate": 0.001,
        "kwargs": None,
        "epochs": 2,
        "stop_condition": None,
        "num_workers": None,
        "batch_size": 1,
    }
    assert trainer.optimizer_manifest == expected_optimizer_manifest

    # dummy training
    trainer.train()
    # check if the trainer exited without any errors, check if .finished file is created
    assert trainer.pl_trainer.state.finished
    assert Path(f"{trainer.current['run_dir']}/.finished").exists()

    # check if the best model is saved
    assert Path(f"{trainer.current['run_dir']}/checkpoints/best_model.pth").exists()
    assert Path(f"{trainer.current['run_dir']}/checkpoints/last_model.pth").exists()

    # check if the kim model is saved, default folder is kim-model
    trainer.save_kim_model()
    assert Path(f"kim-model/model.pt").exists()
    assert Path(f"kim-model/kliff_graph.param").exists()
    assert Path(f"kim-model/CMakeLists.txt").exists()

    # check restart
    # TODO: implement restart test
