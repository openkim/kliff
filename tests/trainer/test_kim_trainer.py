import tarfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from kliff.models import KIMModel
from kliff.trainer.kim_trainer import KIMTrainer

def test_trainer():
    """
    Basic tests for proper initialization of the Trainer module
    """
    manifest_file_template = (
        Path(__file__)
        .parents[1]
        .joinpath("test_data/trainer_data/training_manifest_ase_kim.yaml.tpl")
    )
    data_file = Path(__file__).parents[1].joinpath("test_data/configs/Si_4.xyz")

    manifest_data = yaml.safe_load(open(manifest_file_template, "r"))

    manifest_data["dataset"]["path"] = str(data_file)

    manifest_file = (
        Path(__file__)
        .parents[1]
        .joinpath("test_data/trainer_data/manifest_ase_kim.yaml")
    )
    with open(manifest_file, "w") as f:
        yaml.dump(manifest_data, f)

    model = KIMModel("SW_StillingerWeber_1985_Si__MO_405512056662_006")

    manifest = yaml.safe_load(open(manifest_file, "r"))

    trainer = KIMTrainer(manifest)

    # check basic initialization
    assert trainer.model.model_name == model.model_name

    default_model_params = model.model_params
    trainer_params = trainer.model.model_params

    assert np.allclose(trainer_params["A"], default_model_params["A"])
    assert np.allclose(trainer_params["B"], default_model_params["B"])
    assert np.allclose(trainer_params["p"], default_model_params["p"])
    assert np.allclose(trainer_params["q"], default_model_params["q"])
    assert np.allclose(trainer_params["gamma"], default_model_params["gamma"])
    assert np.allclose(trainer_params["cutoff"], default_model_params["cutoff"])
    assert np.allclose(trainer_params["lambda"], default_model_params["lambda"])
    assert np.allclose(trainer_params["costheta0"], default_model_params["costheta0"])
    assert np.allclose(trainer.model.model_params["sigma"], np.log(2.0))

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

    # check parameter settings
    expected_parameter_manifest = [
        "A",
        "B",
        {
            "sigma": {
                "transform_name": "LogParameterTransform",
                "value": 2.0,
                "bounds": [[1.0, 10.0]],
            }
        },
    ]

    assert trainer.transform_manifest["parameter"] == expected_parameter_manifest

    expected_loss_manifest = {
        "function": "MSE",
        "weights": {"config": 1.0, "energy": 1.0, "forces": 1.0},
        "normalize_per_atom": True,
    }
    print( trainer.loss_manifest, expected_loss_manifest)
    assert trainer.loss_manifest == expected_loss_manifest

    # dataset samples
    assert trainer.dataset_sample_manifest["train_size"] == 3
    assert trainer.dataset_sample_manifest["val_size"] == 0
    assert trainer.dataset_sample_manifest["val_indices"] is None
    assert isinstance(trainer.dataset_sample_manifest["train_indices"], np.ndarray)

    # check optimizer settings
    expected_optimizer_manifest = {
        "name": "L-BFGS-B",
        "learning_rate": None,
        "kwargs": {"tol": 1e-06},
        "epochs": 1000,
        "num_workers": 2,
        "batch_size": 1,
    }
    assert trainer.optimizer_manifest == expected_optimizer_manifest

    # dummy training
    trainer.train()
    # check if the trainer exited without any errors, check if .finished file is created
    assert trainer.result.success

    # check if the kim model is saved, default folder is kim-model
    trainer.save_kim_model()

    # assert
    assert Path(
        f"SW_StillingerWeber_trained_1985_Si__MO_405512056662_006.tar.gz"
    ).exists()
