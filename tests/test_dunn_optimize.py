from pathlib import Path

import numpy as np
import pytest
import torch

from kliff.dataset import Dataset
from kliff.legacy import nn
from kliff.legacy.calculators import CalculatorTorch
from kliff.legacy.descriptors import SymmetryFunction
from kliff.legacy.loss import Loss
from kliff.models import NeuralNetwork


@pytest.fixture(scope="session")
def descriptor() -> SymmetryFunction:
    return SymmetryFunction(
        cut_name="cos",
        cut_dists={"Si-Si": 5.0},
        hyperparams="set30",
        normalize=True,
    )


@pytest.fixture(scope="session")
def dataset(test_data_dir: Path):
    """Si_4 configs shipped with KLIFF test data."""
    return Dataset.from_path(test_data_dir / "configs" / "Si_4").get_configs()


@pytest.fixture(scope="session")
def nn_model(descriptor) -> NeuralNetwork:
    torch.manual_seed(0)
    hidden = 8
    net = NeuralNetwork(descriptor)
    net.add_layers(
        nn.Linear(descriptor.get_size(), hidden),
        nn.Tanh(),
        nn.Linear(hidden, 1),
    )
    return net


@pytest.fixture(scope="session")
def calculator(nn_model, dataset):
    calc = CalculatorTorch(nn_model)
    _ = calc.create(dataset, use_energy=True, use_forces=True)
    return calc


def test_dunn_optimize(calculator):
    """Simple ‘does Adam run’ smoke test."""
    loss = Loss(calculator)
    res = loss.minimize("Adam", lr=1e-2, num_epochs=10, batch_size=1)
    assert True  # training went successfully


def test_dunn_optimize_per_atom_log(calculator, tmp_path):
    """
    Same as above but with per-atom logging switched on.
    We additionally verify that the LMDB file was created
    and has at least one record.
    """
    log_per_atom_path = tmp_path / "log_per_atom"
    loss = Loss(
        calculator,
        log_per_atom_pred=True,
        log_per_atom_pred_path=str(log_per_atom_path),
    )
    res = loss.minimize("Adam", lr=1e-2, num_epochs=10, batch_size=1)

    lmdb_file = log_per_atom_path / "per_atom_pred_database.lmdb"
    assert lmdb_file.exists()

    # Optional: open and check record count
    import lmdb

    env = lmdb.open(str(lmdb_file), readonly=True, lock=False, subdir=False)
    with env.begin() as txn:
        n_records = sum(1 for _ in txn.cursor())
    assert n_records > 0
