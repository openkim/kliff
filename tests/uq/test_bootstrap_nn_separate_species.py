import os
from pathlib import Path

import numpy as np
import pytest

from kliff import nn
from kliff.calculators.calculator_torch import CalculatorTorchSeparateSpecies
from kliff.descriptors import SymmetryFunction
from kliff.loss import Loss
from kliff.models import NeuralNetwork
from kliff.uq.bootstrap import BootstrapNeuralNetworkModel

seed = 1717
np.random.seed(seed)

# Number of bootstrap samples
nsamples = np.random.randint(1, 5)
# Number of nodes of the model
N = np.random.randint(5, 10)
# Optimizer settings
min_kwargs = dict(method="Adam", num_epochs=10, batch_size=100, lr=0.001)


@pytest.fixture(scope="session")
def descriptor():
    """Return atomic descriptor."""
    return SymmetryFunction(
        cut_name="cos",
        cut_dists={"Si-Si": 5.0, "C-C": 5.0, "Si-C": 5.0},
        hyperparams="set30",
        normalize=True,
    )


@pytest.fixture(scope="session")
def model_si(descriptor):
    """Return a model for Si."""
    model = NeuralNetwork(descriptor)
    model.add_layers(
        nn.Linear(descriptor.get_size(), N),
        nn.Tanh(),
        nn.Linear(N, 1),
    )
    return model


@pytest.fixture(scope="session")
def model_c(descriptor):
    """Return a model for C."""
    model = NeuralNetwork(descriptor)
    model.add_layers(
        nn.Linear(descriptor.get_size(), N),
        nn.Tanh(),
        nn.Linear(N, 1),
    )
    return model


@pytest.fixture(scope="session")
def calc(uq_test_configs, model_si, model_c):
    """Calculator for the test."""
    calculator = CalculatorTorchSeparateSpecies({"Si": model_si, "C": model_c})
    _ = calculator.create(uq_test_configs, use_energy=False, use_forces=True)
    return calculator


@pytest.fixture(scope="session")
def BS(calc, uq_nn_orig_state_filename):
    """Return a Bootstrap object."""
    loss = Loss(calc)
    _ = loss.minimize(**min_kwargs)
    return BootstrapNeuralNetworkModel(
        loss, orig_state_filename=uq_nn_orig_state_filename
    )


def test_model(BS):
    """Test the model property."""
    # models
    assert len(BS.model) == 2, "There should be 2 elements in BS.model"
    # species
    assert np.all(
        [sp in ["Si", "C"] for sp in BS._species]
    ), "There are species not recorded"


def test_original_state(BS, uq_nn_orig_state_filename):
    """Test we export the original states for all models when we instantiate the class."""
    # check shape
    assert (
        len(BS.orig_state_filename) == 2
    ), "There should be 2 elements in orig_state_filename"
    # check elements
    splitted_path = os.path.splitext(uq_nn_orig_state_filename)
    fnames = [
        Path(splitted_path[0] + f"_{el}" + splitted_path[1]) for el in ["Si", "C"]
    ]
    assert np.all(
        [str(fname) in BS.orig_state_filename for fname in fnames]
    ), "Not all original state filename are listed"
    # check if initial states are exported
    assert np.all(
        [fname.exists() for fname in fnames]
    ), "Not all original state is not exported"


def test_run(BS, calc):
    """Test the method to run the calculation."""
    BS.generate_bootstrap_compute_arguments(nsamples)
    BS.run(min_kwargs=min_kwargs)
    # Test the shape of ensembles
    samples = BS.samples
    shape = samples.shape
    exp_shape = (nsamples, calc.get_size_opt_params()[-1])
    assert (
        shape == exp_shape
    ), f"Samples doesn't have the right shape; expected {exp_shape}, got {shape}"
    assert BS._nsamples_done == nsamples
