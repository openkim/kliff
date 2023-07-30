"""
TODO:
- [] Test if the implementation works with model with multiple elements.
"""

from pathlib import Path

import numpy as np
import pytest

from kliff import nn
from kliff.calculators import CalculatorTorch
from kliff.dataset import Dataset
from kliff.descriptors import SymmetryFunction
from kliff.loss import Loss
from kliff.models import NeuralNetwork
from kliff.uq.bootstrap import (
    Bootstrap,
    BootstrapError,
    BootstrapNeuralNetworkModel,
    bootstrap_cas_generator_neuralnetwork,
)

seed = 1717
np.random.seed(seed)

# descriptor
descriptor = SymmetryFunction(
    cut_name="cos", cut_dists={"Si-Si": 5.0}, hyperparams="set30", normalize=True
)

# model
N1 = np.random.randint(5, 10)
model = NeuralNetwork(descriptor)
model.add_layers(
    nn.Linear(descriptor.get_size(), N1),
    nn.Tanh(),
    nn.Linear(N1, 1),
)

# training set
FILE_DIR = Path(__file__).absolute().parent  # Directory of test file
path = FILE_DIR.parent.joinpath("configs_extxyz/Si_4")
data = Dataset(path)
configs = data.get_configs()

# calculators
calc = CalculatorTorch(model)
_ = calc.create(configs, use_energy=False, use_forces=True)
fingerprints = calc.get_fingerprints()
nfingerprints = len(fingerprints)

loss = Loss(calc)
min_kwargs = dict(method="Adam", num_epochs=10, batch_size=100, lr=0.001)
result = loss.minimize(**min_kwargs)

orig_state_filename = FILE_DIR / "orig_model.pkl"
BS = Bootstrap(loss, orig_state_filename=orig_state_filename)
nsamples = np.random.randint(1, 5)


def test_wrapper():
    """Test if the Bootstrap class wrapper instantiate the correct class."""
    assert isinstance(
        BS, BootstrapNeuralNetworkModel
    ), "Wrapper should instantiate BootstrapNeuralNetworkModel"


def test_error():
    """Test if BootstrapError is raised when we try to call run before generating
    bootstrap compute arguments.
    """
    with pytest.raises(BootstrapError):
        BS.run(min_kwargs=min_kwargs)


def test_original_state():
    """Test we export the original state when we instantiate the class."""
    assert orig_state_filename.exists(), "Original state is not exported"


def test_bootstrap_cas_generator():
    """Test the generator function generate the same number of bootstrap compute arguments
    samples as requested.
    """
    # Test the shape of bootstrap cas samples with default arguments
    BS.generate_bootstrap_compute_arguments(nsamples)
    bootstrap_cas = BS.bootstrap_compute_arguments
    assert (
        len(bootstrap_cas) == nsamples
    ), "The number of generated cas is not the same as requested, check the generator"
    assert np.all(
        [len(bs_cas) == nfingerprints for _, bs_cas in bootstrap_cas.items()]
    ), "For each sample, generator should generate the same number of cas as the original"
    assert (
        BS._nsamples_prepared == nsamples
    ), "`_nsamples_prepared` property doesn't work"

    # Test the shape of bootstrap cas samples if we specify the number of cas to generate
    nfp = nfingerprints - 1
    bootstrap_cas_2 = bootstrap_cas_generator_neuralnetwork(
        nsamples, fingerprints, nfingerprints=nfp
    )
    assert np.all(
        [len(bs_cas) == nfp for _, bs_cas in bootstrap_cas_2.items()]
    ), "Generator doesn't generate the same number of cas as requested for each sample"


def test_run():
    """Test the method to run the calculation."""
    BS.run(min_kwargs=min_kwargs)
    # Test the shape of ensembles
    samples = BS.samples
    shape = samples.shape
    exp_shape = (nsamples, calc.get_size_opt_params()[-1])
    assert (
        shape == exp_shape
    ), f"Samples doesn't have the right shape; expected {exp_shape}, got {shape}"
    assert BS._nsamples_done == nsamples


def test_appending_cas():
    """If we call the generate method again, test if it appends the newly generated
    bootstrap compute arguments to the previously generated list.
    """
    BS.generate_bootstrap_compute_arguments(nsamples)
    assert (
        BS._nsamples_prepared == 2 * nsamples
    ), "The newly generated cas is not appended to the old list"


def test_save_load_cas():
    """Test the function to load bootstrap compute arguments."""
    filename = FILE_DIR / "bootstrap_cas.json"
    BS.save_bootstrap_compute_arguments(filename)
    BS.load_bootstrap_compute_arguments(filename)
    filename.unlink()
    assert (
        BS._nsamples_prepared == 4 * nsamples
    ), "Problem with function to load bootstrap cas"


def test_reset():
    """Test the reset method."""
    BS.reset()
    # Check reset bootstrap samples
    assert BS._nsamples_prepared == 0, "Reset bootstrap cas failed"
    assert BS._nsamples_done == 0, "Reset ensembles failed"
