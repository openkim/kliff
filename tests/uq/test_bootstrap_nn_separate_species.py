import os
from pathlib import Path

import numpy as np

from kliff import nn
from kliff.calculators.calculator_torch import CalculatorTorchSeparateSpecies
from kliff.dataset import Dataset
from kliff.descriptors import SymmetryFunction
from kliff.loss import Loss
from kliff.models import NeuralNetwork
from kliff.uq.bootstrap import BootstrapNeuralNetworkModel

seed = 1717
np.random.seed(seed)

# descriptor
descriptor = SymmetryFunction(
    cut_name="cos",
    cut_dists={"Si-Si": 5.0, "C-C": 5.0, "Si-C": 5.0},
    hyperparams="set30",
    normalize=True,
)

# models
# Si
N1 = np.random.randint(5, 10)
model_si = NeuralNetwork(descriptor)
model_si.add_layers(
    nn.Linear(descriptor.get_size(), N1),
    nn.Tanh(),
    nn.Linear(N1, 1),
)

# C
model_c = NeuralNetwork(descriptor)
model_c.add_layers(
    nn.Linear(descriptor.get_size(), N1),
    nn.Tanh(),
    nn.Linear(N1, 1),
)

# training set
FILE_DIR = Path(__file__).absolute().parent  # Directory of test file
path = FILE_DIR.parent.joinpath("test_data/configs/SiC_4")
data = Dataset(path)
configs = data.get_configs()

# calculators
calc = CalculatorTorchSeparateSpecies({"Si": model_si, "C": model_c})
_ = calc.create(configs, use_energy=False, use_forces=True)

loss = Loss(calc)
min_kwargs = dict(method="Adam", num_epochs=10, batch_size=100, lr=0.001)
result = loss.minimize(**min_kwargs)

orig_state_filename = FILE_DIR / "orig_model.pkl"
BS = BootstrapNeuralNetworkModel(loss, orig_state_filename=orig_state_filename)
nsamples = np.random.randint(1, 5)


def test_model():
    """Test the model property."""
    # models
    assert len(BS.model) == 2, "There should be 2 elements in BS.model"
    # species
    assert np.all(
        [sp in ["Si", "C"] for sp in BS._species]
    ), "There are species not recorded"


def test_original_state():
    """Test we export the original states for all models when we instantiate the class."""
    # check shape
    assert (
        len(BS.orig_state_filename) == 2
    ), "There should be 2 elements in orig_state_filename"
    # check elements
    splitted_path = os.path.splitext(orig_state_filename)
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


def test_run():
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
