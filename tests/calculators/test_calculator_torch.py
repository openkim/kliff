from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from kliff import nn
from kliff.calculators import CalculatorTorch
from kliff.dataset import Dataset
from kliff.descriptors import SymmetryFunction
from kliff.models import NeuralNetwork

# model
descriptor = SymmetryFunction(
    cut_name="cos", cut_dists={"Si-Si": 5.0}, hyperparams="set30", normalize=True
)

N1 = np.random.randint(5, 10)
N2 = np.random.randint(5, 10)
model = NeuralNetwork(descriptor)
model.add_layers(
    # first hidden layer
    nn.Linear(descriptor.get_size(), N1),
    nn.Tanh(),
    # second hidden layer
    nn.Linear(N1, N2),
    nn.Tanh(),
    # output layer
    nn.Linear(N2, 1),
)

# training set
path = Path(__file__).absolute().parents[1].joinpath("configs_extxyz/Si_4")
data = Dataset(path)
configs = data.get_configs()

# calculator
calc = CalculatorTorch(model, gpu=False)
_ = calc.create(configs, reuse=False)
loader = calc.get_compute_arguments(batch_size=100)

# data on parameter sizes
exp_sizes = [
    torch.Size([N1, 30]),
    torch.Size([N1]),
    torch.Size([N2, N1]),
    torch.Size([N2]),
    torch.Size([1, N2]),
    torch.Size([1]),
]
exp_nparams_per_layer = [N1 * 30, N1, N2 * N1, N2, N2, 1]
exp_nparams_total = np.sum(exp_nparams_per_layer)

# parameters to try
p0 = np.zeros(exp_nparams_total)
p1 = np.ones(exp_nparams_total)


# Test if the functions to update parameters work


def test_get_parameters_sizes():
    """
    Test if the function to get parameters sizes works.

    Given the descriptor and the number of nodes in each layer, we can in principle find
    the expected size of the parameters in each layer.
    """
    sizes, nparams_per_layer, nparams_total = calc.get_size_opt_params()
    assert sizes == exp_sizes, "The sizes retrieved are incorrect"
    assert (
        nparams_per_layer == exp_nparams_per_layer
    ), "The numbers of parameters per layer are incorrect"
    assert nparams_total == exp_nparams_total, "Total number of parameters is incorrect"


def test_parameter_values():
    """
    Test if the parameter values are updated.

    This is done by calling `calc.update_model_params` and set the parameter values to
    zeros. Then, when we retrieve the parameters, we should also get zeros.
    """
    # Update the parameters
    calc.update_model_params(p0)
    # Check if the parameters retrieved are all zeros
    assert np.all(
        calc.get_opt_params() == p0
    ), "Either `update_model_params` or `get_opt_params` not working"


def test_predictions_change():
    """
    Test if changing parameters affect the predictions.

    There are two steps of this test. The first one, if we set all the parameters to be
    zero, then the (forces) predictions should also be zero.

    Then, if we change the parameters to some other values, the predictions should change
    and they should not be zero, unless there is something special with the
    configurations.
    """
    # Test if predictions are zeros when all parameters are zeros
    calc.update_model_params(p0)
    for batch in loader:
        calc.compute(batch)
    # We will only look at the forces
    forces0 = calc.results["forces"]
    all_zeros = []
    for f0 in forces0:
        all_zeros.append(Tensor.all(f0 == 0.0))
    assert np.all(all_zeros), (
        "Problem in predicitons calculation: "
        + "there are non-zero forces when all parameters are zero"
    )

    # Test if predictions change when we change parameters
    calc.update_model_params(p1)
    for batch in loader:
        calc.compute(batch)
    forces1 = calc.results["forces"]
    change = []
    for f0, f1 in zip(forces0, forces1):
        change.append(not Tensor.all(f0 - f1 == 0.0))
    # Use any since there might be special configurations
    assert np.any(change), "Changing parameters doesn't change predictions"


if __name__ == "__main__":
    test_get_parameters_sizes()
    test_parameter_values()
    test_predictions_change()
