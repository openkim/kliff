from pathlib import Path
import pytest

import numpy as np
from scipy.optimize.optimize import OptimizeResult

from kliff.calculators.calculator import Calculator, _WrapperCalculator
from kliff.dataset import Dataset
from kliff.loss import Loss
from kliff.models import KIMModel
from kliff.uq.bootstrap import (
    Bootstrap,
    BootstrapEmpiricalModel,
    BootstrapError,
    bootstrap_cas_generator_empirical,
)

seed = 1717
np.random.seed(seed)

# model
modelname = "SW_StillingerWeber_1985_Si__MO_405512056662_006"
model = KIMModel(modelname)
model.set_opt_params(A=[["default"]])

# training set
FILE_DIR = Path(__file__).absolute().parent  # Directory of test file
path = FILE_DIR.parent.joinpath("configs_extxyz/Si_4")
data = Dataset(path)
configs = data.get_configs()

# calculators
# forces
calc_forces = Calculator(model)
cas_forces = calc_forces.create(configs, use_energy=False, use_forces=True)
ncas_forces = len(calc_forces.get_compute_arguments())
calc_energy = Calculator(model)
cas_energy = calc_energy.create(configs, use_energy=True, use_forces=False)
ncas_energy = len(calc_energy.get_compute_arguments())
calc_comb = _WrapperCalculator(calculators=[calc_energy, calc_forces])


# Single calculator
# loss
min_kwargs = dict(method="lm")
loss_forces = Loss(calc_forces)
loss_forces.minimize(**min_kwargs)
# initial state
orig_params = calc_forces.get_opt_params()

# Bootstrap class
BS_1calc = Bootstrap(loss_forces)
nsamples = np.random.randint(1, 5)


# Multiple calculators
# loss
loss_comb = Loss(calc_comb)
loss_comb.minimize(**min_kwargs)

# Bootstrap class
BS_2calc = Bootstrap(loss_comb)


def test_wrapper():
    """Test if the Bootstrap class wrapper instantiate the correct class."""
    assert isinstance(
        BS_1calc, BootstrapEmpiricalModel
    ), "Wrapper should instantiate BootstrapEmpiricalModel"


def test_error():
    """Test if BootstrapError is raised when we try to call run before generating
    bootstrap compute arguments.
    """
    with pytest.raises(BootstrapError):
        BS_1calc.run(min_kwargs=min_kwargs)


def test_bootstrap_cas_generator():
    """Test the shape of generated bootstrap compute arguments."""
    BS_1calc.generate_bootstrap_compute_arguments(nsamples)
    bootstrap_cas = BS_1calc.bootstrap_compute_arguments
    # Test the shape of bootstrap cas samples with default arguments
    assert (
        len(bootstrap_cas) == nsamples
    ), "The number of generated cas is not the same as requested, check the generator"
    assert np.all(
        [
            [len(bs_cas) == ncas_forces for bs_cas in bootstrap_cas[ii]]
            for ii in range(nsamples)
        ]
    ), "For each sample, generator should generate the same number of cas as the original"
    assert (
        BS_1calc._nsamples_prepared == nsamples
    ), "`_nsamples_prepared` property doesn't work"

    # Test the shape of bootstrap cas samples if we specify the number of cas to generate
    ncas = ncas_forces - 1
    bootstrap_cas_2 = bootstrap_cas_generator_empirical(nsamples, [cas_forces], ncas=ncas)
    assert np.all(
        [
            [len(bs_cas) == ncas for bs_cas in bootstrap_cas_2[ii]]
            for ii in range(nsamples)
        ]
    ), "Generator doesn't generate the same number of cas as requested for each sample"


def test_callback():
    """Test if callback function works and can break the loop in run method."""

    def callback(_, opt):
        assert isinstance(opt, OptimizeResult), "Callback cannot capture the run"
        return True

    BS_1calc.run(min_kwargs=min_kwargs, callback=callback)
    assert (
        len(BS_1calc.samples) == 1
    ), "Callback function cannot break the loop in run method."


def test_run():
    """Test the method to run the calculation."""
    BS_1calc.run(min_kwargs=min_kwargs)
    # Test the shape of ensembles
    samples = BS_1calc.samples
    shape = samples.shape
    exp_shape = (nsamples, len(orig_params))
    assert (
        shape == exp_shape
    ), f"Samples doesn't have the right shape; expected {exp_shape}, got {shape}"
    assert BS_1calc._nsamples_done == shape[0]
    # Assert restore loss
    assert np.all(
        BS_1calc.calculator.get_opt_params() == orig_params
    ), "Parameters are not restored to initial state"


def test_appending_cas():
    """If we call the generate method again, test if it appends the newly generated
    bootstrap compute arguments to the previously generated list.
    """
    BS_1calc.generate_bootstrap_compute_arguments(nsamples)
    assert (
        BS_1calc._nsamples_prepared == 2 * nsamples
    ), "The newly generated cas is not appended to the old list"


def test_save_load_cas():
    """Test the function to load bootstrap compute arguments."""
    filename = FILE_DIR / "bootstrap_cas.json"
    BS_1calc.save_bootstrap_compute_arguments(filename)
    BS_1calc.load_bootstrap_compute_arguments(filename)
    filename.unlink()
    assert (
        BS_1calc._nsamples_prepared == 4 * nsamples
    ), "Problem with function to load bootstrap cas"


def test_reset():
    """Test the reset method."""
    BS_1calc.reset()
    # Check reset parameters
    assert np.all(
        BS_1calc.calculator.get_opt_params() == orig_params
    ), "Parameters are not restored to initial state"
    # Check reset bootstrap samples
    assert BS_1calc._nsamples_prepared == 0, "Reset bootstrap cas failed"
    assert BS_1calc._nsamples_done == 0, "Reset ensembles failed"


def test_multi_calc_cas_generator():
    """Test the shape of generated bootstrap compute arguments when we use multiple
    calculators.
    """
    BS_2calc.generate_bootstrap_compute_arguments(nsamples)
    bootstrap_cas = BS_2calc.bootstrap_compute_arguments
    assert (
        len(bootstrap_cas) == nsamples
    ), "The number of generated cas is not the same as requested, check the generator"
    assert (
        len(bootstrap_cas[0]) == 2
    ), "For each sample, the generator should generate cas for each calculator"
    assert (
        len(bootstrap_cas[0][0]) + len(bootstrap_cas[0][1]) == ncas_energy + ncas_forces
    ), "For each sample, generator should generate the same number of cas in total as the original"


if __name__ == "__main__":
    test_wrapper()
    test_error()
    test_bootstrap_cas_generator()
    test_callback()
    test_run()
    test_appending_cas()
    test_save_load_cas()
    test_reset()
    test_multi_calc_cas_generator()
