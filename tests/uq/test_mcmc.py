from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pytest

from kliff.calculators import Calculator
from kliff.dataset import Dataset
from kliff.loss import Loss
from kliff.models import KIMModel
from kliff.uq import MCMC, get_T0

try:
    from kliff.uq.mcmc import PtemceeSampler

    ptemcee_avail = True
except ImportError:
    ptemcee_avail = False

try:
    from kliff.uq.mcmc import EmceeSampler

    emcee_avail = True
except ImportError:
    emcee_avail = False


seed = 1717
np.random.seed(seed)

# model
modelname = "SW_StillingerWeber_1985_Si__MO_405512056662_006"
model = KIMModel(modelname)
model.set_opt_params(A=[["default"]])

# training set
path = Path(__file__).absolute().parents[1].joinpath("configs_extxyz/Si_4")
data = Dataset(path)
configs = data.get_configs()

# calculator
calc = Calculator(model)
ca = calc.create(configs)

# loss
loss = Loss(calc)
loss.minimize(method="lm")

# dimensionality
ndim = calc.get_num_opt_params()
nwalkers = 2 * np.random.randint(1, 3)
ntemps = np.random.randint(2, 4)
nsteps = np.random.randint(5, 10)

# samplers
prior_bounds = np.tile([0, 10], (ndim, 1))

if ptemcee_avail:
    ptsampler = MCMC(
        loss,
        ntemps=ntemps,
        nwalkers=nwalkers,
        logprior_args=(prior_bounds,),
        random=np.random.RandomState(seed),
    )
if emcee_avail:
    sampler = MCMC(
        loss, nwalkers=nwalkers, logprior_args=(prior_bounds,), sampler="emcee"
    )


def test_T0():
    """Test if the function to compute T0 works properly. This is done by comparing T0
    computed using the internal function and computed manually.
    """
    # Using internal function
    T0_internal = get_T0(loss)
    # Compute manually
    xopt = calc.get_opt_params()
    T0_manual = 2 * loss._get_loss(xopt) / len(xopt)
    assert T0_internal == T0_manual, "Internal function to compute T0 doesn't work"


def test_MCMC_wrapper():
    """Test if the MCMC wrapper class returns the correct sampler instance."""
    if ptemcee_avail:
        assert (
            type(ptsampler) == PtemceeSampler
        ), "MCMC should return ``PtemceeSampler`` instance"
    if emcee_avail:
        assert (
            type(sampler) == EmceeSampler
        ), "MCMC should return ``EmceeSampler`` instance"


def test_dimensionality():
    """Test the number of temperatures, walkers, steps, and parameters. This is done by
    comparing the shape of the resulting MCMC chains and the variables used to set these
    dimensions.
    """

    # Test for ptemcee wrapper
    if ptemcee_avail:
        p0 = np.random.uniform(0, 10, (ntemps, nwalkers, ndim))
        ptsampler.run_mcmc(p0=p0, iterations=nsteps)
        assert ptsampler.chain.shape == (
            ntemps,
            nwalkers,
            nsteps,
            ndim,
        ), "Dimensionality from the ptemcee wrapper is not right"
    else:
        print("Skip testing ptemcee; ptemcee is not found")

    # Test for emcee wrapper
    if emcee_avail:
        p0 = np.random.uniform(0, 10, (nwalkers, ndim))
        sampler.run_mcmc(initial_state=p0, nsteps=nsteps)
        assert sampler.chain.shape == (
            nwalkers,
            nsteps,
            ndim,
        ), "Dimensionality from the emcee wrapper is not right"
    else:
        print("Skip testing emcee; emcee is not found")


def test_pool_exception():
    """Test if an exception is raised when declaring the pool prior to instantiating
    ``kliff.uq.MCMC``.
    """
    if ptemcee_avail:
        with pytest.raises(ValueError):
            _ = MCMC(
                loss,
                ntemps=ntemps,
                nwalkers=nwalkers,
                logprior_args=(prior_bounds,),
                pool=Pool(1),
            )
    else:
        print("Skip the test; ptemcee is not found")


def test_sampler_exception():
    """Test if an exception is raised when specifying sampler string other than the
    built-in ones.
    """
    with pytest.raises(ValueError):
        _ = MCMC(
            loss,
            ntemps=ntemps,
            nwalkers=nwalkers,
            logprior_args=(prior_bounds,),
            sampler="dummy",
        )


if __name__ == "__main__":
    test_T0()
    test_MCMC_wrapper()
    test_dimensionality()
    test_pool_exception()
