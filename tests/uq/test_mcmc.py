from multiprocessing import Pool

import numpy as np
import pytest

from kliff.calculators import Calculator
from kliff.loss import Loss
from kliff.uq import MCMC, get_T0

try:
    from kliff.uq.mcmc import PtemceeSampler
except ImportError:
    print("ptemcee is not found")

try:
    from kliff.uq.mcmc import EmceeSampler
except ImportError:
    print("emcee is not found")


seed = 1717
np.random.seed(seed)


# dimensionality --- Number of parameters is defined when creating the calculator
nwalkers = 2 * np.random.randint(1, 3)
ntemps = np.random.randint(2, 4)
nsteps = np.random.randint(5, 10)


@pytest.fixture(scope="module")
def calc(uq_kim_model, uq_test_configs):
    """Create calculator instance."""
    model = uq_kim_model
    # calculator
    calculator = Calculator(model)
    ca = calculator.create(uq_test_configs)
    return calculator


@pytest.fixture(scope="module")
def ndim(calc):
    return calc.get_num_opt_params()


@pytest.fixture(scope="module")
def loss(calc):
    """Create loss instance."""
    L = Loss(calc)
    L.minimize(method="lm")
    return L


@pytest.fixture(scope="module")
def ptemcee_avail():
    """Check if ptemcee is available."""
    try:
        from kliff.uq.mcmc import PtemceeSampler

        return True
    except ModuleNotFoundError:
        return False


@pytest.fixture(scope="module")
def emcee_avail():
    """Check if emcee is available."""
    try:
        from kliff.uq.mcmc import EmceeSampler

        return True
    except ModuleNotFoundError:
        return False


@pytest.fixture(scope="module")
def prior_bounds(ndim):
    """Return prior bounds."""
    return np.tile([0, 10], (ndim, 1))


@pytest.fixture(scope="module")
def ptsampler(ptemcee_avail, loss, prior_bounds):
    """Instantiate PTSampler."""
    print("Instantiating PTSampler")
    return MCMC(
        loss,
        ntemps=ntemps,
        nwalkers=nwalkers,
        logprior_args=(prior_bounds,),
        random=np.random.RandomState(seed),
    )


@pytest.fixture(scope="module")
def sampler(emcee_avail, loss, prior_bounds):
    """Instantiate Emcee Sampler."""
    print("Instantiating Emcee Sampler")
    return MCMC(loss, nwalkers=nwalkers, logprior_args=(prior_bounds,), sampler="emcee")


def test_T0(calc, loss):
    """Test if the function to compute T0 works properly. This is done by comparing T0
    computed using the internal function and computed manually.
    """
    # Using internal function
    T0_internal = get_T0(loss)

    # Compute manually
    xopt = calc.get_opt_params()
    T0_manual = 2 * loss._get_loss(xopt) / len(xopt)
    assert T0_internal == T0_manual, "Internal function to compute T0 doesn't work"


@pytest.mark.skipif(not ptemcee_avail, reason="ptemcee is not found")
def test_MCMC_wrapper1(ptsampler):
    """Test if the MCMC wrapper class returns the correct sampler instance."""
    assert (
        type(ptsampler) == PtemceeSampler
    ), "MCMC should return ``PtemceeSampler`` instance"


@pytest.mark.skipif(not emcee_avail, reason="emcee is not found")
def test_MCMC_wrapper2(sampler):
    assert type(sampler) == EmceeSampler, "MCMC should return ``EmceeSampler`` instance"


@pytest.mark.skipif(not ptemcee_avail, reason="ptemcee is not found")
def test_dimensionality1(ptsampler, ndim):
    """Test the number of temperatures, walkers, steps, and parameters. This is done by
    comparing the shape of the resulting MCMC chains and the variables used to set these
    dimensions.
    """

    # Test for ptemcee wrapper
    p0 = np.random.uniform(0, 10, (ntemps, nwalkers, ndim))
    ptsampler.run_mcmc(p0=p0, iterations=nsteps)
    assert ptsampler.chain.shape == (
        ntemps,
        nwalkers,
        nsteps,
        ndim,
    ), "Dimensionality from the ptemcee wrapper is not right"


@pytest.mark.skipif(not emcee_avail, reason="emcee is not found")
def test_dimensionality2(sampler, ndim):
    # Test for emcee wrapper
    p0 = np.random.uniform(0, 10, (nwalkers, ndim))
    sampler.run_mcmc(initial_state=p0, nsteps=nsteps)
    assert sampler.get_chain().shape == (
        nsteps,
        nwalkers,
        ndim,
    ), "Dimensionality from the emcee wrapper is not right"


@pytest.mark.skipif(not ptemcee_avail, reason="ptemcee is not found")
def test_pool_exception(loss, prior_bounds):
    """Test if an exception is raised when declaring the pool prior to instantiating
    ``kliff.uq.MCMC``.
    """
    with pytest.raises(ValueError):
        _ = MCMC(
            loss,
            ntemps=ntemps,
            nwalkers=nwalkers,
            logprior_args=(prior_bounds,),
            pool=Pool(1),
        )


def test_sampler_exception(loss, prior_bounds):
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
