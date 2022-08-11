import warnings
from typing import Callable, List, Optional

import numpy as np

from kliff.error import report_import_error
from kliff.loss import Loss

try:
    import emcee

    emcee_avail = True
except ImportError:
    emcee_avail = False

try:
    import ptemcee

    ptemcee_avail = True
except ImportError:
    ptemcee_avail = False


def logprior_uniform(x: np.ndarray, bounds: np.ndarray) -> float:
    """Logarithm of the non-normalized joint uniform prior. This is the default prior
    distribution.

    Parameters
    ----------
    x : np.ndarray
        Parameter values to evaluate.
    bounds : np.ndarray (nparams, 2,)
        An array containing the boundaries of the uniform prior. The first column of the
        array contains the lower bounds and the second column contains the upper bounds.

    Returns
    -------
    float:
        Logarithm of the non-normalized joint uniform prior evaluated at parameter ``x``.
    """
    l_bounds, u_bounds = bounds.T
    if all(np.less(x, u_bounds)) and all(np.greater(x, l_bounds)):
        ret = 0.0
    else:
        ret = -np.inf

    return ret


def get_T0(loss: Loss):
    """Compute the natural temperature. The minimum loss is the loss value at the optimal
    parameters.

    Parameters
    ----------
    loss : Loss
        Loss function class from :class:`~kliff.loss.Loss`.

    Returns
    -------
    float
        Value of the natural temperature.
    """
    xopt = loss.calculator.get_opt_params()
    ndim = len(xopt)
    C0 = loss._get_loss(xopt)
    return 2 * C0 / ndim


class MCMC:
    """MCMC sampler class for Bayesian uncertainty quantification.

    This is a wrapper over :class:`PtemceeSampler` and
    :class:`EmceeSampler`. Currently, only these  2 samplers implemented.

    Parameters
    ----------
    loss : Loss
        Loss function class from :class:`~kliff.loss.Loss`.
    nwalkers : Optional[int]
        Number of walkers to simulate. The minimum number of walkers
        is twice the number of parameters. It defaults to this minimum
        value.
    logprior_fn : Optional[Callable]
        A function that evaluate logarithm of the prior
        distribution. The prior doesn't need to be normalized. It
        defaults to a uniform prior over a finite range.
    logprior_args : Optional[tuple]
        Additional positional arguments of the ``logprior_fn``. If the
        default ``logprior_fn`` is used, then the boundaries of the
        uniform prior can be specified here.
    sampler : Optional[str] or sampler instance
        An argument that specifies the MCMC sampler to use. The value can be one of the
        strings ``"ptemcee"`` (the default value) or ``"emcee"``, or a sampler class
        instance. If ``"ptemcee"`` or ``"emcee"`` is given, a respective internal sampler
        class will be uses.
    **kwargs : Optional[dict]
        Additional keyword arguments for ``ptemcee.Sampler`` or ``emcee.EnsembleSampler``.
    """

    builtin_samplers = ["ptemcee", "emcee"]

    def __new__(
        self,
        loss: Loss,
        nwalkers: Optional[int] = None,
        logprior_fn: Optional[Callable] = None,
        logprior_args: Optional[tuple] = None,
        sampler: Optional[str] = "ptemcee",
        **kwargs,
    ):

        if sampler == "ptemcee":
            if ptemcee_avail:
                # Force pool to be declared after MCMC is instantiated
                if "pool" in kwargs:
                    raise ValueError(
                        "Please declare the pool after instantiating ``kliff.uq.MCMC``"
                    )

                # Pre-process keyword arguments
                ntemps = kwargs.pop("ntemps") if "ntemps" in kwargs else 10
                Tmax_ratio = kwargs.pop("Tmax_ratio") if "Tmax_ratio" in kwargs else 1.0
                Tladder = kwargs.pop("Tladder") if "Tladder" in kwargs else None

                return PtemceeSampler(
                    loss,
                    nwalkers,
                    ntemps,
                    Tmax_ratio,
                    Tladder,
                    logprior_fn,
                    logprior_args,
                    **kwargs,
                )
            else:
                report_import_error("ptemcee")
        elif sampler == "emcee":
            if emcee_avail:
                # Force pool to be declared after MCMC is instantiated
                if "pool" in kwargs:
                    raise ValueError(
                        "Please declare the pool after instantiating ``kliff.uq.MCMC``"
                    )

                # Pre-process keyword arguments
                T = kwargs.pop("T") if "T" in kwargs else None

                return EmceeSampler(
                    loss, nwalkers, T, logprior_fn, logprior_args, **kwargs
                )
            else:
                report_import_error("emcee")
        elif isinstance(sampler, str) and sampler not in self.builtin_samplers:
            raise ValueError(
                "Unknown sampler. Please input 'ptemcee', 'emcee' or "
                "some sampler class instance"
            )
        else:
            warnings.warn("The other arguments are ignored")
            return sampler


if ptemcee_avail:

    class PtemceeSampler(ptemcee.Sampler):
        """Sampler class for PTMCMC via ``ptemcee`` Python package.

        Parameters
        ----------
        loss : Loss
            Loss function instance from :class:`~kliff.loss.Loss`.
        nwalkers : Optional[int]
            Number of walkers to simulate. The minimum number of walkers
            is twice the number of parameters. It defaults to this minimum
            value.
        ntemps: Optional[int]
            Number of temperatures to simulate. It defaults to 10.
        Tmax_ratio: Optional[float]
            The ratio between the highest temperature to use and the natural temperature.
            Higher value means that the maximum temperature is higher than :math:`T_0`
            [Frederiksen2004]_. It defaults to 1.0.
        Tladder: Optional[List]
            A list containing the temperature values to use. The values nedd to be
            monotonically increasing or decreasing. It defaults to ``None``, which will
            be generated from ``ntemps`` and ``Tmax_ratio``.
        logprior_fn : Optional[Callable]
            A function that evaluate logarithm of the prior
            distribution. The prior doesn't need to be normalized. It
            defaults to a uniform prior over a finite range.
        logprior_args : Optional[tuple]
            Additional positional arguments of the ``logprior_fn``. If the
            default ``logprior_fn`` is used, then the boundaries of the
            uniform prior can be specified here.
        **kwargs : Optional[dict]
            Additional keyword arguments for ``ptemcee.Sampler``.

        Attributes
        ----------
        loss: Loss
            Loss function instance from :class:`~kliff.loss.Loss`
        T0: float
            Values of the natural temperature, :math:`T_0` [Frederiksen2004]_.
        Tladder: np.ndarray
            An array containing the values of the sampling temperatures.
        sampler: ptemcee.Sampler
            Sampler instance from ``ptemcee.Sampler``
        """

        def __init__(
            self,
            loss: Loss,
            nwalkers: Optional[int] = None,
            ntemps: Optional[int] = 10,
            Tmax_ratio: Optional[float] = 1.0,
            Tladder: Optional[List] = None,
            logprior_fn: Optional[Callable] = None,
            logprior_args: Optional[tuple] = None,
            **kwargs,
        ):
            self.loss = loss

            # Dimensionality
            ndim = loss.calculator.get_num_opt_params()
            nwalkers = 2 * ndim if nwalkers is None else nwalkers

            # Probability
            global loglikelihood_fn

            def loglikelihood_fn(x):
                return _get_loglikelihood(x, loss)

            if logprior_fn is None:
                logprior_fn = logprior_uniform
                if logprior_args is None:
                    logprior_args = (_get_parameter_bounds(loss),)

            # Sampling temperatures
            self.T0 = None
            self.Tladder = self._generate_temp_ladder(ntemps, Tmax_ratio, Tladder)
            betas = 1 / self.Tladder

            super().__init__(
                nwalkers,
                ndim,
                loglikelihood_fn,
                logprior_fn,
                logpargs=logprior_args,
                betas=betas,
                **kwargs,
            )

        def _generate_temp_ladder(self, ntemps, Tmax_ratio, Tladder):
            """Generate temperature ladder"""
            # Only generate temperature ladder when it is not specified.
            if Tladder is None:
                # Compute T0
                self.T0 = get_T0(self.loss)

                Tmax = Tmax_ratio * self.T0
                Tmax_not_T0 = Tmax_ratio != 1.0

                # If Tmax is not T0, then we need to generate 1 less temperature than
                # requested, and append T0 afterward.
                if Tmax_not_T0:
                    ntemps -= 1

                Tladder = np.logspace(0, np.log10(Tmax), ntemps)

                if Tmax_not_T0:
                    Tladder = np.sort(np.append(Tladder, self.T0))
            return Tladder


if emcee_avail:

    class EmceeSampler(emcee.EnsembleSampler):
        """Sampler class for affine invariant MCMC via ``emcee`` Python package.

        Parameters
        ----------
        loss : Loss
            Loss function instance from :class:`~kliff.loss.Loss`.
        nwalkers : Optional[int]
            Number of walkers to simulate. The minimum number of walkers
            is twice the number of parameters. It defaults to this minimum
            value.
        T: Optional[float]
            Sampling temperatures, used to inflate the likelihood function in the MCMC
            sampling. It defaults to the natural temperature :math:`T_0` [Frederiksen2004]_.
        logprior_fn : Optional[Callable]
            A function that evaluate logarithm of the prior
            distribution. The prior doesn't need to be normalized. It
            defaults to a uniform prior over a finite range.
        logprior_args : Optional[tuple]
            Additional positional arguments of the ``logprior_fn``. If the
            default ``logprior_fn`` is used, then the boundaries of the
            uniform prior can be specified here.
        **kwargs : Optional[dict]
            Additional keyword arguments for ``emcee.EnsembleSampler``.

        Attributes
        ----------
        loss: Loss
            Loss function instance from :class:`~kliff.loss.Loss`
        T: float
            Values of the sampling temperature.
        sampler: emcee.EnsembleSampler
            Sampler instance from ``emcee.EnsembleSampler``

        Notes
        -----
        As a convention, KLIFF inflates the likelihood by some sampling temperature, i.e.,
        :math:`L(\\theta) \propto \exp(-C(\\theta) / T)`. As a default, the sampling
        temperature is set to the natural temperature. To use the untempered likelihood
        (:math:`T=1`), user should specify the argument ``T=1``.


        References
        ----------
        .. [Frederiksen2004] S. L. Frederiksen, K. W. Jacobsen, K. S. Brown, and J. P.
           Sethna, “Bayesian Ensemble Approach to Error Estimation of Interatomic
           Potentials,” Phys. Rev. Lett., vol. 93, no. 16, p. 165501, Oct. 2004,
           doi: 10.1103/PhysRevLett.93.165501.
        """

        def __init__(
            self,
            loss: Loss,
            nwalkers: Optional[int] = None,
            T: Optional[float] = None,
            logprior_fn: Optional[Callable] = None,
            logprior_args: Optional[tuple] = None,
            **kwargs,
        ):
            self.loss = loss

            # Dimensionality
            ndim = loss.calculator.get_num_opt_params()
            nwalkers = 2 * ndim if nwalkers is None else nwalkers

            # Probability
            if T is None:
                self.T = get_T0(self.loss)
            else:
                self.T = T
            logl_fn = self._loglikelihood_wrapper
            logp_fn = self._logprior_wrapper(logprior_fn, *logprior_args)

            global logprobability_fn

            def logprobability_fn(x):
                return logl_fn(x) + logp_fn(x)

            super().__init__(nwalkers, ndim, logprobability_fn, **kwargs)

        def _loglikelihood_wrapper(self, x):
            """A wrapper to the log-likelihood function, so that the only argument is the
            parameter values.
            """
            return _get_loglikelihood(x, self.loss, self.T)

        def _logprior_wrapper(self, logprior_fn, *logprior_args):
            """A wapper to the log-prior function, so that the only argument is the
            parameter values.
            """
            if logprior_fn is None:
                if logprior_args is None:
                    logprior_args = (_get_parameter_bounds(self.loss),)
                logprior_fn = logprior_uniform

            def logp(x):
                return logprior_fn(x, *logprior_args)

            return logp


def _get_loglikelihood(x: np.ndarray, loss: Loss, T: Optional[float] = 1.0):
    """Compute the log-likelihood from the cost function. It has an option to temper the
    cost by specifying ``T``.
    """
    cost = loss._get_loss(x)
    logl = -cost / T
    return logl


def _get_parameter_bounds(loss):
    """Get the parameter bounds for the default uniform prior."""
    bounds = loss.calculator.get_opt_params_bounds()
    if np.any(np.isin(bounds, None)):
        raise MCMCError(
            "Improper uniform prior bound is used. Please specify a finite range."
        )
    return np.array(bounds)


class MCMCError(Exception):
    def __init__(self, msg):
        super(MCMCError, self).__init__(msg)
        self.msg = msg
