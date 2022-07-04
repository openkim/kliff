from typing import Optional

import numpy as np

from kliff.error import report_import_error

try:
    import emcee

    emcee_avail = True
except ImportError:
    emcee_avail = False


# Estimate the burn-in time
def mser(
    chain: np.ndarray,
    dmin: Optional[int] = 1,
    dstep: Optional[int] = 10,
    dmax: Optional[int] = -1,
    full_output: Optional[bool] = False,
) -> int:
    """Estimate the equilibration time using marginal standard error rule (MSER). This is
    done by calculating the standard error (square) of ``chain_d``, where ``chain_d``
    contains the last :math:`n-d` element of the chain (n is the total number of
    iterations for each chain), for progresively larger d values, starting from ``dmin``
    upto ``dmax``, incremented by ``dstep``. The SE values are stored in a list. Then we
    search the minimum element in the list and return the index of that element.

    Parameters
    ----------
    chain: 1D np.ndarray
        Array containing the time series.
    dmin: int
        Index where to start the search in the time series.
    dstep: int
        How much to increment the search is done.
    dmax: int
        Index where to stop the search in the time series.
    full_output: bool
        A flag to return the list of squared standard error.

    Returns
    -------
    dstar: int or dict
        Estimate of the equilibration time using MSER. If ``full_output=True``, then a
        dictionary containing the estimated equilibration time and the list of squared
        standard errors will be returned.
    """
    length = len(chain)  # Chain length

    # Compute the SE square
    SE2_list = [
        _standard_error_squared(chain[dd:]) for dd in range(length)[dmin:dmax:dstep]
    ]

    # Get the estimate of the equilibration time, wrt the original time series
    dest = np.argmin(SE2_list)
    dstar = min([dmin + (dest + 1) * dstep, length])

    if full_output:
        return {"dstar": dstar, "SE2": SE2_list}
    else:
        return dstar


# Estimate autocorrelation length
def autocorr(chain: np.ndarray, *args, **kwargs):
    """Use ``emcee`` package to estimate the autocorrelation length.

    Parameters
    ----------
    chain: np.ndarray (nwalkers, nsteps, ndim,)
        Chains from the MCMC simulation. The shape of the chains needs to be
        (nsteps, nwalkers, ndim). Note that the burn-in time needs to be discarded prior
        to this calculation
    args, kwargs
        Additional positional and keyword arguments of ``emcee.autocorr.integrated_time``.

    Returns
    -------
    float or array:
        Estimate of the autocorrelation length for each parameter.
    """
    if emcee_avail:
        chain = np.swapaxes(chain, 0, 1)
        return emcee.autocorr.integrated_time(chain, *args, **kwargs)
    else:
        report_import_error("emcee")


# Assess convergence
def rhat(chain, time_axis: int = 1, return_WB: bool = False):
    """Compute the value of :math:`\\hat{r}` proposed by Brooks and Gelman
    [BrooksGelman1998]_. If the samples come from PTMCMC simulation, then the chain needs
    to be from one of the temperature only.

    Parameters
    ----------
    chain: ndarray
        The MCMC chain as a ndarray, preferrably with the shape
        (nwalkers, nsteps, ndims). However, the shape can also be
        (nsteps, nwalkers, ndims), but the argument time_axis needs to be set
        to 0.
    time_axis: int (optional)
        Axis in which the time series is stored (0 or 1). For emcee results,
        the time series is stored in axis 0, but for ptemcee for a given
        temperature, the time axis is 1.
    return_WB: bool (optional)
        A flag to return covariance matrices within and between chains.

    Returns
    -------
    r: float
        The value of rhat.
    W, B: 2d ndarray
        Matrices of covariance within and between the chains.

    References
    ----------
    .. [BrooksGelman1998]
       Brooks, S.P., Gelman, A., 1998. General Methods for Monitoring Convergence of
       Iterative Simulations. Journal of Computational and Graphical Statistics 7,
       434455. https://doi.org/10.1080/10618600.1998.10474787
    """
    if time_axis == 1:
        # Reshape the chain so that the time axis is in axis 1
        temp = np.swapaxes(chain, 0, 1)
        chain = temp

    m, n, _ = chain.shape
    lambda1, W, B = _lambda1(chain)
    r = 1 - 1 / n + (1 + 1 / m) * lambda1

    if return_WB:
        toreturn = (r, W, B)
    else:
        toreturn = r

    return toreturn


def _standard_error_squared(chain: np.ndarray) -> float:
    """Compute the square of the standard error."""
    nn = len(chain)
    se2 = np.var(chain) / nn
    return se2


def _lambda1(chain):
    """Compute the largest eigenvalue of :math:`W^{-1} B/n`."""
    W = _W(chain)
    B_over_n = _B_over_n(chain)
    V = np.linalg.lstsq(W, B_over_n, rcond=-1)[0]
    s = np.linalg.svd(V, compute_uv=False)
    return np.max(s), W, B_over_n


def _B_over_n(chain):
    """Compute covariance matrix between the chains."""
    return np.cov(np.mean(chain, axis=1), rowvar=False, ddof=1)


def _W(chain):
    """Compute the mean of the covariance matrix within each chain."""
    m, n, nparams = chain.shape
    Wm = np.empty((m, nparams, nparams))
    for walker in range(m):
        Wm[walker] = np.cov((chain[walker]), rowvar=False, ddof=1)
    return np.mean(Wm, axis=0)
