.. _doc.uq:

===============================
Uncertainty Quantification (UQ)
===============================

Uncertainty quantification (UQ) is an emerging field in applied mathematics that aims to
quantify uncertainties in mathematical models as a result from error propagation in the
modeling process. This is especially important since we use the model, i.e., the
potential, to predict material properties that are not used in the
training process. Thus, UQ process is especially important to assess the reliability of
these out-of-sample predictions.

In UQ process, we first quantify the uncertainty of the model parameters. Having found
the parametric uncertainty, then we can propagate the uncertainty of the parameters and
get the uncertainty of the material properties of interest, e.g., by evaluating the
ensemble that is obtained from sampling the distribution of the parameters. As the first
uncertainty propagation is more involved, KLIFF implements tools to quantify the
uncertainty of the parameters.


MCMC
====

The Bayesian Markov chain Monte Carlo (MCMC) is the UQ method of choice for interatomic
potentials. The distribution of the parameters is given by the posterior
:math:`P(\bm \theta | \bm q)`. By Bayes' theorem

.. math::

   P(\bm \theta | \bm q) \propto L(\bm \theta | \bm q) \times \pi(\bm \theta),

where :math:`L(\bm \theta | \bm q)` is the likelihood (which encodes the information
learned from the data) and :math:`\pi(\bm \theta)` is the prior distribution. Then, some
MCMC algorithm is used to sample the posterior and the distribution of the parameters is
inferred from the distribution of the resulting samples.

The likelihood function is given by

.. math::

   L(\bm \theta | \bm q) \propto \exp \left( -\frac{\mathcal{L}(\bm \theta)}{T} \right).

The inclusion of the sampling temperature :math:`T` is to account for model inadequacy,
or bias, in the potential [Kurniawan2022]_. Frederiksen et al. (2004) [Frederiksen2004]_
suggest to estimate the bias by setting the temperature to

.. math::

   T_0 = \frac{2 \mathcal{L}_0}{N},

where :math:`\mathcal{L}_0` is the value of the loss function evaluated at the best fit
parameters and :math:`N` is the number of tunable parameters.

.. seealso::
   For more discussion about this formulation, see [KurniawanKLIFFUQ]_.



.. [Kurniawan2022]
   Kurniawan, Y., Petrie, C.L., Williams Jr., K.J., Transtrum, M.K., Tadmor, E.B.,
   Elliott, R.S., Karls, D.S., Wen, M., 2022. Bayesian, frequentist, and information
   geometric approaches to parametric uncertainty quantification of classical empirical
   interatomic potentials. J. Chem. Phys. https://doi.org/10.1063/5.0084988

.. [Frederiksen2004] S. L. Frederiksen, K. W. Jacobsen, K. S. Brown, and J. P.
   Sethna, “Bayesian Ensemble Approach to Error Estimation of Interatomic
   Potentials,” Phys. Rev. Lett., vol. 93, no. 16, p. 165501, Oct. 2004,
   doi: 10.1103/PhysRevLett.93.165501.

.. [KurniawanKLIFFUQ]
   Kurniawan, Y., Petrie, C.L., Transtrum, M.K., Tadmor, E.B., Elliott, R.S., Karls,
   D.S., Wen, M., 2022. Extending OpenKIM with an Uncertainty Quantification Toolkit for
   Molecular Modeling. arXiv:2206.00578 [physics.comp-ph]



MCMC implementation
===================

In KLIFF, the UQ tools are implemented in :mod:`~kliff.uq`. In the currect version, only
MCMC sampling is implemented, with the integration of other UQ methods will be added in
the future.

For the MCMC sampling, KLIFF adopts parallel-tempered MCMC (PTMCMC) methods, via the
ptemcee_ Python package, as a way to perform MCMC sampling with several different
temperatures. Additionally, multiple parallel walkers are deployed for each sampling
temperature. PTMCMC has been widely used to improve the mixing rate of the sampling.
Additionally, by sampling at several different temperatures, we can assess the effect of
the size of the bias to any conclusion drawn from the samples.

We start the UQ process by instantiating :class:`~kliff.uq.MCMC`,

.. code-block:: python

   from kliff.uq import MCMC

   loss = ...  # define the loss function
   sampler = MCMC(
       loss, nwalkers, logprior_fn, logprior_args, ntemps, Tmax_ratio, Tladder, **kwargs
   )

The available arguments are:

* ``loss``, which is a :class:`~kliff.loss.Loss` instance. This is a required argument
  that is used to construct the untempered likelihood function (:math:`T=1`) and to
  compute :math:`T_0`.
* ``nwalkers`` specifies the number of parallel walkers to run for each sampling
  temperature. As a default, this quantity is set to twice the number of parameters in
  the model.
* ``logprior_fn`` argument allows user to specify the prior distribution to use. The
  function should accept an array of parameter values as an input and compute the
  logarithm of the prior distribution. Note that the distribution doesn't need to be
  normalized. The default prior is a uniform distribution over a finite range. See the
  argument on how to set the boundaries of the uniform prior.
* ``logprior_args`` is a tuple that contains additional positional arguments that is
  needed by ``logprior_fn``. If the default uniform prior is used, then the boundaries of
  the prior need to be specified here as a :math:`N \times 2` array, where the first and
  second columns of the array contain the lower and upper bound for each parameter.
* ``ntemps`` specifies the number of temperatures to simulate.
* ``Tmax_ratio`` is used to set the highest temperature by
  :math:`T_{\text{max}} = T_{\text{max\_ratio}} \times T_0`. An internal function is used
  to construct a list of logarithmically spaced ``ntemps`` points from 1.0 to
  :math:`T_{\text{max}}`, inclusive.
* ``Tladder`` allows user to specify a list of temperatures to use. This argument will
  overwrites ``ntemps`` and ``Tmax_ratio``.
* Other keyword arguments to be passed into ``ptemcee.Sampler`` needs to be specified in
  ``kwargs``.

.. note::
   For PTMCMC run, :class:`~kliff.uq.MCMC` creates a ``ptemcee.Sampler`` instance, which
   can be accesses directly through :attr:`~kliff.uq.PtemceeSampler.sampler`.

.. How to run sampling
After the sampler is created, the MCMC run is done by calling
:meth:`~kliff.uq.MCMC.run_mcmc`.

.. code-block:: python

   p0 = ...  # Define the initial position of each walker
   sampler.run_mcmc(p0, iterations, *args, **kwargs)

This function is the same as ptemcee.Sampler.run_mcmc_. The required arguments are:

* ``p0``, which is a :math:`K \times L \times N` array containing the position of each
  walker for each temperature in parameter space, where :math:`K`, :math:`L`, and
  :math:`N` are the number of temperatures, walkers, and parameters, respectively.
* ``iterations`` specifies the number of MCMC step to take. Since the position is step
  :math:`i` in Markov chain only depends on step :math:`(i-1)`, it is possible to break
  up the MCMC run into smaller batches, with the note that the initial positions of the
  current run needs to be set to the last positions of the previous run.

The resulting chain can be retrieved from :attr:`~kliff.uq.PtemceeSampler.chain` as a
:math:`K \times L \times M \times N` array, where :math:`M` is the total number of
iteratiions.

.. note::
   The result that can be accessed directly from :class:`~kliff.uq.MCMC` is the chain.
   Other properties, such as ``logposterior``, ``loglikelihood``, etc., can still be
   retrieved from :attr:`~kliff.uq.PtemceeSampler.sampler`.

.. _ptemcee: https://github.com/willvousden/ptemcee/tree/1.0.0
.. _ptemcee.Sampler.run_mcmc: https://github.com/willvousden/ptemcee/blob/1.0.0/ptemcee/sampler.py#L272-L279


MCMC analysis
=============

The chains from the MCMC simulation needs to be processed. In a nutshell, the steps to
take are

* Estimate the burn-in time and discard it from the beginning of the chain,
* Estimate the autocorrelation length, :math:`\tau`, and only take every :math:`\tau` step
  from the remaining chain,
* Assess convergence of the samples, i.e., the remaining chain after the two steps above.


Burn-in time
------------

First we need to discard the first few iterations in the beginning of each chain as a
burn-in time. This is similar to the equilibration time in a molecular dynamics
simulation before the measurement. This action also ensure that the result is independent
of the initial positions of the walkers.

KLIFF provides a function to estimate the burn-in time, based on the Marginal Standard
Error Rule (MSER). This can calculation can be done using the function
:func:`~kliff.uq.mcmc_utils.mser`. However, note that this calculation needs to be
performed for each temperature, walker, and parameter dimension separately.

Autocorrelation length
----------------------

In Markov chain, the position at step :math:`i` is not independent from the previous step.
However, after several iterations (denote this number by :math:`\tau`, which is the
autocorrelation length), the walker will "forget" where it started, i.e., the position at
step :math:`i` is independent from step :math:`(i + \tau)`. Thus, we need to only keep
every :math:`\tau \text{-th}` step to obtain the independent, unceorrelated samples.

The estimation of the autocorrelation length in KLIFF is done via the function
:func:`~kliff.uq.mcmc_utils.autocorr`, which wraps over
``emcee.autocorr.integrated_time``. This calculation needs to be done for each temperature
independently. The required input argument is a :math:`L \times \tilde{M} \times N`
array, where :math:`L` and :math:`N` are the number of walkers and parameters,
respectively, and :math:`\tilde{M}` is the remaining number of iterations after
discarding the burn-in time.

Convergence
-----------

Finally, after a sufficient number of iterations, the distribution of the MCMC samples
will converge to the posterior. For multi-chain MCMC simulation, the convergence can be
assessed by calculating the multivariate potential scale reduction factor, denoted by
:math:`\hat{R}^p`. This quantity compares the variance between and within independent
chains. The value of :math:`\hat{R}^p` declines to 1 as the number of iterations goes to
infinity, with a common threshold is about 1.1.

In KLIFF, the function :func:`~kliff.uq.rhat` computes :math:`\hat{R}^p` for one
temperature. The required input argument is a :math:`L \times \tilde{M}^* \times N` array
of independent samples (:math:`\tilde{M}^*` is the number of independent samples in each
walker). When the resulting :math:`\hat{R}^p` values are larger than the threshold
(e.g., 1.1), then the MCMC simulation should be continued until this criteria is
satisfied.

.. note::
   Some sampling temperatures might converge at slower rates compared to the others.
   So, user can terminate the MCMC simulation as long as the samples at the target
   temperatures, e.g., :math:`T_0`, have converged.


.. seealso::
   See the tutorial for running MCMC in :ref:`tut_mcmc`.
