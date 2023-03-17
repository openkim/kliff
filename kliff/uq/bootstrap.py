"""I am still working on this script to do bootstrap UQ."""

import os
import copy
import json
from pathlib import Path

import numpy as np

from kliff.calculators.calculator import Calculator, _WrapperCalculator
from kliff.calculators.calculator_torch import CalculatorTorchSeparateSpecies
from kliff.loss import Loss, energy_forces_residual, energy_residual, forces_residual


def bootstrap_cas_generator_empirical(nsamples, orig_cas):
    """
    Default class to generate bootstrap compute arguments for empirical, physics-based
    model. The compute arguments from all calculators will be combined, then the bootstrap
    sample configurations will be generated from the combined list. Afterwards, the
    configurations will be split into their respective calculators.

    Parameters
    ----------
    nsamples: int
        Number of the bootstrap compute arguments requested.

    orig_cas: list
        The original list of compute arguments. The bootstrap compute arguments will be
        generated from this list. The format of this input is given below::

            orig_cas = [
                [calc0_ca0, calc0_ca1, ...],
                [calc1_ca0, calc1_ca1, ...],
                ...
            ]

    Returns
    -------
    dict
        A set of bootstrap compute arguments, written in a dictionary format, where the
        keys index the bootstrap samples compute arguments::

            bootstrap_configs = {
                0: [[calc0_cas], [calc1_cas]],
                1: [[calc0_cas], [calc1_cas]]
            }

    """
    ncalc = len(orig_cas)  # Number of calculators
    ncas = [len(cas) for cas in orig_cas]  # Number of compute args per calc
    ncas_total = sum(ncas)  # Total number of compute arguments
    # Combine the compute arguments
    comb_orig_cas = np.concatenate((orig_cas))
    # Index of which calculator each ca correspond to
    calc_idx = np.concatenate([[ii] * nc for ii, nc in enumerate(ncas)])
    bootstrap_cas = {}
    for ii in range(nsamples):
        # Generate a bootstrap sample configuration
        # Generate the bootstrap indices
        bootstrap_idx = np.random.choice(range(ncas_total), size=ncas_total, replace=True)
        # From the indices, get bootstrap compute arguments
        comb_bootstrap_cas = [comb_orig_cas[ii] for ii in bootstrap_idx]
        # We also need to deal with the calculator index
        comb_bootstrap_calc_idx = calc_idx[bootstrap_idx]

        # Split the bootstrap cas into separate calculators
        bootstrap_cas_single_sample = [[] for _ in range(ncalc)]
        for idx, ca in zip(comb_bootstrap_calc_idx, comb_bootstrap_cas):
            bootstrap_cas_single_sample[idx].append(ca)

        # Update the bootstrap compute arguments dictionary
        bootstrap_cas.update({ii: bootstrap_cas_single_sample})
    return bootstrap_cas


def get_identifiers_from_compute_arguments(compute_arguments):
    """
    Retrieve the identifiers of a list of compute arguments.

    Parameters
    ----------
    compute_arguments: list
        A list of :class:`~kliff.models.model.ComputeArguments`.

    Returns
    -------
    identifiers: list
        A list of compute arguments' identifiers, which shows the paths to the xyz files.
    """
    identifiers = []
    for cas in compute_arguments:
        # Iterate over compute arguments corresponding to each calculator
        identifiers.append([ca.conf.identifier for ca in cas])
    return identifiers


class BootstrapEmpiricalModel:
    """
    Bootstrap sampler class for empirical, physics-based potentials.

    Parameters
    ----------
    loss: Loss
        Loss function class from :class:`~kliff.loss.Loss`.
    """

    def __init__(self, loss):
        self.loss = loss
        self.calculator = loss.calculator
        # Cache the original parameter values
        self.orig_params = copy.copy(self.calculator.get_opt_params())
        # Cache the original compute arguments
        if isinstance(self.calculator, Calculator):
            self.orig_compute_arguments = [
                copy.copy(self.calculator.get_compute_arguments())
            ]
            self.use_multi_calc = False
        elif isinstance(self.calculator, _WrapperCalculator):
            self.orig_compute_arguments = copy.copy(
                self.calculator.get_compute_arguments(flat=False)
            )
            self.use_multi_calc = True
        self._orig_compute_arguments_identifiers = get_identifiers_from_compute_arguments(
            self.orig_compute_arguments
        )
        # Initiate the bootstrap configurations property
        self.bootstrap_compute_arguments = {}
        self.samples = np.empty((0, self.calculator.get_num_opt_params()))

    @property
    def _nsamples_done(self):
        """
        For each bootstrap compute arguments sample, we need to train the potential using
        these compute arguments. This function retrieve how many bootstrap compute
        arguments samples have been used in the training. This is to help so that we can
        continue the calculation midway.
        """
        return len(self.samples)

    @property
    def _nsamples_prepared(self):
        """
        Get how many bootstrap compute arguments are prepared. This also include those
        compute arguments set that have been evaluated.
        """
        return len(self.bootstrap_compute_arguments)

    def generate_bootstrap_compute_arguments(
        self, nsamples, bootstrap_cas_generator_fn=None, **kwargs
    ):
        """
        Generate bootstrap compute arguments samples. If this function is called multiple,
        say, K times, then it will in total generate: math: `K \times nsamples` bootstrap
        compute arguments samples. That is , consecutive call of this function will append
        the generated compute arguments samples.

        Parameters
        ----------
        nsamples: int
            Number of bootstrap samples to generate.
        bootstrap_cas_generator_fn: callable ``fn(nsamples, **kwargs)`` (optional)
            A function to generate bootstrap compute argument samples. The default
            function combine the compute arguments across all calculators and do sampling
            with replacement from the combined list. Another possible convention is to
            do sampling with replacement on the compute arguments list of each calculator
            separately, in which case a custom function needs to be defined and used.
        kwargs: dict
            Additional keyword arguments to ``bootstrap_cas_generator_fn``.
        """
        # Function to generate bootstrap configurations
        if bootstrap_cas_generator_fn is None:
            bootstrap_cas_generator_fn = bootstrap_cas_generator_empirical
            kwargs = {"orig_cas": self.orig_compute_arguments}

        # Generate a new bootstrap configurations
        new_bootstrap_compute_arguments = bootstrap_cas_generator_fn(nsamples, **kwargs)
        # Update the old list with this new list
        self.bootstrap_compute_arguments = self._update_bootstrap_compute_arguments(
            new_bootstrap_compute_arguments
        )

    def _update_bootstrap_compute_arguments(self, new_bootstrap_compute_arguments):
        """
        Append the new generated bootstrap compute arguments samples to the old list.
        """
        bootstrap_compute_arguments = copy.copy(self.bootstrap_compute_arguments)
        for ii, vals in new_bootstrap_compute_arguments.items():
            iteration = ii + self._nsamples_prepared
            bootstrap_compute_arguments.update({iteration: vals})

        return bootstrap_compute_arguments

    def save_bootstrap_compute_arguments(self, filename):
        """
        Export the generated bootstrap compute arguments as a json file. The json file
        will contain the identifier of the compute arguments for each sample.

        Parameters
        ----------
        filename: str or Path
            Where to export the bootstrap compute arguments samples
        """
        # We cannot directly export the bootstrap compute arguments. Instead, we will
        # first convert it and list the identifiers and export the identifiers.
        # Convert to identifiers
        bootstrap_compute_arguments_identifiers = {}
        for ii in self.bootstrap_compute_arguments:
            bootstrap_compute_arguments_identifiers.update(
                {
                    ii: get_identifiers_from_compute_arguments(
                        self.bootstrap_compute_arguments[ii]
                    )
                }
            )

        with open(filename, "w") as f:
            json.dump(bootstrap_compute_arguments_identifiers, f, indent=4)

    def load_bootstrap_compute_arguments(self, filename):
        """
        Load the bootstrap compute arguments from a json file. If a list of bootstrap
        compute arguments samples exists prior to this function call, then the samples
        read from this file will be appended to the old list.

        Parameters
        ----------
        filename: str or Path
            Name or path of json file to read.

        Returns
        -------
        new_bootstrap_compute_arguments_identifiers: dict
            Dictionary read from the json file.
        """
        # Load the json file
        with open(filename, "r") as f:
            new_bootstrap_compute_arguments_identifiers = json.load(f)

        # The information stored in the json file are the identifiers. We need to
        # convert it back to compute arguments.
        keys = [int(key) for key in new_bootstrap_compute_arguments_identifiers.keys()]
        new_bootstrap_compute_arguments = {}
        # Iterate over sample
        for ii in keys:
            # List of identifier for step ii
            identifiers_ii = new_bootstrap_compute_arguments_identifiers[str(ii)]
            # Iterate over the calculator
            cas_ii = []
            for jj, identifiers_calc in enumerate(identifiers_ii):
                reference = self._orig_compute_arguments_identifiers[jj]
                cas_calc = [
                    self.orig_compute_arguments[jj][reference.index(ss)]
                    for ss in identifiers_calc
                ]
                cas_ii.append(cas_calc)
            # Update the new bootstrap compute arguments dictionary
            new_bootstrap_compute_arguments.update({ii: cas_ii})

        # Update the old list with this new list
        self.bootstrap_compute_arguments = self._update_bootstrap_compute_arguments(
            new_bootstrap_compute_arguments
        )
        return new_bootstrap_compute_arguments_identifiers

    def run(self, initial_guess=None, residual_fn_list=None, min_kwargs={}):
        """
        Iterate over the generated bootstrap compute arguments samples and train the
        potential using each compute arguments sample.

        Parameters
        ----------
        initial_guess: np.ndarray (optional)
            Initial guess of parameters to use for the minimization. It is recommended to
            use the same values as used in the training process if such step is done prior
            to running bootstrap.
        residual_fn_list: list (optional)
            List of residual function to use in each calculator. Currently, this only
            affect the case when multiple calculators are used. If there is only a single
            calculator, don't worry about this argument.
        min_kwargs: dict (optional)
            Keyword arguments for: meth: `~kliff.loss.Loss.minimize`.
        """
        if self._nsamples_prepared == 0:
            # Bootstrap compute arguments have not been generated
            raise BootstrapError("Please generate a bootstrap compute arguments first")

        # Train the model using each bootstrap compute arguments
        for ii in range(self._nsamples_done, self._nsamples_prepared):
            # Update the compute arguments
            if self.use_multi_calc:
                # There are multiple calculators used
                for jj, calc in enumerate(self.calculator.calculators):
                    calc.compute_arguments = self.bootstrap_compute_arguments[ii][jj]
            else:
                self.calculator.compute_arguments = self.bootstrap_compute_arguments[ii][
                    0
                ]

            # Set the initial parameter guess
            if initial_guess is None:
                initial_guess = self.calculator.get_opt_params().copy()
            self.calculator.update_model_params(initial_guess)
            # TODO This assumes that we use the built-in residual functions
            if self.use_multi_calc:
                if residual_fn_list is None:
                    # If multiple calculators are used, we need to update the residual
                    # function used for each configuration. This is to ensure that we use
                    # the correct residual function for each configuration.
                    calc_list = self.calculator.get_calculator_list()
                    residual_fn_list = []
                    for calculator in calc_list:
                        if calculator.use_energy and calculator.use_forces:
                            residual_fn = energy_forces_residual
                        elif calculator.use_energy:
                            residual_fn = energy_residual
                        elif calculator.use_forces:
                            residual_fn = forces_residual
                        else:
                            raise RuntimeError(
                                "Calculator does not use energy or forces."
                            )
                        residual_fn_list.append(residual_fn)
                self.loss.residual_fn = residual_fn_list
            # Minimization
            self.loss.minimize(**min_kwargs)

            # Append the parameters to the samples
            self.samples = np.row_stack(
                (self.samples, self.loss.calculator.get_opt_params())
            )

        # Finishing up
        self.restore_loss()  # Restore the loss function
        return self.samples

    def restore_loss(self):
        """
        Restore the loss function: revert back the compute arguments and the parameters
        to the original state.
        """
        # Restore the parameters and configurations back
        self.calculator.compute_arguments = self.orig_compute_arguments
        self.calculator.update_model_params(self.orig_params)

    def reset(self):
        """
        Reset the bootstrap sampler.
        """
        self.restore_loss()
        self.bootstrap_compute_arguments = {}
        self.samples = np.empty((0, self.calculator.get_num_opt_params()))


def bootstrap_cas_generator_neuralnetwork(nsamples, orig_fingerprints):
    """
    Default class to generate bootstrap compute arguments(fingerprints) for neural
    network model. Currently, we only have a simple case, where we assume that there is
    only a single list of compute arguments and we sample with replacement from the list.

    TODO:
    - [] Work on default bootstrap cas generator function when we have multiple elements,
      i.e., multiple NN models, each for separate element.
    - [] Alternatively, test this implementation for such case.

    Parameters
    ----------
    nsamples: int
        Number of the bootstrap compute arguments requested.

    orig_fingerprints: list
        The original list of compute arguments(fingerprints). The bootstrap compute
        arguments will be generated from this list. The format of this input is given
        below: :

            orig_fingerprints = [ca0, ca1, ...]

    Returns
    -------
    dict
        A set of bootstrap compute arguments(fingerprints), written in a dictionary
       format, where the keys index the bootstrap samples compute arguments::

            bootstrap_configs = {
                0: [ca0, ca1, ...],
                1: [ca0, ca1, ...],
            }

    """
    bootstrap_fingerprints = {}
    nfingerprints = len(orig_fingerprints)
    for ii in range(nsamples):
        # Get 1 sample of bootstrap fingerprints
        bootstrap_fingerprints_single_sample = np.random.choice(
            orig_fingerprints, size=nfingerprints, replace=True
        )
        bootstrap_fingerprints.update({ii: bootstrap_fingerprints_single_sample})
    return bootstrap_fingerprints


def get_identifiers_from_fingerprints(fingerprints):
    """
    Retrieve the identifiers of a list of fingerprints.

    Parameters
    ----------
    fingerprints: list
        A list of fingerprints.

    Returns
    -------
    identifiers: list
        A list of fingerprints' identifiers, which shows the paths to the xyz files.
    """
    identifiers = [fp["configuration"].identifier for fp in fingerprints]
    return identifiers


class BootstrapNeuralNetworkModel:
    """
    Bootstrap sampler class for neural network potentials.

    Parameters
    ----------
    loss: Loss
        Loss function class from :class:`~kliff.loss.Loss`.
    orig_state_filename: str or Path
        Name of the file in which the initial state of the model prior to bootstrapping
        will be stored. This is to use at the end of the bootstrap run to reset the model
        to the initial state.
    """

    def __init__(self, loss, orig_state_filename="orig_model.pkl"):
        self.loss = loss
        self.calculator = self.loss.calculator
        if isinstance(self.calculator, CalculatorTorchSeparateSpecies):
            self._calc_separate_species = True
            self.model = [model[1] for model in self.calculator.models.items()]
            self._species = self.calculator.models
        else:
            self._calc_separate_species = False
            self.model = [self.calculator.model]
        # Cache the original parameter values
        self.orig_params = copy.copy(self.calculator.get_opt_params())
        # Cache the original fingerprints
        self.orig_compute_arguments = self.calculator.get_fingerprints_dataset()
        self._orig_compute_arguments_identifiers = get_identifiers_from_fingerprints(
            self.orig_compute_arguments
        )
        # Initiate the bootstrap configurations property
        self.bootstrap_compute_arguments = {}
        self.samples = np.empty((0, self.calculator.get_num_opt_params()))

        # Save the original state of the model before running bootstrap
        if self._calc_separate_species:
            self.orig_state_filename = []
            for sp, model in zip(self._species, self.model):
                splitted_path = os.path.splitext(orig_state_filename)
                path_with_species = splitted_path[0] + f"_{sp}" + splitted_path[1]
                self.orig_state_filename.append(path_with_species)
                model.save(path_with_species)
        else:
            self.orig_state_filename = [orig_state_filename]
            self.model[0].save(orig_state_filename)

    @property
    def _nsamples_prepared(self):
        """
        Get how many bootstrap compute arguments are prepared. This also include those
        compute arguments set that have been evaluated.
        """
        return len(self.bootstrap_compute_arguments)

    @property
    def _nsamples_done(self):
        """
        For each bootstrap compute arguments sample, we need to train the potential using
        these compute arguments. This function retrieve how many bootstrap compute
        arguments samples have been used in the training. This is to help so that we can
        continue the calculation midway.
        """
        return len(self.samples)

    def generate_bootstrap_compute_arguments(
        self, nsamples, bootstrap_cas_generator_fn=None, **kwargs
    ):
        """
        Generate bootstrap compute arguments (fingerprints) samples. If this function is
        called multiple, say, K times, then it will in total generate
        :math:`K \times nsamples` bootstrap compute arguments samples. That is,
        consecutive call of this function will append the generated compute arguments
        samples.

        Parameters
        ----------
        nsamples: int
            Number of bootstrap samples to generate.
        bootstrap_cas_generator_fn: callable ``fn(nsamples, **kwargs)`` (optional)
            A function to generate bootstrap compute argument samples. The default
            function combine the compute arguments across all calculators and do sampling
            with replacement from the combined list. Another possible convention is to
            do sampling with replacement on the compute arguments list of each calculator
            separately, in which case a custom function needs to be defined and used.
        kwargs: dict
            Additional keyword arguments to ``bootstrap_cas_generator_fn``.
        """
        # Function to generate bootstrap configurations
        if bootstrap_cas_generator_fn is None:
            bootstrap_cas_generator_fn = bootstrap_cas_generator_neuralnetwork
            kwargs = {"orig_fingerprints": self.orig_compute_arguments}

        # Generate a new bootstrap configurations
        new_bootstrap_compute_arguments = bootstrap_cas_generator_fn(nsamples, **kwargs)
        # Update the old list with this new list
        self.bootstrap_compute_arguments = self._update_bootstrap_compute_arguments(
            new_bootstrap_compute_arguments
        )

    def _update_bootstrap_compute_arguments(self, new_bootstrap_compute_arguments):
        """
        Append the new generated bootstrap compute arguments samples to the old list.
        """
        bootstrap_compute_arguments = copy.copy(self.bootstrap_compute_arguments)
        for ii, vals in new_bootstrap_compute_arguments.items():
            iteration = ii + self._nsamples_prepared
            bootstrap_compute_arguments.update({iteration: vals})

        return bootstrap_compute_arguments

    def save_bootstrap_compute_arguments(self, filename):
        """
        Export the generated bootstrap compute arguments as a json file. The json file
        will contain the identifier of the compute arguments for each sample.

        Parameters
        ----------
        filename: str or Path
            Where to export the bootstrap compute arguments samples
        """
        # We cannot directly export the bootstrap fingerprints. Instead, we will
        # first convert it and list the identifiers and export the identifiers.
        # Convert to identifiers
        bootstrap_compute_arguments_identifiers = {}
        for ii in self.bootstrap_compute_arguments:
            bootstrap_compute_arguments_identifiers.update(
                {
                    ii: get_identifiers_from_fingerprints(
                        self.bootstrap_compute_arguments[ii]
                    )
                }
            )

        with open(filename, "w") as f:
            json.dump(bootstrap_compute_arguments_identifiers, f, indent=4)

    def load_bootstrap_compute_arguments(self, filename):
        """
        Load the bootstrap compute arguments from a json file. If a list of bootstrap
        compute arguments samples exists prior to this function call, then the samples
        read from this file will be appended to the old list.

        Parameters
        ----------
        filename: str or Path
            Name or path of json file to read.

        Returns
        -------
        new_bootstrap_compute_arguments_identifiers: dict
            Dictionary read from the json file.
        """
        # Load the json file
        with open(filename, "r") as f:
            new_bootstrap_compute_arguments_identifiers = json.load(f)

        # The information stored in the json file are the identifiers. We need to
        # convert it back to fingerprints.
        keys = [int(key) for key in new_bootstrap_compute_arguments_identifiers.keys()]
        new_bootstrap_compute_arguments = {}
        # Iterate over sample
        for ii in keys:
            # List of identifier for step ii
            identifiers_ii = new_bootstrap_compute_arguments_identifiers[str(ii)]
            reference = self._orig_compute_arguments_identifiers
            fp_ii = [
                self.orig_compute_arguments[reference.index(ss)] for ss in identifiers_ii
            ]
            # Update the new bootstrap fingerprints dictionary
            new_bootstrap_compute_arguments.update({ii: fp_ii})

        # Update the old list with this new list
        self.bootstrap_compute_arguments = self._update_bootstrap_compute_arguments(
            new_bootstrap_compute_arguments
        )
        return new_bootstrap_compute_arguments_identifiers

    def run(self, min_kwargs={}):
        """
        Iterate over the generated bootstrap compute arguments samples and train the
        potential using each compute arguments sample.

        Parameters
        ----------
        min_kwargs: dict
            Keyword arguments for :meth:`~kliff.loss.Loss.minimize`.
        """
        if self._nsamples_prepared == 0:
            # Bootstrap fingerprints have not been generated
            raise BootstrapError("Please generate a bootstrap compute_arguments first")

        # Train the model using each bootstrap fingerprints
        for ii in range(self._nsamples_done, self._nsamples_prepared):
            # Update the fingerprints
            self.calculator.set_compute_arguments(self.bootstrap_compute_arguments[ii])

            for model in self.model:
                # Reset the initial parameters
                for layer in model.layers:
                    try:
                        layer.reset_parameters()
                    except AttributeError:
                        pass
            # Minimization
            self.loss.minimize(**min_kwargs)

            # Append the parameters to the samples
            self.samples = np.row_stack(
                (self.samples, self.loss.calculator.get_opt_params())
            )

        # Finishing up, restore the state
        self.restore_loss()
        return self.samples

    def restore_loss(self):
        """
        Restore the loss function: revert back the compute arguments and the parameters
        to the original state.
        """
        # Restore the parameters and configurations back by loading the original state
        # back.
        for model, fname in zip(self.model, self.orig_state_filename):
            model.load(fname)

    def reset(self):
        """
        Reset the bootstrap sampler.
        """
        self.restore_loss()
        self.bootstrap_compute_arguments = {}
        self.samples = np.empty((0, self.calculator.get_num_opt_params()))


class BootstrapError(Exception):
    def __init__(self, msg):
        super(BootstrapError, self).__init__(msg)
        self.msg = msg
