"""I am still working on this script to do bootstrap UQ."""

from pathlib import Path
import copy
import json

import numpy as np

from kliff.calculators.calculator import Calculator, _WrapperCalculator
from kliff.loss import LossPhysicsMotivatedModel


def default_bootstrap_generator(nsamples, orig_compute_arguments):
    """Let's try to make the format of the bootstrap configurations as a dictionary:
    bootstrap_configs = {
        0: [[compute_arguments_calc0], [compute_arguments_calc1]],
        1: [[compute_arguments_calc0], [compute_arguments_calc1]]
    }
    """
    ncalcs = len(orig_compute_arguments)
    ncas = [len(cas) for cas in orig_compute_arguments]
    bootstrap_compute_arguments = {}
    for ii in range(nsamples):
        # Get 1 sample of bootstrap compute arguments
        bootstrap_cas_single_sample = []
        for jj, cas in enumerate(orig_compute_arguments):
            # Get sets of bootstrap configurations for each calculator
            bootstrap_cas_single_calc = np.random.choice(
                cas, size=ncas[jj], replace=True
            )
            bootstrap_cas_single_sample.append(bootstrap_cas_single_calc)
        # Update the bootstrap compute arguments dictionary
        bootstrap_compute_arguments.update({ii: bootstrap_cas_single_sample})
    return bootstrap_compute_arguments


class Bootstrap:
    def __init__(self, loss):
        self.loss = loss
        self.calc = loss.calculator
        # Cache the original parameter values
        self.orig_params = copy.copy(self.calc.get_opt_params())
        # Cache the original compute arguments
        if isinstance(self.calc, Calculator):
            self.orig_compute_arguments = [copy.copy(self.calc.get_compute_arguments())]
            self.multi_calc = False
        elif isinstance(self.calc, _WrapperCalculator):
            self.orig_compute_arguments = copy.copy(
                self.calc.get_compute_arguments(flat=False)
            )
            self.multi_calc = True
        self._orig_compute_arguments_identifiers = (
            self.convert_compute_arguments_to_identifiers(self.orig_compute_arguments)
        )
        # Initiate the bootstrap configurations property
        self.bootstrap_compute_arguments = {}
        self.samples = np.empty((0, self.calc.get_num_opt_params()))
        self._nsamples_prepared = 0

    @property
    def _nsamples_done(self):
        return len(self.samples)

    def generate_bootstrap_compute_arguments(
        self, nsamples, bootstrap_generator_fn=None, **kwargs
    ):
        # Function to generate bootstrap configurations
        if bootstrap_generator_fn is None:
            bootstrap_generator_fn = default_bootstrap_generator
            kwargs = {"orig_compute_arguments": self.orig_compute_arguments}

        # Generate a new bootstrap configurations
        new_bootstrap_compute_arguments = bootstrap_generator_fn(nsamples, **kwargs)
        # Update the old list with this new list
        self.bootstrap_compute_arguments = self._update_bootstrap_compute_arguments(
            new_bootstrap_compute_arguments
        )
        self._nsamples_prepared += nsamples

    def _update_bootstrap_compute_arguments(self, new_bootstrap_compute_arguments):
        bootstrap_compute_arguments = copy.copy(self.bootstrap_compute_arguments)
        for ii, vals in new_bootstrap_compute_arguments.items():
            iteration = ii + self._nsamples_prepared
            bootstrap_compute_arguments.update({iteration: vals})

        return bootstrap_compute_arguments

    def load_bootstrap_compute_arguments(self, filename):
        """Load the bootstrap compute arguments from a json file."""
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
        self._nsamples_prepared += len(new_bootstrap_compute_arguments)
        return new_bootstrap_compute_arguments_identifiers

    def save_bootstrap_compute_arguments(self, filename):
        """Export the generated bootstrap compute arguments as a json file."""
        # We cannot directly export the bootstrap compute arguments. Instead, we will
        # first convert it and list the identifiers and export the identifiers.
        # Convert to identifiers
        bootstrap_compute_arguments_identifiers = {}
        for ii in self.bootstrap_compute_arguments:
            bootstrap_compute_arguments_identifiers.update(
                {
                    ii: self.convert_compute_arguments_to_identifiers(
                        self.bootstrap_compute_arguments[ii]
                    )
                }
            )

        with open(filename, "w") as f:
            json.dump(bootstrap_compute_arguments_identifiers, f, indent=4)

    def run(self, min_kwargs={}):
        if not isinstance(self.loss, LossPhysicsMotivatedModel):
            raise BootstrapError(
                "Currently only physics-motivated loss function is supported"
            )

        if self._nsamples_prepared == 0:
            # Bootstrap compute arguments have not been generated
            raise BootstrapError("Please generate a bootstrap compute arguments first")

        # Train the model using each bootstrap compute arguments
        for ii in range(self._nsamples_done, self._nsamples_prepared):
            # Update the compute arguments
            if self.multi_calc:
                # There are multiple calculators used
                for jj, calc in enumerate(self.calc.calculators):
                    calc.compute_arguments = self.bootstrap_compute_arguments[ii][jj]
            else:
                self.calc.compute_arguments = self.bootstrap_compute_arguments[ii][0]

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
        # Restore the parameters and configurations back
        self.calc.compute_arguments = self.orig_compute_arguments
        self.calc.update_model_params(self.orig_params)

    def reset(self):
        self.restore_loss()
        self.bootstrap_compute_arguments = {}
        self._nsamples_done = 0
        self._nsamples_prepared = 0

    @staticmethod
    def convert_compute_arguments_to_identifiers(compute_arguments):
        identifiers = []
        for cas in compute_arguments:
            # Iterate over compute arguments corresponding to each calculator
            identifiers.append([ca.conf.identifier for ca in cas])
        return identifiers


class BootstrapError(Exception):
    def __init__(self, msg):
        super(BootstrapError, self).__init__(msg)
        self.msg = msg
