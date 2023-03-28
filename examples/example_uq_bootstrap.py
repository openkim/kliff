"""
.. _tut_bootstrap:

Bootstrapping
=============

In this example, we demonstrate how to perform uncertainty quantification (UQ) using
bootstrap method. We use a Stillinger-Weber (SW) potential for silicon that is archived
in OpenKIM_.

For simplicity, we only set the energy-scaling parameters, i.e., ``A`` and ``lambda`` as
the tunable parameters. These parameters will be calibrated to energies and forces of a
small dataset, consisting of 4 compressed and stretched configurations of diamond silicon
structure.
"""


##########################################################################################
# To start, let's first install the SW model::
#
#    $ kim-api-collections-management install user SW_StillingerWeber_1985_Si__MO_405512056662_006
#
# .. seealso::
#    This installs the model and its driver into the ``User Collection``. See
#    :ref:`install_model` for more information about installing KIM models.


import matplotlib.pyplot as plt
import numpy as np

from kliff.calculators import Calculator
from kliff.dataset import Dataset
from kliff.loss import Loss
from kliff.models import KIMModel
from kliff.uq.bootstrap import BootstrapEmpiricalModel
from kliff.utils import download_dataset

##########################################################################################
# Before running bootstrap, we need to define a loss function and train the model. More
# detail information about this step can be found in :ref:`tut_kim_sw` and
# :ref:`tut_params_transform`.

# Create the model
model = KIMModel(model_name="SW_StillingerWeber_1985_Si__MO_405512056662_006")

# Set the tunable parameters and the initial guess
opt_params = {"A": [["default"]], "lambda": [["default"]]}

model.set_opt_params(**opt_params)
model.echo_opt_params()

# Get the dataset
dataset_path = download_dataset(dataset_name="Si_training_set_4_configs")
# Read the dataset
tset = Dataset(dataset_path)
configs = tset.get_configs()

# Create calculator
calc = Calculator(model)
# Only use the forces data
ca = calc.create(configs, use_energy=False, use_forces=True)

# Instantiate the loss function
residual_data = {"normalize_by_natoms": False}
loss = Loss(calc, residual_data=residual_data)

# Train the model
min_kwargs = dict(method="lm")  # Optimizer setting
loss.minimize(**min_kwargs)
model.echo_opt_params()

##########################################################################################
# To perform UQ by bootstrapping, the general workflow starts by instantiating
# :class:`~kliff.uq.bootstrap.BootstrapEmpiricalModel`, or
# :class:`~kliff.uq.bootstrap.BootstrapNeuralNetworkModel` if using a neural network
# potential.


# It is a good practice to specify the random seed to use in the calculation to generate
# a reproducible simulation.
np.random.seed(1717)

# Instantiate bootstrap class object
BS = BootstrapEmpiricalModel(loss)

##########################################################################################
# Then, we generate some bootstrap compute arguments. This is equivalent to generating
# bootstrap data. Typically, we just need to specify how many bootstrap data samples to
# generate. Additionally, if we call ``generate_bootstrap_compute_arguments`` multiple
# times, the new generated data samples will be appended to the previously generated data
# samples. This is also the behavior if we read the data samples from the previously
# exported file.


# Generate bootstrap compute arguments
BS.generate_bootstrap_compute_arguments(100)

##########################################################################################
# Finally, we will iterate over these bootstrap data samples and train the potential
# using each data sample. The resulting optimal parameters from each data sample give a
# single sample of parameters. By iterating over all data samples, then we will get an
# ensemble of parameters.
#
# We also recommend in using the same optimizer setting as the one used in the model
# training step. This also means to use the same set of initial parameter guess when the
# loss potentially has multiple local minima. For neural network model, we need to reset
# the initial parameter value, which is done internally.


# Run bootstrap
BS.run(min_kwargs=min_kwargs)

##########################################################################################
# The resulting parameter ensemble can be accessed in `BS.samples` as a `np.ndarray`.
# Then, we can plot the distribution of the parameters, as an example, or propagate the
# error to the target quantities we want to study.


# Plot the distribution of the parameters
plt.figure()
plt.plot(*(BS.samples.T), ".", alpha=0.5)
param_names = list(opt_params.keys())
plt.xlabel(param_names[0])
plt.ylabel(param_names[1])
plt.show()

##########################################################################################
# .. _OpenKIM: https://openkim.org
