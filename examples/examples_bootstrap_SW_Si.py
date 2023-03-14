"""
.. _tut_mcmc:

MCMC sampling
=============

In this example, we demonstrate how to perform uncertainty quantification (UQ) using
parallel tempered MCMC (PTMCMC). We use a Stillinger-Weber (SW) potential for silicon
that is archived in OpenKIM_.

For simplicity, we only set the energy-scaling parameters, i.e., ``A`` and ``lambda`` as
the tunable parameters. Furthermore, these parameters are physically constrained to be
positive, thus we will work in log parameterization, i.e. ``log(A)`` and ``log(lambda)``.
These parameters will be calibrated to energies and forces of a small dataset,
consisting of 4 compressed and stretched configurations of diamond silicon structure.
"""


##########################################################################################
# To start, let's first install the SW model::
#
#    $ kim-api-collections-management install user SW_StillingerWeber_1985_Si__MO_405512056662_006
#
# .. seealso::
#    This installs the model and its driver into the ``User Collection``. See
#    :ref:`install_model` for more information about installing KIM models.


import numpy as np
import matplotlib.pyplot as plt

from kliff.calculators import Calculator
from kliff.dataset import Dataset
from kliff.dataset.weight import MagnitudeInverseWeight
from kliff.loss import Loss
from kliff.models import KIMModel
from kliff.models.parameter_transform import LogParameterTransform
from kliff.uq.bootstrap import BootstrapEmpiricalModel
from kliff.utils import download_dataset

##########################################################################################
# Before running MCMC, we need to define a loss function and train the model. More detail
# information about this step can be found in :ref:`tut_kim_sw` and
# :ref:`tut_params_transform`.

# Instantiate a transformation class to do the log parameter transform
param_names = ["A", "lambda"]
params_transform = LogParameterTransform(param_names)

# Create the model
model = KIMModel(
    model_name="SW_StillingerWeber_1985_Si__MO_405512056662_006",
    params_transform=params_transform,
)

# Set the tunable parameters and the initial guess
opt_params = {"A": [["default"]], "lambda": [["default"]]}

model.set_opt_params(**opt_params)
model.echo_opt_params()

# Get the dataset and set the weights
dataset_path = download_dataset(dataset_name="Si_training_set_4_configs")
# Instantiate the weight class
weight = MagnitudeInverseWeight(
    weight_params={
        "energy_weight_params": [0.0, 0.1],
        "forces_weight_params": [0.0, 0.1],
    }
)
# Read the dataset and compute the weight
tset = Dataset(dataset_path, weight=weight)
configs = tset.get_configs()

# Create calculator
calc = Calculator(model)
ca = calc.create(configs)

# Instantiate the loss function
residual_data = {"normalize_by_natoms": False}
loss = Loss(calc, residual_data=residual_data)

# Train the model
min_kwargs = dict(method="lm")
loss.minimize(**min_kwargs)
model.echo_opt_params()


##########################################################################################
BS = BootstrapEmpiricalModel(loss)
BS.generate_bootstrap_compute_arguments(100)
BS.run(min_kwargs=min_kwargs)

plt.figure()
plt.plot(*(BS.samples.T), ".", alpha=0.5)
plt.xlabel(param_names[0])
plt.ylabel(param_names[1])
