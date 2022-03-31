"""
.. _tut_params_transform:
Parameter transformation for the Stillinger-Weber potential
===========================================================

Parameters in the empirical interatomic potential are often restricted by some physical
constraints. As an example, in the Stillinger-Weber (SW) potential, the energy scaling
parameters (e.g., ``A`` and ``B``) and the length scaling parameters (e.g., ``sigma`` and
``gamma``) are constrained to be positive. Due to these constraints, we might want to
work with the log of the parameters, i.e., ``log(A)``, ``log(B)``, ``log(sigma)``, and
``log(gamma)``.

In this tutorial, we show how to apply parameter transformation to the SW potential for
silicon that is archived on OpenKIM_.

Compare this with :ref:`tut_kim_sw`.


.. _OpenKIM: https://openkim.org
"""


##########################################################################################
# To start, let's first install the SW model::
#
#    $ kim-api-collections-management install user SW_StillingerWeber_1985_Si__MO_405512056662_006
#
# .. seealso::
#    This installs the model and its driver into the ``User Collection``. See
#    :ref:`install_model` for more information about installing KIM models.
#

import numpy as np
from kliff.models import KIMModel
from kliff.models.parameter_transform import LogParameterTransform
from kliff.dataset import Dataset
from kliff.utils import download_dataset
from kliff.calculators import Calculator
from kliff.loss import Loss

##########################################################################################
# Before creating a KIM model for the SW potential, we first instantiate the parameter
# transformation class that we want to use. ``kliff`` has a built-in log-transformation,
# however extending it to other parameter transformation can be done by creating a
# subclass from :class:`~kliff.models.parameter_transform.ParameterTransform` class.
#
# To make a direct comparison to :ref:`tut_kim_sw`, in this tutorial we will apply
# log-transformation to parameters ``A``, ``B``, ``sigma``, and ``gamma``, which
# correspond to energy and length scales. We then create a KIM model for this potential
# and print out the ``model parameters``.

logtransform = LogParameterTransform(param_names=["A", "B", "sigma", "gamma"])
model = KIMModel(
    model_name="SW_StillingerWeber_1985_Si__MO_405512056662_006",
    params_transform=logtransform,
)
model.echo_model_params(params_space="original")

# As a default, the method above will print out parameter values in the original,
# untransformed space, i.e., using the original parameterization of the model. If we
# supply the argument ``params_space="transformed"``, then the printed parameter values
# are given in the transformed space, e.g., log space.

##########################################################################################
# Next, we will set up parameters to optimize. Typically, these are set to the same
# parameters we specify when creating the parameter transformation instance. Notice that
# the parameter values we initialize, as well as the lower and upper bounds, are in
# log space.

model.set_opt_params(
    A=[[np.log(5.0), np.log(1.0), np.log(20)]],
    B=[["default"]],
    sigma=[[np.log(2.0951), "fix"]],
    gamma=[[np.log(1.5)]],
)
model.echo_opt_params()

# .. note::
#    ``model.echo_opt_params()`` always displays the parameter values in the transformed
#    space.

##########################################################################################
# Once we set the model and the parameter transformation scheme, then further
# calculations, e.g., training the model, will be performed using the transformed space
# and can be done in the same way as in :ref:`tut_kim_sw`.

# Training set
dataset_path = download_dataset(dataset_name="Si_training_set")
tset = Dataset(dataset_path)
configs = tset.get_configs()

# Calculator
calc = Calculator(model)
_ = calc.create(configs)

# Loss function and model training
steps = 100
residual_data = {"energy_weight": 1.0, "forces_weight": 0.1}
loss = Loss(calc, residual_data=residual_data, nprocs=2)
loss.minimize(method="L-BFGS-B", options={"disp": True, "maxiter": steps})
model.echo_opt_params()

# The optimized parameter values from this model training are the same (very close) as in
# :ref:`tut_kim_sw`, as expected (note that the values displayed above are in log space).
