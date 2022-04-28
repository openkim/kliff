"""
.. _tut_params_transform:

Parameter transformation for the Stillinger-Weber potential
===========================================================

Parameters in the empirical interatomic potential are often restricted by some physical
constraints. As an example, in the Stillinger-Weber (SW) potential, the energy scaling
parameters (e.g., ``A`` and ``B``) and the length scaling parameters (e.g., ``sigma`` and
``gamma``) are constrained to be positive.

Due to these constraints, we might want to work with the log of the parameters, i.e.,
``log(A)``, ``log(B)``, ``log(sigma)``, and ``log(gamma)`` when doing the optimization.
After the optimization, we can transform them back to the original parameter space using
an exponential function, which will guarantee the positiveness of the parameters.

In this tutorial, we show how to apply parameter transformation to the SW potential for
silicon that is archived on OpenKIM_. Compare this with :ref:`tut_kim_sw`.
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
# This is

import numpy as np

from kliff.calculators import Calculator
from kliff.dataset import Dataset
from kliff.dataset.weight import Weight
from kliff.loss import Loss
from kliff.models import KIMModel
from kliff.models.parameter_transform import LogParameterTransform
from kliff.utils import download_dataset

##########################################################################################
# Before creating a KIM model for the SW potential, we first instantiate the parameter
# transformation class that we want to use. ``kliff`` has a built-in log-transformation;
# however, extending it to other parameter transformation can be done by creating a
# subclass of :class:`~kliff.models.parameter_transform.ParameterTransform`.
#
# To make a direct comparison to :ref:`tut_kim_sw`, in this tutorial we will apply
# log-transformation to parameters ``A``, ``B``, ``sigma``, and ``gamma``, which
# correspond to energy and length scales.
#

transform = LogParameterTransform(param_names=["A", "B", "sigma", "gamma"])
model = KIMModel(
    model_name="SW_StillingerWeber_1985_Si__MO_405512056662_006",
    params_transform=transform,
)
model.echo_model_params(params_space="original")


##########################################################################################
# ``model.echo_model_params(params_space="original")`` above will print out parameter
# values in the original, untransformed space, i.e., the original parameterization of
# the model. If we supply the argument ``params_space="transformed"``, then the printed
# parameter values are given in the transformed space, e.g., log space (below). The
# values of the other parameters are not changed.
#

model.echo_model_params(params_space="original")


##########################################################################################
# Compare the output of ``params_space="transformed"`` and ``params_space="original"``,
# you can see that the values of ``A``, ``B``, ``sigma``, and ``gamma`` are in the log
# space after the transformation.

##########################################################################################
# Next, we will set up the initial guess of the parameters to optimize. A value of
# ``"default"`` means the initial guess will be directly taken from the value already in
# the model.
#
# .. note::
#    The parameter values we initialize, as well as the lower and upper bounds, are in
#    transformed space (i.e. log space here).


model.set_opt_params(
    A=[[np.log(5.0), np.log(1.0), np.log(20)]],
    B=[["default"]],
    sigma=[[np.log(2.0951), "fix"]],
    gamma=[[np.log(1.5)]],
)
model.echo_opt_params()

##########################################################################################
# We can show the parameters we’ve just set by ``model.echo_opt_params()``.
#
# .. note::
#    ``model.echo_opt_params()`` always displays the parameter values in the transformed
#    space. And it only shows all the parameters specified to optimize. To show all
#    the parameters, do ``model.echo_model_params(params_space="transformed")``.

##########################################################################################
# Once we set the model and the parameter transformation scheme, then further
# calculations, e.g., training the model, will be performed using the transformed space
# and can be done in the same way as in :ref:`tut_kim_sw`.

# Training set
dataset_path = download_dataset(dataset_name="Si_training_set")
weight = Weight(energy_weight=1.0, forces_weight=0.1)
tset = Dataset(dataset_path, weight)
configs = tset.get_configs()

# Calculator
calc = Calculator(model)
_ = calc.create(configs)

# Loss function and model training
steps = 100
loss = Loss(calc, nprocs=2)
loss.minimize(method="L-BFGS-B", options={"disp": True, "maxiter": steps})

model.echo_model_params(params_space="original")


##########################################################################################
# The optimized parameter values from this model training are very close, if not the
# same, as in :ref:`tut_kim_sw`. This is expected for the simple tutorial example
# considered. But for more complex models, training in a transformed space can make it
# much easier for the optimizer to navigate the parameter space.
#
# .. _OpenKIM: https://openkim.org
#
