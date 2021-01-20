"""
.. _tut_lj:

Train a Lennard-Jones potential
===============================

In this tutorial, we train a Lennard-Jones potential that is build in KLIFF (i.e. not
models archived on OpenKIM_). From a user's perspective, a KLIFF built-in model is not
different from a KIM model.

Compare this with :ref:`tut_kim_sw`.

.. _OpenKIM: https://openkim.org
"""
from kliff.calculators import Calculator
from kliff.dataset import Dataset
from kliff.loss import Loss
from kliff.models import LennardJones
from kliff.utils import download_dataset

# training set
dataset_path = download_dataset(dataset_name="Si_training_set_4_configs")
tset = Dataset(dataset_path)
configs = tset.get_configs()

# calculator
model = LennardJones()
model.echo_model_params()

# fitting parameters
model.set_opt_params(sigma=[["default"]], epsilon=[["default"]])
model.echo_opt_params()

calc = Calculator(model)
calc.create(configs)

# loss
loss = Loss(calc, nprocs=1)
result = loss.minimize(method="L-BFGS-B", options={"disp": True, "maxiter": 10})


# print optimized parameters
model.echo_opt_params()
model.save("kliff_model.yaml")
