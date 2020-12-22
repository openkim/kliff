"""
.. _tut_linear_regression:

Train a linear regression potential
===================================

In this tutorial, we train a linear regression model on the descriptors obtained using the
symmetry functions.
"""

from kliff.calculators import CalculatorTorch
from kliff.dataset import Dataset
from kliff.descriptors import SymmetryFunction
from kliff.models import LinearRegression

descriptor = SymmetryFunction(
    cut_name="cos", cut_dists={"Si-Si": 5.0}, hyperparams="set30", normalize=True
)


model = LinearRegression(descriptor)

# training set
dataset_name = "Si_training_set/varying_alat"
tset = Dataset()
tset.read(dataset_name)
configs = tset.get_configs()
print("Number of configurations:", len(configs))

# calculator
calc = CalculatorTorch(model)
calc.create(configs, reuse=False)


##########################################################################################
# We can train a linear regression model by minimizing a loss function as discussed in
# :ref:`tut_nn`. But linear regression model has analytic solutions, and thus we can train
# the model directly by using this feature. This can be achieved by calling the ``fit()``
# function of its calculator.
#

# fit the model
calc.fit()


# save model
model.save("linear_model.pkl")
