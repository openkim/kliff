"""
.. _tut_linear_regression:

Train a linear regression potential
===================================

In this tutorial, we train a linear regression model on the descriptors obtained using the
symmetry functions.

"""

from kliff.descriptors import SymmetryFunction
from kliff.dataset import Dataset
from kliff.models.model_torch import LinearRegression, CalculatorTorch


descriptor = SymmetryFunction(
    cut_name='cos', cut_dists={'Si-Si': 5.0}, hyperparams='set30', normalize=True
)


model = LinearRegression(descriptor)

# training set
dataset_name = 'Si_training_set/varying_alat'
tset = Dataset()
tset.read(dataset_name)
configs = tset.get_configs()
print('Number of configurations:', len(configs))

# calculator
calc = CalculatorTorch(model)
calc.create(configs, reuse=True)

# fit the model
calc.fit()
