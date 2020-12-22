"""
Continue the training of example_nn_Si.py.

Most of the settings are the same as ``example_nn_Si.py``. We only need to load the
parameters of the model and the state dictionary of the parameter.
"""

from kliff import nn
from kliff.calculators import CalculatorTorch
from kliff.dataset import Dataset
from kliff.descriptors import SymmetryFunction
from kliff.loss import Loss
from kliff.models import NeuralNetwork

descriptor = SymmetryFunction(
    cut_name="cos", cut_dists={"Si-Si": 5.0}, hyperparams="set30", normalize=True
)

N1 = 10
N2 = 10
model = NeuralNetwork(descriptor)
model.add_layers(
    # first hidden layer
    nn.Linear(descriptor.get_size(), N1),
    nn.Tanh(),
    # second hidden layer
    nn.Linear(N1, N2),
    nn.Tanh(),
    # output layer
    nn.Linear(N2, 1),
)
model.set_save_metadata(prefix="./kliff_saved_model", start=5, frequency=2)

##########################################################################################
# Load the parameters from the saved model.
# If we are load a model to continue the training (the case here), ``mode`` needs to be
# set to ``train``; if we load the model for evaluation, it should be ``eval``. For
# fully-connected layer, this actually does not matter. But for dropout and batchNorm
# layers, the two modes are different.
model.load(path="final_model.pkl", mode="train")


# training set
dataset_name = "Si_training_set/varying_alat"
tset = Dataset()
tset.read(dataset_name)
configs = tset.get_configs()

# calculator
calc = CalculatorTorch(model)
calc.create(configs, reuse=True)


# loss
loss = Loss(calc, residual_data={"forces_weight": 0.3})

##########################################################################################
# load the state dictionary of the optimizer.
# We also set ``start_epoch`` to ``10`` such that the epoch number continues from the last
# training.

loss.load_optimizer_stat("optimizer_stat.pkl")
result = loss.minimize(
    method="Adam", num_epochs=10, start_epoch=10, batch_size=100, lr=0.001
)
