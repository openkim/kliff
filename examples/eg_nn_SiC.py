"""
.. _tut_nn_multi_spec:

Train a neural network potential
================================

In this tutorial, we train a neural network (NN) potential for SiC
"""


from kliff import nn
from kliff.calculators.calculator_torch import CalculatorTorchSeparateSpecies
from kliff.dataset import Dataset
from kliff.descriptors import SymmetryFunction
from kliff.loss import Loss
from kliff.models import NeuralNetwork

descriptor = SymmetryFunction(
    cut_name="cos",
    cut_dists={"Si-Si": 5.0, "C-C": 5.0, "Si-C": 5.0},
    hyperparams="set30",
    normalize=True,
)

N1 = 10
N2 = 10
model_si = NeuralNetwork(descriptor)
model_si.add_layers(
    # first hidden layer
    nn.Linear(descriptor.get_size(), N1),
    nn.Tanh(),
    # second hidden layer
    nn.Linear(N1, N2),
    nn.Tanh(),
    # output layer
    nn.Linear(N2, 1),
)
model_si.set_save_metadata(prefix="./kliff_saved_model_si", start=5, frequency=2)


N1 = 10
N2 = 10
model_c = NeuralNetwork(descriptor)
model_c.add_layers(
    # first hidden layer
    nn.Linear(descriptor.get_size(), N1),
    nn.Tanh(),
    # second hidden layer
    nn.Linear(N1, N2),
    nn.Tanh(),
    # output layer
    nn.Linear(N2, 1),
)
model_c.set_save_metadata(prefix="./kliff_saved_model_c", start=5, frequency=2)


# training set
dataset_name = "SiC_training_set"
tset = Dataset()
tset.read(dataset_name)
configs = tset.get_configs()

# calculator
calc = CalculatorTorchSeparateSpecies({"Si": model_si, "C": model_c})
calc.create(configs, reuse=True)

# loss
loss = Loss(calc, residual_data={"forces_weight": 0.3})
result = loss.minimize(method="Adam", num_epochs=10, batch_size=4, lr=0.001)


##########################################################################################
# We can save the trained model to disk, and later can load it back if we want. We can
# also write the trained model to a KIM model such that it can be used in other simulation
# codes such as LAMMPS via the KIM API.

model_si.save("./final_model_si.pkl")
model_c.save("./final_model_c.pkl")
loss.save_optimizer_stat("./optimizer_stat.pkl")
