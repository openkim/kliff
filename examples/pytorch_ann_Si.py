import sys
import numpy as np
import torch.nn as nn
from kliff.dataset import DataSet
from kliff.loss import Loss
from kliff.descriptors.symmetry_function import Set51
from kliff.pytorch_neuralnetwork import NeuralNetwork
from kliff.pytorch_neuralnetwork import PytorchANNCalculator


descriptor = Set51(cutvalue={'Si-Si': 5.0}, normalize=True, grad=True)
model = NeuralNetwork(descriptor)

# dropout for input fingerprint
model.add_layer(nn.Linear(51, 30))
model.add_layer(nn.Sigmoid())
model.add_layer(nn.Linear(30, 30))
model.add_layer(nn.Sigmoid())
model.add_layer(nn.Linear(30, 1))


# training set
tset = DataSet()
tset.read('../tests/configs_extxyz/Si_4')
configs = tset.get_configurations()


# calculator
calc = PytorchANNCalculator(model, num_epochs=10, batch_size=2)
calc.create(configs)


# loss
loss = Loss(calc)
# result = loss.minimize(method='L-BFGS-B', options={'disp': True, 'maxiter': 2})
result = loss.minimize(method='AdamOptimizer', learning_rate=1e-3)
