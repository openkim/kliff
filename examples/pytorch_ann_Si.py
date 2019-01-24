import sys
import numpy as np
import torch
import torch.nn as nn
from kliff.dataset import DataSet
from kliff.loss import Loss
from kliff.fingerprints import Fingerprints
from kliff.descriptors.symmetry_function import Set51
from kliff.pytorch_neuralnetwork import PytorchANNCalculator
from kliff.pytorch_neuralnetwork import NeuralNetwork


descriptor = Set51(cutvalue={'Si-Si': 5.0})
fps = Fingerprints(descriptor, normalize=True, fit_forces=True)

model = NeuralNetwork(fingerprints=fps)

# dropout for input fingerprint
model.add_layer(nn.Linear(51, 30))


# training set
tset = DataSet()
tset.read('../tests/configs_extxyz/Si_4')
configs = tset.get_configurations()


# calculator
calc = PytorchANNCalculator(model, num_epochs=100, batch_size=2)
calc.create(configs)


# loss
loss = Loss(calc)
# result = loss.minimize(method='L-BFGS-B', options={'disp': True, 'maxiter': 2})
result = loss.minimize(method='AdamOptimizer', learning_rate=1e-3)


print('Results:', result)
