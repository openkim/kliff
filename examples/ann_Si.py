import sys
import numpy as np
import tensorflow as tf
import kliff
from kliff.dataset import DataSet
from kliff.loss import Loss
from kliff.fingerprints import Fingerprints
from kliff.descriptors.symmetry_function import Set51
import kliff.neuralnetwork as nn

kliff.logger.set_level('debug')

descriptor = Set51(cutvalue={'Si-Si': 5.0})
fps = Fingerprints(descriptor, normalize=True, fit_forces=True, dtype=tf.float32)

model = nn.NeuralNetwork(fingerprints=fps)

# dropout for input fingerprint
model.add_layer(nn.Dropout(keep_ratio=1.))

# first hidden layer
model.add_layer(nn.Dense(num_units=30))
model.add_layer(nn.Dropout(keep_ratio=0.9))

# second hidden layer
model.add_layer(nn.Dense(num_units=30))
model.add_layer(nn.Dropout(keep_ratio=0.9))

# output layer
model.add_layer(nn.Output())


# training set
tset = DataSet()
tset.read('../tests/configs_extxyz/Si_4')
configs = tset.get_configurations()


# calculator
calc = nn.ANNCalculator(model, num_epoches=100, batch_size=4)
calc.create(configs)


# loss
loss = Loss(calc)
#result = loss.minimize(method='L-BFGS-B', options={'disp': True, 'maxiter': 2})
result = loss.minimize(method='AdamOptimizer', learning_rate=1e-3)
