from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import tensorflow as tf
from kliff.dataset import DataSet
from kliff.loss import Loss
from kliff.fingerprints import Fingerprints
from kliff.descriptor.symmetryfunction import Set51
from kliff.neuralnetwork import Dense
from kliff.neuralnetwork import Dropout
from kliff.neuralnetwork import Output
from kliff.neuralnetwork import NeuralNetwork
from kliff.neuralnetwork import ANNCalculator


descriptor = Set51(cutvalue={'Si-Si': 5.0})
fps = Fingerprints(descriptor, normalize=True, fit_forces=True, dtype=tf.float32)

model = NeuralNetwork(fingerprints=fps)

# dropout for input fingerprint
model.add_layer(Dropout(keep_ratio=1.))

# first hidden layer
model.add_layer(Dense(num_units=30))
model.add_layer(Dropout(keep_ratio=0.9))

# second hidden layer
model.add_layer(Dense(num_units=30))
model.add_layer(Dropout(keep_ratio=0.9))

# output layer
model.add_layer(Output())


# training set
tset = DataSet()
tset.read('training_set/Si_T300_4')
configs = tset.get_configurations()


# calculator
calc = ANNCalculator(model, num_epoches=100, batch_size=4)
calc.create(configs)


# loss
loss = Loss(model, calc)
#result = loss.minimize(method='L-BFGS-B', options={'disp': True, 'maxiter': 2})
result = loss.minimize(method='AdamOptimizer', learning_rate=1e-3)



print('Results:', result)
print('Fitting done')
