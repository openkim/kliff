import sys
import numpy as np
from kliff.descriptors import SymmetryFunction
from kliff.dataset import DataSet
from kliff.loss import Loss
import kliff.neuralnetwork as nn


descriptor = SymmetryFunction(cut_name='cos', cut_dists={'Si-Si': 5.0},
                              hyperparams='set31', normalize=True)
desc_size = len(descriptor)

# dropout for input fingerprint
model = nn.NeuralNetwork(descriptor)
model.add_layers(nn.Linear(desc_size, 10),
                 nn.Tanh(),
                 # nn.Dropout(p=0.1),
                 nn.Linear(10, 10),
                 nn.Tanh(),
                 # nn.Dropout(p=0.1),
                 nn.Linear(10, 1))


# training set
tset = DataSet()
tset.read('Si_training_set/varying_alat')
configs = tset.get_configs()


# calculator
calc = nn.PytorchANNCalculator(model)
calc.create(configs, use_forces=True, reuse=False)


# loss
loss = Loss(calc, residual_data={'forces_weight': 0.3})

#result = loss.minimize(method='SGD', num_epochs=10, batch_size=100, lr=0.1)
# result = loss.minimize(method='LBFGS', num_epochs=100, batch_size=400,
#                       lr=0.001)
result = loss.minimize(method='Adam', num_epochs=10, batch_size=100, lr=0.01)


model.save('./saved_model.pt')
model.write_kim_model()
