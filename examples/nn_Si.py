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
model.add_layers(nn.Linear(desc_size, 30),
                 nn.Sigmoid(),
                 nn.Dropout(p=0.1),
                 nn.Linear(30, 30),
                 nn.Sigmoid(),
                 nn.Dropout(p=0.1),
                 nn.Linear(30, 1))


# training set
tset = DataSet()
tset.read('../tests/configs_extxyz/Si_4')
configs = tset.get_configs()


# calculator
calc = nn.PytorchANNCalculator(model)
calc.create(configs, use_forces=True)


# loss
loss = Loss(calc)
# result = loss.minimize(method='L-BFGS-B', options={'disp': True, 'maxiter': 2})
result = loss.minimize(method='SGD', num_epochs=10, batch_size=2,
                       lr=0.001, momentum=0.9)

model.save('./saved_model.pt')
model.write_kim_model()
