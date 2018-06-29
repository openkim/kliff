from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import time
from tipp.dataset import DataSet
from tipp.modelparameters import ModelParameters
from tipp.kimcalculator import KIMCalculator
from tipp.loss import Loss
from tipp.loss import energy_forces_residual


start_time = time.time()

# KIM model parameters
model = 'Three_Body_Stillinger_Weber_Si__MO_000000111111_000'
params = ModelParameters(model)
params.echo_avail_params()
fname = 'input/Si_SW_init_guess.txt'
params.read(fname)
params.echo_params()


# training set
tset = DataSet()
tset.read('training_set/Si_T300_4')
configs = tset.get_configurations()


# calculator
calc = KIMCalculator(model)
calc.create(configs)


# loss
loss = Loss(params, calc, residual_fn=energy_forces_residual)
#result = loss.minimize(method='lm')
result = loss.minimize(method='L-BFGS-B', options={'disp':True, 'maxiter':100 })
#print(result)


# print optimized parameters
params.echo_params()
print('--- running time: {} seconds ---'.format(time.time() - start_time))

