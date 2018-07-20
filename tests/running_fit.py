from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import time
from kliff.dataset import DataSet
from kliff.modelparameters import ModelParameters
from kliff.kimcalculator import KIMCalculator
from kliff.loss import Loss
from kliff.loss import energy_forces_residual


start_time = time.time()

# KIM model parameters
model = 'Three_Body_Stillinger_Weber_Si__MO_405512056662_004'
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
with Loss(params, calc, residual_fn=energy_forces_residual) as loss:
  #result = loss.minimize(method='lm')
  result = loss.minimize(method='L-BFGS-B', options={'disp':True, 'maxiter':100 })
  #print(result)


# print optimized parameters
params.echo_params()
print('--- running time: {} seconds ---'.format(time.time() - start_time))

