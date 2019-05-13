import kliff
from kliff.dataset import DataSet
from kliff.calculator import Calculator
from kliff.models import LennardJones
from kliff.loss import Loss

kliff.logger.set_level('debug')

# training set
tset = DataSet()
tset.read('../tests/configs_extxyz/Si_4/')
configs = tset.get_configs()

# calculator
model = LennardJones()
# model.echo_model_params()

# fitting parameters
model.set_fitting_params(sigma=[['default']], epsilon=[['default']])
# model.echo_fitting_params()

calc = Calculator(model)
calc.create(configs)

# loss
with Loss(calc, nprocs=1) as loss:
    result = loss.minimize(method='L-BFGS-B', options={'disp': True, 'maxiter': 10})


# print optimized parameters
model.echo_fitting_params()
model.save_model_params()
