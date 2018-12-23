import kliff
from kliff.dataset import DataSet
from kliff.calculators import LennardJones
from kliff.loss import Loss

kliff.logger.set_level('debug')

# training set
tset = DataSet()
tset.read('../tests/calculators/Si_T300_4/')
configs = tset.get_configurations()

# calculator
calc = LennardJones()
# calc.echo_model_params()
calc.create(configs)

# fitting parameters
calc.set_fitting_params(sigma=[['default']], epsilon=[['default']])
# calc.echo_fitting_params()

# loss
with Loss(calc, nprocs=1) as loss:
    result = loss.minimize(method='L-BFGS-B',
                           options={'disp': True, 'maxiter': 10})


# print optimized parameters
calc.echo_fitting_params()
