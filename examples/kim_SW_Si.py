import kliff
from kliff.dataset import DataSet
from kliff.models import KIM
from kliff.calculator import Calculator
from kliff.loss import Loss

kliff.logger.set_level('debug')


# training set
tset = DataSet()
tset.read('../tests/configs_extxyz/Si_4/')
configs = tset.get_configurations()


# calculator
model = KIM(model_name='Three_Body_Stillinger_Weber_Si__MO_405512056662_004')
# calc.echo_model_params()

# fitting parameters
model.set_fitting_params(
    A=[[16.0, 1., 20]],
    B=[['DEFAULT']],
    sigma=[[2.0951, 'fix']],
    gamma=[[2.51412]])

# "lambda" is a python keyword, cannot use the above method
model.set_one_fitting_params(name='lambda', settings=[[45.5322]])

# calc.echo_fitting_params()


calc = Calculator(model)
calc.create(configs)


# loss
with Loss(calc, nprocs=2) as loss:
    result = loss.minimize(method='L-BFGS-B',
                           options={'disp': True, 'maxiter': 10})


# print optimized parameters
model.echo_fitting_params()
model.write_kim_model()
