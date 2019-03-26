import kliff
from kliff.dataset import DataSet
from kliff.models import KIM
from kliff.calculator import Calculator
from kliff.loss import Loss


# setting logger info
kliff.logger.set_level('info')


# create a KIM model
model = KIM(model_name='Three_Body_Stillinger_Weber_Si__MO_405512056662_004')

# print parameters that are available for fitting
model.echo_model_params()

# fitting parameters
model.set_fitting_params(
    A=[[5.0, 1., 20]],
    B=[['default']],
    sigma=[[2.0951, 'fix']],
    gamma=[[1.5]])

# print fitting parameters
model.echo_fitting_params()


# training set
dataset_name = 'Si_training_set'
tset = DataSet()
tset.read(dataset_name)
configs = tset.get_configurations()


# calculator
calc = Calculator(model)
calc.create(configs)


# loss
steps = 100
loss = Loss(calc, nprocs=1)
loss.minimize(method='L-BFGS-B', options={'disp': True, 'maxiter': steps})


# print optimized parameters
model.echo_fitting_params()

# save model for later retraining
model.save('kliff_model.pkl')

# create a kim model
model.write_kim_model()
