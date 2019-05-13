import kliff
from kliff.dataset import DataSet
from kliff.calculators import KIM
from kliff.calculators import WrapperCalculator
from kliff.loss import Loss

kliff.logger.set_level('debug')


# training set
tset = DataSet()
tset.read('../tests/calculators/Si_T300_4/')
configs = tset.get_configs()

# calculators
calc1 = KIM(model_name='Three_Body_Stillinger_Weber_Si__MO_405512056662_004')
calc1.create(configs)
calc1.set_fitting_params(A=[[16.0, 1.0, 20]], B=[['DEFAULT']])
calc2 = KIM(model_name='Three_Body_Stillinger_Weber_Si__MO_405512056662_004')
calc2.create(configs)
calc2.set_fitting_params(sigma=[[2.0951, 'fix']], gamma=[[1.1]])

# wrapper
calc = WrapperCalculator(calc1, calc2)

# loss
with Loss(calc, nprocs=2) as loss:
    result = loss.minimize(method='L-BFGS-B', options={'disp': True, 'maxiter': 10})
