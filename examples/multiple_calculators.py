from kliff.calculators.calculator import Calculator, _WrapperCalculator
from kliff.dataset import Dataset
from kliff.loss import Loss

# training set
tset = Dataset("Si_training_set")
configs = tset.get_configs()

# calculators
calc1 = Calculator(model="Three_Body_Stillinger_Weber_Si__MO_405512056662_004")
calc1.create(configs)
calc1.set_fitting_params(A=[[16.0, 1.0, 20]], B=[["DEFAULT"]])
calc2 = Calculator(model="Three_Body_Stillinger_Weber_Si__MO_405512056662_004")
calc2.create(configs)
calc2.set_fitting_params(sigma=[[2.0951, "fix"]], gamma=[[1.1]])

# wrapper
calc = _WrapperCalculator(calc1, calc2)

# loss
with Loss(calc, nprocs=2) as loss:
    result = loss.minimize(method="L-BFGS-B", options={"disp": True, "maxiter": 10})
