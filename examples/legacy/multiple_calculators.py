from kliff.ase.calculators.calculator import Calculator, _WrapperCalculator
from kliff.dataset import Dataset
from kliff.ase.loss import Loss
from kliff.models import KIMModel

# training set
tset = Dataset("../Si_training_set")
configs = tset.get_configs()
configs1 = configs[len(configs) // 2 :]
configs2 = configs[: len(configs) // 2]

# models
model1 = KIMModel(model_name="SW_StillingerWeber_1985_Si__MO_405512056662_006")
model1.set_opt_params(A=[[16.0, 1.0, 20]], B=[["DEFAULT"]])
model2 = KIMModel(model_name="SW_StillingerWeber_1985_Si__MO_405512056662_006")
model2.set_opt_params(sigma=[[2.0951, "fix"]], gamma=[[1.1]])

# calculators
calc1 = Calculator(model=model1)
calc1.create(configs1, use_energy=True, use_forces=False)

calc2 = Calculator(model=model2)
calc2.create(configs1, use_energy=True, use_forces=True)

# wrapper
calc = _WrapperCalculator(calculators=[calc1, calc2])

if __name__ == "__main__":
    loss = Loss(calc, nprocs=2)
    loss.minimize(method="L-BFGS-B", options={"disp": True, "maxiter": 10})
