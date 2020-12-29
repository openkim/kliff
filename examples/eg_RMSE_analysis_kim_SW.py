from kliff.analyzers import EnergyForcesRMSE
from kliff.calculators import Calculator
from kliff.dataset import Dataset
from kliff.models import KIMModel

model = KIMModel(model_name="SW_StillingerWeber_1985_Si__MO_405512056662_005")
model.load("kliff_model.pkl")

tset = Dataset("Si_training_set")
configs = tset.get_configs()

calc = Calculator(model)
calc.create(configs)

analyzer = EnergyForcesRMSE(calc)
analyzer.run(verbose=2, sort="energy")
