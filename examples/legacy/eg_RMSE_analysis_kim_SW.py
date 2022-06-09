"""
Compute the root-mean-square error (RMSE) of a model prediction and reference values in
the dataset.
"""

from kliff.analyzers import EnergyForcesRMSE
from kliff.calculators import Calculator
from kliff.dataset import Dataset
from kliff.models import KIMModel
from kliff.utils import download_dataset

model = KIMModel(model_name="SW_StillingerWeber_1985_Si__MO_405512056662_006")

# load the trained model back
# model.load("kliff_model.yaml")


dataset_path = download_dataset(dataset_name="Si_training_set_4_configs")
tset = Dataset(dataset_path)
configs = tset.get_configs()

calc = Calculator(model)
calc.create(configs)

analyzer = EnergyForcesRMSE(calc)
analyzer.run(verbose=2, sort="energy")
