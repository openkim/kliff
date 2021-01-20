"""
Fisher information for the SW potential.

See `A force-matching Stillinger-Weber potential for MoS2: Parameterization and Fisher
information theory based sensitivity analysis <https://doi.org/10.1063/1.5007842>`_
"""


from kliff.analyzers import Fisher
from kliff.calculators import Calculator
from kliff.dataset import Dataset
from kliff.models import KIMModel
from kliff.utils import download_dataset

##########################################################################################
# Select the parameters that will be used to compute the Fisher information. Only
# parameters specified below will be use, others will be kept fixed. The size of the
# Fisher information matrix will be equal to the total size of the parameters specified
# here.
model = KIMModel(model_name="SW_StillingerWeber_1985_Si__MO_405512056662_005")
model.set_opt_params(
    A=[["default"]], B=[["default"]], sigma=[["default"]], gamma=[["default"]]
)

# dataset
dataset_path = download_dataset(dataset_name="Si_training_set_4_configs")
tset = Dataset(dataset_path)
configs = tset.get_configs()

# calculator
calc = Calculator(model)
calc.create(configs)

##########################################################################################
# Fisher information analyzer.
analyzer = Fisher(calc)
analyzer.run()
