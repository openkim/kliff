"""
Fisher information for the SW potential.
"""


from kliff.analyzers import Fisher
from kliff.calculators import Calculator
from kliff.dataset import Dataset
from kliff.models import KIM

##########################################################################################
# Select the parameters that will be used to compute the Fisher information. Only
# parameters specified below will be use, others will be kept fixed. The size of the
# Fisher information matrix will be equal to the total size of the parameters specified
# here.
model = KIM(model_name="SW_StillingerWeber_1985_Si__MO_405512056662_005")
model.set_fitting_params(
    A=[["default"]], B=[["default"]], sigma=[["default"]], gamma=[["default"]]
)

# dataset
dataset_name = "tmp_tset"
tset = Dataset()
tset.read(dataset_name)
configs = tset.get_configs()

# calculator
calc = Calculator(model)
calc.create(configs)

##########################################################################################
# Fisher information analyzer.
analyzer = Fisher(calc)
analyzer.run()
