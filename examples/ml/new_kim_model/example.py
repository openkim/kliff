from kliff.models import KIMModel
from kliff.dataset import Dataset
from kliff.ml.opt import OptimizerScipy
from kliff.utils import download_dataset

dataset_path = download_dataset(dataset_name="Si_training_set")
# ds = Dataset(colabfit_database="colabfit_database", colabfit_dataset="my_dataset")
ds = Dataset(dataset_path)

model = KIMModel("SW_StillingerWeber_1985_Si__MO_405512056662_006")

model.echo_model_params()
model.set_opt_params(
    A=[[5.0, 1.0, 20]], B=[["default"]], sigma=[[2.0951, "fix"]], gamma=[[1.5]])
model.echo_opt_params()

params = model.parameters()

weight = {"energy": 1.0, "forces": 1.0, "stress": 0.0}

opt = OptimizerScipy(model, params, ds[0:10], optimizer="L-BFGS-B", maxiter=400, weights=weight, verbose=True, tol=1e-6,
                     target_property=["energy", "forces"])
opt.minimize()

model.write_kim_model("KIM_OPT_MODEL")
