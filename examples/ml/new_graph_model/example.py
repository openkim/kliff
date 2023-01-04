# KLIFF Dependencies
from kliff.dataset import Dataset
from kliff.ml import TrainingWheels
from kliff.ml import OptimizerTorch

# PyTorch Dependencies
import torch

from simple_equiv_model import EGNN4KLIFF

torch.set_default_tensor_type(torch.DoubleTensor)

# ------------------ Generalized Database ------------------ #
# Load from COLABFIT database
ds = Dataset(colabfit_database="colabfit_database", colabfit_dataset="my_si_dataset")

# ------------------ Define Model -------------------------- #
model = EGNN4KLIFF(3, 10, 3)

# ------------------ Define TrainingWheels ----------------- #
model_tw = TrainingWheels.init_graph(model, 3.77, 3, ["Si"])

print(model_tw.get_parameters())

# ------------------  Weights ------------------------------ #
weights = {"energy": 1.0, "forces": 0.5}

# ------------------ Optimize ------------------------------ #
model_optimizer = OptimizerTorch(model_tw, ds[0:10], target_property=["energy", "forces"],
                                 epochs=10, weights=weights)
model_optimizer.print_loss = True
model_optimizer.minimize()

# ------------------ Save Model ---------------------------- #
model_tw.save_kim_model("KIM_GRAPH_MODEL")
