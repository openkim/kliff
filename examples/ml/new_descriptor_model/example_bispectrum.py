# Import KLIFF requirements 
# Essential
from kliff.dataset import Dataset
from kliff.ml import TrainingWheels  # Model adapter and export
from kliff.ml.libdescriptor import Descriptor  # New descriptor workflow
from kliff.ml.libdescriptor import libdescriptor as lds
# Optional
from kliff.ml import OptimizerTorch  # Optimizer workflow

# Additional requirements
import numpy as np
import torch
from torch import nn

from collections import OrderedDict

# Defaults all parameters to double
torch.set_default_tensor_type(torch.DoubleTensor)

ds = Dataset(colabfit_database="colabfit_database", colabfit_dataset="jax_si")

# Descriptor with cutoff = 3.0 A, for a system of Si atoms, using Symmetry Functions 
# as the descriptor of choice, and inbuilt hyperparameter set "set51".  
bparam = OrderedDict({
    "jmax":4,
    "rfac0":0.99363,
    "diagonalstyle":3,
    "use_shared_array":False,
    "rmin0":0.0,
    "switch_flag":1,
    "bzero_flag":0})


bs_kind = lds.AvailableDescriptors(1)
print(bs_kind)

desc = Descriptor(
    cutoff=3.77, 
    species=["Si"], 
    descriptor="Bispectrum", 
    hyperparameters="bs_defaults")
print(desc)
fwd = desc.forward(ds[0])
print(fwd)
print(desc.backward(ds[0], np.ones_like(fwd)))
desc.write_kim_params("./")
