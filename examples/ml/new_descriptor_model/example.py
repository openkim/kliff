"""
This is a simple example of how to use the new descriptor based model using new
 libdescriptor interface. In this example we will use the implicit optimizer 
 interface which is simple and suitable for most optimizations. For more 
 complicated workflows, and loss functions, it is advisable to use explicit 
 interface.

The dataset supports input from both ColabFit dataset, as well has conventional
file system based IO. 
"""

# Import KLIFF requirements 
from kliff.dataset import Dataset
from kliff.ml import TrainingWheels  # Model adapter and export
from kliff.ml import OptimizerTorch  # Optimizer workflow
from kliff.ml.libdescriptor import Descriptor  # New descriptor workflow

# Additional requirements
import numpy as np
import torch
from torch import nn

# Defaults all parameters to double
torch.set_default_tensor_type(torch.DoubleTensor)

# Check for available descriptors:
Descriptor.show_available_descriptors()

# Descriptor with cutoff = 3.0 A, for a system of Si atoms, using Symmetry Functions 
# as the descriptor of choice, and inbuilt hyperparameter set "set51".  
desc = Descriptor(cutoff=3.0, species=["Si"], descriptor="SymmetryFunctions", hyperparameters="set51")

# Stream dataset from ColabFit Database, named "colabfit_database", and Si dataset 
# named "my_si_dataset" [Note: your dataset name may vary].
dataset = Dataset(colabfit_database="colabfit_database", colabfit_dataset="my_si_dataset")

# AD aided descriptor computation:
# Given a configuration, descriptor can be computed using simple `forward` function
new_desc = desc.forward(dataset[0])

# You can compute torch-like Jacobian-vector products by using `backward` function
grad_new_desc = desc.backward(dataset[0], np.ones_like(new_desc))

# Define a simple Pytorch NN model with one hidden layer, and input width of descriptor
model = nn.Sequential(nn.Linear(desc.width, 10), nn.ReLU(), nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 1))

# Convert the descriptor to KLIFF compatible trainable model using TrainingWheels
# It ensures smoother flow of gradients from Configuration object, and Descriptor module
model_tw = TrainingWheels.from_descriptor_instance(model, desc)

# Trainable parameters
print(model_tw.parameters)

# Energy and force weights
weights = {"energy": 1.0, "forces": 0.5}

# Define optimizer with dataset and properties to optimize on
model_optimizer = OptimizerTorch(model_tw, dataset[0:10], target_property=["energy", "forces"], epochs=10,
                                 weights=weights)

# Print epoch-by-epoch progress to stdout
model_optimizer.print_loss = True

# Optimize the model
model_optimizer.minimize()

# Once optimized, save the model as OpenKIM model
model_tw.save_kim_model("KIM_DESC_MODEL")

# Now to use the model in production, give the following command in terminal
# kim-api-collections-management install user  KIM_DESC_MODEL 

# Comparison with old KLIFF Descriptors ==========================================
# from kliff.legacy.descriptors import SymmetryFunction
# desc_kliff = SymmetryFunction({"Si-Si":3.0},"cos","set51")
# old_desc,jacobian_old_desc,_ = desc_kliff.transform(dataset[0],fit_forces=True)
# print(np.max(np.abs(new_desc - old_desc)))
# print(np.max(np.abs(np.einsum("ijk,ij->k", jacobian_old_desc, 
#                               np.ones((64,51))).reshape(64,3) - grad_new_desc)))
# ================================================================================
