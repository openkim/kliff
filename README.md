# KIM-based Learning-Integrated Fitting Framework (KLIFF)

This is a development fork of KLIFF for adding more ML capabilities and PyTorch integration.
All the changes will be integrated in KLIFF soon.
It integrates: 
1. the new KIM `TorchMLModelDriver`, which interfaces between KIM-API and PyTorch models, 
2. `libdescriptor` library, which is the new unified auto-differentiated descriptor library, and
3. `colabfit-tools`, which provides ability to directly interact with datasets on ColabFit exchange.

There exist two versions of these modifications, `Ver1` and `Ver2`.
While both versions contains same functionality, the difference is backward API compatibility with KLIFF.
`Ver1` maintains full-backward API compatibility, with all the changes being purely additive. 
But this also results in confusing two-way mishmash of doing things.
**This version is temporary and is only there for a risk-free alpha evaluation**.

`Ver2` breaks the old API and highlights the major design ideas for next revision, where functional design approach for 
ML centric applications is highlighted. It offers cleaner API which reflects more of the popular workflows.
It is not a pure additive change in sense that your old scripts might not work out of the box, but no previous 
functionality has been removed. The original KLIFF modules which are being redesigned are kept in `legacy` submodule. 
**This is the recommended version for future-proof applications**.

Both versions live on different branches, and can be accessed as `master-V1` and `master-V2` for `Ver1` and `Ver2` respectively.

### Installation from source
```
git clone https://github.com/ipcamit/kliff
cd kliff
git checkout master-V2 # master-V1 for Ver1
pip install .
```

To train a KIM model, `kim-api` and `kimpy` are needed; to train a machine learning
model, `PyTorch` is needed. For Graph neural networks you would need `PyTorch Geometric` and dependencies.

## Quick examples on how to train your models:

### Neural network based model

```python
# Import KLIFF requirements 
from kliff.dataset import Dataset
from kliff.ml import TrainingWheels  # Model adapter and export
from kliff.ml import OptimizerTorch  # Optimizer workflow
from kliff.ml.libdescriptor import Descriptor  # New descriptor workflow

from torch import nn

# Descriptor with cutoff = 3.0 A, for a system of Si atoms, using Symmetry Functions 
# as the descriptor of choice, and inbuilt hyperparameter set "set51".  
desc = Descriptor(cutoff=3.0, species=["Si"], descriptor="SymmetryFunctions", hyperparameters="set51")

# Stream dataset from ColabFit Database, named "colabfit_database", and Si dataset 
dataset = Dataset(colabfit_database="colabfit_database", colabfit_dataset="my_si_dataset")

# Define a simple Pytorch NN model with one hidden layer, and input width of descriptor
model = nn.Sequential(nn.Linear(desc.width, 10), nn.ReLU(), nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 1))

# Convert the descriptor to KLIFF compatible trainable model using TrainingWheels
# It ensures smoother flow of gradients from Configuration object, and Descriptor module
model_tw = TrainingWheels.from_descriptor_instance(model, desc)

# Energy and force weights
weights = {"energy": 1.0, "forces": 0.5}

# Define optimizer with dataset and properties to optimize on
model_optimizer = OptimizerTorch(model_tw, dataset[0:10], target_property=["energy", "forces"], epochs=10,
                                 weights=weights)
# Optimize the model
model_optimizer.minimize()

# Once optimized, save the model as OpenKIM model
model_tw.save_kim_model("KIM_DESC_MODEL")

```

#### OpenKIM based model
```python
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

```


## Why you want to use KLIFF (or not use it)

- Interacting seamlessly with[ KIM](https://openkim.org), the fitted model can
  be readily used in simulation codes such as LAMMPS and ASE via the `KIM API`
- Creating mixed PM and NN models
- High level API, fitting with a few lines of codes
- Low level API for creating complex NN models
- [PyTorch](https://pytorch.org) backend for NN (include GPU training)


## Citing KLIFF

```
@Article{wen2022kliff,
  title   = {{KLIFF}: A framework to develop physics-based and machine learning interatomic potentials},
  author  = {Mingjian Wen and Yaser Afshar and Ryan S. Elliott and Ellad B. Tadmor},
  journal = {Computer Physics Communications},
  volume  = {272},
  pages   = {108218},
  year    = {2022},
  doi     = {10.1016/j.cpc.2021.108218},
}
```
