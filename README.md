# KIM-based Learning-Integrated Fitting Framework (KLIFF)

KLIFF is an interatomic potential fitting package that can be used to fit physics-motivated (PM) potentials, as well as 
machine learning potentials such as the neural network (NN) models.

KLIFF can be used to fit both Physics motivated and Machine Learning based Portable OpenKIM models.
It utilizes the KIM-API model drivers to interface with the portable models and ML capabilities are driven by the PyTorch
integration.

Added KLIFF capabilities include:
1. Ability to export TorchScript ML models to KIM-API using the [`TorchMLModelDriver`](https://github.com/ipcamit/colabfit-model-driver) 
2. Integration with [`libdescriptor`](https://github.com/ipcamit/libdescriptor) library, which is the new unified auto-differentiated descriptor library, and
3. Data streaming capabilities from ColabFit database instance using [`colabfit-tools`](https://github.com/colabfit/colabfit-tools).

## Installation
The latest version of KLIF can be installed directly from source using `pip`:

```bash
pip install git+https://github.com/openkim/kliff.git@ml
```
> Note: The `ml` branch is the latest development branch, but has not yet been merged into the `master` branch.

## Dependencies
Listed below are the major dependencies of KLIFF, along with their intended use. For a complete list of dependencies, see the `requirements.txt` file.

Essential dependencies:

| Dependency | Version    | Usage                                                            |
|------------|------------|------------------------------------------------------------------|
| numpy      | =>1.22.0   | General Python numerical requirement                             |
| scipy      | =>1.10.0   | Optimization of Physics based models                             |
| loguru     | =>0.5.3    | Logging                                                          |
| pybind11   | =>2.10     | Python bindings for C++ code                                     |
| monty      | =>2022.9.9 | General Python utilities (for implementing posrtale `parameters` |
| pyyaml     | =>5.4.1    | YAML file parsing                                                |
| loguru     | =>0.6.0    | Logging                                                          |
| kimpy      | =>2.1.0    | KIM API bindings (for interacting with KIM Portable models       |
| KIM-API    | =>2.2.0    | KIM API backend for Physics based models (needed by KIMPY)       |
| torch      | =>1.10.0   | PyTorch backend for ML models                                    |

Optional dependencies (You can still use KLIFF without these, but some features will be disabled):

| Dependency      | Version  | Usage                                                                 |
|-----------------|----------|-----------------------------------------------------------------------|
| torch-geometric | =>1.7.2  | PyTorch Supported GNN backend, (used for GNN and graph dataset format |
| libdescriptor   | =>0.5.0  | Unified auto-differentiable descriptor library                        |
| colabfit-tools  | =>0.1.0  | ColabFit database tools                                               |
| torch-scatter   | =>2.0.9  | High performance scatter-gather operations needed for torch-geometric |
| torch-sparse    | =>0.6.12 | High performance sparse matrix operations needed for torch-geometric  |
| ASE             | =>3.21.1 | ASE backend as an alternative configuration file-io parser            |


## Quick examples on how to train your models:
KLIFF supports several workflows for training your models. You can train KIM models using ASE `calculator` like objects 
for more object-oriented workflow, or use KIM Models as a functor/closure datastructure for more 
functional approach. For most common workflows, KLIFF provides a `Optimizer` classes that can be used to train your models.
Below are some examples of how to use these workflows in KLIFF to train your models.

### Neural network based model

```python
# Import KLIFF requirements 
from kliff.dataset import Dataset
from kliff.ml import TrainingWheels  # Model adapter and export
from kliff.ml import OptimizerTorch  # Optimizer workflow
from kliff.ml import Descriptor  # New descriptor workflow

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
model_tw.save_kim_model("MLMODEL__MO_000000000000_000")

```

#### Physics based model
##### 1. Conventional workflow
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

model.write_kim_model("SW__MO_000000000000_000")
```

##### 2. Object-oriented approach
```python
from kliff.ase.calculators import Calculator
from kliff.dataset import Dataset
from kliff.ase.loss import Loss
from kliff.models import KIMModel

# Define model and parameters
model = KIMModel(model_name="SW_StillingerWeber_1985_Si__MO_405512056662_006")
model.set_opt_params(
    A=[[5.0, 1.0, 20]], B=[["default"]], sigma=[[2.0951, "fix"]], gamma=[[1.5]]
)

# Stream dataset from ColabFit Database, named "colabfit_database", and Si dataset 
dataset = Dataset(colabfit_database="colabfit_database", colabfit_dataset="my_si_dataset")

# Define ASE calculator
calc = Calculator(model)
_ = calc.create(dataset.get_configs())

# Optimize the model
steps = 100
loss = Loss(calc, nprocs=2)
loss.minimize(method="L-BFGS-B", options={"disp": True, "maxiter": steps})

# Export the model
model.write_kim_model()
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
