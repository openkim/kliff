# KIM-based Learning-Integrated Fitting Framework (KLIFF)

[![Build Status](https://travis-ci.com/mjwen/kliff.svg?branch=master)](https://travis-ci.com/mjwen/kliff)
[![Python package](https://github.com/mjwen/kliff/workflows/Python%20package/badge.svg)](https://github.com/mjwen/kliff/actions)
[![Documentation Status](https://readthedocs.org/projects/kliff/badge/?version=latest)](https://kliff.readthedocs.io/en/latest/?badge=latest)
[![Anaconda-Server Badge](https://img.shields.io/conda/vn/conda-forge/kliff.svg)](https://anaconda.org/conda-forge/kliff)
[![PyPI](https://img.shields.io/pypi/v/kliff.svg)](https://pypi.python.org/pypi/kliff)

### Documentation at: <https://kliff.readthedocs.io>

KLIFF is an interatomic potential fitting package that can be used to fit
physics-motivated (PM) potentials, as well as machine learning potentials such
as the neural network (NN) models.

## Install 

### Using conda
```sh
conda intall -c conda-forge kliff
```

### Using pip
```sh
pip install kliff
```

### From source 
```
git clone https://github.com/mjwen/kliff
pip install ./kliff
```

To train a KIM model, `kim-api` and `kimpy` are needed; to train a machine learning 
model, `PyTorch` is needed. For more information on installing these packages, see 
[Installation](https://kliff.readthedocs.io/en/latest/installation.html).

## A quick example to train a neural network potential

```python
from kliff import nn
from kliff.calculators import CalculatorTorch
from kliff.descriptors import SymmetryFunction
from kliff.dataset import Dataset
from kliff.models import NeuralNetwork
from kliff.loss import Loss

# Descriptor to featurize atomic configurations  
descriptor = SymmetryFunction(
    cut_name="cos", cut_dists={"Si-Si": 5.0}, hyperparams="set51", normalize=True
)

# Fully-connected neural network model with 2 hidden layers, each with 10 units 
N1 = 10
N2 = 10
model = NeuralNetwork(descriptor)
model.add_layers(
    # first hidden layer
    nn.Linear(descriptor.get_size(), N1),
    nn.Tanh(),
    # second hidden layer
    nn.Linear(N1, N2),
    nn.Tanh(),
    # output layer
    nn.Linear(N2, 1),
)

# Training set (can be downloaded at: https://github.com/mjwen/kliff/blob/master/examples/Si_training_set.tar.gz)
dataset_name = "Si_training_set/varying_alat"
train_set = Dataset()
train_set.read(dataset_name)
configs = train_set.get_configs()

# Set up calculator to compute energy and forces for atomic configurations in the 
# training set using the neural network model
calc = CalculatorTorch(model)
calc.create(configs)

# Define a loss function and train the model by minimizing the loss 
loss = Loss(calc, residual_data={"forces_weight": 0.3})
result = loss.minimize(method="Adam", num_epochs=10, batch_size=100, lr=0.001)

# Write trained model as a KIM model to be used in other codes such as LAMMPS ans ASE
model.write_kim_model()
```

Detailed explanation and more tutorial examples can be found in the 
[documentation](https://kliff.readthedocs.io/en/latest/tutorials.html). 

## Why you want to use KLIFF (or not use it)

- Interacting seamlessly with[ KIM](https://openkim.org), the fitted model can
  be readily used in simulation codes such as LAMMPS and ASE via the `KIM API`
- Creating mixed PM and NN models
- High level API, fitting with a few lines of codes
- Low level API for creating complex NN models
- Parallel execution
- [PyTorch](https://pytorch.org) backend for NN


## Contact

Mingjian Wen (wenxx151@gmail.com)