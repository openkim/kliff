# The Interatomic Potential Package (TIPP)

TIPP is an interatomic potential fitting library that can be used to fit physics-motivated (PM) models and machine learning potentials such as the Aritifical Neural Network (ANN) models.

## Why you want to use TIPP (or don't use it)

- Interacting seamlessly with[ KIM](https://openkim.org), the fitted model can be readily used in simulation codes such as LAMMPS and ASE via the `KIM API`
- Creating mixed PM and ANN models
- High level API, fitting with a few lines of codes
- Also provides low level API for creating complex ANN models
- Parallel execution
- [TensorFlow](https://www.tensorflow.org) backend for ANN

## Installation

(optional) Create a python virtual environment

1. Install from source

    ```
    $ pip install git+https://github.com/mjwen/TIPP.git@master
    ```

