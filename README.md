# KIM-based Learning-Integrated Fitting Framework (KLIFF)

[![Build Status](https://travis-ci.com/mjwen/kliff.svg?branch=master)](https://travis-ci.com/mjwen/kliff)
[![Documentation Status](https://readthedocs.org/projects/kliff/badge/?version=latest)](https://kliff.readthedocs.io/en/latest/?badge=latest)


KLIFF is an interatomic potential fitting package that can be used to fit
physics-motivated (PM) potentials, as well as machine learning potentials such
as the neural network (NN) models.

**Documentation at: <https://kliff.readthedocs.io>**

## Why you want to use KLIFF (or not use it)

- Interacting seamlessly with[ KIM](https://openkim.org), the fitted model can
  be readily used in simulation codes such as LAMMPS and ASE via the `KIM API`
- Creating mixed PM and NN models
- High level API, fitting with a few lines of codes
- Low level API for creating complex NN models
- Parallel execution
- [PyTorch](https://pytorch.org) backend for NN

