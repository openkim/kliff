.. _changelog:

==========
Change Log
==========

v0.4.0 (2022/04/27)

Added
-----
- Add ParameterTransform class to transform parameters into a different space (e.g. log
  space) @yonatank93
- Add Weight class to set weight for energy/forces/stress. This is not backward
  compatible, which changes the signature of the residual function. Previously, in a
  residual function, the weights are passed in via the `data` argument, but now, its
  passed in via an instance of the Weight class. @yonatank93

Fixed
-----
- Fix checking cutoff entry @adityakavalur
- Fix energy_residual_fn and forces_residual_fn to weigh different component

Updated
-------
- Change to use precommit GH action to check code format


v0.3.3 (2022/03/25)
===================

Fixed
-----
- Fix neighlist (even after v0.3.2, the problem can still happen). Now neighlist is the
  same as kimpy


v0.3.2 (2022/03/01)
===================

Added
-----
- Enable params_relation_callback() for KIM model

Fixed
-----
- Fix neighbor list segfault due to numerical error for 1D and 2D cases


v0.3.1 (2021/11/20)
===================

- add gpu training for NN model; set the ``gpu`` parameter of a calculator (e.g.
  ``CalculatorTorch(model, gpu=True)``) to use it
- add pyproject.toml, requirements.txt, dependabot.yml to config repo
- switch to ``furo`` doc theme
- changed: compute grad of energy wrt desc in batch mode (NN calculator)
- fix: set `fingerprints_filename` and load descriptor state dict when reuse fingerprints
  (NN calculator)


v0.3.0 (2021/08/03)
===================

- change license to LGPL
- set default optimizer
- put ``kimpy`` code in ``try except`` block
- add ``state_dict`` for descriptors and save it together with model
- change to use ``loguru`` for logging and allows user to set log level


v0.2.2 (2021/05/24)
===================

- update to be compatible with ``kimpy v2.0.0``


v0.2.1 (2021/05/24)
===================

- update to be compatible with ``kimpy v2.0.0``
- use entry ``entry_points`` to handle command line tool
- rename ``utils`` to ``devtool``


v0.2.0 (2021/01/19)
===================

- add type hint for all codes
- reorganize model and parameters to make it more robust
- add more docstring for many undocumented class and functions


v0.1.7 (2020/12/20)
===================

- add GitHub actions to automatically deploy to PyPI
- add a simple example to README


v0.1.5 (2020/2/13)
==================

- add neighborlist utility, making NN model independent on kimpy
- add calculator to deal with multiple species for NN model
- update dropout layer to be compatible with the pytorch 1.3


v0.1.4 (2019/8/24)
==================

- add support for the geodesic Levenberg-Marquardt minimization algorithm
- add command line tool ``model`` to inquire available parameters of KIM model


v0.1.3 (2019/8/19)
==================

- add RMSE and Fisher information analyzers
- allow configuration weight for ML models
- add write optimizer state dictionary for ML models
- combine functions ``generate_training_fingerprints()`` and
  ``generate_test_fingerprints()`` of descriptor to ``generate_fingerprints()``
  (supporting passing mean and stdev file)
- rewrite symmetry descriptors to share with KIM driver


v0.1.2 (2019/6/27)
==================

- MPI parallelization for physics-based models
- reorganize machine learning related files
- various bug fixes
- API changes
  * class ``DataSet`` renamed to ``Dataset``
  * class ``Calculator`` moved to module ``calculators`` from module ``calculator``


v0.1.1 (2019/5/13)
==================

- KLIFF available from PyPI now. Using ``$pip install kliff`` to install.
- Use SW model from the KIM website in tutorial.
- Format code with ``black``.


v0.1.0 (2019/3/29)
==================
First official release, but API is not guaranteed to be stable.

- Add more docs to :ref:`reference`.


v0.0.1 (2019/1/1)
=================
Pre-release.
