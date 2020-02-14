.. _changelog:

==========
Change Log
==========

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
