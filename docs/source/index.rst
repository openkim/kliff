.. kliff documentation master file, created by
   sphinx-quickstart on Fri Sep 16 18:18:16 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

KLIFF -- KIM-based Learning-Integrated Fitting Framework
========================================================

KLIFF is an interatomic potential fitting package that can be used to fit both
physics-motivated potentials (e.g. the Stillinger-Weber potential) and machine learning
potentials (e.g. neural network potential).
The trained potential can be deployed with the **kim-api**, which is supported by major
simulation codes such as **LAMMPS**, **ASE**, **DL_POLY**, and **GULP** among others.



.. toctree::
    :maxdepth: 1

    changelog


.. toctree::
    :caption: The Basics
    :maxdepth: 2

    installation
    tutorials
    theory
    modules/modules


.. toctree::
    :caption: Advanced Topics
    :maxdepth: 2

    howto/howto
    command_line
    contributing_guide


.. toctree::
    :caption: Extra Information
    :maxdepth: 2

    faq
    apidoc/kliff
    Fork KLIFF on GitHub <https://github.com/mjwen/kliff>




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

