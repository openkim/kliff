.. kliff documentation master file, created by
   sphinx-quickstart on Fri Sep 16 18:18:16 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

KIM-based Learning-Integrated Fitting Framework (KLIFF)
=======================================================

KLIFF is the ``KIM-based Learning-Integrated Fitting Framework`` for interatomic
potentials.
It can be used to fit both physics-motivated potentials and machine learning
potentials like the nueral network potentials.
The trained potential can be deployed with the kim-api, which is supported by major
simulation codes such as **LAMMPS**, **ASE**, **DL_POLY**, and **GULP** among others.


.. toctree::
    :name: intro
    :maxdepth: 1

    about
    changelog


.. toctree::
    :caption: Contents
    :name: contents
    :titlesonly:

    installation
    tutorials/tutorials
    theory
    modules/modules
    command_line

.. toctree::
    :caption: Extra Information
    :name: extra
    :maxdepth: 1

    faq
    apidoc/kliff
    builddoc




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

