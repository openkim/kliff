KLIFF -- KIM-based Learning-Integrated Fitting Framework
========================================================

KLIFF is an interatomic potential fitting package that can be used to fit both
physics-motivated potentials (e.g. the Stillinger-Weber potential) and machine learning
potentials (e.g. neural network potential).
The trained potential can be deployed with the KIM-API_, which is supported by major
simulation codes such as LAMMPS_, ASE_, DL_POLY_, and GULP_ among others.

.. _KIM-API: https://openkim.org/kim-api/
.. _LAMMPS: https://lammps.sandia.gov/
.. _ASE: https://wiki.fysik.dtu.dk/ase/
.. _DL_POLY: https://www.scd.stfc.ac.uk/Pages/DL_POLY.aspx/
.. _GULP: http://gulp.curtin.edu.au/gulp/


.. toctree::
    :maxdepth: 1

.. toctree::
    :caption: The Basics
    :maxdepth: 2

    installation
    introduction
    theory


.. toctree::
    :caption: Advanced Topics
    :maxdepth: 2

    advanced/lammps
    howto/howto
    command_line
    contributing_guide

.. toctree::
    :caption: UQ and Legacy Modules
    :maxdepth: 2

    legacy
    tutorials

.. toctree::
    :caption: Extra Information
    :maxdepth: 1

    changelog
    faq
    apidoc/kliff
    GitHub Repository <https://github.com/openkim/kliff>

If you find KLIFF useful in your research, please cite:

.. code-block:: text

   @Article{wen2022kliff,
     title   = {{KLIFF}: A framework to develop physics-based and machine learning interatomic potentials},
     author  = {Mingjian Wen and Yaser Afshar and Ryan S. Elliott and Ellad B. Tadmor},
     journal = {Computer Physics Communications},
     volume  = {272},
     pages   = {108218},
     year    = {2022},
     doi     = {10.1016/j.cpc.2021.108218},
    }




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
