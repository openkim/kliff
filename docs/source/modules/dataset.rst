.. _doc.dataset:

=======
Dataset
=======

This Module contains classes and functions to deal with dataset.

A dataset is comprised of a set of configurations, which provide the training data
to optimize potential (parameters) or provide the test data to test the quality of
potential.

A configuration should have three ``lattice vectors`` of the simulation cell,
flags to indicate periodic boundary conditions (``PBC``), ``species`` and
``coordinates`` of all atoms. These collectively define a configuration and are,
typically, considered as the input in terms of potential fitting.
A configuration should also contain a set of output (target), which the
optimization algorithm adjust the potential (parameters) to match.
For example, if the force-matching scheme is used for the fitting, the output can be
the forces on individual atoms. The currently supported outputs include
``energy``, ``forces``, and ``stress``.

.. seealso::
    See :class:`kliff.dataset.Configuration` for a complete list of the member
    functions of the `Configuration` class.

To create a data, do:

.. code-block:: python

    from kliff.dataset import DataSet
    path = 'path_to_my_dataset_files'
    dset = Dataset()
    dset.read(path, format='extxyz')

where ``path`` is a file storing a configuration or a directory containing multiple
files. If given a directory, all the files in this directory and its subdirectories
with the extension corresponding to the specified format will be read. For
example, if ``format='extxyz'``, all the files with an extension ``.xyz`` in
``path`` and its subdirectories will be read.

The size of the dataset can be obtained by:

.. code-block:: python

    dset_size = dset.get_num_configs()

and a list of configurations constituting the dataset can be obtained by:

.. code-block:: python

    configs = dset.get_configs()

.. seealso::
    See :class:`kliff.dataset.DataSet` for a complete list of the member functions
    of the `DataSet` class.


Inspect dataset
===============
KLIFF provides a command line tool to get a statistics of a dataset of files.
For example, for the
:download:`Si_training_set.tar.gz <https://raw.githubusercontent.com/mjwen/kliff/pytorch/examples/Si_training_set.tar.gz>`
(the tarball can be extracted by: ``$ tar xzf Si_training_set.tar.gz``), running::

    $ kliff dataset --count Si_training_set

prints out the below information::

    ================================================================================
                                 KLIFF Dataset Count

    Notation: "──dir_name (a/b)"
    a: number of .xyz files in the directory "dir_name"
    b: number of .xyz files in the directory "dir_name" and its subdirectories

    Si_training_set (0/1000)
    ├──NVT_runs (600/600)
    └──varying_alat (400/400)

    ================================================================================



Dataset Format
==============

More than often, your dataset is generated from first-principles calculations
using packages like `VASP`, `SIESTA`, and `Quantum Espresso` among others. Their
output file format may not be support by KLIFF. You can use parse these output to
get the necessary data, and then convert to the format supported by KLIFF using the
functions :func:`kliff.dataset.write_config` and :func:`kliff.dataset.read_config`.


Currently supported dataset format include:

- extended XYZ (.xyz)


Extended XYZ
------------

The Extended XYZ format is an enhanced version of the `basic XYZ format`_ that
allows extra columns to be present in the file for additional per-atom properties as
well as standardizing the format of the comment line to include the cell lattice
and other per-frame parameters. It typically has the ``.xyz`` extension.

It would be easy to explain the format with an example. Below is an example of
the extended XYZ format supported by KLIFF::

    8
    Lattice="4.8879 0 0 0 4.8879 0 0 0 4.8879"  PBC="1 1 1"  Energy=-29.3692121943  Properties=species:S:1:pos:R:3:force:R:3
    Si    0.00000e+00   0.00000e+00   0.00000e+00  2.66454e-15  -8.32667e-17   4.02456e-16
    Si    2.44395e+00   2.44395e+00   0.00000e+00  1.62370e-15   7.21645e-16   8.46653e-16
    Si    0.00000e+00   2.44395e+00   2.44395e+00  0.00000e+00   3.60822e-16   2.01228e-16
    Si    2.44395e+00   0.00000e+00   2.44395e+00  1.33227e-15  -4.44089e-16   8.74350e-16
    Si    1.22198e+00   1.22198e+00   1.22198e+00  4.44089e-15   1.80411e-16   1.87350e-16
    Si    3.66593e+00   3.66593e+00   1.22198e+00  9.29812e-16  -2.67841e-15  -3.22659e-16
    Si    1.22198e+00   3.66593e+00   3.66593e+00  5.55112e-17   3.96905e-15   8.87786e-16
    Si    3.66593e+00   1.22198e+00   3.66593e+00 -2.60902e-15  -9.43690e-16   6.37999e-16


- The first line list the number of atoms in the system.
- The second line follow the ``key=value`` structure. if a ``value`` contains any
  space (e.g. ``Lattice``), it should be placed in the quotation marks ``" "``.
  The supported keys are:

  - ``Lattice`` represents the there Cartesian lattice vectors: the first 3
    numbers denote :math:`\bm a_1`, the next three numbers denote :math:`\bm a_2`,
    and the last 3 numbers denote :math:`\bm a_3`. Note that :math:`\bm a_1`,
    :math:`\bm a_2`, and :math:`\bm a_3` should follow the right-hand rule such that
    the volume of the cell can be obtained by :math:`(\bm a_1\times \bm a_2)\cdot \bm a_3`.
  - ``PBC``. Three integers of ``1`` or ``0`` (or three characters of ``T`` or ``F``)
    to indicate whether to use periodic boundary conditions along :math:`\bm a_1`,
    :math:`\bm a_2`, and :math:`$\bm a_3$`, respectively.
  - ``Energy``. A real value of the total potential energy of the system.
  - ``Properties`` provides information of the names, size, and types of the data
    that are listed in the body part of the file. For example, the ``Properties`` in
    the above example means that the atomic species information (a string) is listed
    in the first column of the body, the next three columns list the atomic
    coordinates, and the last three columns list the forces on atoms.

Each line in the body lists the information, indicated by ``Properties`` in the
second line, for one atom in the system, taking the form::

    species  x  y  z  fx  fy  fz

The coordinates ``x  y  z`` should be given in Cartesian values, not fractional
values.

.. note::
    An atomic configuration stored in the extended XYZ format can be visualized
    using the OVITO_ program.

.. _basic XYZ format: https://en.wikipedia.org/wiki/XYZ_file_format
.. _OVITO: http://ovito.org

