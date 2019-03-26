.. module:: doc.dataset

=======
Dataset
=======


Extended XYZ format
===================

The Extended XYZ format is an enhanced version of the `basic XYZ format`_ that
allows extra columns to be present in the file for additional per-atom properties as
well as standardizing the format of the comment line to include the cell lattice
and other per-frame parameters.

Below is an example of the extended XYZ format supported by KLIFF::

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
    numbers denote :math:`a_1`, the next three numbers denote :math:`a_2`, and the
    last 3 numbers denote :math:`a_3`. Note that :math:`a_1`, :math:`a_2`, and
    :math:`a_3` should follow the right-hand rule such that the volume of the cell
    can be obtained by :math:`(a_1\times a_2)\cdot a_3`.
  - ``PBC``. Three integers of ``1`` or ``0`` (or three characters of ``T`` or ``F``)
    to indicate whether to use periodic boundary conditions along :math:`a_1`,
    :math:`a_2`, and :math:`a_3`, respectively.
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

