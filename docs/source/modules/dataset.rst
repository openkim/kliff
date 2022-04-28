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

    from kliff.dataset import Dataset
    path = 'path_to_my_dataset_files'
    dset = Dataset(path, format='extxyz')


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
    See :class:`kliff.dataset.Dataset` for a complete list of the member functions
    of the `Dataset` class.


Inspect dataset
===============
KLIFF provides a command line tool to get a statistics of a dataset of files.
For example, for the
:download:`Si_training_set.tar.gz <https://raw.githubusercontent.com/openkim/kliff/pytorch/examples/Si_training_set.tar.gz>`
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

  - ``Lattice`` represents the three Cartesian lattice vectors: the first 3
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
values. The forces ``fx fy fz`` can be skipped if you do not want to use them.

.. note::
    An atomic configuration stored in the extended XYZ format can be visualized
    using the OVITO_ program.

.. _basic XYZ format: https://en.wikipedia.org/wiki/XYZ_file_format
.. _OVITO: http://ovito.org


.. _doc.dataset.weight:

Weight
======

As mentioned in :ref:`theory`, the reference :math:`\bm q` can be any material
properties, which can carry different physical units. The weight in the loss function
can be used to put quantities with different units on a common scale. The weights also
give us access to set which properties or configurations are more important, for example,
in developing a potential for a certain application
(see :ref:`doc.dataset.weight.define_your_weight_class`).

KLIFF uses weight class to compute and store the weight information for each
configuration. The basic structure of the class is shown below.

.. code-block:: python

    class Weight():
	"""A class to deal with weights for each configuration."""

	def __init__(self):
	    #... Do necessary steps to initialize the class

        def compute_weight(self, config):
	    #... Compute the weights for the given configutation

	@property
	def some_weight(self):
	    #... Add properties to retrieve the weight values


Default weight class
---------------------

KLIFF has several built-in weight classes. As a default, KLIFF uses :class:`kliff.dataset.weight.Weight`,
which put a single weight for each property.

.. code-block:: python

    from kliff.dataset import Dataset
    from kliff.dataset.weight import  Weight

    path = 'path_to_my_dataset_files'
    weight = Weight()
    dset = Dataset(path, weight=weight, format='extxyz')

    # Retrieve the weights
    config_weight = configs[0].config_weight
    energy_weight = configs[0].energy_weight
    forces_weight = configs[0].forces_weight
    stress_weight = configs[0].stress_weight

``config_weight`` is the weight for the configuration and ``energy_weight``,
``forces_weight``, and ``stress_weigth`` are the weights for energy, forces, and stress,
respectively. The default value for each weight is 1.0.

One can also specify different values for these weights. For example, one might want to
weigh the energy 10 times as the forces. It can be done by specifying the weight values
while instantiating :class:`kliff.dataset.weight.Weight`.

.. code-block:: python

    weight = Weight(
        config_weight=1.0, energy_weight=10.0, forces_weight=1.0, stress_weight=1.0
    )

.. note::
    Another use case is if one wants to, for example, exclude the energy in the loss
    function, which can be done by setting ``energy_weight=0.0``.


Magnitude-inverse weight
------------------------

KLIFF also provides another weight class that computes the weight based on the magnitude
of the data, applying different weight on each data point. The weight calculation is
motivated by formulation suggested by Lenosky et al. [lenosky1997]_,

.. math::

    \frac{1}{w_i}^2 = c_1^2 + c_2^2 \| \bm p_i \|^2

:math:`c_1` and :math:`c_2` are parameters to compute the weight. They can be thought as
a padding and a fractional scaling terms. When :math:`\bm p_i` corresponds to energy,
the norm is the absolute value of the energy. When :math:`\bm p_i` correspond to forces,
the norm is a vector norm of the force vector acting on the corresponding atom. This also
mean that each force component acting on the same atom will have the same weight. If
:math:`\bm p_i` correspond to stress, then the norm is a Frobenius norm of the stress
tensor, giving the same weight for each component in the stress tensor.

To use this weight, we instantiate :class:`~kliff.dataset.weight.MagnitudeInverseWeight`
weight class:

.. code-block:: python

    from kliff.dataset.weight import MagnitudeInverseWeight
    weight = MagnitudeInverseWeight(
        config_weight=1.0,
	weight_params={
            "energy_weight_params": [c1e, c2e],
            "forces_weight_params": [c1f, c2f],
            "stress_weight_params": [c1s, c2s],
	}
    )

``config_weight`` specifies the weight for the entire configuration.

``weight_params`` is a dictionary containing :math:`c_1` and :math:`c_2` for energy,
forces, and stress. The default value is:

.. code-block:: python

    weight_params = {
	"energy_weight_params": [1.0, 0.0],
	"forces_weight_params": [1.0, 0.0],
	"stress_weight_params": [1.0, 0.0],
    }

Additionally, for each key, we can pass in a ``float``,  which set the value of
:math:`c_1` with :math:`c_2=0.0`.

.. [lenosky1997]
   Lenosky, T.J., Kress, J.D., Kwon, I., Voter, A.F., Edwards, B., Richards, D.F., Yang,
   S., Adams, J.B., 1997. Highly optimized tight-binding model of silicon. Phys. Rev. B
   55, 15281544. https://doi.org/10.1103/PhysRevB.55.1528


.. _doc.dataset.weight.define_your_weight_class:

Define your weight class
------------------------

We can also define a custom weight class to use in KLIFF. As an example, suppose we are
developing a potential that will be used to investigate fracture properties. The training
sets includes both configurations with and without cracks. For this application, we might
want to put larger weights for the configurations with cracks. Below is an example of
weight class that achieve this goal.

.. code-block:: python

    from kliff.dataset.weight import Weight

    class WeightForCracks(Weight):
        """An example weight class that put larger weight on the configurations with
	cracks. This class inherit from ``kliff.dataset.weight.Weight``. We just need to
	modify ``compute_weight`` method to put larger weight for the configurations with
	cracks. Other modifications might need to be done for different weight class.
	"""

	def __init__(self, energy_weight, forces_weight):
            super().__init__(energy_weight=energy_weight, forces_weight=forces_weight)

	def compute_weight(self, config):
	    identifier = config.identifer
	    if 'with_cracks' in identifier:
		self._config_weight = 10.0

With this weight class, we can use the built-in ``residual_fn`` to achieve the same
result as the implementation in :ref:`doc.loss.use_your_own_residual_function`.
