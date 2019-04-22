.. _doc.loss:

====
Loss
====

As discussed in :ref:`theory`, we solve a minimization problem to fit the potential.
For physics-motivated potentials, the geodesic Levenberg-Marquardt (``geodesicLM``)
minimization method (TODO, add link to Mark's stuff) can be used, which has been
shown to perform well for potentials in [wen2016potfit]_. KLIFF also interacts
with SciPy_ to utilize the zoo of optimization methods there.
For machine learning potentials, KLIFF wraps the optimization methods in PyTorch_.

KLIFF provides a uniform interface to use all the optimization methods.
To carry out optimization, first create a loss object:

.. code-block:: python

    from kliff.loss import Loss

    calculator = ...  # create a calculator
    Loss(calculator, nprocs=1, residual_fn=None, residual_data=None)

``calculator`` (discussed in :ref:`doc.calculators`) provides predictions
calculated using a potential and the corresponding reference data via
:meth:`~kliff.calculator.Calculator.get_prediction()` and
:meth:`~kliff.calculator.Calculator.get_reference()`, respectively, which the
optimizer can be used to construct the objective function.

``nprocs`` informs KLIFF the number of cores that KLIFF can use to parallelize
over the dataset to evaluate the objective function.

``residual_data`` is a dictionary that will be used by ``residual_fn`` to compute
the residual. ``residual_data`` is optional, and its default is:

.. code-block:: python

    residual_data = {'energy_weight': 1.0,
                     'forces_weight': 1.0,
                     'stress_weight': 1.0,
                     'normalize_by_natoms': True}

The meanings of these values are made clear in the below discussion.


``residual_fn`` is a function used to compute the residual.
As discussed in :ref:`theory`, the objective function is a sum of the square
of the norm of the residual of each individual configuration, i.e.

.. math::
    \mathcal{L(\bm\theta)} = \frac{1}{2} \sum_{i=1}^{N_p}
    w_i \|\bm u_i\|^2

with the residual

.. math::
    \bm u_i = \bm p_i - \bm q_i ,

in which :math:`\bm p_i` is a vector of predictions computed using the potential
for the :math:`i`-th configuration, and :math:`\bm q_i` is a vector of the
corresponding reference data.
The residual is computed using the ``residual_fn``, which should be of the form

.. code-block:: python

    def residual_fn(identifier, natoms, prediction, reference, data):
        """ A function to compute the residual for a configuration.
        """

         u =  #... compute u based on p (prediction) and q (reference)
              # and it should be a vector
        return u

In the above residual function,

- ``identifier`` is a (unique) ``str`` associated with the configuration, which
  is specified in :class:`~kliff.dataset.Configuration`. If it is not provided
  there, ``identifier`` is default to the path to the file that storing the
  configuration, e.g. ``Si_training_set/NVT_runs/T300_step100.xyz``.
- ``natoms`` is an ``int`` denoting the number of atoms in the configuration.
- ``prediction`` is a vector of the prediction :math:`\bm p` computed from the
  potential.
- ``reference`` is a vector of the corresponding reference data :math:`\bm q`.
- ``data`` is ``residual_data`` provided at the initialization of ``Loss``.
  ``residual_data`` is a dictionary, with which the user can provide extra
  information to ``residual_fn``.

``residual_fn`` is also optional, and it defaults to :func:`~kliff.loss.energy_forces_residual`
discussed below.


Built-in residual function
==========================
KLIFF provides a number of residual functions readily to be plugged into ``Loss``
and let the wheel spin. For example, the :func:`~kliff.loss.energy_forces_residual`
that construct the residual using energy and forces is defined as (in a nutshell):

.. code-block:: python

    def energy_forces_residual(identifier, natoms, prediction, reference, data):

        # prepare weight based on user provided data
        energy_weight = data['energy_weight']
        forces_weight = data['forces_weight']
        normalize_by_natoms  = data['normalize_by_natoms']
        if energy_weight is None:
            energy_weight = 1.
        if forces_weight is None:
            forces_weight = 1.
        if normalize_by_natoms:
            energy_weight /= natoms
            forces_weight /= natoms

        # obtain residual and properly normalize it
        residual = prediction - reference
        residual[0] *= energy_weight
        residual[1:] *= forces_weight

        return residual

This residual function can weigh ``energy`` and ``forces`` differently, and
enables the normalization of the residual based on the number of atoms.
Normalization by the number of atoms makes each individual configuration in the
training set contributes equally to the loss function; otherwise, configurations
with more atoms will dominate the loss, which (most of the times) is not what we
prefer.

.. note::
    We take the square root of ``energy_weight`` and ``forces_weight`` in
    ``energy_forces_residual``. With this, the final loss is proportional to the
    number of atoms instead of the square of the number of atoms as can be seen
    in the definition of :math:`\mathcal{L(\bm\theta)}`.


One can provide a ``residual_data`` instead of using the default one to control
tune the loss. In the below example, the `energy` is weighted 10 times as the
`forces`.

.. code-block:: python

    from kliff.loss import Loss
    from kliff.loss import energy_forces_residual

    calculator = ...  # create a calculator

    # provide my data
    residual_data = {'energy_weight': 10.0,
                     'forces_weight': 1.0,
                     'normalize_by_natoms': True}
    Loss(calculator, nprocs=1, residual_fn=energy_forces_residual, residual_data=residual_data)


.. warning::
    Even though ``residual_fn`` and ``residual_data`` is optional, we strongly
    recommend the users to explicitly provide them to remainder themselves what
    they are doing as done above.

.. seealso::
    See :mod:`kliff.loss` for other built-in residual functions.


Use your own residual function
==============================

The built-in residual functions treat each configuration in the training set, and
each atom in a configuration equally important. Sometimes, this may not be what
you want. In these cases, you can define and use your own ``residual_fn``.

For example, if you are creating a potential that is going to be used
to investigate fracture properties, and your training set include both
configurations with cracks and configurations without cracks, then you may want to
weigh more for the configurations with cracks.

.. code-block:: python

    from kliff.loss import Loss

    # define my own residual function
    def residual_fn(identifier, natoms, prediction, reference, data):

        energy_weight = 1./natoms
        forces_weight = 1./natoms

        if 'with_cracks' in identifer:
            energy_weight *= 10
            forces_weight *= 10

        # such that the loss is proportional to atoms but not natoms^2
        energy_weight = energy_weight**0.5
        forces_weight = forces_weight**0.5

        # obtain residual and properly normalize it
        residual = prediction - reference
        residual[0] *= energy_weight
        residual[1:] *= forces_weight

        return residual


    calculator = ...  # create a calculator
    Loss(calculator, nprocs=1, residual_fn=residual_fn)


The above code takes advantage of ``identifier`` to distinguish configurations with
cracks and without cracks, and then weigh more for configurations with cracks.

For configurations with cracks, you may even want to weigh more for the atoms near
the creak tip. Then you need to identify which atoms are near the crack tip
and manipulate the corresponding components of ``residual``.


.. note::
    If you are using your own ``residual_fn``, its ``data`` argument can be completed
    ignored since it can be directly provided in your own ``residual_fn``.



.. _PyTorch: https://pytorch.org
.. _SciPy: https://scipy.org
.. _scipy.optimize.minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
.. _scipy.optimize.least_squares: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares

.. [wen2016potfit] Wen, M., Li, J., Brommer, P., Elliott, R.S., Sethna, J.P. and
   Tadmor, E.B., 2016. A KIM-compliant potfit for fitting sloppy interatomic
   potentials: application to the EDIP model for silicon. Modelling and Simulation in
   Materials Science and Engineering, 25(1), p.014001.
