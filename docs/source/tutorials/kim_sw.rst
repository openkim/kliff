.. _tut_kim_sw:

==================================
Train a Stillinger-Weber potential
==================================

In this tutorial, we train a Stillinger-Weber (SW) potential for silicon that is
archived on OpenKIM_. Before getting started to train the SW model, let's first
install the model driver and the parameterized model::

    $ git clone https://github.com/mjwen/Three_Body_Stillinger_Weber__MD_335816936951_003.git
    $ git clone https://github.com/mjwen/Three_Body_Stillinger_Weber_Si__MO_405512056662_004.git
    $ kim-api-v2-collections-management install user ./Three_Body_Stillinger_Weber__MD_335816936951_003
    $ kim-api-v2-collections-management install user ./Three_Body_Stillinger_Weber_Si__MO_405512056662_004

We are going to create potentials for diamond silicon, and fit the potentials
to a training set of energies and forces consisting of compressed and stretched
diamond silicon structures, as well as configurations drawn from molecular dynamics
trajectories at different temperatures.
Download the training set :download:`Si_training_set.tar.gz <https://raw.githubusercontent.com/mjwen/kliff/pytorch/examples/Si_training_set.tar.gz>`
and extract the tarball: ``$ tar xzf Si_training_set.tar.gz``.
If you prefer, you can put the following snippet in your python code to download and
extract the training set automatically:

.. code-block:: python

    import requests
    import tarfile

    tarball_name = 'Si_training_set.tar.gz'
    url = 'https://raw.githubusercontent.com/mjwen/kliff/pytorch/examples/{}'.format(tarball_name)
    r = requests.get(url)
    with open(tarball_name, 'wb') as f:
        f.write(r.content)
    tarball = tarfile.open(tarball_name)
    tarball.extractall()

The data is stored in **extended xyz** format, and see TODO for an introduction
of this format.

.. warning::
    The ``Si_training_set`` is just a toy data set for the purpose to demonstrate
    how to use KLIFF to train potentials. It should not be used to train any
    potential for real simulations.


Model
-----

We first create a KIM model for the SW potential, and print out all the
available parameters that can be optimized (we call this ``model parameters``):
TODO(maybe add a glossary)

.. code-block:: python

    from kliff.models import KIM

    model = KIM(model_name='Three_Body_Stillinger_Weber_Si__MO_405512056662_004')
    model.echo_model_params()

You wll see something like::

    #================================================================================
    # Available parameters to optimize.

    # Model: Three_Body_Stillinger_Weber_Si__MO_405512056662_004
    #================================================================================

    name: A
    value: [15.28484792]
    size: 1
    dtype: Double
    description: Upper-triangular matrix (of size N=1) in row-major storage.  Ordering is according to SpeciesCode values.  For example, to find the parameter related to SpeciesCode 'i' and SpeciesCode 'j' (i <= j), use (zero-based) index = (i*N + j - (i*i + i)/2).

    name: B
    value: [0.60222456]
    size: 1
    dtype: Double
    description: Upper-triangular matrix (of size N=1) in row-major storage.  Ordering is according to SpeciesCode values.  For example, to find the parameter related to SpeciesCode 'i' and SpeciesCode 'j' (i <= j), use (zero-based) index = (i*N + j - (i*i + i)/2).
    .
    .
    .
    name: cutoff
    value: [3.77118]
    size: 1
    dtype: Double
    description: Upper-triangular matrix (of size N=1) in row-major storage.  Ordering is according to SpeciesCode values.  For example, to find the parameter related to SpeciesCode 'i' and SpeciesCode 'j' (i <= j), use (zero-based) index = (i*N + j - (i*i + i)/2).

which gives the ``name``, ``value``, ``size``, ``data type`` and a ``description``
of each parameter.


.. note::
    You can provide a ``path`` argument to the method ``echo_model_params(path)``
    to write the available parameters information to a file indicated by ``path``.

.. note::
    The available parameters information can also by obtained using the **kliff**
    :ref:`cmdlntool`: ``$ kliff model --echo-params Three_Body_Stillinger_Weber_Si__MO_405512056662_004``


Now that you know what parameters are available for fitting, you can optimize all
or a subset of them to reproduce the training set.

.. code-block:: python

    # fitting parameters
    model.set_fitting_params(
        gamma=[[1.5]],
        B=[['default']],
        sigma=[[2.0951, 'fix']],
        A=[[5.0, 1., 20]])

    # print fitting parameters
    model.echo_fitting_params()

Here, we tell KLIFF to fit four parameters ``B``, ``gamma``, ``sigma``, and ``A``
of the SW model. The information for each fitting parameter should be provided as
a list of list, where the size of the outer list should be equal to the ``size`` of
the parameter given by ``model.echo_model_params()``. For each inner list, you can
provide either one, two, or three items.

- One item. You can use a numerical value (e.g. ``gamma``) to provide an initial
  guess of the parameter. Alternatively, the string ``'default'`` can be provided to
  use the default value in the model (e.g. ``B``).
- Two items. The first item should be a numerical value and the second item should
  be the string ``'fix'`` (e.g. ``sigma``), which tells KLIFF to use the value for
  the parameter, but do not optimize it.
- Three items. The first item can be a numerical value or the string ``'default'``,
  having the same meanings as the one item case. In the second and third items, you
  can list the lower and upper bounds for the parameters, respectively. A bound
  could be provided as a numerical values or ``None``. The latter indicates no bound
  is applied.


The call of ``model.echo_fitting_params()`` will print out the fitting parameters
that you require KLIFF to optimize::

    #================================================================================
    # Model parameters that are optimized.
    #================================================================================

    A 1
      5.0000000000000000e+00   1.0000000000000000e+00   2.0000000000000000e+01

    B 1
      6.0222455840000000e-01

    sigma 1
      2.0951000000000000e+00 fix

    gamma 1
      1.5000000000000000e+00

where the number ``1`` after the name of each parameter indicates the size of the
parameter.


.. note::
    The parameters that are not included as a fitting parameter are fixed to the
    default values in the model during the optimization.


Training set
------------

KLIFF has a :class:`DataSet` to deal with the training data (and possibly test
data). For the silicon training set, we can read and process the files by:

.. code-block:: python

    from kliff.dataset import DataSet

    dataset_name = 'Si_training_set'
    tset = DataSet()
    tset.read(dataset_name)
    configs = tset.get_configs()

The ``configs`` in the last line is a list of :class:`Configuration`. Each
configuration is an internal representation of a processed **extended xyz** file,
hosting the species, coordinates, energy, forces, and other related information of
a system of atoms.


Calculator
----------

:class:`Calculator` is the central agent that exchanges information and orchestrate
the operation of the fitting process. It calls the model to compute the energy and
forces and provide this information to the `Loss function`_ (discussed below) to
compute the loss. It also grabs the parameters from the optimizer and update the
parameters stored in the model so that the up-to-date parameters are used the next
time the model is evaluated to compute the energy and forces. The calculator can be
created by:

.. code-block:: python

    from kliff.calculator import Calculator

    calc = Calculator(model)
    calc.create(configs)

where ``calc.create(configs)`` does some initializations for each each configuration
in the training set, such as creating the neighbor list.


Loss function
-------------

KLIFF uses a loss function to quantify the difference between the training set data
and potential predictions and uses minimization algorithms to reduce the loss as
much as possible. KLIFF provides a large number of minimization algorithms by
interacting with SciPy_. For physics-motivated potentials, any algorithm listed on
`scipy.optimize.minimize`_ and `scipy.optimize.least_squares`_ can be used.
In the following code snippet, we create a loss function that uses the ``L-BFGS-B``
minimization algorithm. The minimization will run on 1 processor and a max number of
100 iterations are allowed.

.. code-block:: python

    from kliff.loss import Loss

    steps = 100
    loss = Loss(calc, nprocs=1)
    loss.minimize(method='L-BFGS-B', options={'disp': True, 'maxiter': steps})


You will see something like::

    RUNNING THE L-BFGS-B CODE

               * * *

    Machine precision = 2.220D-16
     N =            3     M =           10

    At X0         0 variables are exactly at the bounds

    At iterate    0    f=  1.65618D+07    |proj g|=  1.63611D+07

    At iterate    1    f=  4.50459D+06    |proj g|=  7.90884D+06
    .
    .
    .
    At iterate   25    f=  3.25435D+03    |proj g|=  1.16308D+02

    At iterate   26    f=  3.25435D+03    |proj g|=  3.06113D+00

    At iterate   27    f=  3.25435D+03    |proj g|=  6.61066D-01

               * * *

    Tit   = total number of iterations
    Tnf   = total number of function evaluations
    Tnint = total number of segments explored during Cauchy searches
    Skip  = number of BFGS updates skipped
    Nact  = number of active bounds at final generalized Cauchy point
    Projg = norm of the final projected gradient
    F     = final function value

               * * *

       N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
        3     27     36     28     0     0   6.611D-01   3.254D+03
      F =   3254.3480974009767

    CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH

     Cauchy                time 0.000E+00 seconds.
     Subspace minimization time 0.000E+00 seconds.
     Line search           time 0.000E+00 seconds.

The minimization stops after running for 27 steps.


**Save trained model**

After the minimization, we'd better save the model, which can be loaded later for
the purpose to do a retraining or evaluations. If satisfied with the fitted model,
you can also write it as a KIM model that can be used with LAMMPS_, GULP_, ASE_,
etc. via the kim-api_.

.. code-block:: python

    model.echo_fitting_params()
    model.save('kliff_model.pkl')
    model.write_kim_model()

The first line of the above code will print out::

    #================================================================================
    # Model parameters that are optimized.
    #================================================================================

    A 1
      1.5008554501462323e+01   1.0000000000000000e+00   2.0000000000000000e+01

    B 1
      5.9537800948866415e-01

    sigma 1
      2.0951000000000000e+00 fix

    gamma 1
      2.4122637121188939e+00

A comparison with the original parameters before carrying out the minimization
shows that we recover the original parameters quite reasonably. The second line
saves the fitted model to a file named ``kliff_model.pkl`` on the disk, and the
third line writes out a KIM potential named
``Three_Body_Stillinger_Weber_Si__MO_405512056662_004_kliff_trained``.

.. seealso::
    For information about how to load a saved model, see :mod:`~doc.modules`.


Putting them all together, we have

.. code-block:: python

    import kliff
    from kliff.dataset import DataSet
    from kliff.models import KIM
    from kliff.calculator import Calculator
    from kliff.loss import Loss


    # setting logger info
    kliff.logger.set_level('info')


    # create a KIM model
    model = KIM(model_name='Three_Body_Stillinger_Weber_Si__MO_405512056662_004')

    # print parameters that are available for fitting
    model.echo_model_params()

    # fitting parameters
    model.set_fitting_params(
        A=[[5.0, 1., 20]],
        B=[['default']],
        sigma=[[2.0951, 'fix']],
        gamma=[[1.5]])

    # print fitting parameters
    model.echo_fitting_params()


    # training set
    dataset_name = 'Si_training_set'
    tset = DataSet()
    tset.read(dataset_name)
    configs = tset.get_configs()


    # calculator
    calc = Calculator(model)
    calc.create(configs)


    # loss
    steps = 100
    loss = Loss(calc, nprocs=1)
    loss.minimize(method='L-BFGS-B', options={'disp': True, 'maxiter': steps})


    # print optimized parameters
    model.echo_fitting_params()

    # save model for later retraining
    model.save('kliff_model.pkl')

    # write KIM model
    model.write_kim_model()



.. _OpenKIM: https://openkim.org
.. _SciPy: https://scipy.org
.. _scipy.optimize.minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
.. _scipy.optimize.least_squares: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
.. _kim-api: https://openkim.org/kim-api/
.. _LAMMPS: https://lammps.sandia.gov
.. _GULP: http://gulp.curtin.edu.au/gulp/
.. _ASE: https://wiki.fysik.dtu.dk/ase/
