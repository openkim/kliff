.. _tutorials:

=========
Tutorials
=========

This section provides two example to use KLIFF to train a physics-motivated
Stillinger-Weber potential for silicon, and a neural network potential for graphite.
Before getting started, make sure that KLIFF is successfully installed as discussed
in :ref:`installation`.


Physics-motivated potential
===========================

In this example, we work with the Stillinger-Weber (SW) potential for silicon that is
archived on OpenKIM_. Before getting started to train the SW model,
let's first install the model driver and the parameterized model::

    $ git clone https://github.com/mjwen/Three_Body_Stillinger_Weber__MD_335816936951_003.git
    $ git clone https://github.com/mjwen/Three_Body_Stillinger_Weber_Si__MO_405512056662_004.git
    $ kim-api-v2-collections-management install user ./Three_Body_Stillinger_Weber__MD_335816936951_003
    $ kim-api-v2-collections-management install user ./Three_Body_Stillinger_Weber_Si__MO_405512056662_004

Then download the :download:`Si training set <https://raw.githubusercontent.com/mjwen/kliff/pytorch/examples/Si_training_set.tar.gz>`
and extract the tarball by: ``$ tar xzf Si_training_set.tar.gz``.
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

The training set consists of energies and forces of compressed and stretched
diamond silicon structures, as well as configurations drawn from molecular dynamics
trajectories at different temperatures.
The data is stored in **extended xyz** format, and see TODO for an introduction
of this format.


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

- One item. You can use a numerical value (e.g. ``gamma``) to provide the initial
  guess of the value. The string ``'default'`` can be provided to use the default
  value in the model (e.g. ``B``).
- Two items. The first item should be a numerical value and the second item should
  be the string ``'fix'`` (e.g. ``sigma``), which tells KLIFF to use the value for
  the parameter, but do not optimize it.
- Three items. The first item can be a numerical value or the string ``'default'``,
  having the same meanings as the one item case. In the second and third items, you
  can list the lower and upper bounds for the parameters, respectively. A bound
  could be provided as a numerical values or ``None``. The latter indicates no bound
  is applied.


.. note::
    The parameters that are not included as a fitting parameter are fixed to the
    default values in the mdoel during the optimization.








Training set
------------



Calculator
----------

Loss function
-------------


.. code-block:: python

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
    configs = tset.get_configurations()


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

    # create a kim model
    model.write_kim_model()













Neural network potential
========================




.. _OpenKIM: https:openkim.org
