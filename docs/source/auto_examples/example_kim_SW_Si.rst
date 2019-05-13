.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_example_kim_SW_Si.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_example_kim_SW_Si.py:


.. _tut_kim_sw:

Train a Stillinger-Weber potential
==================================

In this tutorial, we train a Stillinger-Weber (SW) potential for silicon that is archived
on OpenKIM_.
Before getting started to train the SW model, let's first install the SW # model::

   $ kim-api-collections-management install user SW_StillingerWeber_1985_Si__MO_405512056662_005

.. seealso::
   This installs the model and its driver into the ``User Collection``. See
   :ref:`install_model` for more information about installing KIM models.

We are going to create potentials for diamond silicon, and fit the potentials to a
training set of energies and forces consisting of compressed and stretched diamond
silicon structures, as well as configurations drawn from molecular dynamics trajectories
at different temperatures.
Download the training set :download:`Si_training_set.tar.gz
<https://raw.githubusercontent.com/mjwen/kliff/pytorch/examples/Si_training_set.tar.gz>`
and extract the tarball: ``$ tar xzf Si_training_set.tar.gz``. The data is stored in
**extended xyz** format, and see :ref:`doc.dataset` for more information of this format.

.. warning::
   The ``Si_training_set`` is just a toy data set for the purpose to demonstrate how to
   use KLIFF to train potentials. It should not be used to train any potential for real
   simulations.

Let's first import the modules that will be used in this example.


.. code-block:: default


    from kliff.models import KIM
    from kliff.loss import Loss
    from kliff.calculator import Calculator
    from kliff.dataset import DataSet








Model
-----

We first create a KIM model for the SW potential, and print out all the available
parameters that can be optimized (we call this ``model parameters``).


.. code-block:: default


    model = KIM(model_name='SW_StillingerWeber_1985_Si__MO_405512056662_005')
    model.echo_model_params()






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    #================================================================================
    # Available parameters to optimize.

    # Model: SW_StillingerWeber_1985_Si__MO_405512056662_005
    #================================================================================

    name: A
    value: [15.28484792]
    size: 1
    dtype: Double
    description: Multiplicative factors on the two-body energy function as a whole for each binary species combination. In terms of the original SW parameters, each quantity is equal to A*epsilon for the corresponding species combination. This array corresponds to a lower-triangular matrix (of size N=1) in row-major storage. Ordering is according to SpeciesCode values. For example, to find the parameter related to SpeciesCode 'i' and SpeciesCode 'j' (i >= j), use (zero-based) index = (j*N + i - (j*j + j)/2).

    name: B
    value: [0.60222456]
    size: 1
    dtype: Double
    description: Multiplicative factors on the repulsive term in the two-body energy function for each binary species combination. This array corresponds to a lower-triangular matrix (of size N=1) in row-major storage. Ordering is according to SpeciesCode values. For example, to find the parameter related to SpeciesCode 'i' and SpeciesCode 'j' (i >= j), use (zero-based) index = (j*N + i - (j*j + j)/2).

    name: p
    value: [4.]
    size: 1
    dtype: Double
    description: The exponent of the repulsive term in the two-body energy function is equal to the negative of this parameter. This array corresponds to a lower-triangular matrix (of size N=1) in row-major storage. Ordering is according to SpeciesCode values. For example, to find the parameter related to SpeciesCode 'i' and SpeciesCode 'j' (i >= j), use (zero-based) index = (j*N + i - (j*j + j)/2).

    name: q
    value: [0.]
    size: 1
    dtype: Double
    description: The exponent of the attractive term in the two-body energy function is equal to the negative of this parameter. This array corresponds to a lower-triangular matrix (of size N=1) in row-major storage. Ordering is according to SpeciesCode values. For example, to find the parameter related to SpeciesCode 'i' and SpeciesCode 'j' (i >= j), use (zero-based) index = (j*N + i - (j*j + j)/2).

    name: sigma
    value: [2.0951]
    size: 1
    dtype: Double
    description: Length normalization factors used in the two-body energy function for each binary species combination. This array corresponds to a lower-triangular matrix (of size N=1) in row-major storage. Ordering is according to SpeciesCode values. For example, to find the parameter related to SpeciesCode 'i' and SpeciesCode 'j' (i >= j), use (zero-based) index = (j*N + i - (j*j + j)/2).

    name: gamma
    value: [2.51412]
    size: 1
    dtype: Double
    description: Length normalization factors used in the three-body energy function for each binary species combination. In terms of the original SW parameters, each quantity is equal to gamma*sigma for the corresponding species combination. This array corresponds to a lower-triangular matrix (of size N=1) in row-major storage. Ordering is according to SpeciesCode values. For example, to find the parameter related to SpeciesCode 'i' and SpeciesCode 'j' (i >= j), use (zero-based) index = (j*N + i - (j*j + j)/2).

    name: cutoff
    value: [3.77118]
    size: 1
    dtype: Double
    description: Distances used to determine whether two-body interactions for a pair of atoms occur, as well as to determine whether three-body interactions for a triplet of atoms occur.This array corresponds to a lower-triangular matrix (of size N=1) in row-major storage. Ordering is according to SpeciesCode values. For example, to find the parameter related to SpeciesCode 'i' and SpeciesCode 'j' (i >= j), use (zero-based) index = (j*N + i - (j*j + j)/2).

    name: lambda
    value: [45.5322]
    size: 1
    dtype: Double
    description: Multiplicative factors on the three-body energy function as a whole for each binary species combination. In terms of the original SW parameters, each quantity is equal to lambda*epsilon for the corresponding species combination. For a vertex atom i with neighbors j and k, the value ultimately used for the three-body interactions of bonds ij and ik is given by lambda_ijk = sqrt(lambda_ij*lambda_ik). This array corresponds to a lower-triangular matrix (of size N=1) in row-major storage. Ordering is according to SpeciesCode values. For example, to find the parameter related to SpeciesCode 'i' and SpeciesCode 'j' (i >= j), use (zero-based) index = (j*N + i - (j*j + j)/2).

    name: costheta0
    value: [-0.33333333]
    size: 1
    dtype: Double
    description: Cosine of the energetically preferable angle between bonds which share a common vertex atom. Formally, this is an array which corresponds to a lower-triangular matrix (of size N=1) in row-major storage. Ordering is according to SpeciesCode values. For example, to find the parameter related to SpeciesCode 'i' and SpeciesCode 'j' (i >= j), use (zero-based) index = (j*N + i - (j*j + j)/2). However, the values are still expected to be the same across different species combinations.


The output is generated by the last line, and it tells us the ``name``, ``value``,
``size``, ``data type`` and a ``description`` of each parameter.

.. note::
   You can provide a ``path`` argument to the method ``echo_model_params(path)`` to
   write the available parameters information to a file indicated by ``path``.

.. note::
   The available parameters information can also by obtained using the **kliff**
   :ref:`cmdlntool`:
   ``$ kliff model --echo-params SW_StillingerWeber_1985_Si__MO_405512056662_005``

Now that we know what parameters are available for fitting, we can optimize all or a
subset of them to reproduce the training set.


.. code-block:: default


    model.set_fitting_params(
        A=[[5.0, 1.0, 20]], B=[['default']], sigma=[[2.0951, 'fix']], gamma=[[1.5]]
    )
    model.echo_fitting_params()






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

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


Here, we tell KLIFF to fit four parameters ``B``, ``gamma``, ``sigma``, and ``A`` of the
SW model. The information for each fitting parameter should be provided as a list of
list, where the size of the outer list should be equal to the ``size`` of the parameter
given by ``model.echo_model_params()``. For each inner list, you can provide either one,
two, or three items.

- One item. You can use a numerical value (e.g. ``gamma``) to provide an initial guess
  of the parameter. Alternatively, the string ``'default'`` can be provided to use the
  default value in the model (e.g. ``B``).

- Two items. The first item should be a numerical value and the second item should be
  the string ``'fix'`` (e.g. ``sigma``), which tells KLIFF to use the value for the
  parameter, but do not optimize it.

- Three items. The first item can be a numerical value or the string ``'default'``,
  having the same meanings as the one item case. In the second and third items, you can
  list the lower and upper bounds for the parameters, respectively. A bound could be
  provided as a numerical values or ``None``. The latter indicates no bound is applied.

The call of ``model.echo_fitting_params()`` prints out the fitting parameters that we
require KLIFF to optimize. The number ``1`` after the name of each parameter indicates
the size of the parameter.

.. note::
   The parameters that are not included as a fitting parameter are fixed to the default
   values in the model during the optimization.


Training set
------------

KLIFF has a :class:`~kliff.dataset.DataSet` to deal with the training data (and possibly
test data). For the silicon training set, we can read and process the files by:


.. code-block:: default


    dataset_name = 'Si_training_set'
    tset = DataSet()
    tset.read(dataset_name)
    configs = tset.get_configs()








The ``configs`` in the last line is a list of :class:`~kliff.dataset.Configuration`.
Each configuration is an internal representation of a processed **extended xyz** file,
hosting the species, coordinates, energy, forces, and other related information of a
system of atoms.


Calculator
----------

:class:`~kliff.calculator.Calculator` is the central agent that exchanges information
and orchestrate the operation of the fitting process. It calls the model to compute the
energy and forces and provide this information to the `Loss function`_ (discussed below)
to compute the loss. It also grabs the parameters from the optimizer and update the
parameters stored in the model so that the up-to-date parameters are used the next time
the model is evaluated to compute the energy and forces. The calculator can be created
by:


.. code-block:: default


    calc = Calculator(model)
    calc.create(configs)








where ``calc.create(configs)`` does some initializations for each each
configuration in the training set, such as creating the neighbor list.


Loss function
-------------

KLIFF uses a loss function to quantify the difference between the training set data and
potential predictions and uses minimization algorithms to reduce the loss as much as
possible. KLIFF provides a large number of minimization algorithms by interacting with
SciPy_. For physics-motivated potentials, any algorithm listed on
`scipy.optimize.minimize`_ and `scipy.optimize.least_squares`_ can be used. In the
following code snippet, we create a loss of energy and forces, where the residual
function uses an ``energy_weight`` of ``1.0`` and a ``forces_weight`` of ``0.1``, and
``2`` processors will be used to calculate the loss. The ``L-BFGS-B`` minimization
algorithm is applied to minimize the loss, and the minimization is allowed to run for a
a max number of 100 iterations.


.. code-block:: default


    steps = 100
    residual_data = {'energy_weight': 1.0, 'forces_weight': 0.1}
    loss = Loss(calc, residual_data=residual_data, nprocs=2)
    loss.minimize(method='L-BFGS-B', options={'disp': True, 'maxiter': steps})








The minimization stops after running for 27 steps.  After the minimization, we'd better
save the model, which can be loaded later for the purpose to do a retraining or
evaluations. If satisfied with the fitted model, you can also write it as a KIM model
that can be used with LAMMPS_, GULP_, ASE_, etc. via the kim-api_.


.. code-block:: default


    model.echo_fitting_params()
    model.save('kliff_model.pkl')
    model.write_kim_model()






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    #================================================================================
    # Model parameters that are optimized.
    #================================================================================

    A 1
      1.4938634542014945e+01   1.0000000000000000e+00   2.0000000000000000e+01 

    B 1
      5.8740272882171785e-01 

    sigma 1
      2.0951000000000000e+00 fix 

    gamma 1
      2.2014613016820985e+00


The first line of the above code generates the output.  A comparison with the original
parameters before carrying out the minimization shows that we recover the original
parameters quite reasonably. The second line saves the fitted model to a file named
``kliff_model.pkl`` on the disk, and the third line writes out a KIM potential named
``SW_StillingerWeber_1985_Si__MO_405512056662_005_kliff_trained``.

.. seealso::
   For information about how to load a saved model, see :ref:`doc.modules`.


.. _OpenKIM: https://openkim.org
.. _SciPy: https://scipy.org
.. _scipy.optimize.minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
.. _scipy.optimize.least_squares: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
.. _kim-api: https://openkim.org/kim-api/
.. _LAMMPS: https://lammps.sandia.gov
.. _GULP: http://gulp.curtin.edu.au/gulp/
.. _ASE: https://wiki.fysik.dtu.dk/ase/


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 5 minutes  58.846 seconds)


.. _sphx_glr_download_auto_examples_example_kim_SW_Si.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: example_kim_SW_Si.py <example_kim_SW_Si.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: example_kim_SW_Si.ipynb <example_kim_SW_Si.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
