.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_example_kim_SW_Si.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_example_kim_SW_Si.py:


.. _tut_kim_sw:

Train a Stillinger-Weber potential
==================================

In this tutorial, we train a Stillinger-Weber (SW) potential for silicon that is archived
on OpenKIM_.

Before getting started to train the SW model, let's first install the SW model::

   $ kim-api-collections-management install user SW_StillingerWeber_1985_Si__MO_405512056662_005

.. seealso::
   This installs the model and its driver into the ``User Collection``. See
   :ref:`install_model` for more information about installing KIM models.

We are going to create potentials for diamond silicon, and fit the potentials to a
training set of energies and forces consisting of compressed and stretched diamond
silicon structures, as well as configurations drawn from molecular dynamics trajectories
at different temperatures.
Download the training set :download:`Si_training_set.tar.gz
<https://raw.githubusercontent.com/mjwen/kliff/master/examples/Si_training_set.tar.gz>`
and extract the tarball: ``$ tar xzf Si_training_set.tar.gz``. The data is stored in
**extended xyz** format, and see :ref:`doc.dataset` for more information of this format.

.. warning::
   The ``Si_training_set`` is just a toy data set for the purpose to demonstrate how to
   use KLIFF to train potentials. It should not be used to train any potential for real
   simulations.

Let's first import the modules that will be used in this example.


.. code-block:: default


    from kliff.calculators import Calculator
    from kliff.dataset import Dataset
    from kliff.loss import Loss
    from kliff.models import KIM








Model
-----

We first create a KIM model for the SW potential, and print out all the available
parameters that can be optimized (we call this ``model parameters``).


.. code-block:: default


    model = KIM(model_name="SW_StillingerWeber_1985_Si__MO_405512056662_005")
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
        A=[[5.0, 1.0, 20]], B=[["default"]], sigma=[[2.0951, "fix"]], gamma=[[1.5]]
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

KLIFF has a :class:`~kliff.dataset.Dataset` to deal with the training data (and possibly
test data). For the silicon training set, we can read and process the files by:


.. code-block:: default


    dataset_name = "Si_training_set"
    tset = Dataset()
    tset.read(dataset_name)
    configs = tset.get_configs()






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    1000 configurations read from "Si_training_set"




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






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    [<kliff.models.kim.KIMComputeArguments object at 0x120859090>, <kliff.models.kim.KIMComputeArguments object at 0x11c837890>, <kliff.models.kim.KIMComputeArguments object at 0x120859290>, <kliff.models.kim.KIMComputeArguments object at 0x1208593d0>, <kliff.models.kim.KIMComputeArguments object at 0x120859410>, <kliff.models.kim.KIMComputeArguments object at 0x120859590>, <kliff.models.kim.KIMComputeArguments object at 0x120859510>, <kliff.models.kim.KIMComputeArguments object at 0x120859650>, <kliff.models.kim.KIMComputeArguments object at 0x1208596d0>, <kliff.models.kim.KIMComputeArguments object at 0x1208594d0>, <kliff.models.kim.KIMComputeArguments object at 0x120859750>, <kliff.models.kim.KIMComputeArguments object at 0x120859810>, <kliff.models.kim.KIMComputeArguments object at 0x120859890>, <kliff.models.kim.KIMComputeArguments object at 0x120859910>, <kliff.models.kim.KIMComputeArguments object at 0x120859990>, <kliff.models.kim.KIMComputeArguments object at 0x120859a10>, <kliff.models.kim.KIMComputeArguments object at 0x120859a90>, <kliff.models.kim.KIMComputeArguments object at 0x120859b10>, <kliff.models.kim.KIMComputeArguments object at 0x120859b90>, <kliff.models.kim.KIMComputeArguments object at 0x120859c10>, <kliff.models.kim.KIMComputeArguments object at 0x120859c90>, <kliff.models.kim.KIMComputeArguments object at 0x120859d10>, <kliff.models.kim.KIMComputeArguments object at 0x120859d90>, <kliff.models.kim.KIMComputeArguments object at 0x120859e10>, <kliff.models.kim.KIMComputeArguments object at 0x120859e90>, <kliff.models.kim.KIMComputeArguments object at 0x120859f10>, <kliff.models.kim.KIMComputeArguments object at 0x120859f90>, <kliff.models.kim.KIMComputeArguments object at 0x120859fd0>, <kliff.models.kim.KIMComputeArguments object at 0x120859790>, <kliff.models.kim.KIMComputeArguments object at 0x1206ed150>, <kliff.models.kim.KIMComputeArguments object at 0x1206ed1d0>, <kliff.models.kim.KIMComputeArguments object at 0x1206ed250>, <kliff.models.kim.KIMComputeArguments object at 0x1206ed2d0>, <kliff.models.kim.KIMComputeArguments object at 0x1206ed350>, <kliff.models.kim.KIMComputeArguments object at 0x1206ed3d0>, <kliff.models.kim.KIMComputeArguments object at 0x1206ed450>, <kliff.models.kim.KIMComputeArguments object at 0x1206ed4d0>, <kliff.models.kim.KIMComputeArguments object at 0x1206ed550>, <kliff.models.kim.KIMComputeArguments object at 0x1206ed5d0>, <kliff.models.kim.KIMComputeArguments object at 0x1206ed650>, <kliff.models.kim.KIMComputeArguments object at 0x1206ed6d0>, <kliff.models.kim.KIMComputeArguments object at 0x1206ed750>, <kliff.models.kim.KIMComputeArguments object at 0x1206ed7d0>, <kliff.models.kim.KIMComputeArguments object at 0x1206ed850>, <kliff.models.kim.KIMComputeArguments object at 0x1206ed8d0>, <kliff.models.kim.KIMComputeArguments object at 0x1206ed950>, <kliff.models.kim.KIMComputeArguments object at 0x1206ed9d0>, <kliff.models.kim.KIMComputeArguments object at 0x1206eda50>, <kliff.models.kim.KIMComputeArguments object at 0x1206edad0>, <kliff.models.kim.KIMComputeArguments object at 0x1206edb50>, <kliff.models.kim.KIMComputeArguments object at 0x1206edbd0>, <kliff.models.kim.KIMComputeArguments object at 0x1206edc50>, <kliff.models.kim.KIMComputeArguments object at 0x1206edcd0>, <kliff.models.kim.KIMComputeArguments object at 0x1206edd50>, <kliff.models.kim.KIMComputeArguments object at 0x1206eddd0>, <kliff.models.kim.KIMComputeArguments object at 0x1206ede50>, <kliff.models.kim.KIMComputeArguments object at 0x1206eded0>, <kliff.models.kim.KIMComputeArguments object at 0x1206edf50>, <kliff.models.kim.KIMComputeArguments object at 0x1206edfd0>, <kliff.models.kim.KIMComputeArguments object at 0x1206ed0d0>, <kliff.models.kim.KIMComputeArguments object at 0x11c5d4dd0>, <kliff.models.kim.KIMComputeArguments object at 0x12085f050>, <kliff.models.kim.KIMComputeArguments object at 0x12085f1d0>, <kliff.models.kim.KIMComputeArguments object at 0x12085f250>, <kliff.models.kim.KIMComputeArguments object at 0x12085f2d0>, <kliff.models.kim.KIMComputeArguments object at 0x12085f350>, <kliff.models.kim.KIMComputeArguments object at 0x12085f3d0>, <kliff.models.kim.KIMComputeArguments object at 0x12085f450>, <kliff.models.kim.KIMComputeArguments object at 0x12085f4d0>, <kliff.models.kim.KIMComputeArguments object at 0x12085f550>, <kliff.models.kim.KIMComputeArguments object at 0x12085f5d0>, <kliff.models.kim.KIMComputeArguments object at 0x12085f650>, <kliff.models.kim.KIMComputeArguments object at 0x12085f6d0>, <kliff.models.kim.KIMComputeArguments object at 0x12085f750>, <kliff.models.kim.KIMComputeArguments object at 0x12085f7d0>, <kliff.models.kim.KIMComputeArguments object at 0x12085f850>, <kliff.models.kim.KIMComputeArguments object at 0x12085f8d0>, <kliff.models.kim.KIMComputeArguments object at 0x12085f950>, <kliff.models.kim.KIMComputeArguments object at 0x12085f9d0>, <kliff.models.kim.KIMComputeArguments object at 0x12085fa50>, <kliff.models.kim.KIMComputeArguments object at 0x12085fad0>, <kliff.models.kim.KIMComputeArguments object at 0x12085fb50>, <kliff.models.kim.KIMComputeArguments object at 0x12085fbd0>, <kliff.models.kim.KIMComputeArguments object at 0x12085fc50>, <kliff.models.kim.KIMComputeArguments object at 0x12085fcd0>, <kliff.models.kim.KIMComputeArguments object at 0x12085fd50>, <kliff.models.kim.KIMComputeArguments object at 0x12085fdd0>, <kliff.models.kim.KIMComputeArguments object at 0x12085fe50>, <kliff.models.kim.KIMComputeArguments object at 0x12085fed0>, <kliff.models.kim.KIMComputeArguments object at 0x12085ff50>, <kliff.models.kim.KIMComputeArguments object at 0x12085ffd0>, <kliff.models.kim.KIMComputeArguments object at 0x12085f150>, <kliff.models.kim.KIMComputeArguments object at 0x120870110>, <kliff.models.kim.KIMComputeArguments object at 0x120870190>, <kliff.models.kim.KIMComputeArguments object at 0x120870210>, <kliff.models.kim.KIMComputeArguments object at 0x120870290>, <kliff.models.kim.KIMComputeArguments object at 0x120870310>, <kliff.models.kim.KIMComputeArguments object at 0x120870390>, <kliff.models.kim.KIMComputeArguments object at 0x120870410>, <kliff.models.kim.KIMComputeArguments object at 0x120870490>, <kliff.models.kim.KIMComputeArguments object at 0x120870510>, <kliff.models.kim.KIMComputeArguments object at 0x120870590>, <kliff.models.kim.KIMComputeArguments object at 0x120870610>, <kliff.models.kim.KIMComputeArguments object at 0x120870690>, <kliff.models.kim.KIMComputeArguments object at 0x120870710>, <kliff.models.kim.KIMComputeArguments object at 0x120870790>, <kliff.models.kim.KIMComputeArguments object at 0x120870810>, <kliff.models.kim.KIMComputeArguments object at 0x120870890>, <kliff.models.kim.KIMComputeArguments object at 0x120870910>, <kliff.models.kim.KIMComputeArguments object at 0x120870990>, <kliff.models.kim.KIMComputeArguments object at 0x120870a10>, <kliff.models.kim.KIMComputeArguments object at 0x120870a90>, <kliff.models.kim.KIMComputeArguments object at 0x120870b10>, <kliff.models.kim.KIMComputeArguments object at 0x120870b90>, <kliff.models.kim.KIMComputeArguments object at 0x120870c10>, <kliff.models.kim.KIMComputeArguments object at 0x120870c90>, <kliff.models.kim.KIMComputeArguments object at 0x120870d10>, <kliff.models.kim.KIMComputeArguments object at 0x120870d90>, <kliff.models.kim.KIMComputeArguments object at 0x120870e10>, <kliff.models.kim.KIMComputeArguments object at 0x120870e90>, <kliff.models.kim.KIMComputeArguments object at 0x120870f10>, <kliff.models.kim.KIMComputeArguments object at 0x120870f90>, <kliff.models.kim.KIMComputeArguments object at 0x120870fd0>, <kliff.models.kim.KIMComputeArguments object at 0x120870090>, <kliff.models.kim.KIMComputeArguments object at 0x12087e150>, <kliff.models.kim.KIMComputeArguments object at 0x12087e1d0>, <kliff.models.kim.KIMComputeArguments object at 0x12087e250>, <kliff.models.kim.KIMComputeArguments object at 0x12087e2d0>, <kliff.models.kim.KIMComputeArguments object at 0x12087e350>, <kliff.models.kim.KIMComputeArguments object at 0x12087e3d0>, <kliff.models.kim.KIMComputeArguments object at 0x12087e450>, <kliff.models.kim.KIMComputeArguments object at 0x12087e4d0>, <kliff.models.kim.KIMComputeArguments object at 0x12087e550>, <kliff.models.kim.KIMComputeArguments object at 0x12087e5d0>, <kliff.models.kim.KIMComputeArguments object at 0x12087e650>, <kliff.models.kim.KIMComputeArguments object at 0x12087e6d0>, <kliff.models.kim.KIMComputeArguments object at 0x12087e750>, <kliff.models.kim.KIMComputeArguments object at 0x12087e7d0>, <kliff.models.kim.KIMComputeArguments object at 0x12087e850>, <kliff.models.kim.KIMComputeArguments object at 0x12087e8d0>, <kliff.models.kim.KIMComputeArguments object at 0x12087e950>, <kliff.models.kim.KIMComputeArguments object at 0x12087e9d0>, <kliff.models.kim.KIMComputeArguments object at 0x12087ea50>, <kliff.models.kim.KIMComputeArguments object at 0x12087ead0>, <kliff.models.kim.KIMComputeArguments object at 0x12087eb50>, <kliff.models.kim.KIMComputeArguments object at 0x12087ebd0>, <kliff.models.kim.KIMComputeArguments object at 0x12087ec50>, <kliff.models.kim.KIMComputeArguments object at 0x12087ecd0>, <kliff.models.kim.KIMComputeArguments object at 0x12087ed50>, <kliff.models.kim.KIMComputeArguments object at 0x12087edd0>, <kliff.models.kim.KIMComputeArguments object at 0x12087ee50>, <kliff.models.kim.KIMComputeArguments object at 0x12087eed0>, <kliff.models.kim.KIMComputeArguments object at 0x12087ef50>, <kliff.models.kim.KIMComputeArguments object at 0x12087efd0>, <kliff.models.kim.KIMComputeArguments object at 0x12087e0d0>, <kliff.models.kim.KIMComputeArguments object at 0x12088a090>, <kliff.models.kim.KIMComputeArguments object at 0x12088a190>, <kliff.models.kim.KIMComputeArguments object at 0x12088a210>, <kliff.models.kim.KIMComputeArguments object at 0x12088a290>, <kliff.models.kim.KIMComputeArguments object at 0x12088a310>, <kliff.models.kim.KIMComputeArguments object at 0x12088a390>, <kliff.models.kim.KIMComputeArguments object at 0x12088a410>, <kliff.models.kim.KIMComputeArguments object at 0x12088a490>, <kliff.models.kim.KIMComputeArguments object at 0x12088a510>, <kliff.models.kim.KIMComputeArguments object at 0x12088a590>, <kliff.models.kim.KIMComputeArguments object at 0x12088a610>, <kliff.models.kim.KIMComputeArguments object at 0x12088a690>, <kliff.models.kim.KIMComputeArguments object at 0x12088a710>, <kliff.models.kim.KIMComputeArguments object at 0x12088a790>, <kliff.models.kim.KIMComputeArguments object at 0x12088a810>, <kliff.models.kim.KIMComputeArguments object at 0x12088a890>, <kliff.models.kim.KIMComputeArguments object at 0x12088a910>, <kliff.models.kim.KIMComputeArguments object at 0x12088a990>, <kliff.models.kim.KIMComputeArguments object at 0x12088aa10>, <kliff.models.kim.KIMComputeArguments object at 0x12088aa90>, <kliff.models.kim.KIMComputeArguments object at 0x12088ab10>, <kliff.models.kim.KIMComputeArguments object at 0x12088ab90>, <kliff.models.kim.KIMComputeArguments object at 0x12088ac10>, <kliff.models.kim.KIMComputeArguments object at 0x12088ac90>, <kliff.models.kim.KIMComputeArguments object at 0x12088ad10>, <kliff.models.kim.KIMComputeArguments object at 0x12088ad90>, <kliff.models.kim.KIMComputeArguments object at 0x12088ae10>, <kliff.models.kim.KIMComputeArguments object at 0x12088ae90>, <kliff.models.kim.KIMComputeArguments object at 0x12088af10>, <kliff.models.kim.KIMComputeArguments object at 0x12088af90>, <kliff.models.kim.KIMComputeArguments object at 0x12088afd0>, <kliff.models.kim.KIMComputeArguments object at 0x12088a110>, <kliff.models.kim.KIMComputeArguments object at 0x120898150>, <kliff.models.kim.KIMComputeArguments object at 0x1208981d0>, <kliff.models.kim.KIMComputeArguments object at 0x120898250>, <kliff.models.kim.KIMComputeArguments object at 0x1208982d0>, <kliff.models.kim.KIMComputeArguments object at 0x120898350>, <kliff.models.kim.KIMComputeArguments object at 0x1208983d0>, <kliff.models.kim.KIMComputeArguments object at 0x120898450>, <kliff.models.kim.KIMComputeArguments object at 0x1208984d0>, <kliff.models.kim.KIMComputeArguments object at 0x120898550>, <kliff.models.kim.KIMComputeArguments object at 0x1208985d0>, <kliff.models.kim.KIMComputeArguments object at 0x120898650>, <kliff.models.kim.KIMComputeArguments object at 0x1208986d0>, <kliff.models.kim.KIMComputeArguments object at 0x120898750>, <kliff.models.kim.KIMComputeArguments object at 0x1208987d0>, <kliff.models.kim.KIMComputeArguments object at 0x120898850>, <kliff.models.kim.KIMComputeArguments object at 0x1208988d0>, <kliff.models.kim.KIMComputeArguments object at 0x120898950>, <kliff.models.kim.KIMComputeArguments object at 0x1208989d0>, <kliff.models.kim.KIMComputeArguments object at 0x120898a50>, <kliff.models.kim.KIMComputeArguments object at 0x120898ad0>, <kliff.models.kim.KIMComputeArguments object at 0x120898b50>, <kliff.models.kim.KIMComputeArguments object at 0x120898bd0>, <kliff.models.kim.KIMComputeArguments object at 0x120898c50>, <kliff.models.kim.KIMComputeArguments object at 0x120898cd0>, <kliff.models.kim.KIMComputeArguments object at 0x120898d50>, <kliff.models.kim.KIMComputeArguments object at 0x120898dd0>, <kliff.models.kim.KIMComputeArguments object at 0x120898e50>, <kliff.models.kim.KIMComputeArguments object at 0x120898ed0>, <kliff.models.kim.KIMComputeArguments object at 0x120898f50>, <kliff.models.kim.KIMComputeArguments object at 0x120898fd0>, <kliff.models.kim.KIMComputeArguments object at 0x1208980d0>, <kliff.models.kim.KIMComputeArguments object at 0x1208a5110>, <kliff.models.kim.KIMComputeArguments object at 0x1208a5190>, <kliff.models.kim.KIMComputeArguments object at 0x1208a5210>, <kliff.models.kim.KIMComputeArguments object at 0x1208a5290>, <kliff.models.kim.KIMComputeArguments object at 0x1208a5310>, <kliff.models.kim.KIMComputeArguments object at 0x1208a5390>, <kliff.models.kim.KIMComputeArguments object at 0x1208a5410>, <kliff.models.kim.KIMComputeArguments object at 0x1208a5490>, <kliff.models.kim.KIMComputeArguments object at 0x1208a5510>, <kliff.models.kim.KIMComputeArguments object at 0x1208a5590>, <kliff.models.kim.KIMComputeArguments object at 0x1208a5610>, <kliff.models.kim.KIMComputeArguments object at 0x1208a5690>, <kliff.models.kim.KIMComputeArguments object at 0x1208a5710>, <kliff.models.kim.KIMComputeArguments object at 0x1208a5790>, <kliff.models.kim.KIMComputeArguments object at 0x1208a5810>, <kliff.models.kim.KIMComputeArguments object at 0x1208a5890>, <kliff.models.kim.KIMComputeArguments object at 0x1208a5910>, <kliff.models.kim.KIMComputeArguments object at 0x1208a5990>, <kliff.models.kim.KIMComputeArguments object at 0x1208a5a10>, <kliff.models.kim.KIMComputeArguments object at 0x1208a5a90>, <kliff.models.kim.KIMComputeArguments object at 0x1208a5b10>, <kliff.models.kim.KIMComputeArguments object at 0x1208a5b90>, <kliff.models.kim.KIMComputeArguments object at 0x1208a5c10>, <kliff.models.kim.KIMComputeArguments object at 0x1208a5c90>, <kliff.models.kim.KIMComputeArguments object at 0x1208a5d10>, <kliff.models.kim.KIMComputeArguments object at 0x1208a5d90>, <kliff.models.kim.KIMComputeArguments object at 0x1208a5e10>, <kliff.models.kim.KIMComputeArguments object at 0x1208a5e90>, <kliff.models.kim.KIMComputeArguments object at 0x1208a5f10>, <kliff.models.kim.KIMComputeArguments object at 0x1208a5f90>, <kliff.models.kim.KIMComputeArguments object at 0x1208a5fd0>, <kliff.models.kim.KIMComputeArguments object at 0x1208a5090>, <kliff.models.kim.KIMComputeArguments object at 0x1208b2150>, <kliff.models.kim.KIMComputeArguments object at 0x1208b21d0>, <kliff.models.kim.KIMComputeArguments object at 0x1208b2250>, <kliff.models.kim.KIMComputeArguments object at 0x1208b22d0>, <kliff.models.kim.KIMComputeArguments object at 0x1208b2350>, <kliff.models.kim.KIMComputeArguments object at 0x1208b23d0>, <kliff.models.kim.KIMComputeArguments object at 0x1208b2450>, <kliff.models.kim.KIMComputeArguments object at 0x1208b24d0>, <kliff.models.kim.KIMComputeArguments object at 0x1208b2550>, <kliff.models.kim.KIMComputeArguments object at 0x1208b25d0>, <kliff.models.kim.KIMComputeArguments object at 0x1208b2650>, <kliff.models.kim.KIMComputeArguments object at 0x1208b26d0>, <kliff.models.kim.KIMComputeArguments object at 0x1208b2750>, <kliff.models.kim.KIMComputeArguments object at 0x1208b27d0>, <kliff.models.kim.KIMComputeArguments object at 0x1208b2850>, <kliff.models.kim.KIMComputeArguments object at 0x1208b28d0>, <kliff.models.kim.KIMComputeArguments object at 0x1208b2950>, <kliff.models.kim.KIMComputeArguments object at 0x1208b29d0>, <kliff.models.kim.KIMComputeArguments object at 0x1208b2a50>, <kliff.models.kim.KIMComputeArguments object at 0x1208b2ad0>, <kliff.models.kim.KIMComputeArguments object at 0x1208b2b50>, <kliff.models.kim.KIMComputeArguments object at 0x1208b2bd0>, <kliff.models.kim.KIMComputeArguments object at 0x1208b2c50>, <kliff.models.kim.KIMComputeArguments object at 0x1208b2cd0>, <kliff.models.kim.KIMComputeArguments object at 0x1208b2d50>, <kliff.models.kim.KIMComputeArguments object at 0x1208b2dd0>, <kliff.models.kim.KIMComputeArguments object at 0x1208b2e50>, <kliff.models.kim.KIMComputeArguments object at 0x1208b2ed0>, <kliff.models.kim.KIMComputeArguments object at 0x1208b2f50>, <kliff.models.kim.KIMComputeArguments object at 0x1208b2fd0>, <kliff.models.kim.KIMComputeArguments object at 0x1208b20d0>, <kliff.models.kim.KIMComputeArguments object at 0x1208be110>, <kliff.models.kim.KIMComputeArguments object at 0x1208be190>, <kliff.models.kim.KIMComputeArguments object at 0x1208be210>, <kliff.models.kim.KIMComputeArguments object at 0x1208be290>, <kliff.models.kim.KIMComputeArguments object at 0x1208be310>, <kliff.models.kim.KIMComputeArguments object at 0x1208be390>, <kliff.models.kim.KIMComputeArguments object at 0x1208be410>, <kliff.models.kim.KIMComputeArguments object at 0x1208be490>, <kliff.models.kim.KIMComputeArguments object at 0x1208be510>, <kliff.models.kim.KIMComputeArguments object at 0x1208be590>, <kliff.models.kim.KIMComputeArguments object at 0x1208be610>, <kliff.models.kim.KIMComputeArguments object at 0x1208be690>, <kliff.models.kim.KIMComputeArguments object at 0x1208be710>, <kliff.models.kim.KIMComputeArguments object at 0x1208be790>, <kliff.models.kim.KIMComputeArguments object at 0x1208be810>, <kliff.models.kim.KIMComputeArguments object at 0x1208be890>, <kliff.models.kim.KIMComputeArguments object at 0x1208be910>, <kliff.models.kim.KIMComputeArguments object at 0x1208be990>, <kliff.models.kim.KIMComputeArguments object at 0x1208bea10>, <kliff.models.kim.KIMComputeArguments object at 0x1208bea90>, <kliff.models.kim.KIMComputeArguments object at 0x1208beb10>, <kliff.models.kim.KIMComputeArguments object at 0x1208beb90>, <kliff.models.kim.KIMComputeArguments object at 0x1208bec10>, <kliff.models.kim.KIMComputeArguments object at 0x1208bec90>, <kliff.models.kim.KIMComputeArguments object at 0x1208bed10>, <kliff.models.kim.KIMComputeArguments object at 0x1208bed90>, <kliff.models.kim.KIMComputeArguments object at 0x1208bee10>, <kliff.models.kim.KIMComputeArguments object at 0x1208bee90>, <kliff.models.kim.KIMComputeArguments object at 0x1208bef10>, <kliff.models.kim.KIMComputeArguments object at 0x1208bef90>, <kliff.models.kim.KIMComputeArguments object at 0x1208befd0>, <kliff.models.kim.KIMComputeArguments object at 0x1208be090>, <kliff.models.kim.KIMComputeArguments object at 0x1208cc150>, <kliff.models.kim.KIMComputeArguments object at 0x1208cc1d0>, <kliff.models.kim.KIMComputeArguments object at 0x1208cc250>, <kliff.models.kim.KIMComputeArguments object at 0x1208cc2d0>, <kliff.models.kim.KIMComputeArguments object at 0x1208cc350>, <kliff.models.kim.KIMComputeArguments object at 0x1208cc3d0>, <kliff.models.kim.KIMComputeArguments object at 0x1208cc450>, <kliff.models.kim.KIMComputeArguments object at 0x1208cc4d0>, <kliff.models.kim.KIMComputeArguments object at 0x1208cc550>, <kliff.models.kim.KIMComputeArguments object at 0x1208cc5d0>, <kliff.models.kim.KIMComputeArguments object at 0x1208cc650>, <kliff.models.kim.KIMComputeArguments object at 0x1208cc6d0>, <kliff.models.kim.KIMComputeArguments object at 0x1208cc750>, <kliff.models.kim.KIMComputeArguments object at 0x1208cc7d0>, <kliff.models.kim.KIMComputeArguments object at 0x1208cc850>, <kliff.models.kim.KIMComputeArguments object at 0x1208cc8d0>, <kliff.models.kim.KIMComputeArguments object at 0x1208cc950>, <kliff.models.kim.KIMComputeArguments object at 0x1208cc9d0>, <kliff.models.kim.KIMComputeArguments object at 0x1208cca50>, <kliff.models.kim.KIMComputeArguments object at 0x1208ccad0>, <kliff.models.kim.KIMComputeArguments object at 0x1208ccb50>, <kliff.models.kim.KIMComputeArguments object at 0x1208ccbd0>, <kliff.models.kim.KIMComputeArguments object at 0x1208ccc50>, <kliff.models.kim.KIMComputeArguments object at 0x1208cccd0>, <kliff.models.kim.KIMComputeArguments object at 0x1208ccd50>, <kliff.models.kim.KIMComputeArguments object at 0x1208ccdd0>, <kliff.models.kim.KIMComputeArguments object at 0x1208cce50>, <kliff.models.kim.KIMComputeArguments object at 0x1208cced0>, <kliff.models.kim.KIMComputeArguments object at 0x1208ccf50>, <kliff.models.kim.KIMComputeArguments object at 0x1208ccfd0>, <kliff.models.kim.KIMComputeArguments object at 0x1208cc0d0>, <kliff.models.kim.KIMComputeArguments object at 0x1208d8110>, <kliff.models.kim.KIMComputeArguments object at 0x1208d8190>, <kliff.models.kim.KIMComputeArguments object at 0x1208d8210>, <kliff.models.kim.KIMComputeArguments object at 0x1208d8290>, <kliff.models.kim.KIMComputeArguments object at 0x1208d8310>, <kliff.models.kim.KIMComputeArguments object at 0x1208d8390>, <kliff.models.kim.KIMComputeArguments object at 0x1208d8410>, <kliff.models.kim.KIMComputeArguments object at 0x1208d8490>, <kliff.models.kim.KIMComputeArguments object at 0x1208d8510>, <kliff.models.kim.KIMComputeArguments object at 0x1208d8590>, <kliff.models.kim.KIMComputeArguments object at 0x1208d8610>, <kliff.models.kim.KIMComputeArguments object at 0x1208d8690>, <kliff.models.kim.KIMComputeArguments object at 0x1208d8710>, <kliff.models.kim.KIMComputeArguments object at 0x1208d8790>, <kliff.models.kim.KIMComputeArguments object at 0x1208d8810>, <kliff.models.kim.KIMComputeArguments object at 0x1208d8890>, <kliff.models.kim.KIMComputeArguments object at 0x1208d8910>, <kliff.models.kim.KIMComputeArguments object at 0x1208d8990>, <kliff.models.kim.KIMComputeArguments object at 0x1208d8a10>, <kliff.models.kim.KIMComputeArguments object at 0x1208d8a90>, <kliff.models.kim.KIMComputeArguments object at 0x1208d8b10>, <kliff.models.kim.KIMComputeArguments object at 0x1208d8b90>, <kliff.models.kim.KIMComputeArguments object at 0x1208d8c10>, <kliff.models.kim.KIMComputeArguments object at 0x1208d8c90>, <kliff.models.kim.KIMComputeArguments object at 0x1208d8d10>, <kliff.models.kim.KIMComputeArguments object at 0x1208d8d90>, <kliff.models.kim.KIMComputeArguments object at 0x1208d8e10>, <kliff.models.kim.KIMComputeArguments object at 0x1208d8e90>, <kliff.models.kim.KIMComputeArguments object at 0x1208d8f10>, <kliff.models.kim.KIMComputeArguments object at 0x1208d8f90>, <kliff.models.kim.KIMComputeArguments object at 0x1208d8fd0>, <kliff.models.kim.KIMComputeArguments object at 0x1208d8090>, <kliff.models.kim.KIMComputeArguments object at 0x1208e6150>, <kliff.models.kim.KIMComputeArguments object at 0x1208e61d0>, <kliff.models.kim.KIMComputeArguments object at 0x1208e6250>, <kliff.models.kim.KIMComputeArguments object at 0x1208e62d0>, <kliff.models.kim.KIMComputeArguments object at 0x1208e6350>, <kliff.models.kim.KIMComputeArguments object at 0x1208e63d0>, <kliff.models.kim.KIMComputeArguments object at 0x1208e6450>, <kliff.models.kim.KIMComputeArguments object at 0x1208e64d0>, <kliff.models.kim.KIMComputeArguments object at 0x1208e6550>, <kliff.models.kim.KIMComputeArguments object at 0x1208e65d0>, <kliff.models.kim.KIMComputeArguments object at 0x1208e6650>, <kliff.models.kim.KIMComputeArguments object at 0x1208e66d0>, <kliff.models.kim.KIMComputeArguments object at 0x1208e6750>, <kliff.models.kim.KIMComputeArguments object at 0x1208e67d0>, <kliff.models.kim.KIMComputeArguments object at 0x1208e6850>, <kliff.models.kim.KIMComputeArguments object at 0x1208e68d0>, <kliff.models.kim.KIMComputeArguments object at 0x1208e6950>, <kliff.models.kim.KIMComputeArguments object at 0x1208e69d0>, <kliff.models.kim.KIMComputeArguments object at 0x1208e6a50>, <kliff.models.kim.KIMComputeArguments object at 0x1208e6ad0>, <kliff.models.kim.KIMComputeArguments object at 0x1208e6b50>, <kliff.models.kim.KIMComputeArguments object at 0x1208e6bd0>, <kliff.models.kim.KIMComputeArguments object at 0x1208e6c50>, <kliff.models.kim.KIMComputeArguments object at 0x1208e6cd0>, <kliff.models.kim.KIMComputeArguments object at 0x1208e6d50>, <kliff.models.kim.KIMComputeArguments object at 0x1208e6dd0>, <kliff.models.kim.KIMComputeArguments object at 0x1208e6e50>, <kliff.models.kim.KIMComputeArguments object at 0x1208e6ed0>, <kliff.models.kim.KIMComputeArguments object at 0x1208e6f50>, <kliff.models.kim.KIMComputeArguments object at 0x1208e6fd0>, <kliff.models.kim.KIMComputeArguments object at 0x1208e60d0>, <kliff.models.kim.KIMComputeArguments object at 0x1208f2110>, <kliff.models.kim.KIMComputeArguments object at 0x1208f2190>, <kliff.models.kim.KIMComputeArguments object at 0x1208f2210>, <kliff.models.kim.KIMComputeArguments object at 0x1208f2290>, <kliff.models.kim.KIMComputeArguments object at 0x1208f2310>, <kliff.models.kim.KIMComputeArguments object at 0x1208f2390>, <kliff.models.kim.KIMComputeArguments object at 0x1208f2410>, <kliff.models.kim.KIMComputeArguments object at 0x1208f2490>, <kliff.models.kim.KIMComputeArguments object at 0x1208f2510>, <kliff.models.kim.KIMComputeArguments object at 0x1208f2590>, <kliff.models.kim.KIMComputeArguments object at 0x1208f2610>, <kliff.models.kim.KIMComputeArguments object at 0x1208f2690>, <kliff.models.kim.KIMComputeArguments object at 0x1208f2710>, <kliff.models.kim.KIMComputeArguments object at 0x1208f2790>, <kliff.models.kim.KIMComputeArguments object at 0x1208f2810>, <kliff.models.kim.KIMComputeArguments object at 0x1208f2890>, <kliff.models.kim.KIMComputeArguments object at 0x1208f2910>, <kliff.models.kim.KIMComputeArguments object at 0x1208f2990>, <kliff.models.kim.KIMComputeArguments object at 0x1208f2a10>, <kliff.models.kim.KIMComputeArguments object at 0x1208f2a90>, <kliff.models.kim.KIMComputeArguments object at 0x1208f2b10>, <kliff.models.kim.KIMComputeArguments object at 0x1208f2b90>, <kliff.models.kim.KIMComputeArguments object at 0x1208f2c10>, <kliff.models.kim.KIMComputeArguments object at 0x1208f2c90>, <kliff.models.kim.KIMComputeArguments object at 0x1208f2d10>, <kliff.models.kim.KIMComputeArguments object at 0x1208f2d90>, <kliff.models.kim.KIMComputeArguments object at 0x1208f2e10>, <kliff.models.kim.KIMComputeArguments object at 0x1208f2e90>, <kliff.models.kim.KIMComputeArguments object at 0x1208f2f10>, <kliff.models.kim.KIMComputeArguments object at 0x1208f2f90>, <kliff.models.kim.KIMComputeArguments object at 0x1208f2fd0>, <kliff.models.kim.KIMComputeArguments object at 0x1208f2090>, <kliff.models.kim.KIMComputeArguments object at 0x120900150>, <kliff.models.kim.KIMComputeArguments object at 0x1209001d0>, <kliff.models.kim.KIMComputeArguments object at 0x120900250>, <kliff.models.kim.KIMComputeArguments object at 0x1209002d0>, <kliff.models.kim.KIMComputeArguments object at 0x120900350>, <kliff.models.kim.KIMComputeArguments object at 0x1209003d0>, <kliff.models.kim.KIMComputeArguments object at 0x120900450>, <kliff.models.kim.KIMComputeArguments object at 0x1209004d0>, <kliff.models.kim.KIMComputeArguments object at 0x120900550>, <kliff.models.kim.KIMComputeArguments object at 0x1209005d0>, <kliff.models.kim.KIMComputeArguments object at 0x120900650>, <kliff.models.kim.KIMComputeArguments object at 0x1209006d0>, <kliff.models.kim.KIMComputeArguments object at 0x120900750>, <kliff.models.kim.KIMComputeArguments object at 0x1209007d0>, <kliff.models.kim.KIMComputeArguments object at 0x120900850>, <kliff.models.kim.KIMComputeArguments object at 0x1209008d0>, <kliff.models.kim.KIMComputeArguments object at 0x120900950>, <kliff.models.kim.KIMComputeArguments object at 0x1209009d0>, <kliff.models.kim.KIMComputeArguments object at 0x120900a50>, <kliff.models.kim.KIMComputeArguments object at 0x120900ad0>, <kliff.models.kim.KIMComputeArguments object at 0x120900b50>, <kliff.models.kim.KIMComputeArguments object at 0x120900bd0>, <kliff.models.kim.KIMComputeArguments object at 0x120900c50>, <kliff.models.kim.KIMComputeArguments object at 0x120900cd0>, <kliff.models.kim.KIMComputeArguments object at 0x120900d50>, <kliff.models.kim.KIMComputeArguments object at 0x120900dd0>, <kliff.models.kim.KIMComputeArguments object at 0x120900e50>, <kliff.models.kim.KIMComputeArguments object at 0x120900ed0>, <kliff.models.kim.KIMComputeArguments object at 0x120900f50>, <kliff.models.kim.KIMComputeArguments object at 0x120900fd0>, <kliff.models.kim.KIMComputeArguments object at 0x1209000d0>, <kliff.models.kim.KIMComputeArguments object at 0x12090c110>, <kliff.models.kim.KIMComputeArguments object at 0x12090c190>, <kliff.models.kim.KIMComputeArguments object at 0x12090c210>, <kliff.models.kim.KIMComputeArguments object at 0x12090c290>, <kliff.models.kim.KIMComputeArguments object at 0x12090c310>, <kliff.models.kim.KIMComputeArguments object at 0x12090c390>, <kliff.models.kim.KIMComputeArguments object at 0x12090c410>, <kliff.models.kim.KIMComputeArguments object at 0x12090c490>, <kliff.models.kim.KIMComputeArguments object at 0x12090c510>, <kliff.models.kim.KIMComputeArguments object at 0x12090c590>, <kliff.models.kim.KIMComputeArguments object at 0x12090c610>, <kliff.models.kim.KIMComputeArguments object at 0x12090c690>, <kliff.models.kim.KIMComputeArguments object at 0x12090c710>, <kliff.models.kim.KIMComputeArguments object at 0x12090c790>, <kliff.models.kim.KIMComputeArguments object at 0x12090c810>, <kliff.models.kim.KIMComputeArguments object at 0x12090c890>, <kliff.models.kim.KIMComputeArguments object at 0x12090c910>, <kliff.models.kim.KIMComputeArguments object at 0x12090c990>, <kliff.models.kim.KIMComputeArguments object at 0x12090ca10>, <kliff.models.kim.KIMComputeArguments object at 0x12090ca90>, <kliff.models.kim.KIMComputeArguments object at 0x12090cb10>, <kliff.models.kim.KIMComputeArguments object at 0x12090cb90>, <kliff.models.kim.KIMComputeArguments object at 0x12090cc10>, <kliff.models.kim.KIMComputeArguments object at 0x12090cc90>, <kliff.models.kim.KIMComputeArguments object at 0x12090cd10>, <kliff.models.kim.KIMComputeArguments object at 0x12090cd90>, <kliff.models.kim.KIMComputeArguments object at 0x12090ce10>, <kliff.models.kim.KIMComputeArguments object at 0x12090ce90>, <kliff.models.kim.KIMComputeArguments object at 0x12090cf10>, <kliff.models.kim.KIMComputeArguments object at 0x12090cf90>, <kliff.models.kim.KIMComputeArguments object at 0x12090cfd0>, <kliff.models.kim.KIMComputeArguments object at 0x12090c090>, <kliff.models.kim.KIMComputeArguments object at 0x120919150>, <kliff.models.kim.KIMComputeArguments object at 0x1209191d0>, <kliff.models.kim.KIMComputeArguments object at 0x120919250>, <kliff.models.kim.KIMComputeArguments object at 0x1209192d0>, <kliff.models.kim.KIMComputeArguments object at 0x120919350>, <kliff.models.kim.KIMComputeArguments object at 0x1209193d0>, <kliff.models.kim.KIMComputeArguments object at 0x120919450>, <kliff.models.kim.KIMComputeArguments object at 0x1209194d0>, <kliff.models.kim.KIMComputeArguments object at 0x120919550>, <kliff.models.kim.KIMComputeArguments object at 0x1209195d0>, <kliff.models.kim.KIMComputeArguments object at 0x120919650>, <kliff.models.kim.KIMComputeArguments object at 0x1209196d0>, <kliff.models.kim.KIMComputeArguments object at 0x120919750>, <kliff.models.kim.KIMComputeArguments object at 0x1209197d0>, <kliff.models.kim.KIMComputeArguments object at 0x120919850>, <kliff.models.kim.KIMComputeArguments object at 0x1209198d0>, <kliff.models.kim.KIMComputeArguments object at 0x120919950>, <kliff.models.kim.KIMComputeArguments object at 0x1209199d0>, <kliff.models.kim.KIMComputeArguments object at 0x120919a50>, <kliff.models.kim.KIMComputeArguments object at 0x120919ad0>, <kliff.models.kim.KIMComputeArguments object at 0x120919b50>, <kliff.models.kim.KIMComputeArguments object at 0x120919bd0>, <kliff.models.kim.KIMComputeArguments object at 0x120919c50>, <kliff.models.kim.KIMComputeArguments object at 0x120919cd0>, <kliff.models.kim.KIMComputeArguments object at 0x120919d50>, <kliff.models.kim.KIMComputeArguments object at 0x120919dd0>, <kliff.models.kim.KIMComputeArguments object at 0x120919e50>, <kliff.models.kim.KIMComputeArguments object at 0x120919ed0>, <kliff.models.kim.KIMComputeArguments object at 0x120919f50>, <kliff.models.kim.KIMComputeArguments object at 0x120919fd0>, <kliff.models.kim.KIMComputeArguments object at 0x1209190d0>, <kliff.models.kim.KIMComputeArguments object at 0x120925110>, <kliff.models.kim.KIMComputeArguments object at 0x120925190>, <kliff.models.kim.KIMComputeArguments object at 0x120925210>, <kliff.models.kim.KIMComputeArguments object at 0x120925290>, <kliff.models.kim.KIMComputeArguments object at 0x120925310>, <kliff.models.kim.KIMComputeArguments object at 0x120925390>, <kliff.models.kim.KIMComputeArguments object at 0x120925410>, <kliff.models.kim.KIMComputeArguments object at 0x120925490>, <kliff.models.kim.KIMComputeArguments object at 0x120925510>, <kliff.models.kim.KIMComputeArguments object at 0x120925590>, <kliff.models.kim.KIMComputeArguments object at 0x120925610>, <kliff.models.kim.KIMComputeArguments object at 0x120925690>, <kliff.models.kim.KIMComputeArguments object at 0x120925710>, <kliff.models.kim.KIMComputeArguments object at 0x120925790>, <kliff.models.kim.KIMComputeArguments object at 0x120925810>, <kliff.models.kim.KIMComputeArguments object at 0x120925890>, <kliff.models.kim.KIMComputeArguments object at 0x120925910>, <kliff.models.kim.KIMComputeArguments object at 0x120925990>, <kliff.models.kim.KIMComputeArguments object at 0x120925a10>, <kliff.models.kim.KIMComputeArguments object at 0x120925a90>, <kliff.models.kim.KIMComputeArguments object at 0x120925b10>, <kliff.models.kim.KIMComputeArguments object at 0x120925b90>, <kliff.models.kim.KIMComputeArguments object at 0x120925c10>, <kliff.models.kim.KIMComputeArguments object at 0x120925c90>, <kliff.models.kim.KIMComputeArguments object at 0x120925d10>, <kliff.models.kim.KIMComputeArguments object at 0x120925d90>, <kliff.models.kim.KIMComputeArguments object at 0x120925e10>, <kliff.models.kim.KIMComputeArguments object at 0x120925e90>, <kliff.models.kim.KIMComputeArguments object at 0x120925f10>, <kliff.models.kim.KIMComputeArguments object at 0x120925f90>, <kliff.models.kim.KIMComputeArguments object at 0x120925fd0>, <kliff.models.kim.KIMComputeArguments object at 0x120925090>, <kliff.models.kim.KIMComputeArguments object at 0x120934150>, <kliff.models.kim.KIMComputeArguments object at 0x1209341d0>, <kliff.models.kim.KIMComputeArguments object at 0x120934250>, <kliff.models.kim.KIMComputeArguments object at 0x1209342d0>, <kliff.models.kim.KIMComputeArguments object at 0x120934350>, <kliff.models.kim.KIMComputeArguments object at 0x1209343d0>, <kliff.models.kim.KIMComputeArguments object at 0x120934450>, <kliff.models.kim.KIMComputeArguments object at 0x1209344d0>, <kliff.models.kim.KIMComputeArguments object at 0x120934550>, <kliff.models.kim.KIMComputeArguments object at 0x1209345d0>, <kliff.models.kim.KIMComputeArguments object at 0x120934650>, <kliff.models.kim.KIMComputeArguments object at 0x1209346d0>, <kliff.models.kim.KIMComputeArguments object at 0x120934750>, <kliff.models.kim.KIMComputeArguments object at 0x1209347d0>, <kliff.models.kim.KIMComputeArguments object at 0x120934850>, <kliff.models.kim.KIMComputeArguments object at 0x1209348d0>, <kliff.models.kim.KIMComputeArguments object at 0x120934950>, <kliff.models.kim.KIMComputeArguments object at 0x1209349d0>, <kliff.models.kim.KIMComputeArguments object at 0x120934a50>, <kliff.models.kim.KIMComputeArguments object at 0x120934ad0>, <kliff.models.kim.KIMComputeArguments object at 0x120934b50>, <kliff.models.kim.KIMComputeArguments object at 0x120934bd0>, <kliff.models.kim.KIMComputeArguments object at 0x120934c50>, <kliff.models.kim.KIMComputeArguments object at 0x120934cd0>, <kliff.models.kim.KIMComputeArguments object at 0x120934d50>, <kliff.models.kim.KIMComputeArguments object at 0x120934dd0>, <kliff.models.kim.KIMComputeArguments object at 0x120934e50>, <kliff.models.kim.KIMComputeArguments object at 0x120934ed0>, <kliff.models.kim.KIMComputeArguments object at 0x120934f50>, <kliff.models.kim.KIMComputeArguments object at 0x120934fd0>, <kliff.models.kim.KIMComputeArguments object at 0x1209340d0>, <kliff.models.kim.KIMComputeArguments object at 0x120940110>, <kliff.models.kim.KIMComputeArguments object at 0x120940190>, <kliff.models.kim.KIMComputeArguments object at 0x120940210>, <kliff.models.kim.KIMComputeArguments object at 0x120940290>, <kliff.models.kim.KIMComputeArguments object at 0x120940310>, <kliff.models.kim.KIMComputeArguments object at 0x120940390>, <kliff.models.kim.KIMComputeArguments object at 0x1209403d0>, <kliff.models.kim.KIMComputeArguments object at 0x120940490>, <kliff.models.kim.KIMComputeArguments object at 0x120940510>, <kliff.models.kim.KIMComputeArguments object at 0x120940590>, <kliff.models.kim.KIMComputeArguments object at 0x1209405d0>, <kliff.models.kim.KIMComputeArguments object at 0x120940690>, <kliff.models.kim.KIMComputeArguments object at 0x120940710>, <kliff.models.kim.KIMComputeArguments object at 0x120940790>, <kliff.models.kim.KIMComputeArguments object at 0x1209407d0>, <kliff.models.kim.KIMComputeArguments object at 0x120940890>, <kliff.models.kim.KIMComputeArguments object at 0x120940910>, <kliff.models.kim.KIMComputeArguments object at 0x120940990>, <kliff.models.kim.KIMComputeArguments object at 0x1209409d0>, <kliff.models.kim.KIMComputeArguments object at 0x120940a90>, <kliff.models.kim.KIMComputeArguments object at 0x120940b10>, <kliff.models.kim.KIMComputeArguments object at 0x120940b90>, <kliff.models.kim.KIMComputeArguments object at 0x120940bd0>, <kliff.models.kim.KIMComputeArguments object at 0x120940c90>, <kliff.models.kim.KIMComputeArguments object at 0x120940d10>, <kliff.models.kim.KIMComputeArguments object at 0x120940d90>, <kliff.models.kim.KIMComputeArguments object at 0x120940dd0>, <kliff.models.kim.KIMComputeArguments object at 0x120940e90>, <kliff.models.kim.KIMComputeArguments object at 0x120940f10>, <kliff.models.kim.KIMComputeArguments object at 0x120940f90>, <kliff.models.kim.KIMComputeArguments object at 0x120940090>, <kliff.models.kim.KIMComputeArguments object at 0x120940fd0>, <kliff.models.kim.KIMComputeArguments object at 0x12094d150>, <kliff.models.kim.KIMComputeArguments object at 0x12094d1d0>, <kliff.models.kim.KIMComputeArguments object at 0x12094d210>, <kliff.models.kim.KIMComputeArguments object at 0x12094d2d0>, <kliff.models.kim.KIMComputeArguments object at 0x12094d350>, <kliff.models.kim.KIMComputeArguments object at 0x12094d3d0>, <kliff.models.kim.KIMComputeArguments object at 0x12094d410>, <kliff.models.kim.KIMComputeArguments object at 0x12094d4d0>, <kliff.models.kim.KIMComputeArguments object at 0x12094d550>, <kliff.models.kim.KIMComputeArguments object at 0x12094d5d0>, <kliff.models.kim.KIMComputeArguments object at 0x12094d610>, <kliff.models.kim.KIMComputeArguments object at 0x12094d6d0>, <kliff.models.kim.KIMComputeArguments object at 0x12094d750>, <kliff.models.kim.KIMComputeArguments object at 0x12094d7d0>, <kliff.models.kim.KIMComputeArguments object at 0x12094d810>, <kliff.models.kim.KIMComputeArguments object at 0x12094d8d0>, <kliff.models.kim.KIMComputeArguments object at 0x12094d950>, <kliff.models.kim.KIMComputeArguments object at 0x12094d9d0>, <kliff.models.kim.KIMComputeArguments object at 0x12094da10>, <kliff.models.kim.KIMComputeArguments object at 0x12094dad0>, <kliff.models.kim.KIMComputeArguments object at 0x12094db50>, <kliff.models.kim.KIMComputeArguments object at 0x12094dbd0>, <kliff.models.kim.KIMComputeArguments object at 0x12094dc10>, <kliff.models.kim.KIMComputeArguments object at 0x12094dcd0>, <kliff.models.kim.KIMComputeArguments object at 0x12094dd50>, <kliff.models.kim.KIMComputeArguments object at 0x12094ddd0>, <kliff.models.kim.KIMComputeArguments object at 0x12094de10>, <kliff.models.kim.KIMComputeArguments object at 0x12094ded0>, <kliff.models.kim.KIMComputeArguments object at 0x12094df50>, <kliff.models.kim.KIMComputeArguments object at 0x12094dfd0>, <kliff.models.kim.KIMComputeArguments object at 0x12094d0d0>, <kliff.models.kim.KIMComputeArguments object at 0x120959110>, <kliff.models.kim.KIMComputeArguments object at 0x120959190>, <kliff.models.kim.KIMComputeArguments object at 0x120959210>, <kliff.models.kim.KIMComputeArguments object at 0x120959250>, <kliff.models.kim.KIMComputeArguments object at 0x120959310>, <kliff.models.kim.KIMComputeArguments object at 0x120959390>, <kliff.models.kim.KIMComputeArguments object at 0x120959410>, <kliff.models.kim.KIMComputeArguments object at 0x120959450>, <kliff.models.kim.KIMComputeArguments object at 0x120959510>, <kliff.models.kim.KIMComputeArguments object at 0x120959590>, <kliff.models.kim.KIMComputeArguments object at 0x120959610>, <kliff.models.kim.KIMComputeArguments object at 0x120959650>, <kliff.models.kim.KIMComputeArguments object at 0x120959710>, <kliff.models.kim.KIMComputeArguments object at 0x120959790>, <kliff.models.kim.KIMComputeArguments object at 0x120959810>, <kliff.models.kim.KIMComputeArguments object at 0x120959850>, <kliff.models.kim.KIMComputeArguments object at 0x120959910>, <kliff.models.kim.KIMComputeArguments object at 0x120959990>, <kliff.models.kim.KIMComputeArguments object at 0x120959a10>, <kliff.models.kim.KIMComputeArguments object at 0x120959a50>, <kliff.models.kim.KIMComputeArguments object at 0x120959b10>, <kliff.models.kim.KIMComputeArguments object at 0x120959b90>, <kliff.models.kim.KIMComputeArguments object at 0x120959c10>, <kliff.models.kim.KIMComputeArguments object at 0x120959c50>, <kliff.models.kim.KIMComputeArguments object at 0x120959d10>, <kliff.models.kim.KIMComputeArguments object at 0x120959d90>, <kliff.models.kim.KIMComputeArguments object at 0x120959e10>, <kliff.models.kim.KIMComputeArguments object at 0x120959e50>, <kliff.models.kim.KIMComputeArguments object at 0x120959f10>, <kliff.models.kim.KIMComputeArguments object at 0x120959f90>, <kliff.models.kim.KIMComputeArguments object at 0x120959fd0>, <kliff.models.kim.KIMComputeArguments object at 0x120959090>, <kliff.models.kim.KIMComputeArguments object at 0x120967150>, <kliff.models.kim.KIMComputeArguments object at 0x1209671d0>, <kliff.models.kim.KIMComputeArguments object at 0x120967250>, <kliff.models.kim.KIMComputeArguments object at 0x120967290>, <kliff.models.kim.KIMComputeArguments object at 0x120967350>, <kliff.models.kim.KIMComputeArguments object at 0x1209673d0>, <kliff.models.kim.KIMComputeArguments object at 0x120967450>, <kliff.models.kim.KIMComputeArguments object at 0x120967490>, <kliff.models.kim.KIMComputeArguments object at 0x120967550>, <kliff.models.kim.KIMComputeArguments object at 0x1209675d0>, <kliff.models.kim.KIMComputeArguments object at 0x120967650>, <kliff.models.kim.KIMComputeArguments object at 0x120967690>, <kliff.models.kim.KIMComputeArguments object at 0x120967750>, <kliff.models.kim.KIMComputeArguments object at 0x1209677d0>, <kliff.models.kim.KIMComputeArguments object at 0x120967850>, <kliff.models.kim.KIMComputeArguments object at 0x120967890>, <kliff.models.kim.KIMComputeArguments object at 0x120967950>, <kliff.models.kim.KIMComputeArguments object at 0x1209679d0>, <kliff.models.kim.KIMComputeArguments object at 0x120967a50>, <kliff.models.kim.KIMComputeArguments object at 0x120967a90>, <kliff.models.kim.KIMComputeArguments object at 0x120967b50>, <kliff.models.kim.KIMComputeArguments object at 0x120967bd0>, <kliff.models.kim.KIMComputeArguments object at 0x120967c50>, <kliff.models.kim.KIMComputeArguments object at 0x120967c90>, <kliff.models.kim.KIMComputeArguments object at 0x120967d50>, <kliff.models.kim.KIMComputeArguments object at 0x120967dd0>, <kliff.models.kim.KIMComputeArguments object at 0x120967e50>, <kliff.models.kim.KIMComputeArguments object at 0x120967e90>, <kliff.models.kim.KIMComputeArguments object at 0x120967f50>, <kliff.models.kim.KIMComputeArguments object at 0x120967fd0>, <kliff.models.kim.KIMComputeArguments object at 0x1209670d0>, <kliff.models.kim.KIMComputeArguments object at 0x1209730d0>, <kliff.models.kim.KIMComputeArguments object at 0x120973190>, <kliff.models.kim.KIMComputeArguments object at 0x120973210>, <kliff.models.kim.KIMComputeArguments object at 0x120973290>, <kliff.models.kim.KIMComputeArguments object at 0x1209732d0>, <kliff.models.kim.KIMComputeArguments object at 0x120973390>, <kliff.models.kim.KIMComputeArguments object at 0x120973410>, <kliff.models.kim.KIMComputeArguments object at 0x120973490>, <kliff.models.kim.KIMComputeArguments object at 0x1209734d0>, <kliff.models.kim.KIMComputeArguments object at 0x120973590>, <kliff.models.kim.KIMComputeArguments object at 0x120973610>, <kliff.models.kim.KIMComputeArguments object at 0x120973690>, <kliff.models.kim.KIMComputeArguments object at 0x1209736d0>, <kliff.models.kim.KIMComputeArguments object at 0x120973790>, <kliff.models.kim.KIMComputeArguments object at 0x120973810>, <kliff.models.kim.KIMComputeArguments object at 0x120973890>, <kliff.models.kim.KIMComputeArguments object at 0x1209738d0>, <kliff.models.kim.KIMComputeArguments object at 0x120973990>, <kliff.models.kim.KIMComputeArguments object at 0x120973a10>, <kliff.models.kim.KIMComputeArguments object at 0x120973a90>, <kliff.models.kim.KIMComputeArguments object at 0x120973ad0>, <kliff.models.kim.KIMComputeArguments object at 0x120973b90>, <kliff.models.kim.KIMComputeArguments object at 0x120973c10>, <kliff.models.kim.KIMComputeArguments object at 0x120973c90>, <kliff.models.kim.KIMComputeArguments object at 0x120973cd0>, <kliff.models.kim.KIMComputeArguments object at 0x120973d90>, <kliff.models.kim.KIMComputeArguments object at 0x120973e10>, <kliff.models.kim.KIMComputeArguments object at 0x120973e90>, <kliff.models.kim.KIMComputeArguments object at 0x120973ed0>, <kliff.models.kim.KIMComputeArguments object at 0x120973f90>, <kliff.models.kim.KIMComputeArguments object at 0x120973fd0>, <kliff.models.kim.KIMComputeArguments object at 0x120973090>, <kliff.models.kim.KIMComputeArguments object at 0x120980110>, <kliff.models.kim.KIMComputeArguments object at 0x1209801d0>, <kliff.models.kim.KIMComputeArguments object at 0x120980250>, <kliff.models.kim.KIMComputeArguments object at 0x1209802d0>, <kliff.models.kim.KIMComputeArguments object at 0x120980310>, <kliff.models.kim.KIMComputeArguments object at 0x1209803d0>, <kliff.models.kim.KIMComputeArguments object at 0x120980450>, <kliff.models.kim.KIMComputeArguments object at 0x1209804d0>, <kliff.models.kim.KIMComputeArguments object at 0x120980510>, <kliff.models.kim.KIMComputeArguments object at 0x1209805d0>, <kliff.models.kim.KIMComputeArguments object at 0x120980650>, <kliff.models.kim.KIMComputeArguments object at 0x1209806d0>, <kliff.models.kim.KIMComputeArguments object at 0x120980710>, <kliff.models.kim.KIMComputeArguments object at 0x1209807d0>, <kliff.models.kim.KIMComputeArguments object at 0x120980850>, <kliff.models.kim.KIMComputeArguments object at 0x1209808d0>, <kliff.models.kim.KIMComputeArguments object at 0x120980910>, <kliff.models.kim.KIMComputeArguments object at 0x1209809d0>, <kliff.models.kim.KIMComputeArguments object at 0x120980a50>, <kliff.models.kim.KIMComputeArguments object at 0x120980ad0>, <kliff.models.kim.KIMComputeArguments object at 0x120980b10>, <kliff.models.kim.KIMComputeArguments object at 0x120980bd0>, <kliff.models.kim.KIMComputeArguments object at 0x120980c50>, <kliff.models.kim.KIMComputeArguments object at 0x120980cd0>, <kliff.models.kim.KIMComputeArguments object at 0x120980d10>, <kliff.models.kim.KIMComputeArguments object at 0x120980dd0>, <kliff.models.kim.KIMComputeArguments object at 0x120980e50>, <kliff.models.kim.KIMComputeArguments object at 0x120980ed0>, <kliff.models.kim.KIMComputeArguments object at 0x120980f10>, <kliff.models.kim.KIMComputeArguments object at 0x120980fd0>, <kliff.models.kim.KIMComputeArguments object at 0x1209800d0>, <kliff.models.kim.KIMComputeArguments object at 0x12098d110>, <kliff.models.kim.KIMComputeArguments object at 0x12098d150>, <kliff.models.kim.KIMComputeArguments object at 0x12098d210>, <kliff.models.kim.KIMComputeArguments object at 0x12098d290>, <kliff.models.kim.KIMComputeArguments object at 0x12098d310>, <kliff.models.kim.KIMComputeArguments object at 0x12098d350>, <kliff.models.kim.KIMComputeArguments object at 0x12098d410>, <kliff.models.kim.KIMComputeArguments object at 0x12098d490>, <kliff.models.kim.KIMComputeArguments object at 0x12098d510>, <kliff.models.kim.KIMComputeArguments object at 0x12098d550>, <kliff.models.kim.KIMComputeArguments object at 0x12098d610>, <kliff.models.kim.KIMComputeArguments object at 0x12098d690>, <kliff.models.kim.KIMComputeArguments object at 0x12098d710>, <kliff.models.kim.KIMComputeArguments object at 0x12098d750>, <kliff.models.kim.KIMComputeArguments object at 0x12098d810>, <kliff.models.kim.KIMComputeArguments object at 0x12098d890>, <kliff.models.kim.KIMComputeArguments object at 0x12098d910>, <kliff.models.kim.KIMComputeArguments object at 0x12098d950>, <kliff.models.kim.KIMComputeArguments object at 0x12098da10>, <kliff.models.kim.KIMComputeArguments object at 0x12098da90>, <kliff.models.kim.KIMComputeArguments object at 0x12098db10>, <kliff.models.kim.KIMComputeArguments object at 0x12098db50>, <kliff.models.kim.KIMComputeArguments object at 0x12098dc10>, <kliff.models.kim.KIMComputeArguments object at 0x12098dc90>, <kliff.models.kim.KIMComputeArguments object at 0x12098dd10>, <kliff.models.kim.KIMComputeArguments object at 0x12098dd50>, <kliff.models.kim.KIMComputeArguments object at 0x12098de10>, <kliff.models.kim.KIMComputeArguments object at 0x12098de90>, <kliff.models.kim.KIMComputeArguments object at 0x12098df10>, <kliff.models.kim.KIMComputeArguments object at 0x12098df50>, <kliff.models.kim.KIMComputeArguments object at 0x12098dfd0>, <kliff.models.kim.KIMComputeArguments object at 0x12098d090>, <kliff.models.kim.KIMComputeArguments object at 0x12099b150>, <kliff.models.kim.KIMComputeArguments object at 0x12099b190>, <kliff.models.kim.KIMComputeArguments object at 0x12099b250>, <kliff.models.kim.KIMComputeArguments object at 0x12099b2d0>, <kliff.models.kim.KIMComputeArguments object at 0x12099b350>, <kliff.models.kim.KIMComputeArguments object at 0x12099b390>, <kliff.models.kim.KIMComputeArguments object at 0x12099b450>, <kliff.models.kim.KIMComputeArguments object at 0x12099b4d0>, <kliff.models.kim.KIMComputeArguments object at 0x12099b550>, <kliff.models.kim.KIMComputeArguments object at 0x12099b590>, <kliff.models.kim.KIMComputeArguments object at 0x12099b650>, <kliff.models.kim.KIMComputeArguments object at 0x12099b6d0>, <kliff.models.kim.KIMComputeArguments object at 0x12099b750>, <kliff.models.kim.KIMComputeArguments object at 0x12099b790>, <kliff.models.kim.KIMComputeArguments object at 0x12099b850>, <kliff.models.kim.KIMComputeArguments object at 0x12099b8d0>, <kliff.models.kim.KIMComputeArguments object at 0x12099b950>, <kliff.models.kim.KIMComputeArguments object at 0x12099b990>, <kliff.models.kim.KIMComputeArguments object at 0x12099ba50>, <kliff.models.kim.KIMComputeArguments object at 0x12099bad0>, <kliff.models.kim.KIMComputeArguments object at 0x12099bb50>, <kliff.models.kim.KIMComputeArguments object at 0x12099bb90>, <kliff.models.kim.KIMComputeArguments object at 0x12099bc50>, <kliff.models.kim.KIMComputeArguments object at 0x12099bcd0>, <kliff.models.kim.KIMComputeArguments object at 0x12099bd50>, <kliff.models.kim.KIMComputeArguments object at 0x12099bd90>, <kliff.models.kim.KIMComputeArguments object at 0x12099be50>, <kliff.models.kim.KIMComputeArguments object at 0x12099bed0>, <kliff.models.kim.KIMComputeArguments object at 0x12099bf50>, <kliff.models.kim.KIMComputeArguments object at 0x12099bf90>, <kliff.models.kim.KIMComputeArguments object at 0x12099b0d0>, <kliff.models.kim.KIMComputeArguments object at 0x1209a7110>, <kliff.models.kim.KIMComputeArguments object at 0x1209a7190>, <kliff.models.kim.KIMComputeArguments object at 0x1209a71d0>, <kliff.models.kim.KIMComputeArguments object at 0x1209a7290>, <kliff.models.kim.KIMComputeArguments object at 0x1209a7310>, <kliff.models.kim.KIMComputeArguments object at 0x1209a7390>, <kliff.models.kim.KIMComputeArguments object at 0x1209a73d0>, <kliff.models.kim.KIMComputeArguments object at 0x1209a7490>, <kliff.models.kim.KIMComputeArguments object at 0x1209a7510>, <kliff.models.kim.KIMComputeArguments object at 0x1209a7590>, <kliff.models.kim.KIMComputeArguments object at 0x1209a75d0>, <kliff.models.kim.KIMComputeArguments object at 0x1209a7690>, <kliff.models.kim.KIMComputeArguments object at 0x1209a7710>, <kliff.models.kim.KIMComputeArguments object at 0x1209a7790>, <kliff.models.kim.KIMComputeArguments object at 0x1209a77d0>, <kliff.models.kim.KIMComputeArguments object at 0x1209a7890>, <kliff.models.kim.KIMComputeArguments object at 0x1209a7910>, <kliff.models.kim.KIMComputeArguments object at 0x1209a7990>, <kliff.models.kim.KIMComputeArguments object at 0x1209a79d0>, <kliff.models.kim.KIMComputeArguments object at 0x1209a7a90>, <kliff.models.kim.KIMComputeArguments object at 0x1209a7b10>, <kliff.models.kim.KIMComputeArguments object at 0x1209a7b90>, <kliff.models.kim.KIMComputeArguments object at 0x1209a7bd0>, <kliff.models.kim.KIMComputeArguments object at 0x1209a7c90>, <kliff.models.kim.KIMComputeArguments object at 0x1209a7d10>, <kliff.models.kim.KIMComputeArguments object at 0x1209a7d90>, <kliff.models.kim.KIMComputeArguments object at 0x1209a7dd0>, <kliff.models.kim.KIMComputeArguments object at 0x1209a7e90>, <kliff.models.kim.KIMComputeArguments object at 0x1209a7f10>, <kliff.models.kim.KIMComputeArguments object at 0x1209a7f90>, <kliff.models.kim.KIMComputeArguments object at 0x1209a7090>, <kliff.models.kim.KIMComputeArguments object at 0x1209a7fd0>, <kliff.models.kim.KIMComputeArguments object at 0x1209b4150>, <kliff.models.kim.KIMComputeArguments object at 0x1209b41d0>, <kliff.models.kim.KIMComputeArguments object at 0x1209b4210>, <kliff.models.kim.KIMComputeArguments object at 0x1209b42d0>, <kliff.models.kim.KIMComputeArguments object at 0x1209b4350>, <kliff.models.kim.KIMComputeArguments object at 0x1209b43d0>, <kliff.models.kim.KIMComputeArguments object at 0x1209b4410>, <kliff.models.kim.KIMComputeArguments object at 0x1209b44d0>, <kliff.models.kim.KIMComputeArguments object at 0x1209b4550>, <kliff.models.kim.KIMComputeArguments object at 0x1209b45d0>, <kliff.models.kim.KIMComputeArguments object at 0x1209b4610>, <kliff.models.kim.KIMComputeArguments object at 0x1209b46d0>, <kliff.models.kim.KIMComputeArguments object at 0x1209b4750>, <kliff.models.kim.KIMComputeArguments object at 0x1209b47d0>, <kliff.models.kim.KIMComputeArguments object at 0x1209b4810>, <kliff.models.kim.KIMComputeArguments object at 0x1209b48d0>, <kliff.models.kim.KIMComputeArguments object at 0x1209b4950>, <kliff.models.kim.KIMComputeArguments object at 0x1209b49d0>, <kliff.models.kim.KIMComputeArguments object at 0x1209b4a10>, <kliff.models.kim.KIMComputeArguments object at 0x1209b4ad0>, <kliff.models.kim.KIMComputeArguments object at 0x1209b4b50>, <kliff.models.kim.KIMComputeArguments object at 0x1209b4bd0>, <kliff.models.kim.KIMComputeArguments object at 0x1209b4c10>, <kliff.models.kim.KIMComputeArguments object at 0x1209b4cd0>, <kliff.models.kim.KIMComputeArguments object at 0x1209b4d50>, <kliff.models.kim.KIMComputeArguments object at 0x1209b4dd0>, <kliff.models.kim.KIMComputeArguments object at 0x1209b4e10>, <kliff.models.kim.KIMComputeArguments object at 0x1209b4ed0>, <kliff.models.kim.KIMComputeArguments object at 0x1209b4f50>, <kliff.models.kim.KIMComputeArguments object at 0x1209b4fd0>, <kliff.models.kim.KIMComputeArguments object at 0x1209b40d0>, <kliff.models.kim.KIMComputeArguments object at 0x1209c0110>, <kliff.models.kim.KIMComputeArguments object at 0x1209c0190>, <kliff.models.kim.KIMComputeArguments object at 0x1209c0210>, <kliff.models.kim.KIMComputeArguments object at 0x1209c0250>, <kliff.models.kim.KIMComputeArguments object at 0x1209c0310>, <kliff.models.kim.KIMComputeArguments object at 0x1209c0390>, <kliff.models.kim.KIMComputeArguments object at 0x1209c0410>, <kliff.models.kim.KIMComputeArguments object at 0x1209c0450>, <kliff.models.kim.KIMComputeArguments object at 0x1209c0510>, <kliff.models.kim.KIMComputeArguments object at 0x1209c0590>, <kliff.models.kim.KIMComputeArguments object at 0x1209c0610>, <kliff.models.kim.KIMComputeArguments object at 0x1209c0650>, <kliff.models.kim.KIMComputeArguments object at 0x1209c0710>, <kliff.models.kim.KIMComputeArguments object at 0x1209c0790>, <kliff.models.kim.KIMComputeArguments object at 0x1209c0810>, <kliff.models.kim.KIMComputeArguments object at 0x1209c0850>, <kliff.models.kim.KIMComputeArguments object at 0x1209c0910>, <kliff.models.kim.KIMComputeArguments object at 0x1209c0990>, <kliff.models.kim.KIMComputeArguments object at 0x1209c0a10>, <kliff.models.kim.KIMComputeArguments object at 0x1209c0a50>, <kliff.models.kim.KIMComputeArguments object at 0x1209c0b10>, <kliff.models.kim.KIMComputeArguments object at 0x1209c0b90>, <kliff.models.kim.KIMComputeArguments object at 0x1209c0c10>, <kliff.models.kim.KIMComputeArguments object at 0x1209c0c50>, <kliff.models.kim.KIMComputeArguments object at 0x1209c0d10>, <kliff.models.kim.KIMComputeArguments object at 0x1209c0d90>, <kliff.models.kim.KIMComputeArguments object at 0x1209c0e10>, <kliff.models.kim.KIMComputeArguments object at 0x1209c0e50>, <kliff.models.kim.KIMComputeArguments object at 0x1209c0f10>, <kliff.models.kim.KIMComputeArguments object at 0x1209c0f90>, <kliff.models.kim.KIMComputeArguments object at 0x1209c0fd0>, <kliff.models.kim.KIMComputeArguments object at 0x1209c0090>, <kliff.models.kim.KIMComputeArguments object at 0x1209cf150>, <kliff.models.kim.KIMComputeArguments object at 0x1209cf1d0>, <kliff.models.kim.KIMComputeArguments object at 0x1209cf250>, <kliff.models.kim.KIMComputeArguments object at 0x1209cf290>, <kliff.models.kim.KIMComputeArguments object at 0x1209cf350>, <kliff.models.kim.KIMComputeArguments object at 0x1209cf3d0>, <kliff.models.kim.KIMComputeArguments object at 0x1209cf450>, <kliff.models.kim.KIMComputeArguments object at 0x1209cf490>, <kliff.models.kim.KIMComputeArguments object at 0x1209cf550>, <kliff.models.kim.KIMComputeArguments object at 0x1209cf5d0>, <kliff.models.kim.KIMComputeArguments object at 0x1209cf650>, <kliff.models.kim.KIMComputeArguments object at 0x1209cf690>, <kliff.models.kim.KIMComputeArguments object at 0x1209cf750>, <kliff.models.kim.KIMComputeArguments object at 0x1209cf7d0>, <kliff.models.kim.KIMComputeArguments object at 0x1209cf850>, <kliff.models.kim.KIMComputeArguments object at 0x1209cf890>, <kliff.models.kim.KIMComputeArguments object at 0x1209cf950>, <kliff.models.kim.KIMComputeArguments object at 0x1209cf9d0>, <kliff.models.kim.KIMComputeArguments object at 0x1209cfa50>, <kliff.models.kim.KIMComputeArguments object at 0x1209cfa90>, <kliff.models.kim.KIMComputeArguments object at 0x1209cfb50>, <kliff.models.kim.KIMComputeArguments object at 0x1209cfbd0>, <kliff.models.kim.KIMComputeArguments object at 0x1209cfc50>, <kliff.models.kim.KIMComputeArguments object at 0x1209cfc90>, <kliff.models.kim.KIMComputeArguments object at 0x1209cfd50>, <kliff.models.kim.KIMComputeArguments object at 0x1209cfdd0>, <kliff.models.kim.KIMComputeArguments object at 0x1209cfe50>, <kliff.models.kim.KIMComputeArguments object at 0x1209cfe90>, <kliff.models.kim.KIMComputeArguments object at 0x1209cff50>, <kliff.models.kim.KIMComputeArguments object at 0x1209cffd0>, <kliff.models.kim.KIMComputeArguments object at 0x1209cf0d0>, <kliff.models.kim.KIMComputeArguments object at 0x1209db0d0>, <kliff.models.kim.KIMComputeArguments object at 0x1209db190>, <kliff.models.kim.KIMComputeArguments object at 0x1209db210>, <kliff.models.kim.KIMComputeArguments object at 0x1209db290>, <kliff.models.kim.KIMComputeArguments object at 0x1209db2d0>, <kliff.models.kim.KIMComputeArguments object at 0x1209db390>, <kliff.models.kim.KIMComputeArguments object at 0x1209db410>, <kliff.models.kim.KIMComputeArguments object at 0x1209db490>, <kliff.models.kim.KIMComputeArguments object at 0x1209db4d0>, <kliff.models.kim.KIMComputeArguments object at 0x1209db590>, <kliff.models.kim.KIMComputeArguments object at 0x1209db610>, <kliff.models.kim.KIMComputeArguments object at 0x1209db690>, <kliff.models.kim.KIMComputeArguments object at 0x1209db6d0>, <kliff.models.kim.KIMComputeArguments object at 0x1209db790>, <kliff.models.kim.KIMComputeArguments object at 0x1209db810>, <kliff.models.kim.KIMComputeArguments object at 0x1209db890>, <kliff.models.kim.KIMComputeArguments object at 0x1209db8d0>, <kliff.models.kim.KIMComputeArguments object at 0x1209db990>, <kliff.models.kim.KIMComputeArguments object at 0x1209dba10>, <kliff.models.kim.KIMComputeArguments object at 0x1209dba90>, <kliff.models.kim.KIMComputeArguments object at 0x1209dbad0>, <kliff.models.kim.KIMComputeArguments object at 0x1209dbb90>, <kliff.models.kim.KIMComputeArguments object at 0x1209dbc10>, <kliff.models.kim.KIMComputeArguments object at 0x1209dbc90>, <kliff.models.kim.KIMComputeArguments object at 0x1209dbcd0>, <kliff.models.kim.KIMComputeArguments object at 0x1209dbd90>]



where ``calc.create(configs)`` does some initializations for each
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
algorithm is applied to minimize the loss, and the minimization is allowed to run for
a max number of 100 iterations.


.. code-block:: default


    steps = 100
    residual_data = {"energy_weight": 1.0, "forces_weight": 0.1}
    loss = Loss(calc, residual_data=residual_data, nprocs=2)
    loss.minimize(method="L-BFGS-B", options={"disp": True, "maxiter": steps})






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Start minimization using method: L-BFGS-B.
    Running in multiprocessing mode with 2 processes.

    Finish minimization using method: L-BFGS-B.

          fun: 0.6940780132855155
     hess_inv: <3x3 LbfgsInvHessProduct with dtype=float64>
          jac: array([ 1.17683641e-05, -1.67532654e-04,  9.08162434e-06])
      message: b'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH'
         nfev: 176
          nit: 36
       status: 0
      success: True
            x: array([14.93863516,  0.58740277,  2.20146189])



The minimization stops after running for 27 steps.  After the minimization, we'd better
save the model, which can be loaded later for the purpose to do a retraining or
evaluations. If satisfied with the fitted model, you can also write it as a KIM model
that can be used with LAMMPS_, GULP_, ASE_, etc. via the kim-api_.


.. code-block:: default


    model.echo_fitting_params()
    model.save("kliff_model.pkl")
    model.write_kim_model()
    model.load("kliff_model.pkl")






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    #================================================================================
    # Model parameters that are optimized.
    #================================================================================

    A 1
      1.4938635159369211e+01   1.0000000000000000e+00   2.0000000000000000e+01 

    B 1
      5.8740276691185811e-01 

    sigma 1
      2.0951000000000000e+00 fix 

    gamma 1
      2.2014618855893255e+00 

    KLIFF trained model write to "/Users/mjwen/Applications/kliff/examples/SW_StillingerWeber_1985_Si__MO_405512056662_005_kliff_trained"




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

   **Total running time of the script:** ( 1 minutes  39.395 seconds)


.. _sphx_glr_download_auto_examples_example_kim_SW_Si.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: example_kim_SW_Si.py <example_kim_SW_Si.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: example_kim_SW_Si.ipynb <example_kim_SW_Si.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
