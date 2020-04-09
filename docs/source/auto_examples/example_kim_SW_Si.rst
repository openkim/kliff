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


    from kliff.models import KIM
    from kliff.loss import Loss
    from kliff.calculators import Calculator
    from kliff.dataset import Dataset









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


    [<kliff.models.kim.KIMComputeArguments object at 0x11deaabd0>, <kliff.models.kim.KIMComputeArguments object at 0x11c1a7110>, <kliff.models.kim.KIMComputeArguments object at 0x11deaac10>, <kliff.models.kim.KIMComputeArguments object at 0x11deaac90>, <kliff.models.kim.KIMComputeArguments object at 0x11deaad90>, <kliff.models.kim.KIMComputeArguments object at 0x11deaaf10>, <kliff.models.kim.KIMComputeArguments object at 0x11deaae90>, <kliff.models.kim.KIMComputeArguments object at 0x11deaafd0>, <kliff.models.kim.KIMComputeArguments object at 0x11deaaf50>, <kliff.models.kim.KIMComputeArguments object at 0x11deaae50>, <kliff.models.kim.KIMComputeArguments object at 0x11dd3d110>, <kliff.models.kim.KIMComputeArguments object at 0x11dd3d1d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dd3d250>, <kliff.models.kim.KIMComputeArguments object at 0x11dd3d2d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dd3d350>, <kliff.models.kim.KIMComputeArguments object at 0x11dd3d3d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dd3d450>, <kliff.models.kim.KIMComputeArguments object at 0x11dd3d4d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dd3d550>, <kliff.models.kim.KIMComputeArguments object at 0x11dd3d5d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dd3d650>, <kliff.models.kim.KIMComputeArguments object at 0x11dd3d6d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dd3d750>, <kliff.models.kim.KIMComputeArguments object at 0x11dd3d7d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dd3d850>, <kliff.models.kim.KIMComputeArguments object at 0x11dd3d8d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dd3d950>, <kliff.models.kim.KIMComputeArguments object at 0x11dd3d9d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dd3da50>, <kliff.models.kim.KIMComputeArguments object at 0x11dd3dad0>, <kliff.models.kim.KIMComputeArguments object at 0x11dd3db50>, <kliff.models.kim.KIMComputeArguments object at 0x11dd3dbd0>, <kliff.models.kim.KIMComputeArguments object at 0x11dd3dc50>, <kliff.models.kim.KIMComputeArguments object at 0x11dd3dcd0>, <kliff.models.kim.KIMComputeArguments object at 0x11dd3dd50>, <kliff.models.kim.KIMComputeArguments object at 0x11dd3ddd0>, <kliff.models.kim.KIMComputeArguments object at 0x11dd3de50>, <kliff.models.kim.KIMComputeArguments object at 0x11dd3ded0>, <kliff.models.kim.KIMComputeArguments object at 0x11dd3df50>, <kliff.models.kim.KIMComputeArguments object at 0x11dd3dfd0>, <kliff.models.kim.KIMComputeArguments object at 0x11dd3d150>, <kliff.models.kim.KIMComputeArguments object at 0x11deb4110>, <kliff.models.kim.KIMComputeArguments object at 0x11deb4190>, <kliff.models.kim.KIMComputeArguments object at 0x11deb4210>, <kliff.models.kim.KIMComputeArguments object at 0x11deb4290>, <kliff.models.kim.KIMComputeArguments object at 0x11deb4310>, <kliff.models.kim.KIMComputeArguments object at 0x11deb4390>, <kliff.models.kim.KIMComputeArguments object at 0x11deb4410>, <kliff.models.kim.KIMComputeArguments object at 0x11deb4490>, <kliff.models.kim.KIMComputeArguments object at 0x11deb4510>, <kliff.models.kim.KIMComputeArguments object at 0x11deb4590>, <kliff.models.kim.KIMComputeArguments object at 0x11deb4610>, <kliff.models.kim.KIMComputeArguments object at 0x11deb4690>, <kliff.models.kim.KIMComputeArguments object at 0x11deb4710>, <kliff.models.kim.KIMComputeArguments object at 0x11deb4790>, <kliff.models.kim.KIMComputeArguments object at 0x11deb4810>, <kliff.models.kim.KIMComputeArguments object at 0x11deb4890>, <kliff.models.kim.KIMComputeArguments object at 0x11deb4910>, <kliff.models.kim.KIMComputeArguments object at 0x11deb4990>, <kliff.models.kim.KIMComputeArguments object at 0x11dc33b50>, <kliff.models.kim.KIMComputeArguments object at 0x11deb4a10>, <kliff.models.kim.KIMComputeArguments object at 0x11deb4ad0>, <kliff.models.kim.KIMComputeArguments object at 0x11deb4b50>, <kliff.models.kim.KIMComputeArguments object at 0x11deb4bd0>, <kliff.models.kim.KIMComputeArguments object at 0x11deb4c50>, <kliff.models.kim.KIMComputeArguments object at 0x11deb4cd0>, <kliff.models.kim.KIMComputeArguments object at 0x11deb4d50>, <kliff.models.kim.KIMComputeArguments object at 0x11deb4dd0>, <kliff.models.kim.KIMComputeArguments object at 0x11deb4e50>, <kliff.models.kim.KIMComputeArguments object at 0x11deb4ed0>, <kliff.models.kim.KIMComputeArguments object at 0x11deb4f50>, <kliff.models.kim.KIMComputeArguments object at 0x11deb4fd0>, <kliff.models.kim.KIMComputeArguments object at 0x11deb4a50>, <kliff.models.kim.KIMComputeArguments object at 0x11dec2110>, <kliff.models.kim.KIMComputeArguments object at 0x11dec2190>, <kliff.models.kim.KIMComputeArguments object at 0x11dec2210>, <kliff.models.kim.KIMComputeArguments object at 0x11dec2290>, <kliff.models.kim.KIMComputeArguments object at 0x11dec2310>, <kliff.models.kim.KIMComputeArguments object at 0x11dec2390>, <kliff.models.kim.KIMComputeArguments object at 0x11dec2410>, <kliff.models.kim.KIMComputeArguments object at 0x11dec2490>, <kliff.models.kim.KIMComputeArguments object at 0x11dec2510>, <kliff.models.kim.KIMComputeArguments object at 0x11dec2590>, <kliff.models.kim.KIMComputeArguments object at 0x11dec2610>, <kliff.models.kim.KIMComputeArguments object at 0x11dec2690>, <kliff.models.kim.KIMComputeArguments object at 0x11dec2710>, <kliff.models.kim.KIMComputeArguments object at 0x11dec2790>, <kliff.models.kim.KIMComputeArguments object at 0x11dec2810>, <kliff.models.kim.KIMComputeArguments object at 0x11dec2890>, <kliff.models.kim.KIMComputeArguments object at 0x11dec2910>, <kliff.models.kim.KIMComputeArguments object at 0x11dec2990>, <kliff.models.kim.KIMComputeArguments object at 0x11dec2a10>, <kliff.models.kim.KIMComputeArguments object at 0x11dec2a90>, <kliff.models.kim.KIMComputeArguments object at 0x11dec2b10>, <kliff.models.kim.KIMComputeArguments object at 0x11dec2b90>, <kliff.models.kim.KIMComputeArguments object at 0x11dec2c10>, <kliff.models.kim.KIMComputeArguments object at 0x11dec2c90>, <kliff.models.kim.KIMComputeArguments object at 0x11dec2d10>, <kliff.models.kim.KIMComputeArguments object at 0x11dec2d90>, <kliff.models.kim.KIMComputeArguments object at 0x11dec2e10>, <kliff.models.kim.KIMComputeArguments object at 0x11dec2e90>, <kliff.models.kim.KIMComputeArguments object at 0x11dec2f10>, <kliff.models.kim.KIMComputeArguments object at 0x11dec2f90>, <kliff.models.kim.KIMComputeArguments object at 0x11dec2fd0>, <kliff.models.kim.KIMComputeArguments object at 0x11dec2050>, <kliff.models.kim.KIMComputeArguments object at 0x11ded0150>, <kliff.models.kim.KIMComputeArguments object at 0x11ded01d0>, <kliff.models.kim.KIMComputeArguments object at 0x11ded0250>, <kliff.models.kim.KIMComputeArguments object at 0x11ded02d0>, <kliff.models.kim.KIMComputeArguments object at 0x11ded0350>, <kliff.models.kim.KIMComputeArguments object at 0x11ded03d0>, <kliff.models.kim.KIMComputeArguments object at 0x11ded0450>, <kliff.models.kim.KIMComputeArguments object at 0x11ded04d0>, <kliff.models.kim.KIMComputeArguments object at 0x11ded0550>, <kliff.models.kim.KIMComputeArguments object at 0x11ded05d0>, <kliff.models.kim.KIMComputeArguments object at 0x11ded0650>, <kliff.models.kim.KIMComputeArguments object at 0x11ded06d0>, <kliff.models.kim.KIMComputeArguments object at 0x11ded0750>, <kliff.models.kim.KIMComputeArguments object at 0x11ded07d0>, <kliff.models.kim.KIMComputeArguments object at 0x11ded0850>, <kliff.models.kim.KIMComputeArguments object at 0x11ded08d0>, <kliff.models.kim.KIMComputeArguments object at 0x11ded0950>, <kliff.models.kim.KIMComputeArguments object at 0x11ded09d0>, <kliff.models.kim.KIMComputeArguments object at 0x11ded0a50>, <kliff.models.kim.KIMComputeArguments object at 0x11ded0ad0>, <kliff.models.kim.KIMComputeArguments object at 0x11ded0b50>, <kliff.models.kim.KIMComputeArguments object at 0x11ded0bd0>, <kliff.models.kim.KIMComputeArguments object at 0x11ded0c50>, <kliff.models.kim.KIMComputeArguments object at 0x11ded0cd0>, <kliff.models.kim.KIMComputeArguments object at 0x11ded0d50>, <kliff.models.kim.KIMComputeArguments object at 0x11ded0dd0>, <kliff.models.kim.KIMComputeArguments object at 0x11ded0e50>, <kliff.models.kim.KIMComputeArguments object at 0x11ded0ed0>, <kliff.models.kim.KIMComputeArguments object at 0x11ded0f50>, <kliff.models.kim.KIMComputeArguments object at 0x11ded0fd0>, <kliff.models.kim.KIMComputeArguments object at 0x11ded00d0>, <kliff.models.kim.KIMComputeArguments object at 0x11deda110>, <kliff.models.kim.KIMComputeArguments object at 0x11deda190>, <kliff.models.kim.KIMComputeArguments object at 0x11deda210>, <kliff.models.kim.KIMComputeArguments object at 0x11deda290>, <kliff.models.kim.KIMComputeArguments object at 0x11deda310>, <kliff.models.kim.KIMComputeArguments object at 0x11deda390>, <kliff.models.kim.KIMComputeArguments object at 0x11deda410>, <kliff.models.kim.KIMComputeArguments object at 0x11deda490>, <kliff.models.kim.KIMComputeArguments object at 0x11deda510>, <kliff.models.kim.KIMComputeArguments object at 0x11deda590>, <kliff.models.kim.KIMComputeArguments object at 0x11deda610>, <kliff.models.kim.KIMComputeArguments object at 0x11deda690>, <kliff.models.kim.KIMComputeArguments object at 0x11deda710>, <kliff.models.kim.KIMComputeArguments object at 0x11deda790>, <kliff.models.kim.KIMComputeArguments object at 0x11deda810>, <kliff.models.kim.KIMComputeArguments object at 0x11deda890>, <kliff.models.kim.KIMComputeArguments object at 0x11deda910>, <kliff.models.kim.KIMComputeArguments object at 0x11deda990>, <kliff.models.kim.KIMComputeArguments object at 0x11dedaa10>, <kliff.models.kim.KIMComputeArguments object at 0x11dedaa90>, <kliff.models.kim.KIMComputeArguments object at 0x11dedab10>, <kliff.models.kim.KIMComputeArguments object at 0x11dedab90>, <kliff.models.kim.KIMComputeArguments object at 0x11dedac10>, <kliff.models.kim.KIMComputeArguments object at 0x11dedac90>, <kliff.models.kim.KIMComputeArguments object at 0x11dedad10>, <kliff.models.kim.KIMComputeArguments object at 0x11dedad90>, <kliff.models.kim.KIMComputeArguments object at 0x11dedae10>, <kliff.models.kim.KIMComputeArguments object at 0x11dedae90>, <kliff.models.kim.KIMComputeArguments object at 0x11dedaf10>, <kliff.models.kim.KIMComputeArguments object at 0x11dedaf90>, <kliff.models.kim.KIMComputeArguments object at 0x11dedafd0>, <kliff.models.kim.KIMComputeArguments object at 0x11deda090>, <kliff.models.kim.KIMComputeArguments object at 0x11dee8150>, <kliff.models.kim.KIMComputeArguments object at 0x11dee81d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dee8250>, <kliff.models.kim.KIMComputeArguments object at 0x11dee82d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dee8350>, <kliff.models.kim.KIMComputeArguments object at 0x11dee83d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dee8450>, <kliff.models.kim.KIMComputeArguments object at 0x11dee84d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dee8550>, <kliff.models.kim.KIMComputeArguments object at 0x11dee85d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dee8650>, <kliff.models.kim.KIMComputeArguments object at 0x11dee86d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dee8750>, <kliff.models.kim.KIMComputeArguments object at 0x11dee87d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dee8850>, <kliff.models.kim.KIMComputeArguments object at 0x11dee88d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dee8950>, <kliff.models.kim.KIMComputeArguments object at 0x11dee89d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dee8a50>, <kliff.models.kim.KIMComputeArguments object at 0x11dee8ad0>, <kliff.models.kim.KIMComputeArguments object at 0x11dee8b50>, <kliff.models.kim.KIMComputeArguments object at 0x11dee8bd0>, <kliff.models.kim.KIMComputeArguments object at 0x11dee8c50>, <kliff.models.kim.KIMComputeArguments object at 0x11dee8cd0>, <kliff.models.kim.KIMComputeArguments object at 0x11dee8d50>, <kliff.models.kim.KIMComputeArguments object at 0x11dee8dd0>, <kliff.models.kim.KIMComputeArguments object at 0x11dee8e50>, <kliff.models.kim.KIMComputeArguments object at 0x11dee8ed0>, <kliff.models.kim.KIMComputeArguments object at 0x11dee8f50>, <kliff.models.kim.KIMComputeArguments object at 0x11dee8fd0>, <kliff.models.kim.KIMComputeArguments object at 0x11dee8050>, <kliff.models.kim.KIMComputeArguments object at 0x11def6110>, <kliff.models.kim.KIMComputeArguments object at 0x11def6190>, <kliff.models.kim.KIMComputeArguments object at 0x11def6210>, <kliff.models.kim.KIMComputeArguments object at 0x11def6290>, <kliff.models.kim.KIMComputeArguments object at 0x11def6310>, <kliff.models.kim.KIMComputeArguments object at 0x11def6390>, <kliff.models.kim.KIMComputeArguments object at 0x11def6410>, <kliff.models.kim.KIMComputeArguments object at 0x11def6490>, <kliff.models.kim.KIMComputeArguments object at 0x11def6510>, <kliff.models.kim.KIMComputeArguments object at 0x11def6590>, <kliff.models.kim.KIMComputeArguments object at 0x11def6610>, <kliff.models.kim.KIMComputeArguments object at 0x11def6690>, <kliff.models.kim.KIMComputeArguments object at 0x11def6710>, <kliff.models.kim.KIMComputeArguments object at 0x11def6790>, <kliff.models.kim.KIMComputeArguments object at 0x11def6810>, <kliff.models.kim.KIMComputeArguments object at 0x11def6890>, <kliff.models.kim.KIMComputeArguments object at 0x11def6910>, <kliff.models.kim.KIMComputeArguments object at 0x11def6990>, <kliff.models.kim.KIMComputeArguments object at 0x11def6a10>, <kliff.models.kim.KIMComputeArguments object at 0x11def6a90>, <kliff.models.kim.KIMComputeArguments object at 0x11def6b10>, <kliff.models.kim.KIMComputeArguments object at 0x11def6b90>, <kliff.models.kim.KIMComputeArguments object at 0x11def6c10>, <kliff.models.kim.KIMComputeArguments object at 0x11def6c90>, <kliff.models.kim.KIMComputeArguments object at 0x11def6d10>, <kliff.models.kim.KIMComputeArguments object at 0x11def6d90>, <kliff.models.kim.KIMComputeArguments object at 0x11def6e10>, <kliff.models.kim.KIMComputeArguments object at 0x11def6e90>, <kliff.models.kim.KIMComputeArguments object at 0x11def6f10>, <kliff.models.kim.KIMComputeArguments object at 0x11def6f90>, <kliff.models.kim.KIMComputeArguments object at 0x11def6fd0>, <kliff.models.kim.KIMComputeArguments object at 0x11def6050>, <kliff.models.kim.KIMComputeArguments object at 0x11df02150>, <kliff.models.kim.KIMComputeArguments object at 0x11df021d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df02250>, <kliff.models.kim.KIMComputeArguments object at 0x11df022d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df02350>, <kliff.models.kim.KIMComputeArguments object at 0x11df023d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df02450>, <kliff.models.kim.KIMComputeArguments object at 0x11df024d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df02550>, <kliff.models.kim.KIMComputeArguments object at 0x11df025d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df02650>, <kliff.models.kim.KIMComputeArguments object at 0x11df026d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df02750>, <kliff.models.kim.KIMComputeArguments object at 0x11df027d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df02850>, <kliff.models.kim.KIMComputeArguments object at 0x11df028d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df02950>, <kliff.models.kim.KIMComputeArguments object at 0x11df029d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df02a50>, <kliff.models.kim.KIMComputeArguments object at 0x11df02ad0>, <kliff.models.kim.KIMComputeArguments object at 0x11df02b50>, <kliff.models.kim.KIMComputeArguments object at 0x11df02bd0>, <kliff.models.kim.KIMComputeArguments object at 0x11df02c50>, <kliff.models.kim.KIMComputeArguments object at 0x11df02cd0>, <kliff.models.kim.KIMComputeArguments object at 0x11df02d50>, <kliff.models.kim.KIMComputeArguments object at 0x11df02dd0>, <kliff.models.kim.KIMComputeArguments object at 0x11df02e50>, <kliff.models.kim.KIMComputeArguments object at 0x11df02ed0>, <kliff.models.kim.KIMComputeArguments object at 0x11df02f50>, <kliff.models.kim.KIMComputeArguments object at 0x11df02fd0>, <kliff.models.kim.KIMComputeArguments object at 0x11df020d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df0e110>, <kliff.models.kim.KIMComputeArguments object at 0x11df0e190>, <kliff.models.kim.KIMComputeArguments object at 0x11df0e210>, <kliff.models.kim.KIMComputeArguments object at 0x11df0e290>, <kliff.models.kim.KIMComputeArguments object at 0x11df0e310>, <kliff.models.kim.KIMComputeArguments object at 0x11df0e390>, <kliff.models.kim.KIMComputeArguments object at 0x11df0e410>, <kliff.models.kim.KIMComputeArguments object at 0x11df0e490>, <kliff.models.kim.KIMComputeArguments object at 0x11df0e510>, <kliff.models.kim.KIMComputeArguments object at 0x11df0e590>, <kliff.models.kim.KIMComputeArguments object at 0x11df0e610>, <kliff.models.kim.KIMComputeArguments object at 0x11df0e690>, <kliff.models.kim.KIMComputeArguments object at 0x11df0e710>, <kliff.models.kim.KIMComputeArguments object at 0x11df0e790>, <kliff.models.kim.KIMComputeArguments object at 0x11df0e810>, <kliff.models.kim.KIMComputeArguments object at 0x11df0e890>, <kliff.models.kim.KIMComputeArguments object at 0x11df0e910>, <kliff.models.kim.KIMComputeArguments object at 0x11df0e990>, <kliff.models.kim.KIMComputeArguments object at 0x11df0ea10>, <kliff.models.kim.KIMComputeArguments object at 0x11df0ea90>, <kliff.models.kim.KIMComputeArguments object at 0x11df0eb10>, <kliff.models.kim.KIMComputeArguments object at 0x11df0eb90>, <kliff.models.kim.KIMComputeArguments object at 0x11df0ec10>, <kliff.models.kim.KIMComputeArguments object at 0x11df0ec90>, <kliff.models.kim.KIMComputeArguments object at 0x11df0ed10>, <kliff.models.kim.KIMComputeArguments object at 0x11df0ed90>, <kliff.models.kim.KIMComputeArguments object at 0x11df0ee10>, <kliff.models.kim.KIMComputeArguments object at 0x11df0ee90>, <kliff.models.kim.KIMComputeArguments object at 0x11df0ef10>, <kliff.models.kim.KIMComputeArguments object at 0x11df0ef90>, <kliff.models.kim.KIMComputeArguments object at 0x11df0efd0>, <kliff.models.kim.KIMComputeArguments object at 0x11df0e090>, <kliff.models.kim.KIMComputeArguments object at 0x11df1c150>, <kliff.models.kim.KIMComputeArguments object at 0x11df1c1d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df1c250>, <kliff.models.kim.KIMComputeArguments object at 0x11df1c2d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df1c350>, <kliff.models.kim.KIMComputeArguments object at 0x11df1c3d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df1c450>, <kliff.models.kim.KIMComputeArguments object at 0x11df1c4d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df1c550>, <kliff.models.kim.KIMComputeArguments object at 0x11df1c5d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df1c650>, <kliff.models.kim.KIMComputeArguments object at 0x11df1c6d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df1c750>, <kliff.models.kim.KIMComputeArguments object at 0x11df1c7d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df1c850>, <kliff.models.kim.KIMComputeArguments object at 0x11df1c8d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df1c950>, <kliff.models.kim.KIMComputeArguments object at 0x11df1c9d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df1ca50>, <kliff.models.kim.KIMComputeArguments object at 0x11df1cad0>, <kliff.models.kim.KIMComputeArguments object at 0x11df1cb50>, <kliff.models.kim.KIMComputeArguments object at 0x11df1cbd0>, <kliff.models.kim.KIMComputeArguments object at 0x11df1cc50>, <kliff.models.kim.KIMComputeArguments object at 0x11df1ccd0>, <kliff.models.kim.KIMComputeArguments object at 0x11df1cd50>, <kliff.models.kim.KIMComputeArguments object at 0x11df1cdd0>, <kliff.models.kim.KIMComputeArguments object at 0x11df1ce50>, <kliff.models.kim.KIMComputeArguments object at 0x11df1ced0>, <kliff.models.kim.KIMComputeArguments object at 0x11df1cf50>, <kliff.models.kim.KIMComputeArguments object at 0x11df1cfd0>, <kliff.models.kim.KIMComputeArguments object at 0x11df1c050>, <kliff.models.kim.KIMComputeArguments object at 0x11df2a110>, <kliff.models.kim.KIMComputeArguments object at 0x11df2a190>, <kliff.models.kim.KIMComputeArguments object at 0x11df2a210>, <kliff.models.kim.KIMComputeArguments object at 0x11df2a290>, <kliff.models.kim.KIMComputeArguments object at 0x11df2a310>, <kliff.models.kim.KIMComputeArguments object at 0x11df2a390>, <kliff.models.kim.KIMComputeArguments object at 0x11df2a410>, <kliff.models.kim.KIMComputeArguments object at 0x11df2a490>, <kliff.models.kim.KIMComputeArguments object at 0x11df2a510>, <kliff.models.kim.KIMComputeArguments object at 0x11df2a590>, <kliff.models.kim.KIMComputeArguments object at 0x11df2a610>, <kliff.models.kim.KIMComputeArguments object at 0x11df2a690>, <kliff.models.kim.KIMComputeArguments object at 0x11df2a710>, <kliff.models.kim.KIMComputeArguments object at 0x11df2a790>, <kliff.models.kim.KIMComputeArguments object at 0x11df2a810>, <kliff.models.kim.KIMComputeArguments object at 0x11df2a890>, <kliff.models.kim.KIMComputeArguments object at 0x11df2a910>, <kliff.models.kim.KIMComputeArguments object at 0x11df2a990>, <kliff.models.kim.KIMComputeArguments object at 0x11df2aa10>, <kliff.models.kim.KIMComputeArguments object at 0x11df2aa90>, <kliff.models.kim.KIMComputeArguments object at 0x11df2ab10>, <kliff.models.kim.KIMComputeArguments object at 0x11df2ab90>, <kliff.models.kim.KIMComputeArguments object at 0x11df2ac10>, <kliff.models.kim.KIMComputeArguments object at 0x11df2ac90>, <kliff.models.kim.KIMComputeArguments object at 0x11df2ad10>, <kliff.models.kim.KIMComputeArguments object at 0x11df2ad90>, <kliff.models.kim.KIMComputeArguments object at 0x11df2ae10>, <kliff.models.kim.KIMComputeArguments object at 0x11df2ae90>, <kliff.models.kim.KIMComputeArguments object at 0x11df2af10>, <kliff.models.kim.KIMComputeArguments object at 0x11df2af90>, <kliff.models.kim.KIMComputeArguments object at 0x11df2afd0>, <kliff.models.kim.KIMComputeArguments object at 0x11df2a050>, <kliff.models.kim.KIMComputeArguments object at 0x11df36150>, <kliff.models.kim.KIMComputeArguments object at 0x11df361d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df36250>, <kliff.models.kim.KIMComputeArguments object at 0x11df362d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df36350>, <kliff.models.kim.KIMComputeArguments object at 0x11df363d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df36450>, <kliff.models.kim.KIMComputeArguments object at 0x11df364d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df36550>, <kliff.models.kim.KIMComputeArguments object at 0x11df365d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df36650>, <kliff.models.kim.KIMComputeArguments object at 0x11df366d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df36750>, <kliff.models.kim.KIMComputeArguments object at 0x11df367d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df36850>, <kliff.models.kim.KIMComputeArguments object at 0x11df368d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df36950>, <kliff.models.kim.KIMComputeArguments object at 0x11df369d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df36a50>, <kliff.models.kim.KIMComputeArguments object at 0x11df36ad0>, <kliff.models.kim.KIMComputeArguments object at 0x11df36b50>, <kliff.models.kim.KIMComputeArguments object at 0x11df36bd0>, <kliff.models.kim.KIMComputeArguments object at 0x11df36c50>, <kliff.models.kim.KIMComputeArguments object at 0x11df36cd0>, <kliff.models.kim.KIMComputeArguments object at 0x11df36d50>, <kliff.models.kim.KIMComputeArguments object at 0x11df36dd0>, <kliff.models.kim.KIMComputeArguments object at 0x11df36e50>, <kliff.models.kim.KIMComputeArguments object at 0x11df36ed0>, <kliff.models.kim.KIMComputeArguments object at 0x11df36f50>, <kliff.models.kim.KIMComputeArguments object at 0x11df36fd0>, <kliff.models.kim.KIMComputeArguments object at 0x11df36050>, <kliff.models.kim.KIMComputeArguments object at 0x11df42110>, <kliff.models.kim.KIMComputeArguments object at 0x11df42190>, <kliff.models.kim.KIMComputeArguments object at 0x11df42210>, <kliff.models.kim.KIMComputeArguments object at 0x11df42290>, <kliff.models.kim.KIMComputeArguments object at 0x11df42310>, <kliff.models.kim.KIMComputeArguments object at 0x11df42390>, <kliff.models.kim.KIMComputeArguments object at 0x11df42410>, <kliff.models.kim.KIMComputeArguments object at 0x11df42490>, <kliff.models.kim.KIMComputeArguments object at 0x11df42510>, <kliff.models.kim.KIMComputeArguments object at 0x11df42590>, <kliff.models.kim.KIMComputeArguments object at 0x11df42610>, <kliff.models.kim.KIMComputeArguments object at 0x11df42690>, <kliff.models.kim.KIMComputeArguments object at 0x11df42710>, <kliff.models.kim.KIMComputeArguments object at 0x11df42790>, <kliff.models.kim.KIMComputeArguments object at 0x11df42810>, <kliff.models.kim.KIMComputeArguments object at 0x11df42890>, <kliff.models.kim.KIMComputeArguments object at 0x11df42910>, <kliff.models.kim.KIMComputeArguments object at 0x11df42990>, <kliff.models.kim.KIMComputeArguments object at 0x11df42a10>, <kliff.models.kim.KIMComputeArguments object at 0x11df42a90>, <kliff.models.kim.KIMComputeArguments object at 0x11df42b10>, <kliff.models.kim.KIMComputeArguments object at 0x11df42b90>, <kliff.models.kim.KIMComputeArguments object at 0x11df42c10>, <kliff.models.kim.KIMComputeArguments object at 0x11df42c90>, <kliff.models.kim.KIMComputeArguments object at 0x11df42d10>, <kliff.models.kim.KIMComputeArguments object at 0x11df42d90>, <kliff.models.kim.KIMComputeArguments object at 0x11df42e10>, <kliff.models.kim.KIMComputeArguments object at 0x11df42e90>, <kliff.models.kim.KIMComputeArguments object at 0x11df42f10>, <kliff.models.kim.KIMComputeArguments object at 0x11df42f90>, <kliff.models.kim.KIMComputeArguments object at 0x11df42fd0>, <kliff.models.kim.KIMComputeArguments object at 0x11df42050>, <kliff.models.kim.KIMComputeArguments object at 0x11df50150>, <kliff.models.kim.KIMComputeArguments object at 0x11df501d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df50250>, <kliff.models.kim.KIMComputeArguments object at 0x11df502d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df50350>, <kliff.models.kim.KIMComputeArguments object at 0x11df503d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df50450>, <kliff.models.kim.KIMComputeArguments object at 0x11df504d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df50550>, <kliff.models.kim.KIMComputeArguments object at 0x11df505d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df50650>, <kliff.models.kim.KIMComputeArguments object at 0x11df506d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df50750>, <kliff.models.kim.KIMComputeArguments object at 0x11df507d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df50850>, <kliff.models.kim.KIMComputeArguments object at 0x11df508d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df50950>, <kliff.models.kim.KIMComputeArguments object at 0x11df509d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df50a50>, <kliff.models.kim.KIMComputeArguments object at 0x11df50ad0>, <kliff.models.kim.KIMComputeArguments object at 0x11df50b50>, <kliff.models.kim.KIMComputeArguments object at 0x11df50bd0>, <kliff.models.kim.KIMComputeArguments object at 0x11df50c50>, <kliff.models.kim.KIMComputeArguments object at 0x11df50cd0>, <kliff.models.kim.KIMComputeArguments object at 0x11df50d50>, <kliff.models.kim.KIMComputeArguments object at 0x11df50dd0>, <kliff.models.kim.KIMComputeArguments object at 0x11df50e50>, <kliff.models.kim.KIMComputeArguments object at 0x11df50ed0>, <kliff.models.kim.KIMComputeArguments object at 0x11df50f50>, <kliff.models.kim.KIMComputeArguments object at 0x11df50fd0>, <kliff.models.kim.KIMComputeArguments object at 0x11df500d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df5e110>, <kliff.models.kim.KIMComputeArguments object at 0x11df5e190>, <kliff.models.kim.KIMComputeArguments object at 0x11df5e210>, <kliff.models.kim.KIMComputeArguments object at 0x11df5e290>, <kliff.models.kim.KIMComputeArguments object at 0x11df5e310>, <kliff.models.kim.KIMComputeArguments object at 0x11df5e390>, <kliff.models.kim.KIMComputeArguments object at 0x11df5e410>, <kliff.models.kim.KIMComputeArguments object at 0x11df5e490>, <kliff.models.kim.KIMComputeArguments object at 0x11df5e510>, <kliff.models.kim.KIMComputeArguments object at 0x11df5e590>, <kliff.models.kim.KIMComputeArguments object at 0x11df5e610>, <kliff.models.kim.KIMComputeArguments object at 0x11df5e690>, <kliff.models.kim.KIMComputeArguments object at 0x11df5e710>, <kliff.models.kim.KIMComputeArguments object at 0x11df5e790>, <kliff.models.kim.KIMComputeArguments object at 0x11df5e810>, <kliff.models.kim.KIMComputeArguments object at 0x11df5e890>, <kliff.models.kim.KIMComputeArguments object at 0x11df5e910>, <kliff.models.kim.KIMComputeArguments object at 0x11df5e990>, <kliff.models.kim.KIMComputeArguments object at 0x11df5ea10>, <kliff.models.kim.KIMComputeArguments object at 0x11df5ea90>, <kliff.models.kim.KIMComputeArguments object at 0x11df5eb10>, <kliff.models.kim.KIMComputeArguments object at 0x11df5eb90>, <kliff.models.kim.KIMComputeArguments object at 0x11df5ec10>, <kliff.models.kim.KIMComputeArguments object at 0x11df5ec90>, <kliff.models.kim.KIMComputeArguments object at 0x11df5ed10>, <kliff.models.kim.KIMComputeArguments object at 0x11df5ed90>, <kliff.models.kim.KIMComputeArguments object at 0x11df5ee10>, <kliff.models.kim.KIMComputeArguments object at 0x11df5ee90>, <kliff.models.kim.KIMComputeArguments object at 0x11df5ef10>, <kliff.models.kim.KIMComputeArguments object at 0x11df5ef90>, <kliff.models.kim.KIMComputeArguments object at 0x11df5efd0>, <kliff.models.kim.KIMComputeArguments object at 0x11df5e090>, <kliff.models.kim.KIMComputeArguments object at 0x11df6a150>, <kliff.models.kim.KIMComputeArguments object at 0x11df6a1d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df6a250>, <kliff.models.kim.KIMComputeArguments object at 0x11df6a2d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df6a350>, <kliff.models.kim.KIMComputeArguments object at 0x11df6a3d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df6a450>, <kliff.models.kim.KIMComputeArguments object at 0x11df6a4d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df6a550>, <kliff.models.kim.KIMComputeArguments object at 0x11df6a5d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df6a650>, <kliff.models.kim.KIMComputeArguments object at 0x11df6a6d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df6a750>, <kliff.models.kim.KIMComputeArguments object at 0x11df6a7d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df6a850>, <kliff.models.kim.KIMComputeArguments object at 0x11df6a8d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df6a950>, <kliff.models.kim.KIMComputeArguments object at 0x11df6a9d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df6aa50>, <kliff.models.kim.KIMComputeArguments object at 0x11df6aad0>, <kliff.models.kim.KIMComputeArguments object at 0x11df6ab50>, <kliff.models.kim.KIMComputeArguments object at 0x11df6abd0>, <kliff.models.kim.KIMComputeArguments object at 0x11df6ac50>, <kliff.models.kim.KIMComputeArguments object at 0x11df6acd0>, <kliff.models.kim.KIMComputeArguments object at 0x11df6ad50>, <kliff.models.kim.KIMComputeArguments object at 0x11df6add0>, <kliff.models.kim.KIMComputeArguments object at 0x11df6ae50>, <kliff.models.kim.KIMComputeArguments object at 0x11df6aed0>, <kliff.models.kim.KIMComputeArguments object at 0x11df6af50>, <kliff.models.kim.KIMComputeArguments object at 0x11df6afd0>, <kliff.models.kim.KIMComputeArguments object at 0x11df6a050>, <kliff.models.kim.KIMComputeArguments object at 0x11df76110>, <kliff.models.kim.KIMComputeArguments object at 0x11df76190>, <kliff.models.kim.KIMComputeArguments object at 0x11df76210>, <kliff.models.kim.KIMComputeArguments object at 0x11df76290>, <kliff.models.kim.KIMComputeArguments object at 0x11df76310>, <kliff.models.kim.KIMComputeArguments object at 0x11df76390>, <kliff.models.kim.KIMComputeArguments object at 0x11df76410>, <kliff.models.kim.KIMComputeArguments object at 0x11df76490>, <kliff.models.kim.KIMComputeArguments object at 0x11df76510>, <kliff.models.kim.KIMComputeArguments object at 0x11df76590>, <kliff.models.kim.KIMComputeArguments object at 0x11df76610>, <kliff.models.kim.KIMComputeArguments object at 0x11df76690>, <kliff.models.kim.KIMComputeArguments object at 0x11df76710>, <kliff.models.kim.KIMComputeArguments object at 0x11df76790>, <kliff.models.kim.KIMComputeArguments object at 0x11df76810>, <kliff.models.kim.KIMComputeArguments object at 0x11df76890>, <kliff.models.kim.KIMComputeArguments object at 0x11df76910>, <kliff.models.kim.KIMComputeArguments object at 0x11df76990>, <kliff.models.kim.KIMComputeArguments object at 0x11df76a10>, <kliff.models.kim.KIMComputeArguments object at 0x11df76a90>, <kliff.models.kim.KIMComputeArguments object at 0x11df76b10>, <kliff.models.kim.KIMComputeArguments object at 0x11df76b90>, <kliff.models.kim.KIMComputeArguments object at 0x11df76c10>, <kliff.models.kim.KIMComputeArguments object at 0x11df76c90>, <kliff.models.kim.KIMComputeArguments object at 0x11df76d10>, <kliff.models.kim.KIMComputeArguments object at 0x11df76d90>, <kliff.models.kim.KIMComputeArguments object at 0x11df76e10>, <kliff.models.kim.KIMComputeArguments object at 0x11df76e90>, <kliff.models.kim.KIMComputeArguments object at 0x11df76f10>, <kliff.models.kim.KIMComputeArguments object at 0x11df76f90>, <kliff.models.kim.KIMComputeArguments object at 0x11df76fd0>, <kliff.models.kim.KIMComputeArguments object at 0x11df76050>, <kliff.models.kim.KIMComputeArguments object at 0x11df83150>, <kliff.models.kim.KIMComputeArguments object at 0x11df831d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df83250>, <kliff.models.kim.KIMComputeArguments object at 0x11df832d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df83350>, <kliff.models.kim.KIMComputeArguments object at 0x11df833d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df83450>, <kliff.models.kim.KIMComputeArguments object at 0x11df834d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df83550>, <kliff.models.kim.KIMComputeArguments object at 0x11df835d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df83650>, <kliff.models.kim.KIMComputeArguments object at 0x11df836d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df83750>, <kliff.models.kim.KIMComputeArguments object at 0x11df837d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df83850>, <kliff.models.kim.KIMComputeArguments object at 0x11df838d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df83950>, <kliff.models.kim.KIMComputeArguments object at 0x11df839d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df83a50>, <kliff.models.kim.KIMComputeArguments object at 0x11df83ad0>, <kliff.models.kim.KIMComputeArguments object at 0x11df83b50>, <kliff.models.kim.KIMComputeArguments object at 0x11df83bd0>, <kliff.models.kim.KIMComputeArguments object at 0x11df83c50>, <kliff.models.kim.KIMComputeArguments object at 0x11df83cd0>, <kliff.models.kim.KIMComputeArguments object at 0x11df83d50>, <kliff.models.kim.KIMComputeArguments object at 0x11df83dd0>, <kliff.models.kim.KIMComputeArguments object at 0x11df83e50>, <kliff.models.kim.KIMComputeArguments object at 0x11df83ed0>, <kliff.models.kim.KIMComputeArguments object at 0x11df83f50>, <kliff.models.kim.KIMComputeArguments object at 0x11df83fd0>, <kliff.models.kim.KIMComputeArguments object at 0x11df83050>, <kliff.models.kim.KIMComputeArguments object at 0x11df91110>, <kliff.models.kim.KIMComputeArguments object at 0x11df91190>, <kliff.models.kim.KIMComputeArguments object at 0x11df91210>, <kliff.models.kim.KIMComputeArguments object at 0x11df91290>, <kliff.models.kim.KIMComputeArguments object at 0x11df91310>, <kliff.models.kim.KIMComputeArguments object at 0x11df91390>, <kliff.models.kim.KIMComputeArguments object at 0x11df91410>, <kliff.models.kim.KIMComputeArguments object at 0x11df91490>, <kliff.models.kim.KIMComputeArguments object at 0x11df91510>, <kliff.models.kim.KIMComputeArguments object at 0x11df91590>, <kliff.models.kim.KIMComputeArguments object at 0x11df91610>, <kliff.models.kim.KIMComputeArguments object at 0x11df91690>, <kliff.models.kim.KIMComputeArguments object at 0x11df91710>, <kliff.models.kim.KIMComputeArguments object at 0x11df91790>, <kliff.models.kim.KIMComputeArguments object at 0x11df91810>, <kliff.models.kim.KIMComputeArguments object at 0x11df91890>, <kliff.models.kim.KIMComputeArguments object at 0x11df91910>, <kliff.models.kim.KIMComputeArguments object at 0x11df91990>, <kliff.models.kim.KIMComputeArguments object at 0x11df91a10>, <kliff.models.kim.KIMComputeArguments object at 0x11df91a90>, <kliff.models.kim.KIMComputeArguments object at 0x11df91b10>, <kliff.models.kim.KIMComputeArguments object at 0x11df91b90>, <kliff.models.kim.KIMComputeArguments object at 0x11df91c10>, <kliff.models.kim.KIMComputeArguments object at 0x11df91c90>, <kliff.models.kim.KIMComputeArguments object at 0x11df91d10>, <kliff.models.kim.KIMComputeArguments object at 0x11df91d50>, <kliff.models.kim.KIMComputeArguments object at 0x11df91e10>, <kliff.models.kim.KIMComputeArguments object at 0x11df91e90>, <kliff.models.kim.KIMComputeArguments object at 0x11df91f10>, <kliff.models.kim.KIMComputeArguments object at 0x11df91f50>, <kliff.models.kim.KIMComputeArguments object at 0x11df91fd0>, <kliff.models.kim.KIMComputeArguments object at 0x11df91090>, <kliff.models.kim.KIMComputeArguments object at 0x11df9e150>, <kliff.models.kim.KIMComputeArguments object at 0x11df9e190>, <kliff.models.kim.KIMComputeArguments object at 0x11df9e250>, <kliff.models.kim.KIMComputeArguments object at 0x11df9e2d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df9e350>, <kliff.models.kim.KIMComputeArguments object at 0x11df9e390>, <kliff.models.kim.KIMComputeArguments object at 0x11df9e450>, <kliff.models.kim.KIMComputeArguments object at 0x11df9e4d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df9e550>, <kliff.models.kim.KIMComputeArguments object at 0x11df9e590>, <kliff.models.kim.KIMComputeArguments object at 0x11df9e650>, <kliff.models.kim.KIMComputeArguments object at 0x11df9e6d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df9e750>, <kliff.models.kim.KIMComputeArguments object at 0x11df9e790>, <kliff.models.kim.KIMComputeArguments object at 0x11df9e850>, <kliff.models.kim.KIMComputeArguments object at 0x11df9e8d0>, <kliff.models.kim.KIMComputeArguments object at 0x11df9e950>, <kliff.models.kim.KIMComputeArguments object at 0x11df9e990>, <kliff.models.kim.KIMComputeArguments object at 0x11df9ea50>, <kliff.models.kim.KIMComputeArguments object at 0x11df9ead0>, <kliff.models.kim.KIMComputeArguments object at 0x11df9eb50>, <kliff.models.kim.KIMComputeArguments object at 0x11df9eb90>, <kliff.models.kim.KIMComputeArguments object at 0x11df9ec50>, <kliff.models.kim.KIMComputeArguments object at 0x11df9ecd0>, <kliff.models.kim.KIMComputeArguments object at 0x11df9ed50>, <kliff.models.kim.KIMComputeArguments object at 0x11df9ed90>, <kliff.models.kim.KIMComputeArguments object at 0x11df9ee50>, <kliff.models.kim.KIMComputeArguments object at 0x11df9eed0>, <kliff.models.kim.KIMComputeArguments object at 0x11df9ef50>, <kliff.models.kim.KIMComputeArguments object at 0x11df9ef90>, <kliff.models.kim.KIMComputeArguments object at 0x11df9e0d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfaa110>, <kliff.models.kim.KIMComputeArguments object at 0x11dfaa190>, <kliff.models.kim.KIMComputeArguments object at 0x11dfaa1d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfaa290>, <kliff.models.kim.KIMComputeArguments object at 0x11dfaa310>, <kliff.models.kim.KIMComputeArguments object at 0x11dfaa390>, <kliff.models.kim.KIMComputeArguments object at 0x11dfaa3d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfaa490>, <kliff.models.kim.KIMComputeArguments object at 0x11dfaa510>, <kliff.models.kim.KIMComputeArguments object at 0x11dfaa590>, <kliff.models.kim.KIMComputeArguments object at 0x11dfaa5d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfaa690>, <kliff.models.kim.KIMComputeArguments object at 0x11dfaa710>, <kliff.models.kim.KIMComputeArguments object at 0x11dfaa790>, <kliff.models.kim.KIMComputeArguments object at 0x11dfaa7d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfaa890>, <kliff.models.kim.KIMComputeArguments object at 0x11dfaa910>, <kliff.models.kim.KIMComputeArguments object at 0x11dfaa990>, <kliff.models.kim.KIMComputeArguments object at 0x11dfaa9d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfaaa90>, <kliff.models.kim.KIMComputeArguments object at 0x11dfaab10>, <kliff.models.kim.KIMComputeArguments object at 0x11dfaab90>, <kliff.models.kim.KIMComputeArguments object at 0x11dfaabd0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfaac90>, <kliff.models.kim.KIMComputeArguments object at 0x11dfaad10>, <kliff.models.kim.KIMComputeArguments object at 0x11dfaad90>, <kliff.models.kim.KIMComputeArguments object at 0x11dfaadd0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfaae90>, <kliff.models.kim.KIMComputeArguments object at 0x11dfaaf10>, <kliff.models.kim.KIMComputeArguments object at 0x11dfaaf90>, <kliff.models.kim.KIMComputeArguments object at 0x11dfaa050>, <kliff.models.kim.KIMComputeArguments object at 0x11dfaafd0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfb7150>, <kliff.models.kim.KIMComputeArguments object at 0x11dfb71d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfb7210>, <kliff.models.kim.KIMComputeArguments object at 0x11dfb72d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfb7350>, <kliff.models.kim.KIMComputeArguments object at 0x11dfb73d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfb7410>, <kliff.models.kim.KIMComputeArguments object at 0x11dfb74d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfb7550>, <kliff.models.kim.KIMComputeArguments object at 0x11dfb75d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfb7610>, <kliff.models.kim.KIMComputeArguments object at 0x11dfb76d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfb7750>, <kliff.models.kim.KIMComputeArguments object at 0x11dfb77d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfb7810>, <kliff.models.kim.KIMComputeArguments object at 0x11dfb78d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfb7950>, <kliff.models.kim.KIMComputeArguments object at 0x11dfb79d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfb7a10>, <kliff.models.kim.KIMComputeArguments object at 0x11dfb7ad0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfb7b50>, <kliff.models.kim.KIMComputeArguments object at 0x11dfb7bd0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfb7c10>, <kliff.models.kim.KIMComputeArguments object at 0x11dfb7cd0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfb7d50>, <kliff.models.kim.KIMComputeArguments object at 0x11dfb7dd0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfb7e10>, <kliff.models.kim.KIMComputeArguments object at 0x11dfb7ed0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfb7f50>, <kliff.models.kim.KIMComputeArguments object at 0x11dfb7fd0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfb7050>, <kliff.models.kim.KIMComputeArguments object at 0x11dfc5110>, <kliff.models.kim.KIMComputeArguments object at 0x11dfc5190>, <kliff.models.kim.KIMComputeArguments object at 0x11dfc5210>, <kliff.models.kim.KIMComputeArguments object at 0x11dfc5250>, <kliff.models.kim.KIMComputeArguments object at 0x11dfc5310>, <kliff.models.kim.KIMComputeArguments object at 0x11dfc5390>, <kliff.models.kim.KIMComputeArguments object at 0x11dfc5410>, <kliff.models.kim.KIMComputeArguments object at 0x11dfc5450>, <kliff.models.kim.KIMComputeArguments object at 0x11dfc5510>, <kliff.models.kim.KIMComputeArguments object at 0x11dfc5590>, <kliff.models.kim.KIMComputeArguments object at 0x11dfc5610>, <kliff.models.kim.KIMComputeArguments object at 0x11dfc5650>, <kliff.models.kim.KIMComputeArguments object at 0x11dfc5710>, <kliff.models.kim.KIMComputeArguments object at 0x11dfc5790>, <kliff.models.kim.KIMComputeArguments object at 0x11dfc5810>, <kliff.models.kim.KIMComputeArguments object at 0x11dfc5850>, <kliff.models.kim.KIMComputeArguments object at 0x11dfc5910>, <kliff.models.kim.KIMComputeArguments object at 0x11dfc5990>, <kliff.models.kim.KIMComputeArguments object at 0x11dfc5a10>, <kliff.models.kim.KIMComputeArguments object at 0x11dfc5a50>, <kliff.models.kim.KIMComputeArguments object at 0x11dfc5b10>, <kliff.models.kim.KIMComputeArguments object at 0x11dfc5b90>, <kliff.models.kim.KIMComputeArguments object at 0x11dfc5c10>, <kliff.models.kim.KIMComputeArguments object at 0x11dfc5c50>, <kliff.models.kim.KIMComputeArguments object at 0x11dfc5d10>, <kliff.models.kim.KIMComputeArguments object at 0x11dfc5d90>, <kliff.models.kim.KIMComputeArguments object at 0x11dfc5e10>, <kliff.models.kim.KIMComputeArguments object at 0x11dfc5e50>, <kliff.models.kim.KIMComputeArguments object at 0x11dfc5f10>, <kliff.models.kim.KIMComputeArguments object at 0x11dfc5f90>, <kliff.models.kim.KIMComputeArguments object at 0x11dfc5fd0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfc5090>, <kliff.models.kim.KIMComputeArguments object at 0x11dfd1150>, <kliff.models.kim.KIMComputeArguments object at 0x11dfd11d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfd1250>, <kliff.models.kim.KIMComputeArguments object at 0x11dfd1290>, <kliff.models.kim.KIMComputeArguments object at 0x11dfd1350>, <kliff.models.kim.KIMComputeArguments object at 0x11dfd13d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfd1450>, <kliff.models.kim.KIMComputeArguments object at 0x11dfd1490>, <kliff.models.kim.KIMComputeArguments object at 0x11dfd1550>, <kliff.models.kim.KIMComputeArguments object at 0x11dfd15d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfd1650>, <kliff.models.kim.KIMComputeArguments object at 0x11dfd1690>, <kliff.models.kim.KIMComputeArguments object at 0x11dfd1750>, <kliff.models.kim.KIMComputeArguments object at 0x11dfd17d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfd1850>, <kliff.models.kim.KIMComputeArguments object at 0x11dfd1890>, <kliff.models.kim.KIMComputeArguments object at 0x11dfd1950>, <kliff.models.kim.KIMComputeArguments object at 0x11dfd19d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfd1a50>, <kliff.models.kim.KIMComputeArguments object at 0x11dfd1a90>, <kliff.models.kim.KIMComputeArguments object at 0x11dfd1b50>, <kliff.models.kim.KIMComputeArguments object at 0x11dfd1bd0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfd1c50>, <kliff.models.kim.KIMComputeArguments object at 0x11dfd1c90>, <kliff.models.kim.KIMComputeArguments object at 0x11dfd1d50>, <kliff.models.kim.KIMComputeArguments object at 0x11dfd1dd0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfd1e50>, <kliff.models.kim.KIMComputeArguments object at 0x11dfd1e90>, <kliff.models.kim.KIMComputeArguments object at 0x11dfd1f50>, <kliff.models.kim.KIMComputeArguments object at 0x11dfd1fd0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfd1050>, <kliff.models.kim.KIMComputeArguments object at 0x11dfdd0d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfdd190>, <kliff.models.kim.KIMComputeArguments object at 0x11dfdd210>, <kliff.models.kim.KIMComputeArguments object at 0x11dfdd290>, <kliff.models.kim.KIMComputeArguments object at 0x11dfdd2d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfdd390>, <kliff.models.kim.KIMComputeArguments object at 0x11dfdd410>, <kliff.models.kim.KIMComputeArguments object at 0x11dfdd490>, <kliff.models.kim.KIMComputeArguments object at 0x11dfdd4d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfdd590>, <kliff.models.kim.KIMComputeArguments object at 0x11dfdd610>, <kliff.models.kim.KIMComputeArguments object at 0x11dfdd690>, <kliff.models.kim.KIMComputeArguments object at 0x11dfdd6d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfdd790>, <kliff.models.kim.KIMComputeArguments object at 0x11dfdd810>, <kliff.models.kim.KIMComputeArguments object at 0x11dfdd890>, <kliff.models.kim.KIMComputeArguments object at 0x11dfdd8d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfdd990>, <kliff.models.kim.KIMComputeArguments object at 0x11dfdda10>, <kliff.models.kim.KIMComputeArguments object at 0x11dfdda90>, <kliff.models.kim.KIMComputeArguments object at 0x11dfddad0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfddb90>, <kliff.models.kim.KIMComputeArguments object at 0x11dfddc10>, <kliff.models.kim.KIMComputeArguments object at 0x11dfddc90>, <kliff.models.kim.KIMComputeArguments object at 0x11dfddcd0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfddd90>, <kliff.models.kim.KIMComputeArguments object at 0x11dfdde10>, <kliff.models.kim.KIMComputeArguments object at 0x11dfdde90>, <kliff.models.kim.KIMComputeArguments object at 0x11dfdded0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfddf90>, <kliff.models.kim.KIMComputeArguments object at 0x11dfddfd0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfdd050>, <kliff.models.kim.KIMComputeArguments object at 0x11dfeb110>, <kliff.models.kim.KIMComputeArguments object at 0x11dfeb1d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfeb250>, <kliff.models.kim.KIMComputeArguments object at 0x11dfeb2d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfeb310>, <kliff.models.kim.KIMComputeArguments object at 0x11dfeb3d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfeb450>, <kliff.models.kim.KIMComputeArguments object at 0x11dfeb4d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfeb510>, <kliff.models.kim.KIMComputeArguments object at 0x11dfeb5d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfeb650>, <kliff.models.kim.KIMComputeArguments object at 0x11dfeb6d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfeb710>, <kliff.models.kim.KIMComputeArguments object at 0x11dfeb7d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfeb850>, <kliff.models.kim.KIMComputeArguments object at 0x11dfeb8d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfeb910>, <kliff.models.kim.KIMComputeArguments object at 0x11dfeb9d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfeba50>, <kliff.models.kim.KIMComputeArguments object at 0x11dfebad0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfebb10>, <kliff.models.kim.KIMComputeArguments object at 0x11dfebbd0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfebc50>, <kliff.models.kim.KIMComputeArguments object at 0x11dfebcd0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfebd10>, <kliff.models.kim.KIMComputeArguments object at 0x11dfebdd0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfebe50>, <kliff.models.kim.KIMComputeArguments object at 0x11dfebed0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfebf10>, <kliff.models.kim.KIMComputeArguments object at 0x11dfebfd0>, <kliff.models.kim.KIMComputeArguments object at 0x11dfeb0d0>, <kliff.models.kim.KIMComputeArguments object at 0x11dff8110>, <kliff.models.kim.KIMComputeArguments object at 0x11dff8150>, <kliff.models.kim.KIMComputeArguments object at 0x11dff8210>, <kliff.models.kim.KIMComputeArguments object at 0x11dff8290>, <kliff.models.kim.KIMComputeArguments object at 0x11dff8310>, <kliff.models.kim.KIMComputeArguments object at 0x11dff8350>, <kliff.models.kim.KIMComputeArguments object at 0x11dff8410>, <kliff.models.kim.KIMComputeArguments object at 0x11dff8490>, <kliff.models.kim.KIMComputeArguments object at 0x11dff8510>, <kliff.models.kim.KIMComputeArguments object at 0x11dff8550>, <kliff.models.kim.KIMComputeArguments object at 0x11dff8610>, <kliff.models.kim.KIMComputeArguments object at 0x11dff8690>, <kliff.models.kim.KIMComputeArguments object at 0x11dff8710>, <kliff.models.kim.KIMComputeArguments object at 0x11dff8750>, <kliff.models.kim.KIMComputeArguments object at 0x11dff8810>, <kliff.models.kim.KIMComputeArguments object at 0x11dff8890>, <kliff.models.kim.KIMComputeArguments object at 0x11dff8910>, <kliff.models.kim.KIMComputeArguments object at 0x11dff8950>, <kliff.models.kim.KIMComputeArguments object at 0x11dff8a10>, <kliff.models.kim.KIMComputeArguments object at 0x11dff8a90>, <kliff.models.kim.KIMComputeArguments object at 0x11dff8b10>, <kliff.models.kim.KIMComputeArguments object at 0x11dff8b50>, <kliff.models.kim.KIMComputeArguments object at 0x11dff8c10>, <kliff.models.kim.KIMComputeArguments object at 0x11dff8c90>, <kliff.models.kim.KIMComputeArguments object at 0x11dff8d10>, <kliff.models.kim.KIMComputeArguments object at 0x11dff8d50>, <kliff.models.kim.KIMComputeArguments object at 0x11dff8e10>, <kliff.models.kim.KIMComputeArguments object at 0x11dff8e90>, <kliff.models.kim.KIMComputeArguments object at 0x11dff8f10>, <kliff.models.kim.KIMComputeArguments object at 0x11dff8f50>, <kliff.models.kim.KIMComputeArguments object at 0x11dff8fd0>, <kliff.models.kim.KIMComputeArguments object at 0x11dff8090>, <kliff.models.kim.KIMComputeArguments object at 0x11e005150>, <kliff.models.kim.KIMComputeArguments object at 0x11e005190>, <kliff.models.kim.KIMComputeArguments object at 0x11e005250>, <kliff.models.kim.KIMComputeArguments object at 0x11e0052d0>, <kliff.models.kim.KIMComputeArguments object at 0x11e005350>, <kliff.models.kim.KIMComputeArguments object at 0x11e005390>, <kliff.models.kim.KIMComputeArguments object at 0x11e005450>, <kliff.models.kim.KIMComputeArguments object at 0x11e0054d0>, <kliff.models.kim.KIMComputeArguments object at 0x11e005550>, <kliff.models.kim.KIMComputeArguments object at 0x11e005590>, <kliff.models.kim.KIMComputeArguments object at 0x11e005650>, <kliff.models.kim.KIMComputeArguments object at 0x11e0056d0>, <kliff.models.kim.KIMComputeArguments object at 0x11e005750>, <kliff.models.kim.KIMComputeArguments object at 0x11e005790>, <kliff.models.kim.KIMComputeArguments object at 0x11e005850>, <kliff.models.kim.KIMComputeArguments object at 0x11e0058d0>, <kliff.models.kim.KIMComputeArguments object at 0x11e005950>, <kliff.models.kim.KIMComputeArguments object at 0x11e005990>, <kliff.models.kim.KIMComputeArguments object at 0x11e005a50>, <kliff.models.kim.KIMComputeArguments object at 0x11e005ad0>, <kliff.models.kim.KIMComputeArguments object at 0x11e005b50>, <kliff.models.kim.KIMComputeArguments object at 0x11e005b90>, <kliff.models.kim.KIMComputeArguments object at 0x11e005c50>, <kliff.models.kim.KIMComputeArguments object at 0x11e005cd0>, <kliff.models.kim.KIMComputeArguments object at 0x11e005d50>, <kliff.models.kim.KIMComputeArguments object at 0x11e005d90>, <kliff.models.kim.KIMComputeArguments object at 0x11e005e50>, <kliff.models.kim.KIMComputeArguments object at 0x11e005ed0>, <kliff.models.kim.KIMComputeArguments object at 0x11e005f50>, <kliff.models.kim.KIMComputeArguments object at 0x11e005f90>, <kliff.models.kim.KIMComputeArguments object at 0x11e005050>, <kliff.models.kim.KIMComputeArguments object at 0x11e011110>, <kliff.models.kim.KIMComputeArguments object at 0x11e011190>, <kliff.models.kim.KIMComputeArguments object at 0x11e0111d0>, <kliff.models.kim.KIMComputeArguments object at 0x11e011290>, <kliff.models.kim.KIMComputeArguments object at 0x11e011310>, <kliff.models.kim.KIMComputeArguments object at 0x11e011390>, <kliff.models.kim.KIMComputeArguments object at 0x11e0113d0>, <kliff.models.kim.KIMComputeArguments object at 0x11e011490>, <kliff.models.kim.KIMComputeArguments object at 0x11e011510>, <kliff.models.kim.KIMComputeArguments object at 0x11e011590>, <kliff.models.kim.KIMComputeArguments object at 0x11e0115d0>, <kliff.models.kim.KIMComputeArguments object at 0x11e011690>, <kliff.models.kim.KIMComputeArguments object at 0x11e011710>, <kliff.models.kim.KIMComputeArguments object at 0x11e011790>, <kliff.models.kim.KIMComputeArguments object at 0x11e0117d0>, <kliff.models.kim.KIMComputeArguments object at 0x11e011890>, <kliff.models.kim.KIMComputeArguments object at 0x11e011910>, <kliff.models.kim.KIMComputeArguments object at 0x11e011990>, <kliff.models.kim.KIMComputeArguments object at 0x11e0119d0>, <kliff.models.kim.KIMComputeArguments object at 0x11e011a90>, <kliff.models.kim.KIMComputeArguments object at 0x11e011b10>, <kliff.models.kim.KIMComputeArguments object at 0x11e011b90>, <kliff.models.kim.KIMComputeArguments object at 0x11e011bd0>, <kliff.models.kim.KIMComputeArguments object at 0x11e011c90>, <kliff.models.kim.KIMComputeArguments object at 0x11e011d10>, <kliff.models.kim.KIMComputeArguments object at 0x11e011d90>, <kliff.models.kim.KIMComputeArguments object at 0x11e011dd0>, <kliff.models.kim.KIMComputeArguments object at 0x11e011e90>, <kliff.models.kim.KIMComputeArguments object at 0x11e011f10>, <kliff.models.kim.KIMComputeArguments object at 0x11e011f90>, <kliff.models.kim.KIMComputeArguments object at 0x11e011050>, <kliff.models.kim.KIMComputeArguments object at 0x11e011fd0>, <kliff.models.kim.KIMComputeArguments object at 0x11e01e150>, <kliff.models.kim.KIMComputeArguments object at 0x11e01e1d0>, <kliff.models.kim.KIMComputeArguments object at 0x11e01e210>, <kliff.models.kim.KIMComputeArguments object at 0x11e01e2d0>, <kliff.models.kim.KIMComputeArguments object at 0x11e01e350>, <kliff.models.kim.KIMComputeArguments object at 0x11e01e3d0>, <kliff.models.kim.KIMComputeArguments object at 0x11e01e410>, <kliff.models.kim.KIMComputeArguments object at 0x11e01e4d0>, <kliff.models.kim.KIMComputeArguments object at 0x11e01e550>, <kliff.models.kim.KIMComputeArguments object at 0x11e01e5d0>, <kliff.models.kim.KIMComputeArguments object at 0x11e01e610>, <kliff.models.kim.KIMComputeArguments object at 0x11e01e6d0>, <kliff.models.kim.KIMComputeArguments object at 0x11e01e750>, <kliff.models.kim.KIMComputeArguments object at 0x11e01e7d0>, <kliff.models.kim.KIMComputeArguments object at 0x11e01e810>, <kliff.models.kim.KIMComputeArguments object at 0x11e01e8d0>, <kliff.models.kim.KIMComputeArguments object at 0x11e01e950>, <kliff.models.kim.KIMComputeArguments object at 0x11e01e9d0>, <kliff.models.kim.KIMComputeArguments object at 0x11e01ea10>, <kliff.models.kim.KIMComputeArguments object at 0x11e01ead0>, <kliff.models.kim.KIMComputeArguments object at 0x11e01eb50>, <kliff.models.kim.KIMComputeArguments object at 0x11e01ebd0>, <kliff.models.kim.KIMComputeArguments object at 0x11e01ec10>, <kliff.models.kim.KIMComputeArguments object at 0x11e01ecd0>, <kliff.models.kim.KIMComputeArguments object at 0x11e01ed50>, <kliff.models.kim.KIMComputeArguments object at 0x11e01edd0>, <kliff.models.kim.KIMComputeArguments object at 0x11e01ee10>, <kliff.models.kim.KIMComputeArguments object at 0x11e01eed0>, <kliff.models.kim.KIMComputeArguments object at 0x11e01ef50>, <kliff.models.kim.KIMComputeArguments object at 0x11e01efd0>, <kliff.models.kim.KIMComputeArguments object at 0x11e01e050>, <kliff.models.kim.KIMComputeArguments object at 0x11e02b110>, <kliff.models.kim.KIMComputeArguments object at 0x11e02b190>, <kliff.models.kim.KIMComputeArguments object at 0x11e02b210>, <kliff.models.kim.KIMComputeArguments object at 0x11e02b250>, <kliff.models.kim.KIMComputeArguments object at 0x11e02b310>, <kliff.models.kim.KIMComputeArguments object at 0x11e02b390>, <kliff.models.kim.KIMComputeArguments object at 0x11e02b410>, <kliff.models.kim.KIMComputeArguments object at 0x11e02b450>, <kliff.models.kim.KIMComputeArguments object at 0x11e02b510>, <kliff.models.kim.KIMComputeArguments object at 0x11e02b590>, <kliff.models.kim.KIMComputeArguments object at 0x11e02b610>, <kliff.models.kim.KIMComputeArguments object at 0x11e02b650>, <kliff.models.kim.KIMComputeArguments object at 0x11e02b710>, <kliff.models.kim.KIMComputeArguments object at 0x11e02b790>, <kliff.models.kim.KIMComputeArguments object at 0x11e02b810>, <kliff.models.kim.KIMComputeArguments object at 0x11e02b850>, <kliff.models.kim.KIMComputeArguments object at 0x11e02b910>, <kliff.models.kim.KIMComputeArguments object at 0x11e02b990>, <kliff.models.kim.KIMComputeArguments object at 0x11e02ba10>, <kliff.models.kim.KIMComputeArguments object at 0x11e02ba50>, <kliff.models.kim.KIMComputeArguments object at 0x11e02bb10>, <kliff.models.kim.KIMComputeArguments object at 0x11e02bb90>, <kliff.models.kim.KIMComputeArguments object at 0x11e02bc10>, <kliff.models.kim.KIMComputeArguments object at 0x11e02bc50>, <kliff.models.kim.KIMComputeArguments object at 0x11e02bd10>, <kliff.models.kim.KIMComputeArguments object at 0x11e02bd90>, <kliff.models.kim.KIMComputeArguments object at 0x11e02be10>, <kliff.models.kim.KIMComputeArguments object at 0x11e02be50>, <kliff.models.kim.KIMComputeArguments object at 0x11e02bf10>, <kliff.models.kim.KIMComputeArguments object at 0x11e02bf90>, <kliff.models.kim.KIMComputeArguments object at 0x11e02bfd0>, <kliff.models.kim.KIMComputeArguments object at 0x11e02b050>, <kliff.models.kim.KIMComputeArguments object at 0x11e039150>, <kliff.models.kim.KIMComputeArguments object at 0x11e0391d0>, <kliff.models.kim.KIMComputeArguments object at 0x11e039250>, <kliff.models.kim.KIMComputeArguments object at 0x11e039290>, <kliff.models.kim.KIMComputeArguments object at 0x11e039350>, <kliff.models.kim.KIMComputeArguments object at 0x11e0393d0>, <kliff.models.kim.KIMComputeArguments object at 0x11e039450>, <kliff.models.kim.KIMComputeArguments object at 0x11e039490>, <kliff.models.kim.KIMComputeArguments object at 0x11e039550>, <kliff.models.kim.KIMComputeArguments object at 0x11e0395d0>, <kliff.models.kim.KIMComputeArguments object at 0x11e039650>, <kliff.models.kim.KIMComputeArguments object at 0x11e039690>, <kliff.models.kim.KIMComputeArguments object at 0x11e039750>]



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

   **Total running time of the script:** ( 1 minutes  33.965 seconds)


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
