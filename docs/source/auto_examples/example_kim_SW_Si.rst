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


    [<kliff.models.kim.KIMComputeArguments object at 0x129f59048>, <kliff.models.kim.KIMComputeArguments object at 0x129f590f0>, <kliff.models.kim.KIMComputeArguments object at 0x129f592b0>, <kliff.models.kim.KIMComputeArguments object at 0x129f59320>, <kliff.models.kim.KIMComputeArguments object at 0x129f59390>, <kliff.models.kim.KIMComputeArguments object at 0x129f59400>, <kliff.models.kim.KIMComputeArguments object at 0x129f59470>, <kliff.models.kim.KIMComputeArguments object at 0x129f594e0>, <kliff.models.kim.KIMComputeArguments object at 0x129f59550>, <kliff.models.kim.KIMComputeArguments object at 0x129f595c0>, <kliff.models.kim.KIMComputeArguments object at 0x129f59630>, <kliff.models.kim.KIMComputeArguments object at 0x129f596a0>, <kliff.models.kim.KIMComputeArguments object at 0x129f59710>, <kliff.models.kim.KIMComputeArguments object at 0x129f59780>, <kliff.models.kim.KIMComputeArguments object at 0x129f597f0>, <kliff.models.kim.KIMComputeArguments object at 0x129f59860>, <kliff.models.kim.KIMComputeArguments object at 0x129f598d0>, <kliff.models.kim.KIMComputeArguments object at 0x129f59940>, <kliff.models.kim.KIMComputeArguments object at 0x129f599b0>, <kliff.models.kim.KIMComputeArguments object at 0x129f59a20>, <kliff.models.kim.KIMComputeArguments object at 0x129f59a90>, <kliff.models.kim.KIMComputeArguments object at 0x129f59b00>, <kliff.models.kim.KIMComputeArguments object at 0x129f59b70>, <kliff.models.kim.KIMComputeArguments object at 0x129f59be0>, <kliff.models.kim.KIMComputeArguments object at 0x129f59c50>, <kliff.models.kim.KIMComputeArguments object at 0x129f59cc0>, <kliff.models.kim.KIMComputeArguments object at 0x129f59d30>, <kliff.models.kim.KIMComputeArguments object at 0x129f59da0>, <kliff.models.kim.KIMComputeArguments object at 0x129f59e10>, <kliff.models.kim.KIMComputeArguments object at 0x129f59e80>, <kliff.models.kim.KIMComputeArguments object at 0x129f59ef0>, <kliff.models.kim.KIMComputeArguments object at 0x129f59f60>, <kliff.models.kim.KIMComputeArguments object at 0x129f59160>, <kliff.models.kim.KIMComputeArguments object at 0x129f59fd0>, <kliff.models.kim.KIMComputeArguments object at 0x129f590b8>, <kliff.models.kim.KIMComputeArguments object at 0x129fc6080>, <kliff.models.kim.KIMComputeArguments object at 0x129fc60f0>, <kliff.models.kim.KIMComputeArguments object at 0x129fc6160>, <kliff.models.kim.KIMComputeArguments object at 0x129fc61d0>, <kliff.models.kim.KIMComputeArguments object at 0x129fc6240>, <kliff.models.kim.KIMComputeArguments object at 0x129fc62b0>, <kliff.models.kim.KIMComputeArguments object at 0x129fc6320>, <kliff.models.kim.KIMComputeArguments object at 0x129fc6390>, <kliff.models.kim.KIMComputeArguments object at 0x129fc6400>, <kliff.models.kim.KIMComputeArguments object at 0x129fc6470>, <kliff.models.kim.KIMComputeArguments object at 0x129fc64e0>, <kliff.models.kim.KIMComputeArguments object at 0x129fc6550>, <kliff.models.kim.KIMComputeArguments object at 0x129fc65c0>, <kliff.models.kim.KIMComputeArguments object at 0x129fc6630>, <kliff.models.kim.KIMComputeArguments object at 0x129fc66a0>, <kliff.models.kim.KIMComputeArguments object at 0x129fc6710>, <kliff.models.kim.KIMComputeArguments object at 0x129fc6780>, <kliff.models.kim.KIMComputeArguments object at 0x129fc67f0>, <kliff.models.kim.KIMComputeArguments object at 0x129fc6860>, <kliff.models.kim.KIMComputeArguments object at 0x129fc68d0>, <kliff.models.kim.KIMComputeArguments object at 0x129fc6940>, <kliff.models.kim.KIMComputeArguments object at 0x129fc69b0>, <kliff.models.kim.KIMComputeArguments object at 0x129fc6a20>, <kliff.models.kim.KIMComputeArguments object at 0x129fc6a90>, <kliff.models.kim.KIMComputeArguments object at 0x129fc6b00>, <kliff.models.kim.KIMComputeArguments object at 0x129fc6b70>, <kliff.models.kim.KIMComputeArguments object at 0x129fc6be0>, <kliff.models.kim.KIMComputeArguments object at 0x129fc6c50>, <kliff.models.kim.KIMComputeArguments object at 0x129fc6cc0>, <kliff.models.kim.KIMComputeArguments object at 0x129fc6d30>, <kliff.models.kim.KIMComputeArguments object at 0x129fc6da0>, <kliff.models.kim.KIMComputeArguments object at 0x129fc6e10>, <kliff.models.kim.KIMComputeArguments object at 0x129fc6e80>, <kliff.models.kim.KIMComputeArguments object at 0x129fc6ef0>, <kliff.models.kim.KIMComputeArguments object at 0x129fc6f60>, <kliff.models.kim.KIMComputeArguments object at 0x129fc6048>, <kliff.models.kim.KIMComputeArguments object at 0x129fd3080>, <kliff.models.kim.KIMComputeArguments object at 0x129fd30f0>, <kliff.models.kim.KIMComputeArguments object at 0x129fd3160>, <kliff.models.kim.KIMComputeArguments object at 0x129fd31d0>, <kliff.models.kim.KIMComputeArguments object at 0x129fd3240>, <kliff.models.kim.KIMComputeArguments object at 0x129fd32b0>, <kliff.models.kim.KIMComputeArguments object at 0x129fd3320>, <kliff.models.kim.KIMComputeArguments object at 0x129fd3390>, <kliff.models.kim.KIMComputeArguments object at 0x129fd3400>, <kliff.models.kim.KIMComputeArguments object at 0x129fd3470>, <kliff.models.kim.KIMComputeArguments object at 0x129fd34e0>, <kliff.models.kim.KIMComputeArguments object at 0x129fd3550>, <kliff.models.kim.KIMComputeArguments object at 0x129fd35c0>, <kliff.models.kim.KIMComputeArguments object at 0x129fd3630>, <kliff.models.kim.KIMComputeArguments object at 0x129fd36a0>, <kliff.models.kim.KIMComputeArguments object at 0x129fd3710>, <kliff.models.kim.KIMComputeArguments object at 0x129fd3780>, <kliff.models.kim.KIMComputeArguments object at 0x129fd37f0>, <kliff.models.kim.KIMComputeArguments object at 0x129fd3860>, <kliff.models.kim.KIMComputeArguments object at 0x129fd38d0>, <kliff.models.kim.KIMComputeArguments object at 0x129fd3940>, <kliff.models.kim.KIMComputeArguments object at 0x129fd39b0>, <kliff.models.kim.KIMComputeArguments object at 0x129fd3a20>, <kliff.models.kim.KIMComputeArguments object at 0x129fd3a90>, <kliff.models.kim.KIMComputeArguments object at 0x129fd3b00>, <kliff.models.kim.KIMComputeArguments object at 0x129fd3b70>, <kliff.models.kim.KIMComputeArguments object at 0x129fd3be0>, <kliff.models.kim.KIMComputeArguments object at 0x129fd3c50>, <kliff.models.kim.KIMComputeArguments object at 0x129fd3cc0>, <kliff.models.kim.KIMComputeArguments object at 0x129fd3d30>, <kliff.models.kim.KIMComputeArguments object at 0x129fd3da0>, <kliff.models.kim.KIMComputeArguments object at 0x129fd3e10>, <kliff.models.kim.KIMComputeArguments object at 0x129fd3e80>, <kliff.models.kim.KIMComputeArguments object at 0x129fd3ef0>, <kliff.models.kim.KIMComputeArguments object at 0x129fd3f60>, <kliff.models.kim.KIMComputeArguments object at 0x129fd3048>, <kliff.models.kim.KIMComputeArguments object at 0x129fe0080>, <kliff.models.kim.KIMComputeArguments object at 0x129fe00f0>, <kliff.models.kim.KIMComputeArguments object at 0x129fe0160>, <kliff.models.kim.KIMComputeArguments object at 0x129fe01d0>, <kliff.models.kim.KIMComputeArguments object at 0x129fe0240>, <kliff.models.kim.KIMComputeArguments object at 0x129fe02b0>, <kliff.models.kim.KIMComputeArguments object at 0x129fe0320>, <kliff.models.kim.KIMComputeArguments object at 0x129fe0390>, <kliff.models.kim.KIMComputeArguments object at 0x129fe0400>, <kliff.models.kim.KIMComputeArguments object at 0x129fe0470>, <kliff.models.kim.KIMComputeArguments object at 0x129fe04e0>, <kliff.models.kim.KIMComputeArguments object at 0x129fe0550>, <kliff.models.kim.KIMComputeArguments object at 0x129fe05c0>, <kliff.models.kim.KIMComputeArguments object at 0x129fe0630>, <kliff.models.kim.KIMComputeArguments object at 0x129fe06a0>, <kliff.models.kim.KIMComputeArguments object at 0x129fe0710>, <kliff.models.kim.KIMComputeArguments object at 0x129fe0780>, <kliff.models.kim.KIMComputeArguments object at 0x129fe07f0>, <kliff.models.kim.KIMComputeArguments object at 0x129fe0860>, <kliff.models.kim.KIMComputeArguments object at 0x129fe08d0>, <kliff.models.kim.KIMComputeArguments object at 0x129fe0940>, <kliff.models.kim.KIMComputeArguments object at 0x129fe09b0>, <kliff.models.kim.KIMComputeArguments object at 0x129fe0a20>, <kliff.models.kim.KIMComputeArguments object at 0x129fe0a90>, <kliff.models.kim.KIMComputeArguments object at 0x129fe0b00>, <kliff.models.kim.KIMComputeArguments object at 0x129fe0b70>, <kliff.models.kim.KIMComputeArguments object at 0x129fe0be0>, <kliff.models.kim.KIMComputeArguments object at 0x129fe0c50>, <kliff.models.kim.KIMComputeArguments object at 0x129fe0cc0>, <kliff.models.kim.KIMComputeArguments object at 0x129fe0d30>, <kliff.models.kim.KIMComputeArguments object at 0x129fe0da0>, <kliff.models.kim.KIMComputeArguments object at 0x129fe0e10>, <kliff.models.kim.KIMComputeArguments object at 0x129fe0e80>, <kliff.models.kim.KIMComputeArguments object at 0x129fe0ef0>, <kliff.models.kim.KIMComputeArguments object at 0x129fe0f60>, <kliff.models.kim.KIMComputeArguments object at 0x129fe0048>, <kliff.models.kim.KIMComputeArguments object at 0x129fee080>, <kliff.models.kim.KIMComputeArguments object at 0x129fee0f0>, <kliff.models.kim.KIMComputeArguments object at 0x129fee160>, <kliff.models.kim.KIMComputeArguments object at 0x129fee1d0>, <kliff.models.kim.KIMComputeArguments object at 0x129fee240>, <kliff.models.kim.KIMComputeArguments object at 0x129fee2b0>, <kliff.models.kim.KIMComputeArguments object at 0x129fee320>, <kliff.models.kim.KIMComputeArguments object at 0x129fee390>, <kliff.models.kim.KIMComputeArguments object at 0x129fee400>, <kliff.models.kim.KIMComputeArguments object at 0x129fee470>, <kliff.models.kim.KIMComputeArguments object at 0x129fee4e0>, <kliff.models.kim.KIMComputeArguments object at 0x129fee550>, <kliff.models.kim.KIMComputeArguments object at 0x129fee5c0>, <kliff.models.kim.KIMComputeArguments object at 0x129fee630>, <kliff.models.kim.KIMComputeArguments object at 0x129fee6a0>, <kliff.models.kim.KIMComputeArguments object at 0x129fee710>, <kliff.models.kim.KIMComputeArguments object at 0x129fee780>, <kliff.models.kim.KIMComputeArguments object at 0x129fee7f0>, <kliff.models.kim.KIMComputeArguments object at 0x129fee860>, <kliff.models.kim.KIMComputeArguments object at 0x129fee8d0>, <kliff.models.kim.KIMComputeArguments object at 0x129fee940>, <kliff.models.kim.KIMComputeArguments object at 0x129fee9b0>, <kliff.models.kim.KIMComputeArguments object at 0x129feea20>, <kliff.models.kim.KIMComputeArguments object at 0x129feea90>, <kliff.models.kim.KIMComputeArguments object at 0x129feeb00>, <kliff.models.kim.KIMComputeArguments object at 0x129feeb70>, <kliff.models.kim.KIMComputeArguments object at 0x129feebe0>, <kliff.models.kim.KIMComputeArguments object at 0x129feec50>, <kliff.models.kim.KIMComputeArguments object at 0x129feecc0>, <kliff.models.kim.KIMComputeArguments object at 0x129feed30>, <kliff.models.kim.KIMComputeArguments object at 0x129feeda0>, <kliff.models.kim.KIMComputeArguments object at 0x129feee10>, <kliff.models.kim.KIMComputeArguments object at 0x129feee80>, <kliff.models.kim.KIMComputeArguments object at 0x129feeef0>, <kliff.models.kim.KIMComputeArguments object at 0x129feef60>, <kliff.models.kim.KIMComputeArguments object at 0x129fee048>, <kliff.models.kim.KIMComputeArguments object at 0x129ffc080>, <kliff.models.kim.KIMComputeArguments object at 0x129ffc0f0>, <kliff.models.kim.KIMComputeArguments object at 0x129ffc160>, <kliff.models.kim.KIMComputeArguments object at 0x129ffc1d0>, <kliff.models.kim.KIMComputeArguments object at 0x129ffc240>, <kliff.models.kim.KIMComputeArguments object at 0x129ffc2b0>, <kliff.models.kim.KIMComputeArguments object at 0x129ffc320>, <kliff.models.kim.KIMComputeArguments object at 0x129ffc390>, <kliff.models.kim.KIMComputeArguments object at 0x129ffc400>, <kliff.models.kim.KIMComputeArguments object at 0x129ffc470>, <kliff.models.kim.KIMComputeArguments object at 0x129ffc4e0>, <kliff.models.kim.KIMComputeArguments object at 0x129ffc550>, <kliff.models.kim.KIMComputeArguments object at 0x129ffc5c0>, <kliff.models.kim.KIMComputeArguments object at 0x129ffc630>, <kliff.models.kim.KIMComputeArguments object at 0x129ffc6a0>, <kliff.models.kim.KIMComputeArguments object at 0x129ffc710>, <kliff.models.kim.KIMComputeArguments object at 0x129ffc780>, <kliff.models.kim.KIMComputeArguments object at 0x129ffc7f0>, <kliff.models.kim.KIMComputeArguments object at 0x129ffc860>, <kliff.models.kim.KIMComputeArguments object at 0x129ffc8d0>, <kliff.models.kim.KIMComputeArguments object at 0x129ffc940>, <kliff.models.kim.KIMComputeArguments object at 0x129ffc9b0>, <kliff.models.kim.KIMComputeArguments object at 0x129ffca20>, <kliff.models.kim.KIMComputeArguments object at 0x129ffca90>, <kliff.models.kim.KIMComputeArguments object at 0x129ffcb00>, <kliff.models.kim.KIMComputeArguments object at 0x129ffcb70>, <kliff.models.kim.KIMComputeArguments object at 0x129ffcbe0>, <kliff.models.kim.KIMComputeArguments object at 0x129ffcc50>, <kliff.models.kim.KIMComputeArguments object at 0x129ffccc0>, <kliff.models.kim.KIMComputeArguments object at 0x129ffcd30>, <kliff.models.kim.KIMComputeArguments object at 0x129ffcda0>, <kliff.models.kim.KIMComputeArguments object at 0x129ffce10>, <kliff.models.kim.KIMComputeArguments object at 0x129ffce80>, <kliff.models.kim.KIMComputeArguments object at 0x129ffcef0>, <kliff.models.kim.KIMComputeArguments object at 0x129ffcf60>, <kliff.models.kim.KIMComputeArguments object at 0x129ffc048>, <kliff.models.kim.KIMComputeArguments object at 0x12a009080>, <kliff.models.kim.KIMComputeArguments object at 0x12a0090f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a009160>, <kliff.models.kim.KIMComputeArguments object at 0x12a0091d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a009240>, <kliff.models.kim.KIMComputeArguments object at 0x12a0092b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a009320>, <kliff.models.kim.KIMComputeArguments object at 0x12a009390>, <kliff.models.kim.KIMComputeArguments object at 0x12a009400>, <kliff.models.kim.KIMComputeArguments object at 0x12a009470>, <kliff.models.kim.KIMComputeArguments object at 0x12a0094e0>, <kliff.models.kim.KIMComputeArguments object at 0x12a009550>, <kliff.models.kim.KIMComputeArguments object at 0x12a0095c0>, <kliff.models.kim.KIMComputeArguments object at 0x12a009630>, <kliff.models.kim.KIMComputeArguments object at 0x12a0096a0>, <kliff.models.kim.KIMComputeArguments object at 0x12a009710>, <kliff.models.kim.KIMComputeArguments object at 0x12a009780>, <kliff.models.kim.KIMComputeArguments object at 0x12a0097f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a009860>, <kliff.models.kim.KIMComputeArguments object at 0x12a0098d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a009940>, <kliff.models.kim.KIMComputeArguments object at 0x12a0099b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a009a20>, <kliff.models.kim.KIMComputeArguments object at 0x12a009a90>, <kliff.models.kim.KIMComputeArguments object at 0x12a009b00>, <kliff.models.kim.KIMComputeArguments object at 0x12a009b70>, <kliff.models.kim.KIMComputeArguments object at 0x12a009be0>, <kliff.models.kim.KIMComputeArguments object at 0x12a009c50>, <kliff.models.kim.KIMComputeArguments object at 0x12a009cc0>, <kliff.models.kim.KIMComputeArguments object at 0x12a009d30>, <kliff.models.kim.KIMComputeArguments object at 0x12a009da0>, <kliff.models.kim.KIMComputeArguments object at 0x12a009e10>, <kliff.models.kim.KIMComputeArguments object at 0x12a009e80>, <kliff.models.kim.KIMComputeArguments object at 0x12a009ef0>, <kliff.models.kim.KIMComputeArguments object at 0x12a009f60>, <kliff.models.kim.KIMComputeArguments object at 0x12a009048>, <kliff.models.kim.KIMComputeArguments object at 0x12a017080>, <kliff.models.kim.KIMComputeArguments object at 0x12a0170f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a017160>, <kliff.models.kim.KIMComputeArguments object at 0x12a0171d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a017240>, <kliff.models.kim.KIMComputeArguments object at 0x12a0172b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a017320>, <kliff.models.kim.KIMComputeArguments object at 0x12a017390>, <kliff.models.kim.KIMComputeArguments object at 0x12a017400>, <kliff.models.kim.KIMComputeArguments object at 0x12a017470>, <kliff.models.kim.KIMComputeArguments object at 0x12a0174e0>, <kliff.models.kim.KIMComputeArguments object at 0x12a017550>, <kliff.models.kim.KIMComputeArguments object at 0x12a0175c0>, <kliff.models.kim.KIMComputeArguments object at 0x12a017630>, <kliff.models.kim.KIMComputeArguments object at 0x12a0176a0>, <kliff.models.kim.KIMComputeArguments object at 0x12a017710>, <kliff.models.kim.KIMComputeArguments object at 0x12a017780>, <kliff.models.kim.KIMComputeArguments object at 0x12a0177f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a017860>, <kliff.models.kim.KIMComputeArguments object at 0x12a0178d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a017940>, <kliff.models.kim.KIMComputeArguments object at 0x12a0179b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a017a20>, <kliff.models.kim.KIMComputeArguments object at 0x12a017a90>, <kliff.models.kim.KIMComputeArguments object at 0x12a017b00>, <kliff.models.kim.KIMComputeArguments object at 0x12a017b70>, <kliff.models.kim.KIMComputeArguments object at 0x12a017be0>, <kliff.models.kim.KIMComputeArguments object at 0x12a017c50>, <kliff.models.kim.KIMComputeArguments object at 0x12a017cc0>, <kliff.models.kim.KIMComputeArguments object at 0x12a017d30>, <kliff.models.kim.KIMComputeArguments object at 0x12a017da0>, <kliff.models.kim.KIMComputeArguments object at 0x12a017e10>, <kliff.models.kim.KIMComputeArguments object at 0x12a017e80>, <kliff.models.kim.KIMComputeArguments object at 0x12a017ef0>, <kliff.models.kim.KIMComputeArguments object at 0x12a017f60>, <kliff.models.kim.KIMComputeArguments object at 0x12a017048>, <kliff.models.kim.KIMComputeArguments object at 0x12a026080>, <kliff.models.kim.KIMComputeArguments object at 0x12a0260f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a026160>, <kliff.models.kim.KIMComputeArguments object at 0x12a0261d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a026240>, <kliff.models.kim.KIMComputeArguments object at 0x12a0262b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a026320>, <kliff.models.kim.KIMComputeArguments object at 0x12a026390>, <kliff.models.kim.KIMComputeArguments object at 0x12a026400>, <kliff.models.kim.KIMComputeArguments object at 0x12a026470>, <kliff.models.kim.KIMComputeArguments object at 0x12a0264e0>, <kliff.models.kim.KIMComputeArguments object at 0x12a026550>, <kliff.models.kim.KIMComputeArguments object at 0x12a0265c0>, <kliff.models.kim.KIMComputeArguments object at 0x12a026630>, <kliff.models.kim.KIMComputeArguments object at 0x12a0266a0>, <kliff.models.kim.KIMComputeArguments object at 0x12a026710>, <kliff.models.kim.KIMComputeArguments object at 0x12a026780>, <kliff.models.kim.KIMComputeArguments object at 0x12a0267f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a026860>, <kliff.models.kim.KIMComputeArguments object at 0x12a0268d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a026940>, <kliff.models.kim.KIMComputeArguments object at 0x12a0269b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a026a20>, <kliff.models.kim.KIMComputeArguments object at 0x12a026a90>, <kliff.models.kim.KIMComputeArguments object at 0x12a026b00>, <kliff.models.kim.KIMComputeArguments object at 0x12a026b70>, <kliff.models.kim.KIMComputeArguments object at 0x12a026be0>, <kliff.models.kim.KIMComputeArguments object at 0x12a026c50>, <kliff.models.kim.KIMComputeArguments object at 0x12a026cc0>, <kliff.models.kim.KIMComputeArguments object at 0x12a026d30>, <kliff.models.kim.KIMComputeArguments object at 0x12a026da0>, <kliff.models.kim.KIMComputeArguments object at 0x12a026e10>, <kliff.models.kim.KIMComputeArguments object at 0x12a026e80>, <kliff.models.kim.KIMComputeArguments object at 0x12a026ef0>, <kliff.models.kim.KIMComputeArguments object at 0x12a026f60>, <kliff.models.kim.KIMComputeArguments object at 0x12a026048>, <kliff.models.kim.KIMComputeArguments object at 0x12a033080>, <kliff.models.kim.KIMComputeArguments object at 0x12a0330f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a033160>, <kliff.models.kim.KIMComputeArguments object at 0x12a0331d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a033240>, <kliff.models.kim.KIMComputeArguments object at 0x12a0332b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a033320>, <kliff.models.kim.KIMComputeArguments object at 0x12a033390>, <kliff.models.kim.KIMComputeArguments object at 0x12a033400>, <kliff.models.kim.KIMComputeArguments object at 0x12a033470>, <kliff.models.kim.KIMComputeArguments object at 0x12a0334e0>, <kliff.models.kim.KIMComputeArguments object at 0x12a033550>, <kliff.models.kim.KIMComputeArguments object at 0x12a0335c0>, <kliff.models.kim.KIMComputeArguments object at 0x12a033630>, <kliff.models.kim.KIMComputeArguments object at 0x12a0336a0>, <kliff.models.kim.KIMComputeArguments object at 0x12a033710>, <kliff.models.kim.KIMComputeArguments object at 0x12a033780>, <kliff.models.kim.KIMComputeArguments object at 0x12a0337f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a033860>, <kliff.models.kim.KIMComputeArguments object at 0x12a0338d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a033940>, <kliff.models.kim.KIMComputeArguments object at 0x12a0339b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a033a20>, <kliff.models.kim.KIMComputeArguments object at 0x12a033a90>, <kliff.models.kim.KIMComputeArguments object at 0x12a033b00>, <kliff.models.kim.KIMComputeArguments object at 0x12a033b70>, <kliff.models.kim.KIMComputeArguments object at 0x12a033be0>, <kliff.models.kim.KIMComputeArguments object at 0x12a033c50>, <kliff.models.kim.KIMComputeArguments object at 0x12a033cc0>, <kliff.models.kim.KIMComputeArguments object at 0x12a033d30>, <kliff.models.kim.KIMComputeArguments object at 0x12a033da0>, <kliff.models.kim.KIMComputeArguments object at 0x12a033e10>, <kliff.models.kim.KIMComputeArguments object at 0x12a033e80>, <kliff.models.kim.KIMComputeArguments object at 0x12a033ef0>, <kliff.models.kim.KIMComputeArguments object at 0x12a033f60>, <kliff.models.kim.KIMComputeArguments object at 0x12a033048>, <kliff.models.kim.KIMComputeArguments object at 0x12a03f080>, <kliff.models.kim.KIMComputeArguments object at 0x12a03f0f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a03f160>, <kliff.models.kim.KIMComputeArguments object at 0x12a03f1d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a03f240>, <kliff.models.kim.KIMComputeArguments object at 0x12a03f2b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a03f320>, <kliff.models.kim.KIMComputeArguments object at 0x12a03f390>, <kliff.models.kim.KIMComputeArguments object at 0x12a03f400>, <kliff.models.kim.KIMComputeArguments object at 0x12a03f470>, <kliff.models.kim.KIMComputeArguments object at 0x12a03f4e0>, <kliff.models.kim.KIMComputeArguments object at 0x12a03f550>, <kliff.models.kim.KIMComputeArguments object at 0x12a03f5c0>, <kliff.models.kim.KIMComputeArguments object at 0x12a03f630>, <kliff.models.kim.KIMComputeArguments object at 0x12a03f6a0>, <kliff.models.kim.KIMComputeArguments object at 0x12a03f710>, <kliff.models.kim.KIMComputeArguments object at 0x12a03f780>, <kliff.models.kim.KIMComputeArguments object at 0x12a03f7f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a03f860>, <kliff.models.kim.KIMComputeArguments object at 0x12a03f8d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a03f940>, <kliff.models.kim.KIMComputeArguments object at 0x12a03f9b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a03fa20>, <kliff.models.kim.KIMComputeArguments object at 0x12a03fa90>, <kliff.models.kim.KIMComputeArguments object at 0x12a03fb00>, <kliff.models.kim.KIMComputeArguments object at 0x12a03fb70>, <kliff.models.kim.KIMComputeArguments object at 0x12a03fbe0>, <kliff.models.kim.KIMComputeArguments object at 0x12a03fc50>, <kliff.models.kim.KIMComputeArguments object at 0x12a03fcc0>, <kliff.models.kim.KIMComputeArguments object at 0x12a03fd30>, <kliff.models.kim.KIMComputeArguments object at 0x12a03fda0>, <kliff.models.kim.KIMComputeArguments object at 0x12a03fe10>, <kliff.models.kim.KIMComputeArguments object at 0x12a03fe80>, <kliff.models.kim.KIMComputeArguments object at 0x12a03fef0>, <kliff.models.kim.KIMComputeArguments object at 0x12a03ff60>, <kliff.models.kim.KIMComputeArguments object at 0x12a03f048>, <kliff.models.kim.KIMComputeArguments object at 0x12a04d080>, <kliff.models.kim.KIMComputeArguments object at 0x12a04d0f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a04d160>, <kliff.models.kim.KIMComputeArguments object at 0x12a04d1d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a04d240>, <kliff.models.kim.KIMComputeArguments object at 0x12a04d2b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a04d320>, <kliff.models.kim.KIMComputeArguments object at 0x12a04d390>, <kliff.models.kim.KIMComputeArguments object at 0x12a04d400>, <kliff.models.kim.KIMComputeArguments object at 0x12a04d470>, <kliff.models.kim.KIMComputeArguments object at 0x12a04d4e0>, <kliff.models.kim.KIMComputeArguments object at 0x12a04d550>, <kliff.models.kim.KIMComputeArguments object at 0x12a04d5c0>, <kliff.models.kim.KIMComputeArguments object at 0x12a04d630>, <kliff.models.kim.KIMComputeArguments object at 0x12a04d6a0>, <kliff.models.kim.KIMComputeArguments object at 0x12a04d710>, <kliff.models.kim.KIMComputeArguments object at 0x12a04d780>, <kliff.models.kim.KIMComputeArguments object at 0x12a04d7f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a04d860>, <kliff.models.kim.KIMComputeArguments object at 0x12a04d8d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a04d940>, <kliff.models.kim.KIMComputeArguments object at 0x12a04d9b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a04da20>, <kliff.models.kim.KIMComputeArguments object at 0x12a04da90>, <kliff.models.kim.KIMComputeArguments object at 0x12a04db00>, <kliff.models.kim.KIMComputeArguments object at 0x12a04db70>, <kliff.models.kim.KIMComputeArguments object at 0x12a04dbe0>, <kliff.models.kim.KIMComputeArguments object at 0x12a04dc50>, <kliff.models.kim.KIMComputeArguments object at 0x12a04dcc0>, <kliff.models.kim.KIMComputeArguments object at 0x12a04dd30>, <kliff.models.kim.KIMComputeArguments object at 0x12a04dda0>, <kliff.models.kim.KIMComputeArguments object at 0x12a04de10>, <kliff.models.kim.KIMComputeArguments object at 0x12a04de80>, <kliff.models.kim.KIMComputeArguments object at 0x12a04def0>, <kliff.models.kim.KIMComputeArguments object at 0x12a04df60>, <kliff.models.kim.KIMComputeArguments object at 0x12a04d048>, <kliff.models.kim.KIMComputeArguments object at 0x12a05c080>, <kliff.models.kim.KIMComputeArguments object at 0x12a05c0f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a05c160>, <kliff.models.kim.KIMComputeArguments object at 0x12a05c1d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a05c240>, <kliff.models.kim.KIMComputeArguments object at 0x12a05c2b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a05c320>, <kliff.models.kim.KIMComputeArguments object at 0x12a05c390>, <kliff.models.kim.KIMComputeArguments object at 0x12a05c400>, <kliff.models.kim.KIMComputeArguments object at 0x12a05c470>, <kliff.models.kim.KIMComputeArguments object at 0x12a05c4e0>, <kliff.models.kim.KIMComputeArguments object at 0x12a05c550>, <kliff.models.kim.KIMComputeArguments object at 0x12a05c5c0>, <kliff.models.kim.KIMComputeArguments object at 0x12a05c630>, <kliff.models.kim.KIMComputeArguments object at 0x12a05c6a0>, <kliff.models.kim.KIMComputeArguments object at 0x12a05c710>, <kliff.models.kim.KIMComputeArguments object at 0x12a05c780>, <kliff.models.kim.KIMComputeArguments object at 0x12a05c7f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a05c860>, <kliff.models.kim.KIMComputeArguments object at 0x12a05c8d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a05c940>, <kliff.models.kim.KIMComputeArguments object at 0x12a05c9b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a05ca20>, <kliff.models.kim.KIMComputeArguments object at 0x12a05ca90>, <kliff.models.kim.KIMComputeArguments object at 0x12a05cb00>, <kliff.models.kim.KIMComputeArguments object at 0x12a05cb70>, <kliff.models.kim.KIMComputeArguments object at 0x12a05cbe0>, <kliff.models.kim.KIMComputeArguments object at 0x12a05cc50>, <kliff.models.kim.KIMComputeArguments object at 0x12a05ccc0>, <kliff.models.kim.KIMComputeArguments object at 0x12a05cd30>, <kliff.models.kim.KIMComputeArguments object at 0x12a05cda0>, <kliff.models.kim.KIMComputeArguments object at 0x12a05ce10>, <kliff.models.kim.KIMComputeArguments object at 0x12a05ce80>, <kliff.models.kim.KIMComputeArguments object at 0x12a05cef0>, <kliff.models.kim.KIMComputeArguments object at 0x12a05cf60>, <kliff.models.kim.KIMComputeArguments object at 0x12a05c0b8>, <kliff.models.kim.KIMComputeArguments object at 0x12a069080>, <kliff.models.kim.KIMComputeArguments object at 0x12a0690f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a069160>, <kliff.models.kim.KIMComputeArguments object at 0x12a0691d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a069240>, <kliff.models.kim.KIMComputeArguments object at 0x12a0692b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a069320>, <kliff.models.kim.KIMComputeArguments object at 0x12a069390>, <kliff.models.kim.KIMComputeArguments object at 0x12a069400>, <kliff.models.kim.KIMComputeArguments object at 0x12a069470>, <kliff.models.kim.KIMComputeArguments object at 0x12a0694e0>, <kliff.models.kim.KIMComputeArguments object at 0x12a069550>, <kliff.models.kim.KIMComputeArguments object at 0x12a0695c0>, <kliff.models.kim.KIMComputeArguments object at 0x12a069630>, <kliff.models.kim.KIMComputeArguments object at 0x12a0696a0>, <kliff.models.kim.KIMComputeArguments object at 0x12a069710>, <kliff.models.kim.KIMComputeArguments object at 0x12a069780>, <kliff.models.kim.KIMComputeArguments object at 0x12a0697f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a069860>, <kliff.models.kim.KIMComputeArguments object at 0x12a0698d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a069940>, <kliff.models.kim.KIMComputeArguments object at 0x12a0699b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a069a20>, <kliff.models.kim.KIMComputeArguments object at 0x12a069a90>, <kliff.models.kim.KIMComputeArguments object at 0x12a069b00>, <kliff.models.kim.KIMComputeArguments object at 0x12a069b70>, <kliff.models.kim.KIMComputeArguments object at 0x12a069be0>, <kliff.models.kim.KIMComputeArguments object at 0x12a069c50>, <kliff.models.kim.KIMComputeArguments object at 0x12a069cc0>, <kliff.models.kim.KIMComputeArguments object at 0x12a069d30>, <kliff.models.kim.KIMComputeArguments object at 0x12a069da0>, <kliff.models.kim.KIMComputeArguments object at 0x12a069e10>, <kliff.models.kim.KIMComputeArguments object at 0x12a069e80>, <kliff.models.kim.KIMComputeArguments object at 0x12a069ef0>, <kliff.models.kim.KIMComputeArguments object at 0x12a069f60>, <kliff.models.kim.KIMComputeArguments object at 0x12a0690b8>, <kliff.models.kim.KIMComputeArguments object at 0x12a077080>, <kliff.models.kim.KIMComputeArguments object at 0x12a0770f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a077160>, <kliff.models.kim.KIMComputeArguments object at 0x12a0771d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a077240>, <kliff.models.kim.KIMComputeArguments object at 0x12a0772b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a077320>, <kliff.models.kim.KIMComputeArguments object at 0x12a077390>, <kliff.models.kim.KIMComputeArguments object at 0x12a077400>, <kliff.models.kim.KIMComputeArguments object at 0x12a077470>, <kliff.models.kim.KIMComputeArguments object at 0x12a0774e0>, <kliff.models.kim.KIMComputeArguments object at 0x12a077550>, <kliff.models.kim.KIMComputeArguments object at 0x12a0775c0>, <kliff.models.kim.KIMComputeArguments object at 0x12a077630>, <kliff.models.kim.KIMComputeArguments object at 0x12a0776a0>, <kliff.models.kim.KIMComputeArguments object at 0x12a077710>, <kliff.models.kim.KIMComputeArguments object at 0x12a077780>, <kliff.models.kim.KIMComputeArguments object at 0x12a0777f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a077860>, <kliff.models.kim.KIMComputeArguments object at 0x12a0778d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a077940>, <kliff.models.kim.KIMComputeArguments object at 0x12a0779b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a077a20>, <kliff.models.kim.KIMComputeArguments object at 0x12a077a90>, <kliff.models.kim.KIMComputeArguments object at 0x12a077b00>, <kliff.models.kim.KIMComputeArguments object at 0x12a077b70>, <kliff.models.kim.KIMComputeArguments object at 0x12a077be0>, <kliff.models.kim.KIMComputeArguments object at 0x12a077c50>, <kliff.models.kim.KIMComputeArguments object at 0x12a077cc0>, <kliff.models.kim.KIMComputeArguments object at 0x12a077d30>, <kliff.models.kim.KIMComputeArguments object at 0x12a077da0>, <kliff.models.kim.KIMComputeArguments object at 0x12a077e10>, <kliff.models.kim.KIMComputeArguments object at 0x12a077e80>, <kliff.models.kim.KIMComputeArguments object at 0x12a077ef0>, <kliff.models.kim.KIMComputeArguments object at 0x12a077f60>, <kliff.models.kim.KIMComputeArguments object at 0x12a0770b8>, <kliff.models.kim.KIMComputeArguments object at 0x12a086080>, <kliff.models.kim.KIMComputeArguments object at 0x12a0860f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a086160>, <kliff.models.kim.KIMComputeArguments object at 0x12a0861d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a086240>, <kliff.models.kim.KIMComputeArguments object at 0x12a0862b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a086320>, <kliff.models.kim.KIMComputeArguments object at 0x12a086390>, <kliff.models.kim.KIMComputeArguments object at 0x12a086400>, <kliff.models.kim.KIMComputeArguments object at 0x12a086470>, <kliff.models.kim.KIMComputeArguments object at 0x12a0864e0>, <kliff.models.kim.KIMComputeArguments object at 0x12a086550>, <kliff.models.kim.KIMComputeArguments object at 0x12a0865c0>, <kliff.models.kim.KIMComputeArguments object at 0x12a086630>, <kliff.models.kim.KIMComputeArguments object at 0x12a0866a0>, <kliff.models.kim.KIMComputeArguments object at 0x12a086710>, <kliff.models.kim.KIMComputeArguments object at 0x12a086780>, <kliff.models.kim.KIMComputeArguments object at 0x12a0867f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a086860>, <kliff.models.kim.KIMComputeArguments object at 0x12a0868d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a086940>, <kliff.models.kim.KIMComputeArguments object at 0x12a0869b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a086a20>, <kliff.models.kim.KIMComputeArguments object at 0x12a086a90>, <kliff.models.kim.KIMComputeArguments object at 0x12a086b00>, <kliff.models.kim.KIMComputeArguments object at 0x12a086b70>, <kliff.models.kim.KIMComputeArguments object at 0x12a086be0>, <kliff.models.kim.KIMComputeArguments object at 0x12a086c50>, <kliff.models.kim.KIMComputeArguments object at 0x12a086cc0>, <kliff.models.kim.KIMComputeArguments object at 0x12a086d30>, <kliff.models.kim.KIMComputeArguments object at 0x12a086da0>, <kliff.models.kim.KIMComputeArguments object at 0x12a086e10>, <kliff.models.kim.KIMComputeArguments object at 0x12a086e80>, <kliff.models.kim.KIMComputeArguments object at 0x12a086ef0>, <kliff.models.kim.KIMComputeArguments object at 0x12a086f60>, <kliff.models.kim.KIMComputeArguments object at 0x12a0860b8>, <kliff.models.kim.KIMComputeArguments object at 0x12a093080>, <kliff.models.kim.KIMComputeArguments object at 0x12a0930f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a093160>, <kliff.models.kim.KIMComputeArguments object at 0x12a0931d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a093240>, <kliff.models.kim.KIMComputeArguments object at 0x12a0932b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a093320>, <kliff.models.kim.KIMComputeArguments object at 0x12a093390>, <kliff.models.kim.KIMComputeArguments object at 0x12a093400>, <kliff.models.kim.KIMComputeArguments object at 0x12a093470>, <kliff.models.kim.KIMComputeArguments object at 0x12a0934e0>, <kliff.models.kim.KIMComputeArguments object at 0x12a093550>, <kliff.models.kim.KIMComputeArguments object at 0x12a0935c0>, <kliff.models.kim.KIMComputeArguments object at 0x12a093630>, <kliff.models.kim.KIMComputeArguments object at 0x12a0936a0>, <kliff.models.kim.KIMComputeArguments object at 0x12a093710>, <kliff.models.kim.KIMComputeArguments object at 0x12a093780>, <kliff.models.kim.KIMComputeArguments object at 0x12a0937f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a093860>, <kliff.models.kim.KIMComputeArguments object at 0x12a0938d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a093940>, <kliff.models.kim.KIMComputeArguments object at 0x12a0939b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a093a20>, <kliff.models.kim.KIMComputeArguments object at 0x12a093a90>, <kliff.models.kim.KIMComputeArguments object at 0x12a093b00>, <kliff.models.kim.KIMComputeArguments object at 0x12a093b70>, <kliff.models.kim.KIMComputeArguments object at 0x12a093be0>, <kliff.models.kim.KIMComputeArguments object at 0x12a093c50>, <kliff.models.kim.KIMComputeArguments object at 0x12a093cc0>, <kliff.models.kim.KIMComputeArguments object at 0x12a093d30>, <kliff.models.kim.KIMComputeArguments object at 0x12a093da0>, <kliff.models.kim.KIMComputeArguments object at 0x12a093e10>, <kliff.models.kim.KIMComputeArguments object at 0x12a093e80>, <kliff.models.kim.KIMComputeArguments object at 0x12a093ef0>, <kliff.models.kim.KIMComputeArguments object at 0x12a093f60>, <kliff.models.kim.KIMComputeArguments object at 0x12a0930b8>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a0080>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a00f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a0160>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a01d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a0240>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a02b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a0320>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a0390>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a0400>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a0470>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a04e0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a0550>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a05c0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a0630>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a06a0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a0710>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a0780>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a07f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a0860>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a08d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a0940>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a09b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a0a20>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a0a90>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a0b00>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a0b70>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a0be0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a0c50>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a0cc0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a0d30>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a0da0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a0e10>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a0e80>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a0ef0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a0f60>, <kliff.models.kim.KIMComputeArguments object at 0x12a0a00b8>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ae080>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ae0f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ae160>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ae1d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ae240>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ae2b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ae320>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ae390>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ae400>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ae470>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ae4e0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ae550>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ae5c0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ae630>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ae6a0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ae710>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ae780>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ae7f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ae860>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ae8d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ae940>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ae9b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0aea20>, <kliff.models.kim.KIMComputeArguments object at 0x12a0aea90>, <kliff.models.kim.KIMComputeArguments object at 0x12a0aeb00>, <kliff.models.kim.KIMComputeArguments object at 0x12a0aeb70>, <kliff.models.kim.KIMComputeArguments object at 0x12a0aebe0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0aec50>, <kliff.models.kim.KIMComputeArguments object at 0x12a0aecc0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0aed30>, <kliff.models.kim.KIMComputeArguments object at 0x12a0aeda0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0aee10>, <kliff.models.kim.KIMComputeArguments object at 0x12a0aee80>, <kliff.models.kim.KIMComputeArguments object at 0x12a0aeef0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0aef60>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ae048>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bc080>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bc0f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bc160>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bc1d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bc240>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bc2b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bc320>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bc390>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bc400>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bc470>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bc4e0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bc550>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bc5c0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bc630>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bc6a0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bc710>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bc780>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bc7f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bc860>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bc8d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bc940>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bc9b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bca20>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bca90>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bcb00>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bcb70>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bcbe0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bcc50>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bccc0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bcd30>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bcda0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bce10>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bce80>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bcef0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bcf60>, <kliff.models.kim.KIMComputeArguments object at 0x12a0bc048>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c9080>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c90f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c9160>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c91d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c9240>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c92b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c9320>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c9390>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c9400>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c9470>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c94e0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c9550>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c95c0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c9630>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c96a0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c9710>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c9780>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c97f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c9860>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c98d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c9940>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c99b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c9a20>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c9a90>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c9b00>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c9b70>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c9be0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c9c50>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c9cc0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c9d30>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c9da0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c9e10>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c9e80>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c9ef0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c9f60>, <kliff.models.kim.KIMComputeArguments object at 0x12a0c9048>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d7080>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d70f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d7160>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d71d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d7240>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d72b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d7320>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d7390>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d7400>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d7470>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d74e0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d7550>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d75c0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d7630>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d76a0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d7710>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d7780>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d77f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d7860>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d78d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d7940>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d79b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d7a20>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d7a90>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d7b00>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d7b70>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d7be0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d7c50>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d7cc0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d7d30>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d7da0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d7e10>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d7e80>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d7ef0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d7f60>, <kliff.models.kim.KIMComputeArguments object at 0x12a0d7048>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e6080>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e60f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e6160>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e61d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e6240>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e62b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e6320>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e6390>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e6400>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e6470>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e64e0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e6550>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e65c0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e6630>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e66a0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e6710>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e6780>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e67f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e6860>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e68d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e6940>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e69b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e6a20>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e6a90>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e6b00>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e6b70>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e6be0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e6c50>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e6cc0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e6d30>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e6da0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e6e10>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e6e80>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e6ef0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e6f60>, <kliff.models.kim.KIMComputeArguments object at 0x12a0e6048>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f3048>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f30f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f3160>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f31d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f3240>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f32b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f3320>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f3390>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f3400>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f3470>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f34e0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f3550>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f35c0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f3630>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f36a0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f3710>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f3780>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f37f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f3860>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f38d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f3940>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f39b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f3a20>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f3a90>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f3b00>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f3b70>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f3be0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f3c50>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f3cc0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f3d30>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f3da0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f3e10>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f3e80>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f3ef0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f3f60>, <kliff.models.kim.KIMComputeArguments object at 0x12a0f30b8>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ff080>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ff0f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ff160>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ff1d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ff240>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ff2b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ff320>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ff390>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ff400>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ff470>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ff4e0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ff550>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ff5c0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ff630>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ff6a0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ff710>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ff780>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ff7f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ff860>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ff8d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ff940>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ff9b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ffa20>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ffa90>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ffb00>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ffb70>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ffbe0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ffc50>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ffcc0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ffd30>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ffda0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ffe10>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ffe80>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ffef0>, <kliff.models.kim.KIMComputeArguments object at 0x12a0fff60>, <kliff.models.kim.KIMComputeArguments object at 0x12a0ff048>, <kliff.models.kim.KIMComputeArguments object at 0x12a10d080>, <kliff.models.kim.KIMComputeArguments object at 0x12a10d0f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a10d160>, <kliff.models.kim.KIMComputeArguments object at 0x12a10d1d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a10d240>, <kliff.models.kim.KIMComputeArguments object at 0x12a10d2b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a10d320>, <kliff.models.kim.KIMComputeArguments object at 0x12a10d390>, <kliff.models.kim.KIMComputeArguments object at 0x12a10d400>, <kliff.models.kim.KIMComputeArguments object at 0x12a10d470>, <kliff.models.kim.KIMComputeArguments object at 0x12a10d4e0>, <kliff.models.kim.KIMComputeArguments object at 0x12a10d550>, <kliff.models.kim.KIMComputeArguments object at 0x12a10d5c0>, <kliff.models.kim.KIMComputeArguments object at 0x12a10d630>, <kliff.models.kim.KIMComputeArguments object at 0x12a10d6a0>, <kliff.models.kim.KIMComputeArguments object at 0x12a10d710>, <kliff.models.kim.KIMComputeArguments object at 0x12a10d780>, <kliff.models.kim.KIMComputeArguments object at 0x12a10d7f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a10d860>, <kliff.models.kim.KIMComputeArguments object at 0x12a10d8d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a10d940>, <kliff.models.kim.KIMComputeArguments object at 0x12a10d9b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a10da20>, <kliff.models.kim.KIMComputeArguments object at 0x12a10da90>, <kliff.models.kim.KIMComputeArguments object at 0x12a10db00>, <kliff.models.kim.KIMComputeArguments object at 0x12a10db70>, <kliff.models.kim.KIMComputeArguments object at 0x12a10dbe0>, <kliff.models.kim.KIMComputeArguments object at 0x12a10dc50>, <kliff.models.kim.KIMComputeArguments object at 0x12a10dcc0>, <kliff.models.kim.KIMComputeArguments object at 0x12a10dd30>, <kliff.models.kim.KIMComputeArguments object at 0x12a10dda0>, <kliff.models.kim.KIMComputeArguments object at 0x12a10de10>, <kliff.models.kim.KIMComputeArguments object at 0x12a10de80>, <kliff.models.kim.KIMComputeArguments object at 0x12a10def0>, <kliff.models.kim.KIMComputeArguments object at 0x12a10df60>, <kliff.models.kim.KIMComputeArguments object at 0x12a10d048>, <kliff.models.kim.KIMComputeArguments object at 0x12a11c080>, <kliff.models.kim.KIMComputeArguments object at 0x12a11c0f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a11c160>, <kliff.models.kim.KIMComputeArguments object at 0x12a11c1d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a11c240>, <kliff.models.kim.KIMComputeArguments object at 0x12a11c2b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a11c320>, <kliff.models.kim.KIMComputeArguments object at 0x12a11c390>, <kliff.models.kim.KIMComputeArguments object at 0x12a11c400>, <kliff.models.kim.KIMComputeArguments object at 0x12a11c470>, <kliff.models.kim.KIMComputeArguments object at 0x12a11c4e0>, <kliff.models.kim.KIMComputeArguments object at 0x12a11c550>, <kliff.models.kim.KIMComputeArguments object at 0x12a11c5c0>, <kliff.models.kim.KIMComputeArguments object at 0x12a11c630>, <kliff.models.kim.KIMComputeArguments object at 0x12a11c6a0>, <kliff.models.kim.KIMComputeArguments object at 0x12a11c710>, <kliff.models.kim.KIMComputeArguments object at 0x12a11c780>, <kliff.models.kim.KIMComputeArguments object at 0x12a11c7f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a11c860>, <kliff.models.kim.KIMComputeArguments object at 0x12a11c8d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a11c940>, <kliff.models.kim.KIMComputeArguments object at 0x12a11c9b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a11ca20>, <kliff.models.kim.KIMComputeArguments object at 0x12a11ca90>, <kliff.models.kim.KIMComputeArguments object at 0x12a11cb00>, <kliff.models.kim.KIMComputeArguments object at 0x12a11cb70>, <kliff.models.kim.KIMComputeArguments object at 0x12a11cbe0>, <kliff.models.kim.KIMComputeArguments object at 0x12a11cc50>, <kliff.models.kim.KIMComputeArguments object at 0x12a11ccc0>, <kliff.models.kim.KIMComputeArguments object at 0x12a11cd30>, <kliff.models.kim.KIMComputeArguments object at 0x12a11cda0>, <kliff.models.kim.KIMComputeArguments object at 0x12a11ce10>, <kliff.models.kim.KIMComputeArguments object at 0x12a11ce80>, <kliff.models.kim.KIMComputeArguments object at 0x12a11cef0>, <kliff.models.kim.KIMComputeArguments object at 0x12a11cf60>, <kliff.models.kim.KIMComputeArguments object at 0x12a11c048>, <kliff.models.kim.KIMComputeArguments object at 0x12a129080>, <kliff.models.kim.KIMComputeArguments object at 0x12a1290f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a129160>, <kliff.models.kim.KIMComputeArguments object at 0x12a1291d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a129240>, <kliff.models.kim.KIMComputeArguments object at 0x12a1292b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a129320>, <kliff.models.kim.KIMComputeArguments object at 0x12a129390>, <kliff.models.kim.KIMComputeArguments object at 0x12a129400>, <kliff.models.kim.KIMComputeArguments object at 0x12a129470>, <kliff.models.kim.KIMComputeArguments object at 0x12a1294e0>, <kliff.models.kim.KIMComputeArguments object at 0x12a129550>, <kliff.models.kim.KIMComputeArguments object at 0x12a1295c0>, <kliff.models.kim.KIMComputeArguments object at 0x12a129630>, <kliff.models.kim.KIMComputeArguments object at 0x12a1296a0>, <kliff.models.kim.KIMComputeArguments object at 0x12a129710>, <kliff.models.kim.KIMComputeArguments object at 0x12a129780>, <kliff.models.kim.KIMComputeArguments object at 0x12a1297f0>, <kliff.models.kim.KIMComputeArguments object at 0x12a129860>, <kliff.models.kim.KIMComputeArguments object at 0x12a1298d0>, <kliff.models.kim.KIMComputeArguments object at 0x12a129940>, <kliff.models.kim.KIMComputeArguments object at 0x12a1299b0>, <kliff.models.kim.KIMComputeArguments object at 0x12a129a20>, <kliff.models.kim.KIMComputeArguments object at 0x12a129a90>, <kliff.models.kim.KIMComputeArguments object at 0x12a129b00>, <kliff.models.kim.KIMComputeArguments object at 0x12a129b70>, <kliff.models.kim.KIMComputeArguments object at 0x12a129be0>, <kliff.models.kim.KIMComputeArguments object at 0x12a129c50>, <kliff.models.kim.KIMComputeArguments object at 0x12a129cc0>]



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

   **Total running time of the script:** ( 1 minutes  40.195 seconds)


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

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
