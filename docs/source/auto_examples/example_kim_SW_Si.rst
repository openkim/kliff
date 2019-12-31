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

KLIFF has a :class:`~kliff.dataset.Dataset` to deal with the training data (and possibly
test data). For the silicon training set, we can read and process the files by:


.. code-block:: default


    dataset_name = 'Si_training_set'
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


    [<kliff.models.kim.KIMComputeArguments object at 0x1026497320>, <kliff.models.kim.KIMComputeArguments object at 0x10264975c0>, <kliff.models.kim.KIMComputeArguments object at 0x1026497208>, <kliff.models.kim.KIMComputeArguments object at 0x10264976a0>, <kliff.models.kim.KIMComputeArguments object at 0x1026497748>, <kliff.models.kim.KIMComputeArguments object at 0x10264977f0>, <kliff.models.kim.KIMComputeArguments object at 0x10264978d0>, <kliff.models.kim.KIMComputeArguments object at 0x1026497978>, <kliff.models.kim.KIMComputeArguments object at 0x1026497a20>, <kliff.models.kim.KIMComputeArguments object at 0x1026497b00>, <kliff.models.kim.KIMComputeArguments object at 0x1026497ba8>, <kliff.models.kim.KIMComputeArguments object at 0x1026497c50>, <kliff.models.kim.KIMComputeArguments object at 0x1026497d30>, <kliff.models.kim.KIMComputeArguments object at 0x1026497dd8>, <kliff.models.kim.KIMComputeArguments object at 0x1026497e80>, <kliff.models.kim.KIMComputeArguments object at 0x1026497ef0>, <kliff.models.kim.KIMComputeArguments object at 0x1026497f28>, <kliff.models.kim.KIMComputeArguments object at 0x1026497240>, <kliff.models.kim.KIMComputeArguments object at 0x1026497fd0>, <kliff.models.kim.KIMComputeArguments object at 0x10264974a8>, <kliff.models.kim.KIMComputeArguments object at 0x1026497198>, <kliff.models.kim.KIMComputeArguments object at 0x1026507080>, <kliff.models.kim.KIMComputeArguments object at 0x10265070f0>, <kliff.models.kim.KIMComputeArguments object at 0x1026507160>, <kliff.models.kim.KIMComputeArguments object at 0x10265071d0>, <kliff.models.kim.KIMComputeArguments object at 0x1026507240>, <kliff.models.kim.KIMComputeArguments object at 0x10265072b0>, <kliff.models.kim.KIMComputeArguments object at 0x1026507320>, <kliff.models.kim.KIMComputeArguments object at 0x1026507390>, <kliff.models.kim.KIMComputeArguments object at 0x1026507400>, <kliff.models.kim.KIMComputeArguments object at 0x1026507470>, <kliff.models.kim.KIMComputeArguments object at 0x10265074e0>, <kliff.models.kim.KIMComputeArguments object at 0x1026507550>, <kliff.models.kim.KIMComputeArguments object at 0x10265075c0>, <kliff.models.kim.KIMComputeArguments object at 0x1026507630>, <kliff.models.kim.KIMComputeArguments object at 0x10265076a0>, <kliff.models.kim.KIMComputeArguments object at 0x1026507710>, <kliff.models.kim.KIMComputeArguments object at 0x1026507780>, <kliff.models.kim.KIMComputeArguments object at 0x10265077f0>, <kliff.models.kim.KIMComputeArguments object at 0x1026507860>, <kliff.models.kim.KIMComputeArguments object at 0x10265078d0>, <kliff.models.kim.KIMComputeArguments object at 0x1026507940>, <kliff.models.kim.KIMComputeArguments object at 0x10265079b0>, <kliff.models.kim.KIMComputeArguments object at 0x1026507a20>, <kliff.models.kim.KIMComputeArguments object at 0x1026507a90>, <kliff.models.kim.KIMComputeArguments object at 0x1026507b00>, <kliff.models.kim.KIMComputeArguments object at 0x1026507b70>, <kliff.models.kim.KIMComputeArguments object at 0x1026507be0>, <kliff.models.kim.KIMComputeArguments object at 0x1026507c50>, <kliff.models.kim.KIMComputeArguments object at 0x1026507cc0>, <kliff.models.kim.KIMComputeArguments object at 0x1026507d30>, <kliff.models.kim.KIMComputeArguments object at 0x1026507da0>, <kliff.models.kim.KIMComputeArguments object at 0x1026507e10>, <kliff.models.kim.KIMComputeArguments object at 0x1026507e80>, <kliff.models.kim.KIMComputeArguments object at 0x1026507ef0>, <kliff.models.kim.KIMComputeArguments object at 0x1026507f60>, <kliff.models.kim.KIMComputeArguments object at 0x1026507048>, <kliff.models.kim.KIMComputeArguments object at 0x1026514080>, <kliff.models.kim.KIMComputeArguments object at 0x10265140f0>, <kliff.models.kim.KIMComputeArguments object at 0x1026514160>, <kliff.models.kim.KIMComputeArguments object at 0x10265141d0>, <kliff.models.kim.KIMComputeArguments object at 0x1026514240>, <kliff.models.kim.KIMComputeArguments object at 0x10265142b0>, <kliff.models.kim.KIMComputeArguments object at 0x1026514320>, <kliff.models.kim.KIMComputeArguments object at 0x1026514390>, <kliff.models.kim.KIMComputeArguments object at 0x1026514400>, <kliff.models.kim.KIMComputeArguments object at 0x1026514470>, <kliff.models.kim.KIMComputeArguments object at 0x10265144e0>, <kliff.models.kim.KIMComputeArguments object at 0x1026514550>, <kliff.models.kim.KIMComputeArguments object at 0x10265145c0>, <kliff.models.kim.KIMComputeArguments object at 0x1026514630>, <kliff.models.kim.KIMComputeArguments object at 0x10265146a0>, <kliff.models.kim.KIMComputeArguments object at 0x1026514710>, <kliff.models.kim.KIMComputeArguments object at 0x1026514780>, <kliff.models.kim.KIMComputeArguments object at 0x10265147f0>, <kliff.models.kim.KIMComputeArguments object at 0x1026514860>, <kliff.models.kim.KIMComputeArguments object at 0x10265148d0>, <kliff.models.kim.KIMComputeArguments object at 0x1026514940>, <kliff.models.kim.KIMComputeArguments object at 0x10265149b0>, <kliff.models.kim.KIMComputeArguments object at 0x1026514a20>, <kliff.models.kim.KIMComputeArguments object at 0x1026514a90>, <kliff.models.kim.KIMComputeArguments object at 0x1026514b00>, <kliff.models.kim.KIMComputeArguments object at 0x1026514b70>, <kliff.models.kim.KIMComputeArguments object at 0x1026514be0>, <kliff.models.kim.KIMComputeArguments object at 0x1026514c50>, <kliff.models.kim.KIMComputeArguments object at 0x1026514cc0>, <kliff.models.kim.KIMComputeArguments object at 0x1026514d30>, <kliff.models.kim.KIMComputeArguments object at 0x1026514da0>, <kliff.models.kim.KIMComputeArguments object at 0x1026514e10>, <kliff.models.kim.KIMComputeArguments object at 0x1026514e80>, <kliff.models.kim.KIMComputeArguments object at 0x1026514ef0>, <kliff.models.kim.KIMComputeArguments object at 0x1026514f60>, <kliff.models.kim.KIMComputeArguments object at 0x1026514048>, <kliff.models.kim.KIMComputeArguments object at 0x1026523080>, <kliff.models.kim.KIMComputeArguments object at 0x10265230f0>, <kliff.models.kim.KIMComputeArguments object at 0x1026523160>, <kliff.models.kim.KIMComputeArguments object at 0x10265231d0>, <kliff.models.kim.KIMComputeArguments object at 0x1026523240>, <kliff.models.kim.KIMComputeArguments object at 0x10265232b0>, <kliff.models.kim.KIMComputeArguments object at 0x1026523320>, <kliff.models.kim.KIMComputeArguments object at 0x1026523390>, <kliff.models.kim.KIMComputeArguments object at 0x1026523400>, <kliff.models.kim.KIMComputeArguments object at 0x1026523470>, <kliff.models.kim.KIMComputeArguments object at 0x10265234e0>, <kliff.models.kim.KIMComputeArguments object at 0x1026523550>, <kliff.models.kim.KIMComputeArguments object at 0x10265235c0>, <kliff.models.kim.KIMComputeArguments object at 0x1026523630>, <kliff.models.kim.KIMComputeArguments object at 0x10265236a0>, <kliff.models.kim.KIMComputeArguments object at 0x1026523710>, <kliff.models.kim.KIMComputeArguments object at 0x1026523780>, <kliff.models.kim.KIMComputeArguments object at 0x10265237f0>, <kliff.models.kim.KIMComputeArguments object at 0x1026523860>, <kliff.models.kim.KIMComputeArguments object at 0x10265238d0>, <kliff.models.kim.KIMComputeArguments object at 0x1026523940>, <kliff.models.kim.KIMComputeArguments object at 0x10265239b0>, <kliff.models.kim.KIMComputeArguments object at 0x1026523a20>, <kliff.models.kim.KIMComputeArguments object at 0x1026523a90>, <kliff.models.kim.KIMComputeArguments object at 0x1026523b00>, <kliff.models.kim.KIMComputeArguments object at 0x1026523b70>, <kliff.models.kim.KIMComputeArguments object at 0x1026523be0>, <kliff.models.kim.KIMComputeArguments object at 0x1026523c50>, <kliff.models.kim.KIMComputeArguments object at 0x1026523cc0>, <kliff.models.kim.KIMComputeArguments object at 0x1026523d30>, <kliff.models.kim.KIMComputeArguments object at 0x1026523da0>, <kliff.models.kim.KIMComputeArguments object at 0x1026523e10>, <kliff.models.kim.KIMComputeArguments object at 0x1026523e80>, <kliff.models.kim.KIMComputeArguments object at 0x1026523ef0>, <kliff.models.kim.KIMComputeArguments object at 0x1026523f60>, <kliff.models.kim.KIMComputeArguments object at 0x1026523048>, <kliff.models.kim.KIMComputeArguments object at 0x1026531080>, <kliff.models.kim.KIMComputeArguments object at 0x10265310f0>, <kliff.models.kim.KIMComputeArguments object at 0x1026531160>, <kliff.models.kim.KIMComputeArguments object at 0x10265311d0>, <kliff.models.kim.KIMComputeArguments object at 0x1026531240>, <kliff.models.kim.KIMComputeArguments object at 0x10265312b0>, <kliff.models.kim.KIMComputeArguments object at 0x1026531320>, <kliff.models.kim.KIMComputeArguments object at 0x1026531390>, <kliff.models.kim.KIMComputeArguments object at 0x1026531400>, <kliff.models.kim.KIMComputeArguments object at 0x1026531470>, <kliff.models.kim.KIMComputeArguments object at 0x10265314e0>, <kliff.models.kim.KIMComputeArguments object at 0x1026531550>, <kliff.models.kim.KIMComputeArguments object at 0x10265315c0>, <kliff.models.kim.KIMComputeArguments object at 0x1026531630>, <kliff.models.kim.KIMComputeArguments object at 0x10265316a0>, <kliff.models.kim.KIMComputeArguments object at 0x1026531710>, <kliff.models.kim.KIMComputeArguments object at 0x1026531780>, <kliff.models.kim.KIMComputeArguments object at 0x10265317f0>, <kliff.models.kim.KIMComputeArguments object at 0x1026531860>, <kliff.models.kim.KIMComputeArguments object at 0x10265318d0>, <kliff.models.kim.KIMComputeArguments object at 0x1026531940>, <kliff.models.kim.KIMComputeArguments object at 0x10265319b0>, <kliff.models.kim.KIMComputeArguments object at 0x1026531a20>, <kliff.models.kim.KIMComputeArguments object at 0x1026531a90>, <kliff.models.kim.KIMComputeArguments object at 0x1026531b00>, <kliff.models.kim.KIMComputeArguments object at 0x1026531b70>, <kliff.models.kim.KIMComputeArguments object at 0x1026531be0>, <kliff.models.kim.KIMComputeArguments object at 0x1026531c50>, <kliff.models.kim.KIMComputeArguments object at 0x1026531cc0>, <kliff.models.kim.KIMComputeArguments object at 0x1026531d30>, <kliff.models.kim.KIMComputeArguments object at 0x1026531da0>, <kliff.models.kim.KIMComputeArguments object at 0x1026531e10>, <kliff.models.kim.KIMComputeArguments object at 0x1026531e80>, <kliff.models.kim.KIMComputeArguments object at 0x1026531ef0>, <kliff.models.kim.KIMComputeArguments object at 0x1026531f60>, <kliff.models.kim.KIMComputeArguments object at 0x1026531048>, <kliff.models.kim.KIMComputeArguments object at 0x102653e048>, <kliff.models.kim.KIMComputeArguments object at 0x102653e0f0>, <kliff.models.kim.KIMComputeArguments object at 0x102653e160>, <kliff.models.kim.KIMComputeArguments object at 0x102653e1d0>, <kliff.models.kim.KIMComputeArguments object at 0x102653e240>, <kliff.models.kim.KIMComputeArguments object at 0x102653e2b0>, <kliff.models.kim.KIMComputeArguments object at 0x102653e320>, <kliff.models.kim.KIMComputeArguments object at 0x102653e390>, <kliff.models.kim.KIMComputeArguments object at 0x102653e400>, <kliff.models.kim.KIMComputeArguments object at 0x102653e470>, <kliff.models.kim.KIMComputeArguments object at 0x102653e4e0>, <kliff.models.kim.KIMComputeArguments object at 0x102653e550>, <kliff.models.kim.KIMComputeArguments object at 0x102653e5c0>, <kliff.models.kim.KIMComputeArguments object at 0x102653e630>, <kliff.models.kim.KIMComputeArguments object at 0x102653e6a0>, <kliff.models.kim.KIMComputeArguments object at 0x102653e710>, <kliff.models.kim.KIMComputeArguments object at 0x102653e780>, <kliff.models.kim.KIMComputeArguments object at 0x102653e7f0>, <kliff.models.kim.KIMComputeArguments object at 0x102653e860>, <kliff.models.kim.KIMComputeArguments object at 0x102653e8d0>, <kliff.models.kim.KIMComputeArguments object at 0x102653e940>, <kliff.models.kim.KIMComputeArguments object at 0x102653e9b0>, <kliff.models.kim.KIMComputeArguments object at 0x102653ea20>, <kliff.models.kim.KIMComputeArguments object at 0x102653ea90>, <kliff.models.kim.KIMComputeArguments object at 0x102653eb00>, <kliff.models.kim.KIMComputeArguments object at 0x102653eb70>, <kliff.models.kim.KIMComputeArguments object at 0x102653ebe0>, <kliff.models.kim.KIMComputeArguments object at 0x102653ec50>, <kliff.models.kim.KIMComputeArguments object at 0x102653ecc0>, <kliff.models.kim.KIMComputeArguments object at 0x102653ed30>, <kliff.models.kim.KIMComputeArguments object at 0x102653eda0>, <kliff.models.kim.KIMComputeArguments object at 0x102653ee10>, <kliff.models.kim.KIMComputeArguments object at 0x102653ee80>, <kliff.models.kim.KIMComputeArguments object at 0x102653eef0>, <kliff.models.kim.KIMComputeArguments object at 0x102653ef60>, <kliff.models.kim.KIMComputeArguments object at 0x102653e0b8>, <kliff.models.kim.KIMComputeArguments object at 0x102654c080>, <kliff.models.kim.KIMComputeArguments object at 0x102654c0f0>, <kliff.models.kim.KIMComputeArguments object at 0x102654c160>, <kliff.models.kim.KIMComputeArguments object at 0x102654c1d0>, <kliff.models.kim.KIMComputeArguments object at 0x102654c240>, <kliff.models.kim.KIMComputeArguments object at 0x102654c2b0>, <kliff.models.kim.KIMComputeArguments object at 0x102654c320>, <kliff.models.kim.KIMComputeArguments object at 0x102654c390>, <kliff.models.kim.KIMComputeArguments object at 0x102654c400>, <kliff.models.kim.KIMComputeArguments object at 0x102654c470>, <kliff.models.kim.KIMComputeArguments object at 0x102654c4e0>, <kliff.models.kim.KIMComputeArguments object at 0x102654c550>, <kliff.models.kim.KIMComputeArguments object at 0x102654c5c0>, <kliff.models.kim.KIMComputeArguments object at 0x102654c630>, <kliff.models.kim.KIMComputeArguments object at 0x102654c6a0>, <kliff.models.kim.KIMComputeArguments object at 0x102654c710>, <kliff.models.kim.KIMComputeArguments object at 0x102654c780>, <kliff.models.kim.KIMComputeArguments object at 0x102654c7f0>, <kliff.models.kim.KIMComputeArguments object at 0x102654c860>, <kliff.models.kim.KIMComputeArguments object at 0x102654c8d0>, <kliff.models.kim.KIMComputeArguments object at 0x102654c940>, <kliff.models.kim.KIMComputeArguments object at 0x102654c9b0>, <kliff.models.kim.KIMComputeArguments object at 0x102654ca20>, <kliff.models.kim.KIMComputeArguments object at 0x102654ca90>, <kliff.models.kim.KIMComputeArguments object at 0x102654cb00>, <kliff.models.kim.KIMComputeArguments object at 0x102654cb70>, <kliff.models.kim.KIMComputeArguments object at 0x102654cbe0>, <kliff.models.kim.KIMComputeArguments object at 0x102654cc50>, <kliff.models.kim.KIMComputeArguments object at 0x102654ccc0>, <kliff.models.kim.KIMComputeArguments object at 0x102654cd30>, <kliff.models.kim.KIMComputeArguments object at 0x102654cda0>, <kliff.models.kim.KIMComputeArguments object at 0x102654ce10>, <kliff.models.kim.KIMComputeArguments object at 0x102654ce80>, <kliff.models.kim.KIMComputeArguments object at 0x102654cef0>, <kliff.models.kim.KIMComputeArguments object at 0x102654cf60>, <kliff.models.kim.KIMComputeArguments object at 0x102654c048>, <kliff.models.kim.KIMComputeArguments object at 0x102655a080>, <kliff.models.kim.KIMComputeArguments object at 0x102655a0f0>, <kliff.models.kim.KIMComputeArguments object at 0x102655a160>, <kliff.models.kim.KIMComputeArguments object at 0x102655a1d0>, <kliff.models.kim.KIMComputeArguments object at 0x102655a240>, <kliff.models.kim.KIMComputeArguments object at 0x102655a2b0>, <kliff.models.kim.KIMComputeArguments object at 0x102655a320>, <kliff.models.kim.KIMComputeArguments object at 0x102655a390>, <kliff.models.kim.KIMComputeArguments object at 0x102655a400>, <kliff.models.kim.KIMComputeArguments object at 0x102655a470>, <kliff.models.kim.KIMComputeArguments object at 0x102655a4e0>, <kliff.models.kim.KIMComputeArguments object at 0x102655a550>, <kliff.models.kim.KIMComputeArguments object at 0x102655a5c0>, <kliff.models.kim.KIMComputeArguments object at 0x102655a630>, <kliff.models.kim.KIMComputeArguments object at 0x102655a6a0>, <kliff.models.kim.KIMComputeArguments object at 0x102655a710>, <kliff.models.kim.KIMComputeArguments object at 0x102655a780>, <kliff.models.kim.KIMComputeArguments object at 0x102655a7f0>, <kliff.models.kim.KIMComputeArguments object at 0x102655a860>, <kliff.models.kim.KIMComputeArguments object at 0x102655a8d0>, <kliff.models.kim.KIMComputeArguments object at 0x102655a940>, <kliff.models.kim.KIMComputeArguments object at 0x102655a9b0>, <kliff.models.kim.KIMComputeArguments object at 0x102655aa20>, <kliff.models.kim.KIMComputeArguments object at 0x102655aa90>, <kliff.models.kim.KIMComputeArguments object at 0x102655ab00>, <kliff.models.kim.KIMComputeArguments object at 0x102655ab70>, <kliff.models.kim.KIMComputeArguments object at 0x102655abe0>, <kliff.models.kim.KIMComputeArguments object at 0x102655ac50>, <kliff.models.kim.KIMComputeArguments object at 0x102655acc0>, <kliff.models.kim.KIMComputeArguments object at 0x102655ad30>, <kliff.models.kim.KIMComputeArguments object at 0x102655ada0>, <kliff.models.kim.KIMComputeArguments object at 0x102655ae10>, <kliff.models.kim.KIMComputeArguments object at 0x102655ae80>, <kliff.models.kim.KIMComputeArguments object at 0x102655aef0>, <kliff.models.kim.KIMComputeArguments object at 0x102655af60>, <kliff.models.kim.KIMComputeArguments object at 0x102655a048>, <kliff.models.kim.KIMComputeArguments object at 0x1026567080>, <kliff.models.kim.KIMComputeArguments object at 0x10265670f0>, <kliff.models.kim.KIMComputeArguments object at 0x1026567160>, <kliff.models.kim.KIMComputeArguments object at 0x10265671d0>, <kliff.models.kim.KIMComputeArguments object at 0x1026567240>, <kliff.models.kim.KIMComputeArguments object at 0x10265672b0>, <kliff.models.kim.KIMComputeArguments object at 0x1026567320>, <kliff.models.kim.KIMComputeArguments object at 0x1026567390>, <kliff.models.kim.KIMComputeArguments object at 0x1026567400>, <kliff.models.kim.KIMComputeArguments object at 0x1026567470>, <kliff.models.kim.KIMComputeArguments object at 0x10265674e0>, <kliff.models.kim.KIMComputeArguments object at 0x1026567550>, <kliff.models.kim.KIMComputeArguments object at 0x10265675c0>, <kliff.models.kim.KIMComputeArguments object at 0x1026567630>, <kliff.models.kim.KIMComputeArguments object at 0x10265676a0>, <kliff.models.kim.KIMComputeArguments object at 0x1026567710>, <kliff.models.kim.KIMComputeArguments object at 0x1026567780>, <kliff.models.kim.KIMComputeArguments object at 0x10265677f0>, <kliff.models.kim.KIMComputeArguments object at 0x1026567860>, <kliff.models.kim.KIMComputeArguments object at 0x10265678d0>, <kliff.models.kim.KIMComputeArguments object at 0x1026567940>, <kliff.models.kim.KIMComputeArguments object at 0x10265679b0>, <kliff.models.kim.KIMComputeArguments object at 0x1026567a20>, <kliff.models.kim.KIMComputeArguments object at 0x1026567a90>, <kliff.models.kim.KIMComputeArguments object at 0x1026567b00>, <kliff.models.kim.KIMComputeArguments object at 0x1026567b70>, <kliff.models.kim.KIMComputeArguments object at 0x1026567be0>, <kliff.models.kim.KIMComputeArguments object at 0x1026567c50>, <kliff.models.kim.KIMComputeArguments object at 0x1026567cc0>, <kliff.models.kim.KIMComputeArguments object at 0x1026567d30>, <kliff.models.kim.KIMComputeArguments object at 0x1026567da0>, <kliff.models.kim.KIMComputeArguments object at 0x1026567e10>, <kliff.models.kim.KIMComputeArguments object at 0x1026567e80>, <kliff.models.kim.KIMComputeArguments object at 0x1026567ef0>, <kliff.models.kim.KIMComputeArguments object at 0x1026567f60>, <kliff.models.kim.KIMComputeArguments object at 0x1026567048>, <kliff.models.kim.KIMComputeArguments object at 0x1026574080>, <kliff.models.kim.KIMComputeArguments object at 0x10265740f0>, <kliff.models.kim.KIMComputeArguments object at 0x1026574160>, <kliff.models.kim.KIMComputeArguments object at 0x10265741d0>, <kliff.models.kim.KIMComputeArguments object at 0x1026574240>, <kliff.models.kim.KIMComputeArguments object at 0x10265742b0>, <kliff.models.kim.KIMComputeArguments object at 0x1026574320>, <kliff.models.kim.KIMComputeArguments object at 0x1026574390>, <kliff.models.kim.KIMComputeArguments object at 0x1026574400>, <kliff.models.kim.KIMComputeArguments object at 0x1026574470>, <kliff.models.kim.KIMComputeArguments object at 0x10265744e0>, <kliff.models.kim.KIMComputeArguments object at 0x1026574550>, <kliff.models.kim.KIMComputeArguments object at 0x10265745c0>, <kliff.models.kim.KIMComputeArguments object at 0x1026574630>, <kliff.models.kim.KIMComputeArguments object at 0x10265746a0>, <kliff.models.kim.KIMComputeArguments object at 0x1026574710>, <kliff.models.kim.KIMComputeArguments object at 0x1026574780>, <kliff.models.kim.KIMComputeArguments object at 0x10265747f0>, <kliff.models.kim.KIMComputeArguments object at 0x1026574860>, <kliff.models.kim.KIMComputeArguments object at 0x10265748d0>, <kliff.models.kim.KIMComputeArguments object at 0x1026574940>, <kliff.models.kim.KIMComputeArguments object at 0x10265749b0>, <kliff.models.kim.KIMComputeArguments object at 0x1026574a20>, <kliff.models.kim.KIMComputeArguments object at 0x1026574a90>, <kliff.models.kim.KIMComputeArguments object at 0x1026574b00>, <kliff.models.kim.KIMComputeArguments object at 0x1026574b70>, <kliff.models.kim.KIMComputeArguments object at 0x1026574be0>, <kliff.models.kim.KIMComputeArguments object at 0x1026574c50>, <kliff.models.kim.KIMComputeArguments object at 0x1026574cc0>, <kliff.models.kim.KIMComputeArguments object at 0x1026574d30>, <kliff.models.kim.KIMComputeArguments object at 0x1026574da0>, <kliff.models.kim.KIMComputeArguments object at 0x1026574e10>, <kliff.models.kim.KIMComputeArguments object at 0x1026574e80>, <kliff.models.kim.KIMComputeArguments object at 0x1026574ef0>, <kliff.models.kim.KIMComputeArguments object at 0x1026574f60>, <kliff.models.kim.KIMComputeArguments object at 0x10265740b8>, <kliff.models.kim.KIMComputeArguments object at 0x1026582080>, <kliff.models.kim.KIMComputeArguments object at 0x10265820f0>, <kliff.models.kim.KIMComputeArguments object at 0x1026582160>, <kliff.models.kim.KIMComputeArguments object at 0x10265821d0>, <kliff.models.kim.KIMComputeArguments object at 0x1026582240>, <kliff.models.kim.KIMComputeArguments object at 0x10265822b0>, <kliff.models.kim.KIMComputeArguments object at 0x1026582320>, <kliff.models.kim.KIMComputeArguments object at 0x1026582390>, <kliff.models.kim.KIMComputeArguments object at 0x1026582400>, <kliff.models.kim.KIMComputeArguments object at 0x1026582470>, <kliff.models.kim.KIMComputeArguments object at 0x10265824e0>, <kliff.models.kim.KIMComputeArguments object at 0x1026582550>, <kliff.models.kim.KIMComputeArguments object at 0x10265825c0>, <kliff.models.kim.KIMComputeArguments object at 0x1026582630>, <kliff.models.kim.KIMComputeArguments object at 0x10265826a0>, <kliff.models.kim.KIMComputeArguments object at 0x1026582710>, <kliff.models.kim.KIMComputeArguments object at 0x1026582780>, <kliff.models.kim.KIMComputeArguments object at 0x10265827f0>, <kliff.models.kim.KIMComputeArguments object at 0x1026582860>, <kliff.models.kim.KIMComputeArguments object at 0x10265828d0>, <kliff.models.kim.KIMComputeArguments object at 0x1026582940>, <kliff.models.kim.KIMComputeArguments object at 0x10265829b0>, <kliff.models.kim.KIMComputeArguments object at 0x1026582a20>, <kliff.models.kim.KIMComputeArguments object at 0x1026582a90>, <kliff.models.kim.KIMComputeArguments object at 0x1026582b00>, <kliff.models.kim.KIMComputeArguments object at 0x1026582b70>, <kliff.models.kim.KIMComputeArguments object at 0x1026582be0>, <kliff.models.kim.KIMComputeArguments object at 0x1026582c50>, <kliff.models.kim.KIMComputeArguments object at 0x1026582cc0>, <kliff.models.kim.KIMComputeArguments object at 0x1026582d30>, <kliff.models.kim.KIMComputeArguments object at 0x1026582da0>, <kliff.models.kim.KIMComputeArguments object at 0x1026582e10>, <kliff.models.kim.KIMComputeArguments object at 0x1026582e80>, <kliff.models.kim.KIMComputeArguments object at 0x1026582ef0>, <kliff.models.kim.KIMComputeArguments object at 0x1026582f60>, <kliff.models.kim.KIMComputeArguments object at 0x10265820b8>, <kliff.models.kim.KIMComputeArguments object at 0x1026591080>, <kliff.models.kim.KIMComputeArguments object at 0x10265910f0>, <kliff.models.kim.KIMComputeArguments object at 0x1026591160>, <kliff.models.kim.KIMComputeArguments object at 0x10265911d0>, <kliff.models.kim.KIMComputeArguments object at 0x1026591240>, <kliff.models.kim.KIMComputeArguments object at 0x10265912b0>, <kliff.models.kim.KIMComputeArguments object at 0x1026591320>, <kliff.models.kim.KIMComputeArguments object at 0x1026591390>, <kliff.models.kim.KIMComputeArguments object at 0x1026591400>, <kliff.models.kim.KIMComputeArguments object at 0x1026591470>, <kliff.models.kim.KIMComputeArguments object at 0x10265914e0>, <kliff.models.kim.KIMComputeArguments object at 0x1026591550>, <kliff.models.kim.KIMComputeArguments object at 0x10265915c0>, <kliff.models.kim.KIMComputeArguments object at 0x1026591630>, <kliff.models.kim.KIMComputeArguments object at 0x10265916a0>, <kliff.models.kim.KIMComputeArguments object at 0x1026591710>, <kliff.models.kim.KIMComputeArguments object at 0x1026591780>, <kliff.models.kim.KIMComputeArguments object at 0x10265917f0>, <kliff.models.kim.KIMComputeArguments object at 0x1026591860>, <kliff.models.kim.KIMComputeArguments object at 0x10265918d0>, <kliff.models.kim.KIMComputeArguments object at 0x1026591940>, <kliff.models.kim.KIMComputeArguments object at 0x10265919b0>, <kliff.models.kim.KIMComputeArguments object at 0x1026591a20>, <kliff.models.kim.KIMComputeArguments object at 0x1026591a90>, <kliff.models.kim.KIMComputeArguments object at 0x1026591b00>, <kliff.models.kim.KIMComputeArguments object at 0x1026591b70>, <kliff.models.kim.KIMComputeArguments object at 0x1026591be0>, <kliff.models.kim.KIMComputeArguments object at 0x1026591c50>, <kliff.models.kim.KIMComputeArguments object at 0x1026591cc0>, <kliff.models.kim.KIMComputeArguments object at 0x1026591d30>, <kliff.models.kim.KIMComputeArguments object at 0x1026591da0>, <kliff.models.kim.KIMComputeArguments object at 0x1026591e10>, <kliff.models.kim.KIMComputeArguments object at 0x1026591e80>, <kliff.models.kim.KIMComputeArguments object at 0x1026591ef0>, <kliff.models.kim.KIMComputeArguments object at 0x1026591f60>, <kliff.models.kim.KIMComputeArguments object at 0x10265910b8>, <kliff.models.kim.KIMComputeArguments object at 0x102659e080>, <kliff.models.kim.KIMComputeArguments object at 0x102659e0f0>, <kliff.models.kim.KIMComputeArguments object at 0x102659e160>, <kliff.models.kim.KIMComputeArguments object at 0x102659e1d0>, <kliff.models.kim.KIMComputeArguments object at 0x102659e240>, <kliff.models.kim.KIMComputeArguments object at 0x102659e2b0>, <kliff.models.kim.KIMComputeArguments object at 0x102659e320>, <kliff.models.kim.KIMComputeArguments object at 0x102659e390>, <kliff.models.kim.KIMComputeArguments object at 0x102659e400>, <kliff.models.kim.KIMComputeArguments object at 0x102659e470>, <kliff.models.kim.KIMComputeArguments object at 0x102659e4e0>, <kliff.models.kim.KIMComputeArguments object at 0x102659e550>, <kliff.models.kim.KIMComputeArguments object at 0x102659e5c0>, <kliff.models.kim.KIMComputeArguments object at 0x102659e630>, <kliff.models.kim.KIMComputeArguments object at 0x102659e6a0>, <kliff.models.kim.KIMComputeArguments object at 0x102659e710>, <kliff.models.kim.KIMComputeArguments object at 0x102659e780>, <kliff.models.kim.KIMComputeArguments object at 0x102659e7f0>, <kliff.models.kim.KIMComputeArguments object at 0x102659e860>, <kliff.models.kim.KIMComputeArguments object at 0x102659e8d0>, <kliff.models.kim.KIMComputeArguments object at 0x102659e940>, <kliff.models.kim.KIMComputeArguments object at 0x102659e9b0>, <kliff.models.kim.KIMComputeArguments object at 0x102659ea20>, <kliff.models.kim.KIMComputeArguments object at 0x102659ea90>, <kliff.models.kim.KIMComputeArguments object at 0x102659eb00>, <kliff.models.kim.KIMComputeArguments object at 0x102659eb70>, <kliff.models.kim.KIMComputeArguments object at 0x102659ebe0>, <kliff.models.kim.KIMComputeArguments object at 0x102659ec50>, <kliff.models.kim.KIMComputeArguments object at 0x102659ecc0>, <kliff.models.kim.KIMComputeArguments object at 0x102659ed30>, <kliff.models.kim.KIMComputeArguments object at 0x102659eda0>, <kliff.models.kim.KIMComputeArguments object at 0x102659ee10>, <kliff.models.kim.KIMComputeArguments object at 0x102659ee80>, <kliff.models.kim.KIMComputeArguments object at 0x102659eef0>, <kliff.models.kim.KIMComputeArguments object at 0x102659ef60>, <kliff.models.kim.KIMComputeArguments object at 0x102659e0b8>, <kliff.models.kim.KIMComputeArguments object at 0x10265ad080>, <kliff.models.kim.KIMComputeArguments object at 0x10265ad0f0>, <kliff.models.kim.KIMComputeArguments object at 0x10265ad160>, <kliff.models.kim.KIMComputeArguments object at 0x10265ad1d0>, <kliff.models.kim.KIMComputeArguments object at 0x10265ad240>, <kliff.models.kim.KIMComputeArguments object at 0x10265ad2b0>, <kliff.models.kim.KIMComputeArguments object at 0x10265ad320>, <kliff.models.kim.KIMComputeArguments object at 0x10265ad390>, <kliff.models.kim.KIMComputeArguments object at 0x10265ad400>, <kliff.models.kim.KIMComputeArguments object at 0x10265ad470>, <kliff.models.kim.KIMComputeArguments object at 0x10265ad4e0>, <kliff.models.kim.KIMComputeArguments object at 0x10265ad550>, <kliff.models.kim.KIMComputeArguments object at 0x10265ad5c0>, <kliff.models.kim.KIMComputeArguments object at 0x10265ad630>, <kliff.models.kim.KIMComputeArguments object at 0x10265ad6a0>, <kliff.models.kim.KIMComputeArguments object at 0x10265ad710>, <kliff.models.kim.KIMComputeArguments object at 0x10265ad780>, <kliff.models.kim.KIMComputeArguments object at 0x10265ad7f0>, <kliff.models.kim.KIMComputeArguments object at 0x10265ad860>, <kliff.models.kim.KIMComputeArguments object at 0x10265ad8d0>, <kliff.models.kim.KIMComputeArguments object at 0x10265ad940>, <kliff.models.kim.KIMComputeArguments object at 0x10265ad9b0>, <kliff.models.kim.KIMComputeArguments object at 0x10265ada20>, <kliff.models.kim.KIMComputeArguments object at 0x10265ada90>, <kliff.models.kim.KIMComputeArguments object at 0x10265adb00>, <kliff.models.kim.KIMComputeArguments object at 0x10265adb70>, <kliff.models.kim.KIMComputeArguments object at 0x10265adbe0>, <kliff.models.kim.KIMComputeArguments object at 0x10265adc50>, <kliff.models.kim.KIMComputeArguments object at 0x10265adcc0>, <kliff.models.kim.KIMComputeArguments object at 0x10265add30>, <kliff.models.kim.KIMComputeArguments object at 0x10265adda0>, <kliff.models.kim.KIMComputeArguments object at 0x10265ade10>, <kliff.models.kim.KIMComputeArguments object at 0x10265ade80>, <kliff.models.kim.KIMComputeArguments object at 0x10265adef0>, <kliff.models.kim.KIMComputeArguments object at 0x10265adf60>, <kliff.models.kim.KIMComputeArguments object at 0x10265ad0b8>, <kliff.models.kim.KIMComputeArguments object at 0x10265bb080>, <kliff.models.kim.KIMComputeArguments object at 0x10265bb0f0>, <kliff.models.kim.KIMComputeArguments object at 0x10265bb160>, <kliff.models.kim.KIMComputeArguments object at 0x10265bb1d0>, <kliff.models.kim.KIMComputeArguments object at 0x10265bb240>, <kliff.models.kim.KIMComputeArguments object at 0x10265bb2b0>, <kliff.models.kim.KIMComputeArguments object at 0x10265bb320>, <kliff.models.kim.KIMComputeArguments object at 0x10265bb390>, <kliff.models.kim.KIMComputeArguments object at 0x10265bb400>, <kliff.models.kim.KIMComputeArguments object at 0x10265bb470>, <kliff.models.kim.KIMComputeArguments object at 0x10265bb4e0>, <kliff.models.kim.KIMComputeArguments object at 0x10265bb550>, <kliff.models.kim.KIMComputeArguments object at 0x10265bb5c0>, <kliff.models.kim.KIMComputeArguments object at 0x10265bb630>, <kliff.models.kim.KIMComputeArguments object at 0x10265bb6a0>, <kliff.models.kim.KIMComputeArguments object at 0x10265bb710>, <kliff.models.kim.KIMComputeArguments object at 0x10265bb780>, <kliff.models.kim.KIMComputeArguments object at 0x10265bb7f0>, <kliff.models.kim.KIMComputeArguments object at 0x10265bb860>, <kliff.models.kim.KIMComputeArguments object at 0x10265bb8d0>, <kliff.models.kim.KIMComputeArguments object at 0x10265bb940>, <kliff.models.kim.KIMComputeArguments object at 0x10265bb9b0>, <kliff.models.kim.KIMComputeArguments object at 0x10265bba20>, <kliff.models.kim.KIMComputeArguments object at 0x10265bba90>, <kliff.models.kim.KIMComputeArguments object at 0x10265bbb00>, <kliff.models.kim.KIMComputeArguments object at 0x10265bbb70>, <kliff.models.kim.KIMComputeArguments object at 0x10265bbbe0>, <kliff.models.kim.KIMComputeArguments object at 0x10265bbc50>, <kliff.models.kim.KIMComputeArguments object at 0x10265bbcc0>, <kliff.models.kim.KIMComputeArguments object at 0x10265bbd30>, <kliff.models.kim.KIMComputeArguments object at 0x10265bbda0>, <kliff.models.kim.KIMComputeArguments object at 0x10265bbe10>, <kliff.models.kim.KIMComputeArguments object at 0x10265bbe80>, <kliff.models.kim.KIMComputeArguments object at 0x10265bbef0>, <kliff.models.kim.KIMComputeArguments object at 0x10265bbf60>, <kliff.models.kim.KIMComputeArguments object at 0x10265bb0b8>, <kliff.models.kim.KIMComputeArguments object at 0x10265c7080>, <kliff.models.kim.KIMComputeArguments object at 0x10265c70f0>, <kliff.models.kim.KIMComputeArguments object at 0x10265c7160>, <kliff.models.kim.KIMComputeArguments object at 0x10265c71d0>, <kliff.models.kim.KIMComputeArguments object at 0x10265c7240>, <kliff.models.kim.KIMComputeArguments object at 0x10265c72b0>, <kliff.models.kim.KIMComputeArguments object at 0x10265c7320>, <kliff.models.kim.KIMComputeArguments object at 0x10265c7390>, <kliff.models.kim.KIMComputeArguments object at 0x10265c7400>, <kliff.models.kim.KIMComputeArguments object at 0x10265c7470>, <kliff.models.kim.KIMComputeArguments object at 0x10265c74e0>, <kliff.models.kim.KIMComputeArguments object at 0x10265c7550>, <kliff.models.kim.KIMComputeArguments object at 0x10265c75c0>, <kliff.models.kim.KIMComputeArguments object at 0x10265c7630>, <kliff.models.kim.KIMComputeArguments object at 0x10265c76a0>, <kliff.models.kim.KIMComputeArguments object at 0x10265c7710>, <kliff.models.kim.KIMComputeArguments object at 0x10265c7780>, <kliff.models.kim.KIMComputeArguments object at 0x10265c77f0>, <kliff.models.kim.KIMComputeArguments object at 0x10265c7860>, <kliff.models.kim.KIMComputeArguments object at 0x10265c78d0>, <kliff.models.kim.KIMComputeArguments object at 0x10265c7940>, <kliff.models.kim.KIMComputeArguments object at 0x10265c79b0>, <kliff.models.kim.KIMComputeArguments object at 0x10265c7a20>, <kliff.models.kim.KIMComputeArguments object at 0x10265c7a90>, <kliff.models.kim.KIMComputeArguments object at 0x10265c7b00>, <kliff.models.kim.KIMComputeArguments object at 0x10265c7b70>, <kliff.models.kim.KIMComputeArguments object at 0x10265c7be0>, <kliff.models.kim.KIMComputeArguments object at 0x10265c7c50>, <kliff.models.kim.KIMComputeArguments object at 0x10265c7cc0>, <kliff.models.kim.KIMComputeArguments object at 0x10265c7d30>, <kliff.models.kim.KIMComputeArguments object at 0x10265c7da0>, <kliff.models.kim.KIMComputeArguments object at 0x10265c7e10>, <kliff.models.kim.KIMComputeArguments object at 0x10265c7e80>, <kliff.models.kim.KIMComputeArguments object at 0x10265c7ef0>, <kliff.models.kim.KIMComputeArguments object at 0x10265c7f60>, <kliff.models.kim.KIMComputeArguments object at 0x10265c7048>, <kliff.models.kim.KIMComputeArguments object at 0x10265d4080>, <kliff.models.kim.KIMComputeArguments object at 0x10265d40f0>, <kliff.models.kim.KIMComputeArguments object at 0x10265d4160>, <kliff.models.kim.KIMComputeArguments object at 0x10265d41d0>, <kliff.models.kim.KIMComputeArguments object at 0x10265d4240>, <kliff.models.kim.KIMComputeArguments object at 0x10265d42b0>, <kliff.models.kim.KIMComputeArguments object at 0x10265d4320>, <kliff.models.kim.KIMComputeArguments object at 0x10265d4390>, <kliff.models.kim.KIMComputeArguments object at 0x10265d4400>, <kliff.models.kim.KIMComputeArguments object at 0x10265d4470>, <kliff.models.kim.KIMComputeArguments object at 0x10265d44e0>, <kliff.models.kim.KIMComputeArguments object at 0x10265d4550>, <kliff.models.kim.KIMComputeArguments object at 0x10265d45c0>, <kliff.models.kim.KIMComputeArguments object at 0x10265d4630>, <kliff.models.kim.KIMComputeArguments object at 0x10265d46a0>, <kliff.models.kim.KIMComputeArguments object at 0x10265d4710>, <kliff.models.kim.KIMComputeArguments object at 0x10265d4780>, <kliff.models.kim.KIMComputeArguments object at 0x10265d47f0>, <kliff.models.kim.KIMComputeArguments object at 0x10265d4860>, <kliff.models.kim.KIMComputeArguments object at 0x10265d48d0>, <kliff.models.kim.KIMComputeArguments object at 0x10265d4940>, <kliff.models.kim.KIMComputeArguments object at 0x10265d49b0>, <kliff.models.kim.KIMComputeArguments object at 0x10265d4a20>, <kliff.models.kim.KIMComputeArguments object at 0x10265d4a90>, <kliff.models.kim.KIMComputeArguments object at 0x10265d4b00>, <kliff.models.kim.KIMComputeArguments object at 0x10265d4b70>, <kliff.models.kim.KIMComputeArguments object at 0x10265d4be0>, <kliff.models.kim.KIMComputeArguments object at 0x10265d4c50>, <kliff.models.kim.KIMComputeArguments object at 0x10265d4cc0>, <kliff.models.kim.KIMComputeArguments object at 0x10265d4d30>, <kliff.models.kim.KIMComputeArguments object at 0x10265d4da0>, <kliff.models.kim.KIMComputeArguments object at 0x10265d4e10>, <kliff.models.kim.KIMComputeArguments object at 0x10265d4e80>, <kliff.models.kim.KIMComputeArguments object at 0x10265d4ef0>, <kliff.models.kim.KIMComputeArguments object at 0x10265d4f60>, <kliff.models.kim.KIMComputeArguments object at 0x10265d4048>, <kliff.models.kim.KIMComputeArguments object at 0x10265e3080>, <kliff.models.kim.KIMComputeArguments object at 0x10265e30f0>, <kliff.models.kim.KIMComputeArguments object at 0x10265e3160>, <kliff.models.kim.KIMComputeArguments object at 0x10265e31d0>, <kliff.models.kim.KIMComputeArguments object at 0x10265e3240>, <kliff.models.kim.KIMComputeArguments object at 0x10265e32b0>, <kliff.models.kim.KIMComputeArguments object at 0x10265e3320>, <kliff.models.kim.KIMComputeArguments object at 0x10265e3390>, <kliff.models.kim.KIMComputeArguments object at 0x10265e3400>, <kliff.models.kim.KIMComputeArguments object at 0x10265e3470>, <kliff.models.kim.KIMComputeArguments object at 0x10265e34e0>, <kliff.models.kim.KIMComputeArguments object at 0x10265e3550>, <kliff.models.kim.KIMComputeArguments object at 0x10265e35c0>, <kliff.models.kim.KIMComputeArguments object at 0x10265e3630>, <kliff.models.kim.KIMComputeArguments object at 0x10265e36a0>, <kliff.models.kim.KIMComputeArguments object at 0x10265e3710>, <kliff.models.kim.KIMComputeArguments object at 0x10265e3780>, <kliff.models.kim.KIMComputeArguments object at 0x10265e37f0>, <kliff.models.kim.KIMComputeArguments object at 0x10265e3860>, <kliff.models.kim.KIMComputeArguments object at 0x10265e38d0>, <kliff.models.kim.KIMComputeArguments object at 0x10265e3940>, <kliff.models.kim.KIMComputeArguments object at 0x10265e39b0>, <kliff.models.kim.KIMComputeArguments object at 0x10265e3a20>, <kliff.models.kim.KIMComputeArguments object at 0x10265e3a90>, <kliff.models.kim.KIMComputeArguments object at 0x10265e3b00>, <kliff.models.kim.KIMComputeArguments object at 0x10265e3b70>, <kliff.models.kim.KIMComputeArguments object at 0x10265e3be0>, <kliff.models.kim.KIMComputeArguments object at 0x10265e3c50>, <kliff.models.kim.KIMComputeArguments object at 0x10265e3cc0>, <kliff.models.kim.KIMComputeArguments object at 0x10265e3d30>, <kliff.models.kim.KIMComputeArguments object at 0x10265e3da0>, <kliff.models.kim.KIMComputeArguments object at 0x10265e3e10>, <kliff.models.kim.KIMComputeArguments object at 0x10265e3e80>, <kliff.models.kim.KIMComputeArguments object at 0x10265e3ef0>, <kliff.models.kim.KIMComputeArguments object at 0x10265e3f60>, <kliff.models.kim.KIMComputeArguments object at 0x10265e3048>, <kliff.models.kim.KIMComputeArguments object at 0x10265f1080>, <kliff.models.kim.KIMComputeArguments object at 0x10265f10f0>, <kliff.models.kim.KIMComputeArguments object at 0x10265f1160>, <kliff.models.kim.KIMComputeArguments object at 0x10265f11d0>, <kliff.models.kim.KIMComputeArguments object at 0x10265f1240>, <kliff.models.kim.KIMComputeArguments object at 0x10265f12b0>, <kliff.models.kim.KIMComputeArguments object at 0x10265f1320>, <kliff.models.kim.KIMComputeArguments object at 0x10265f1390>, <kliff.models.kim.KIMComputeArguments object at 0x10265f1400>, <kliff.models.kim.KIMComputeArguments object at 0x10265f1470>, <kliff.models.kim.KIMComputeArguments object at 0x10265f14e0>, <kliff.models.kim.KIMComputeArguments object at 0x10265f1550>, <kliff.models.kim.KIMComputeArguments object at 0x10265f15c0>, <kliff.models.kim.KIMComputeArguments object at 0x10265f1630>, <kliff.models.kim.KIMComputeArguments object at 0x10265f16a0>, <kliff.models.kim.KIMComputeArguments object at 0x10265f1710>, <kliff.models.kim.KIMComputeArguments object at 0x10265f1780>, <kliff.models.kim.KIMComputeArguments object at 0x10265f17f0>, <kliff.models.kim.KIMComputeArguments object at 0x10265f1860>, <kliff.models.kim.KIMComputeArguments object at 0x10265f18d0>, <kliff.models.kim.KIMComputeArguments object at 0x10265f1940>, <kliff.models.kim.KIMComputeArguments object at 0x10265f19b0>, <kliff.models.kim.KIMComputeArguments object at 0x10265f1a20>, <kliff.models.kim.KIMComputeArguments object at 0x10265f1a90>, <kliff.models.kim.KIMComputeArguments object at 0x10265f1b00>, <kliff.models.kim.KIMComputeArguments object at 0x10265f1b70>, <kliff.models.kim.KIMComputeArguments object at 0x10265f1be0>, <kliff.models.kim.KIMComputeArguments object at 0x10265f1c50>, <kliff.models.kim.KIMComputeArguments object at 0x10265f1cc0>, <kliff.models.kim.KIMComputeArguments object at 0x10265f1d30>, <kliff.models.kim.KIMComputeArguments object at 0x10265f1da0>, <kliff.models.kim.KIMComputeArguments object at 0x10265f1e10>, <kliff.models.kim.KIMComputeArguments object at 0x10265f1e80>, <kliff.models.kim.KIMComputeArguments object at 0x10265f1ef0>, <kliff.models.kim.KIMComputeArguments object at 0x10265f1f60>, <kliff.models.kim.KIMComputeArguments object at 0x10265f1048>, <kliff.models.kim.KIMComputeArguments object at 0x10265fd080>, <kliff.models.kim.KIMComputeArguments object at 0x10265fd0f0>, <kliff.models.kim.KIMComputeArguments object at 0x10265fd160>, <kliff.models.kim.KIMComputeArguments object at 0x10265fd1d0>, <kliff.models.kim.KIMComputeArguments object at 0x10265fd240>, <kliff.models.kim.KIMComputeArguments object at 0x10265fd2b0>, <kliff.models.kim.KIMComputeArguments object at 0x10265fd320>, <kliff.models.kim.KIMComputeArguments object at 0x10265fd390>, <kliff.models.kim.KIMComputeArguments object at 0x10265fd400>, <kliff.models.kim.KIMComputeArguments object at 0x10265fd470>, <kliff.models.kim.KIMComputeArguments object at 0x10265fd4e0>, <kliff.models.kim.KIMComputeArguments object at 0x10265fd550>, <kliff.models.kim.KIMComputeArguments object at 0x10265fd5c0>, <kliff.models.kim.KIMComputeArguments object at 0x10265fd630>, <kliff.models.kim.KIMComputeArguments object at 0x10265fd6a0>, <kliff.models.kim.KIMComputeArguments object at 0x10265fd710>, <kliff.models.kim.KIMComputeArguments object at 0x10265fd780>, <kliff.models.kim.KIMComputeArguments object at 0x10265fd7f0>, <kliff.models.kim.KIMComputeArguments object at 0x10265fd860>, <kliff.models.kim.KIMComputeArguments object at 0x10265fd8d0>, <kliff.models.kim.KIMComputeArguments object at 0x10265fd940>, <kliff.models.kim.KIMComputeArguments object at 0x10265fd9b0>, <kliff.models.kim.KIMComputeArguments object at 0x10265fda20>, <kliff.models.kim.KIMComputeArguments object at 0x10265fda90>, <kliff.models.kim.KIMComputeArguments object at 0x10265fdb00>, <kliff.models.kim.KIMComputeArguments object at 0x10265fdb70>, <kliff.models.kim.KIMComputeArguments object at 0x10265fdbe0>, <kliff.models.kim.KIMComputeArguments object at 0x10265fdc50>, <kliff.models.kim.KIMComputeArguments object at 0x10265fdcc0>, <kliff.models.kim.KIMComputeArguments object at 0x10265fdd30>, <kliff.models.kim.KIMComputeArguments object at 0x10265fdda0>, <kliff.models.kim.KIMComputeArguments object at 0x10265fde10>, <kliff.models.kim.KIMComputeArguments object at 0x10265fde80>, <kliff.models.kim.KIMComputeArguments object at 0x10265fdef0>, <kliff.models.kim.KIMComputeArguments object at 0x10265fdf60>, <kliff.models.kim.KIMComputeArguments object at 0x10265fd048>, <kliff.models.kim.KIMComputeArguments object at 0x102660c080>, <kliff.models.kim.KIMComputeArguments object at 0x102660c0f0>, <kliff.models.kim.KIMComputeArguments object at 0x102660c160>, <kliff.models.kim.KIMComputeArguments object at 0x102660c1d0>, <kliff.models.kim.KIMComputeArguments object at 0x102660c240>, <kliff.models.kim.KIMComputeArguments object at 0x102660c2b0>, <kliff.models.kim.KIMComputeArguments object at 0x102660c320>, <kliff.models.kim.KIMComputeArguments object at 0x102660c390>, <kliff.models.kim.KIMComputeArguments object at 0x102660c400>, <kliff.models.kim.KIMComputeArguments object at 0x102660c470>, <kliff.models.kim.KIMComputeArguments object at 0x102660c4e0>, <kliff.models.kim.KIMComputeArguments object at 0x102660c550>, <kliff.models.kim.KIMComputeArguments object at 0x102660c5c0>, <kliff.models.kim.KIMComputeArguments object at 0x102660c630>, <kliff.models.kim.KIMComputeArguments object at 0x102660c6a0>, <kliff.models.kim.KIMComputeArguments object at 0x102660c710>, <kliff.models.kim.KIMComputeArguments object at 0x102660c780>, <kliff.models.kim.KIMComputeArguments object at 0x102660c7f0>, <kliff.models.kim.KIMComputeArguments object at 0x102660c860>, <kliff.models.kim.KIMComputeArguments object at 0x102660c8d0>, <kliff.models.kim.KIMComputeArguments object at 0x102660c940>, <kliff.models.kim.KIMComputeArguments object at 0x102660c9b0>, <kliff.models.kim.KIMComputeArguments object at 0x102660ca20>, <kliff.models.kim.KIMComputeArguments object at 0x102660ca90>, <kliff.models.kim.KIMComputeArguments object at 0x102660cb00>, <kliff.models.kim.KIMComputeArguments object at 0x102660cb70>, <kliff.models.kim.KIMComputeArguments object at 0x102660cbe0>, <kliff.models.kim.KIMComputeArguments object at 0x102660cc50>, <kliff.models.kim.KIMComputeArguments object at 0x102660ccc0>, <kliff.models.kim.KIMComputeArguments object at 0x102660cd30>, <kliff.models.kim.KIMComputeArguments object at 0x102660cda0>, <kliff.models.kim.KIMComputeArguments object at 0x102660ce10>, <kliff.models.kim.KIMComputeArguments object at 0x102660ce80>, <kliff.models.kim.KIMComputeArguments object at 0x102660cef0>, <kliff.models.kim.KIMComputeArguments object at 0x102660cf60>, <kliff.models.kim.KIMComputeArguments object at 0x102660c048>, <kliff.models.kim.KIMComputeArguments object at 0x102661a080>, <kliff.models.kim.KIMComputeArguments object at 0x102661a0f0>, <kliff.models.kim.KIMComputeArguments object at 0x102661a160>, <kliff.models.kim.KIMComputeArguments object at 0x102661a1d0>, <kliff.models.kim.KIMComputeArguments object at 0x102661a240>, <kliff.models.kim.KIMComputeArguments object at 0x102661a2b0>, <kliff.models.kim.KIMComputeArguments object at 0x102661a320>, <kliff.models.kim.KIMComputeArguments object at 0x102661a390>, <kliff.models.kim.KIMComputeArguments object at 0x102661a400>, <kliff.models.kim.KIMComputeArguments object at 0x102661a470>, <kliff.models.kim.KIMComputeArguments object at 0x102661a4e0>, <kliff.models.kim.KIMComputeArguments object at 0x102661a550>, <kliff.models.kim.KIMComputeArguments object at 0x102661a5c0>, <kliff.models.kim.KIMComputeArguments object at 0x102661a630>, <kliff.models.kim.KIMComputeArguments object at 0x102661a6a0>, <kliff.models.kim.KIMComputeArguments object at 0x102661a710>, <kliff.models.kim.KIMComputeArguments object at 0x102661a780>, <kliff.models.kim.KIMComputeArguments object at 0x102661a7f0>, <kliff.models.kim.KIMComputeArguments object at 0x102661a860>, <kliff.models.kim.KIMComputeArguments object at 0x102661a8d0>, <kliff.models.kim.KIMComputeArguments object at 0x102661a940>, <kliff.models.kim.KIMComputeArguments object at 0x102661a9b0>, <kliff.models.kim.KIMComputeArguments object at 0x102661aa20>, <kliff.models.kim.KIMComputeArguments object at 0x102661aa90>, <kliff.models.kim.KIMComputeArguments object at 0x102661ab00>, <kliff.models.kim.KIMComputeArguments object at 0x102661ab70>, <kliff.models.kim.KIMComputeArguments object at 0x102661abe0>, <kliff.models.kim.KIMComputeArguments object at 0x102661ac50>, <kliff.models.kim.KIMComputeArguments object at 0x102661acc0>, <kliff.models.kim.KIMComputeArguments object at 0x102661ad30>, <kliff.models.kim.KIMComputeArguments object at 0x102661ada0>, <kliff.models.kim.KIMComputeArguments object at 0x102661ae10>, <kliff.models.kim.KIMComputeArguments object at 0x102661ae80>, <kliff.models.kim.KIMComputeArguments object at 0x102661aef0>, <kliff.models.kim.KIMComputeArguments object at 0x102661af60>, <kliff.models.kim.KIMComputeArguments object at 0x102661a048>, <kliff.models.kim.KIMComputeArguments object at 0x1026627080>, <kliff.models.kim.KIMComputeArguments object at 0x10266270f0>, <kliff.models.kim.KIMComputeArguments object at 0x1026627160>, <kliff.models.kim.KIMComputeArguments object at 0x10266271d0>, <kliff.models.kim.KIMComputeArguments object at 0x1026627240>, <kliff.models.kim.KIMComputeArguments object at 0x10266272b0>, <kliff.models.kim.KIMComputeArguments object at 0x1026627320>, <kliff.models.kim.KIMComputeArguments object at 0x1026627390>, <kliff.models.kim.KIMComputeArguments object at 0x1026627400>, <kliff.models.kim.KIMComputeArguments object at 0x1026627470>, <kliff.models.kim.KIMComputeArguments object at 0x10266274e0>, <kliff.models.kim.KIMComputeArguments object at 0x1026627550>, <kliff.models.kim.KIMComputeArguments object at 0x10266275c0>, <kliff.models.kim.KIMComputeArguments object at 0x1026627630>, <kliff.models.kim.KIMComputeArguments object at 0x10266276a0>, <kliff.models.kim.KIMComputeArguments object at 0x1026627710>, <kliff.models.kim.KIMComputeArguments object at 0x1026627780>, <kliff.models.kim.KIMComputeArguments object at 0x10266277f0>, <kliff.models.kim.KIMComputeArguments object at 0x1026627860>, <kliff.models.kim.KIMComputeArguments object at 0x10266278d0>, <kliff.models.kim.KIMComputeArguments object at 0x1026627940>, <kliff.models.kim.KIMComputeArguments object at 0x10266279b0>, <kliff.models.kim.KIMComputeArguments object at 0x1026627a20>, <kliff.models.kim.KIMComputeArguments object at 0x1026627a90>, <kliff.models.kim.KIMComputeArguments object at 0x1026627b00>, <kliff.models.kim.KIMComputeArguments object at 0x1026627b70>, <kliff.models.kim.KIMComputeArguments object at 0x1026627be0>, <kliff.models.kim.KIMComputeArguments object at 0x1026627c50>, <kliff.models.kim.KIMComputeArguments object at 0x1026627cc0>, <kliff.models.kim.KIMComputeArguments object at 0x1026627d30>, <kliff.models.kim.KIMComputeArguments object at 0x1026627da0>, <kliff.models.kim.KIMComputeArguments object at 0x1026627e10>, <kliff.models.kim.KIMComputeArguments object at 0x1026627e80>, <kliff.models.kim.KIMComputeArguments object at 0x1026627ef0>, <kliff.models.kim.KIMComputeArguments object at 0x1026627f60>, <kliff.models.kim.KIMComputeArguments object at 0x1026627048>, <kliff.models.kim.KIMComputeArguments object at 0x1026634080>, <kliff.models.kim.KIMComputeArguments object at 0x10266340f0>, <kliff.models.kim.KIMComputeArguments object at 0x1026634160>, <kliff.models.kim.KIMComputeArguments object at 0x10266341d0>, <kliff.models.kim.KIMComputeArguments object at 0x1026634240>, <kliff.models.kim.KIMComputeArguments object at 0x10266342b0>, <kliff.models.kim.KIMComputeArguments object at 0x1026634320>, <kliff.models.kim.KIMComputeArguments object at 0x1026634390>, <kliff.models.kim.KIMComputeArguments object at 0x1026634400>, <kliff.models.kim.KIMComputeArguments object at 0x1026634470>, <kliff.models.kim.KIMComputeArguments object at 0x10266344e0>, <kliff.models.kim.KIMComputeArguments object at 0x1026634550>, <kliff.models.kim.KIMComputeArguments object at 0x10266345c0>, <kliff.models.kim.KIMComputeArguments object at 0x1026634630>, <kliff.models.kim.KIMComputeArguments object at 0x10266346a0>, <kliff.models.kim.KIMComputeArguments object at 0x1026634710>, <kliff.models.kim.KIMComputeArguments object at 0x1026634780>, <kliff.models.kim.KIMComputeArguments object at 0x10266347f0>, <kliff.models.kim.KIMComputeArguments object at 0x1026634860>, <kliff.models.kim.KIMComputeArguments object at 0x10266348d0>, <kliff.models.kim.KIMComputeArguments object at 0x1026634940>, <kliff.models.kim.KIMComputeArguments object at 0x10266349b0>, <kliff.models.kim.KIMComputeArguments object at 0x1026634a20>, <kliff.models.kim.KIMComputeArguments object at 0x1026634a90>, <kliff.models.kim.KIMComputeArguments object at 0x1026634b00>, <kliff.models.kim.KIMComputeArguments object at 0x1026634b70>, <kliff.models.kim.KIMComputeArguments object at 0x1026634be0>, <kliff.models.kim.KIMComputeArguments object at 0x1026634c50>, <kliff.models.kim.KIMComputeArguments object at 0x1026634cc0>, <kliff.models.kim.KIMComputeArguments object at 0x1026634d30>, <kliff.models.kim.KIMComputeArguments object at 0x1026634da0>, <kliff.models.kim.KIMComputeArguments object at 0x1026634e10>, <kliff.models.kim.KIMComputeArguments object at 0x1026634e80>, <kliff.models.kim.KIMComputeArguments object at 0x1026634ef0>, <kliff.models.kim.KIMComputeArguments object at 0x1026634f60>, <kliff.models.kim.KIMComputeArguments object at 0x1026634048>, <kliff.models.kim.KIMComputeArguments object at 0x1026642080>, <kliff.models.kim.KIMComputeArguments object at 0x10266420f0>, <kliff.models.kim.KIMComputeArguments object at 0x1026642160>, <kliff.models.kim.KIMComputeArguments object at 0x10266421d0>, <kliff.models.kim.KIMComputeArguments object at 0x1026642240>, <kliff.models.kim.KIMComputeArguments object at 0x10266422b0>, <kliff.models.kim.KIMComputeArguments object at 0x1026642320>, <kliff.models.kim.KIMComputeArguments object at 0x1026642390>, <kliff.models.kim.KIMComputeArguments object at 0x1026642400>, <kliff.models.kim.KIMComputeArguments object at 0x1026642470>, <kliff.models.kim.KIMComputeArguments object at 0x10266424e0>, <kliff.models.kim.KIMComputeArguments object at 0x1026642550>, <kliff.models.kim.KIMComputeArguments object at 0x10266425c0>, <kliff.models.kim.KIMComputeArguments object at 0x1026642630>, <kliff.models.kim.KIMComputeArguments object at 0x10266426a0>, <kliff.models.kim.KIMComputeArguments object at 0x1026642710>, <kliff.models.kim.KIMComputeArguments object at 0x1026642780>, <kliff.models.kim.KIMComputeArguments object at 0x10266427f0>, <kliff.models.kim.KIMComputeArguments object at 0x1026642860>, <kliff.models.kim.KIMComputeArguments object at 0x10266428d0>, <kliff.models.kim.KIMComputeArguments object at 0x1026642940>, <kliff.models.kim.KIMComputeArguments object at 0x10266429b0>, <kliff.models.kim.KIMComputeArguments object at 0x1026642a20>, <kliff.models.kim.KIMComputeArguments object at 0x1026642a90>, <kliff.models.kim.KIMComputeArguments object at 0x1026642b00>, <kliff.models.kim.KIMComputeArguments object at 0x1026642b70>, <kliff.models.kim.KIMComputeArguments object at 0x1026642be0>, <kliff.models.kim.KIMComputeArguments object at 0x1026642c50>, <kliff.models.kim.KIMComputeArguments object at 0x1026642cc0>, <kliff.models.kim.KIMComputeArguments object at 0x1026642d30>, <kliff.models.kim.KIMComputeArguments object at 0x1026642da0>, <kliff.models.kim.KIMComputeArguments object at 0x1026642e10>, <kliff.models.kim.KIMComputeArguments object at 0x1026642e80>, <kliff.models.kim.KIMComputeArguments object at 0x1026642ef0>, <kliff.models.kim.KIMComputeArguments object at 0x1026642f60>, <kliff.models.kim.KIMComputeArguments object at 0x1026642048>, <kliff.models.kim.KIMComputeArguments object at 0x1026650080>, <kliff.models.kim.KIMComputeArguments object at 0x10266500f0>, <kliff.models.kim.KIMComputeArguments object at 0x1026650160>, <kliff.models.kim.KIMComputeArguments object at 0x10266501d0>, <kliff.models.kim.KIMComputeArguments object at 0x1026650240>, <kliff.models.kim.KIMComputeArguments object at 0x10266502b0>, <kliff.models.kim.KIMComputeArguments object at 0x1026650320>, <kliff.models.kim.KIMComputeArguments object at 0x1026650390>, <kliff.models.kim.KIMComputeArguments object at 0x1026650400>, <kliff.models.kim.KIMComputeArguments object at 0x1026650470>, <kliff.models.kim.KIMComputeArguments object at 0x10266504e0>, <kliff.models.kim.KIMComputeArguments object at 0x1026650550>, <kliff.models.kim.KIMComputeArguments object at 0x10266505c0>, <kliff.models.kim.KIMComputeArguments object at 0x1026650630>, <kliff.models.kim.KIMComputeArguments object at 0x10266506a0>, <kliff.models.kim.KIMComputeArguments object at 0x1026650710>, <kliff.models.kim.KIMComputeArguments object at 0x1026650780>, <kliff.models.kim.KIMComputeArguments object at 0x10266507f0>, <kliff.models.kim.KIMComputeArguments object at 0x1026650860>, <kliff.models.kim.KIMComputeArguments object at 0x10266508d0>, <kliff.models.kim.KIMComputeArguments object at 0x1026650940>, <kliff.models.kim.KIMComputeArguments object at 0x10266509b0>, <kliff.models.kim.KIMComputeArguments object at 0x1026650a20>, <kliff.models.kim.KIMComputeArguments object at 0x1026650a90>, <kliff.models.kim.KIMComputeArguments object at 0x1026650b00>, <kliff.models.kim.KIMComputeArguments object at 0x1026650b70>, <kliff.models.kim.KIMComputeArguments object at 0x1026650be0>, <kliff.models.kim.KIMComputeArguments object at 0x1026650c50>, <kliff.models.kim.KIMComputeArguments object at 0x1026650cc0>, <kliff.models.kim.KIMComputeArguments object at 0x1026650d30>, <kliff.models.kim.KIMComputeArguments object at 0x1026650da0>, <kliff.models.kim.KIMComputeArguments object at 0x1026650e10>, <kliff.models.kim.KIMComputeArguments object at 0x1026650e80>, <kliff.models.kim.KIMComputeArguments object at 0x1026650ef0>, <kliff.models.kim.KIMComputeArguments object at 0x1026650f60>, <kliff.models.kim.KIMComputeArguments object at 0x1026650048>, <kliff.models.kim.KIMComputeArguments object at 0x102665d080>, <kliff.models.kim.KIMComputeArguments object at 0x102665d0f0>, <kliff.models.kim.KIMComputeArguments object at 0x102665d160>, <kliff.models.kim.KIMComputeArguments object at 0x102665d1d0>, <kliff.models.kim.KIMComputeArguments object at 0x102665d240>, <kliff.models.kim.KIMComputeArguments object at 0x102665d2b0>, <kliff.models.kim.KIMComputeArguments object at 0x102665d320>, <kliff.models.kim.KIMComputeArguments object at 0x102665d390>, <kliff.models.kim.KIMComputeArguments object at 0x102665d400>, <kliff.models.kim.KIMComputeArguments object at 0x102665d470>, <kliff.models.kim.KIMComputeArguments object at 0x102665d4e0>, <kliff.models.kim.KIMComputeArguments object at 0x102665d550>, <kliff.models.kim.KIMComputeArguments object at 0x102665d5c0>, <kliff.models.kim.KIMComputeArguments object at 0x102665d630>, <kliff.models.kim.KIMComputeArguments object at 0x102665d6a0>, <kliff.models.kim.KIMComputeArguments object at 0x102665d710>, <kliff.models.kim.KIMComputeArguments object at 0x102665d780>, <kliff.models.kim.KIMComputeArguments object at 0x102665d7f0>, <kliff.models.kim.KIMComputeArguments object at 0x102665d860>, <kliff.models.kim.KIMComputeArguments object at 0x102665d8d0>, <kliff.models.kim.KIMComputeArguments object at 0x102665d940>, <kliff.models.kim.KIMComputeArguments object at 0x102665d9b0>, <kliff.models.kim.KIMComputeArguments object at 0x102665da20>, <kliff.models.kim.KIMComputeArguments object at 0x102665da90>, <kliff.models.kim.KIMComputeArguments object at 0x102665db00>, <kliff.models.kim.KIMComputeArguments object at 0x102665db70>, <kliff.models.kim.KIMComputeArguments object at 0x102665dbe0>, <kliff.models.kim.KIMComputeArguments object at 0x102665dc50>, <kliff.models.kim.KIMComputeArguments object at 0x102665dcc0>, <kliff.models.kim.KIMComputeArguments object at 0x102665dd30>, <kliff.models.kim.KIMComputeArguments object at 0x102665dda0>, <kliff.models.kim.KIMComputeArguments object at 0x102665de10>, <kliff.models.kim.KIMComputeArguments object at 0x102665de80>, <kliff.models.kim.KIMComputeArguments object at 0x102665def0>, <kliff.models.kim.KIMComputeArguments object at 0x102665df60>, <kliff.models.kim.KIMComputeArguments object at 0x102665d048>, <kliff.models.kim.KIMComputeArguments object at 0x102666c080>, <kliff.models.kim.KIMComputeArguments object at 0x102666c0f0>, <kliff.models.kim.KIMComputeArguments object at 0x102666c160>, <kliff.models.kim.KIMComputeArguments object at 0x102666c1d0>, <kliff.models.kim.KIMComputeArguments object at 0x102666c240>, <kliff.models.kim.KIMComputeArguments object at 0x102666c2b0>, <kliff.models.kim.KIMComputeArguments object at 0x102666c320>, <kliff.models.kim.KIMComputeArguments object at 0x102666c390>, <kliff.models.kim.KIMComputeArguments object at 0x102666c400>, <kliff.models.kim.KIMComputeArguments object at 0x102666c470>, <kliff.models.kim.KIMComputeArguments object at 0x102666c4e0>, <kliff.models.kim.KIMComputeArguments object at 0x102666c550>, <kliff.models.kim.KIMComputeArguments object at 0x102666c5c0>, <kliff.models.kim.KIMComputeArguments object at 0x102666c630>, <kliff.models.kim.KIMComputeArguments object at 0x102666c6a0>, <kliff.models.kim.KIMComputeArguments object at 0x102666c710>, <kliff.models.kim.KIMComputeArguments object at 0x102666c780>, <kliff.models.kim.KIMComputeArguments object at 0x102666c7f0>, <kliff.models.kim.KIMComputeArguments object at 0x102666c860>, <kliff.models.kim.KIMComputeArguments object at 0x102666c8d0>, <kliff.models.kim.KIMComputeArguments object at 0x102666c940>, <kliff.models.kim.KIMComputeArguments object at 0x102666c9b0>, <kliff.models.kim.KIMComputeArguments object at 0x102666ca20>, <kliff.models.kim.KIMComputeArguments object at 0x102666ca90>, <kliff.models.kim.KIMComputeArguments object at 0x102666cb00>, <kliff.models.kim.KIMComputeArguments object at 0x102666cb70>, <kliff.models.kim.KIMComputeArguments object at 0x102666cbe0>, <kliff.models.kim.KIMComputeArguments object at 0x102666cc50>, <kliff.models.kim.KIMComputeArguments object at 0x102666ccc0>, <kliff.models.kim.KIMComputeArguments object at 0x102666cd30>, <kliff.models.kim.KIMComputeArguments object at 0x102666cda0>, <kliff.models.kim.KIMComputeArguments object at 0x102666ce10>, <kliff.models.kim.KIMComputeArguments object at 0x102666ce80>, <kliff.models.kim.KIMComputeArguments object at 0x102666cef0>, <kliff.models.kim.KIMComputeArguments object at 0x102666cf60>, <kliff.models.kim.KIMComputeArguments object at 0x102666c048>, <kliff.models.kim.KIMComputeArguments object at 0x102667a048>, <kliff.models.kim.KIMComputeArguments object at 0x102667a0f0>, <kliff.models.kim.KIMComputeArguments object at 0x102667a160>, <kliff.models.kim.KIMComputeArguments object at 0x102667a1d0>, <kliff.models.kim.KIMComputeArguments object at 0x102667a240>, <kliff.models.kim.KIMComputeArguments object at 0x102667a2b0>, <kliff.models.kim.KIMComputeArguments object at 0x102667a320>]



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
    residual_data = {'energy_weight': 1.0, 'forces_weight': 0.1}
    loss = Loss(calc, residual_data=residual_data, nprocs=2)
    loss.minimize(method='L-BFGS-B', options={'disp': True, 'maxiter': steps})






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
    model.save('kliff_model.pkl')
    model.write_kim_model()
    model.load('kliff_model.pkl')






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

   **Total running time of the script:** ( 1 minutes  46.919 seconds)


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
