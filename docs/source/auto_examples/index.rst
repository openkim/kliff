:orphan:

=========
Tutorials
=========

To learn how to use KLIFF, begin with the tutorials.

:ref:`tut_kim_sw`: a good entry point to see the basics of training a physics-motivated
potential.

:ref:`tut_params_transform`: it is similar to :ref:`tut_kim_sw`, except that some
parameters are transformed to the log space for optimization.

:ref:`tut_nn`: walks through the steps to train a machine-learning neural network
potential.

:ref:`tut_nn_multi_spec`: similar to :ref:`tut_nn`, but train for a system of multiple
species.

:ref:`tut_lj`: similar to :ref:`tut_kim_sw` (where a KIM model is used), here the
Lennard-Jones model built in KLIFF is used.

More examples can be found at `<https://github.com/openkim/kliff/tree/master/examples>`_.



.. raw:: html

    <div class="sphx-glr-thumbnails">


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we train a linear regression model on the descriptors obtained using the symm...">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_example_linear_regression_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_example_linear_regression.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Train a linear regression potential</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we train a Lennard-Jones potential that is build in KLIFF (i.e. not models ar...">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_example_lennard_jones_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_example_lennard_jones.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Train a Lennard-Jones potential</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we train a Stillinger-Weber (SW) potential for silicon that is archived on Op...">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_example_kim_SW_Si_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_example_kim_SW_Si.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Train a Stillinger-Weber potential</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Parameters in the empirical interatomic potential are often restricted by some physical constra...">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_example_parameter_transform_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_example_parameter_transform.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Parameter transformation for the Stillinger-Weber potential</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we train a neural network (NN) potential for silicon.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_example_nn_Si_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_example_nn_Si.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Train a neural network potential</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we train a neural network (NN) potential for a system containing two species:...">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_example_nn_SiC_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_example_nn_SiC.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Train a neural network potential for SiC</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example, we demonstrate how to perform uncertainty quantification (UQ) using bootstrap ...">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_example_uq_bootstrap_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_example_uq_bootstrap.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Bootstrapping</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example, we demonstrate how to perform uncertainty quantification (UQ) using parallel t...">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_example_uq_mcmc_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_example_uq_mcmc.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">MCMC sampling</div>
    </div>


.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/example_linear_regression
   /auto_examples/example_lennard_jones
   /auto_examples/example_kim_SW_Si
   /auto_examples/example_parameter_transform
   /auto_examples/example_nn_Si
   /auto_examples/example_nn_SiC
   /auto_examples/example_uq_bootstrap
   /auto_examples/example_uq_mcmc


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: auto_examples_python.zip </auto_examples/auto_examples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: auto_examples_jupyter.zip </auto_examples/auto_examples_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
