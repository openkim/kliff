Transforms
==========

:py:mod:`~kliff.transforms` is a collection of commonly used
functions, used to change or transform, the datasets/parameters.
Transforms module is divided as, - Coordinate transforms: Mapping the
coordinates of a configuration to invariant representations, which can
be used in ML models. - Descriptors - Radial Graphs - Properties:
Transform properties associated with the configurations. Often it takes
input as a complete dataset, and aggregate statistics of property of
entire dataset before transformations like normalization -
Parameters ( only available for the physic based models for now )
Transform the parameter space for enabling better
sampling/training.

Configuration Transforms
------------------------

Descriptor
~~~~~~~~~~

The ``Descriptors`` module bridges the
`libdescriptor <https://github.com/openkim/libdescriptor>`__ library
with KLIFF’s data structures (i.e., ``Configuration``,
``NeighborList``). It provides:

-  ``show_available_descriptors()``: A helper function that prints all
   descriptor names.
-  ``Descriptor``:

   -  Takes a ``cutoff``, ``species``, ``descriptor name``, and
      ``hyperparameters``.
   -  Computes descriptors (``forward``) and their derivatives w.r.t.
      atomic coordinates (``backward``).
   -  Can store results directly in the ``Configuration`` object’s
      fingerprint.

-  ``default_hyperparams``: Module containing collection of sane
   defaults for different descriptors

.. tip::

   This module relies on the optional dependency ``libdescriptor``.

   .. code-block:: bash

      conda install -c ipcamit libdescriptor

.. code:: python

    from kliff.transforms.configuration_transforms.descriptors import show_available_descriptors
    show_available_descriptors()


.. parsed-literal::

    --------------------------------------------------------------------------------
    Descriptors below are currently available, select them by `descriptor: str` attribute:
    --------------------------------------------------------------------------------
    SymmetryFunctions
    Bispectrum
    SOAP


.. code:: python

    from kliff.transforms.configuration_transforms.descriptors import Descriptor
    from kliff.transforms.configuration_transforms.default_hyperparams import symmetry_functions_set30
    
    desc = Descriptor(cutoff=3.77, 
                      species=["Si"], 
                      descriptor="SymmetryFunctions", 
                      hyperparameters=symmetry_functions_set30())

This ``Descriptor`` module is designed to work as a thin wrapper over
``libdescriptor`` library, and provides ``forward`` and ``backward``
function for computing the descriptors, and their vector-Jacobian
products for gradient. Given below is a brief overview of how typical ML
potential evaluates forces, and how it is achieved in KLIFF.

Theory of ML with descriptors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Descriptors (:math:`\zeta`) are used in machine learning to transform
raw input features (:math:`\mathbf{\zeta}`) into a higher-dimensional
representation that captures more complex patterns and relationships.
This transformation is particularly useful in various applications,
including molecular dynamics, material science, and geometric deep
learning.

Forward Pass
^^^^^^^^^^^^

1. Descriptor Calculation

   -  The input features :math:`x` (e.g., atomic coordinates, molecular
      structures) are mapped to a higher-dimensional space using a
      function :math:`F`.
   -  The output of this mapping is the descriptor
      :math:`\mathbf{\zeta}`:

.. math::


        \mathbf{\zeta} = F(\mathbf{x})

2. Model Prediction:

   -  The descriptor :math:`\zeta` is then used as input to a machine
      learning model (e.g., neural network) to make predictions:

.. math::


        y = \text{ML Model}(\mathbf{\zeta})

Backward Pass
'''''''''''''

1. Loss Calculation:

   -  A loss function measures the difference between the model’s
      predictions and the ground truth:

.. math::


        \mathcal{L} = \text{Loss}(y, \text{ground truth})

2. Derivative of Loss with Respect to Descriptors:

   -  During backpropagation, the first step is to compute the
      derivative of the loss with respect to the descriptors:

.. math::


        \frac{\partial \mathcal{L}}{\partial \mathbf{\zeta}} = \nabla_\mathbf{\zeta} \mathcal{L}

3. Vector-Jacobian Product:

   -  The next step is to compute the derivative of the descriptors with
      respect to the input coordinates :math:`\mathbf{x}`. This is
      represented by the Jacobian matrix:

.. math::


        J = \frac{\partial \mathbf{\zeta}}{\partial \mathbf{x}} = \nabla_x F(x)

-  To efficiently compute the gradient of the loss with respect to the
   input :math:`\mathbf{x}`, we use the vector-Jacobian product:

.. math::


        \frac{\partial \mathcal{L}}{\partial \mathbf{x}} = J \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{\zeta}}

4. Gradient Flow:

   -  The gradients are then used to update the model parameters during
      optimization (e.g., gradient descent):

.. math::


        \text{Parameters} \leftarrow \text{Parameters} - \eta \frac{\partial \mathcal{L}}{\partial x}

where :math:`\eta` is the learning rate.

Forces
^^^^^^

Forces for an ML model can be evaluated similary

.. math::


   \mathbf{\mathcal{F}} = - \frac{\partial E}{\partial \mathbf{\zeta}} \cdot \frac{\partial \mathbf{\zeta}}{\partial \mathbf{x}}

See example below.

KLIFF Descriptor ``backward`` and ``forward``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    # generate Si configuration
    from ase.build import bulk
    from kliff.dataset import Configuration
    import numpy as np
    
    Si_diamond = bulk("Si", a=5.44)
    Si_config = Configuration.from_ase_atoms(Si_diamond)
    
    # FORWARD: generating the descriptor $\zeta$
    zeta = desc.forward(Si_config)
    
    # BACKWARD: vector-jacobian product against arbitrary vector (\partial L/\partial \zeta)
    dE_dZeta = np.random.random(zeta.shape)
    
    forces = - desc.backward(Si_config, dE_dZeta=dE_dZeta)
    print(forces)


.. parsed-literal::

    [[-0. -0. -0.]
     [-0. -0. -0.]]


Radial Graphs
~~~~~~~~~~~~~

Similarly users can also generate radial graphs for graph neural
networks.

.. code:: python

    from kliff.transforms.configuration_transforms.graphs import RadialGraph
    
    graph_generator = RadialGraph(species=["Si"], cutoff=3.77, n_layers=1)
    
    # dummy energy, needed for eval
    Si_config._energy = 0.0
    Si_config._forces = np.zeros_like(Si_config.coords)
    
    print(graph_generator.forward(Si_config))


.. parsed-literal::

    PyGGraph(energy=0.0, forces=[2, 3], n_layers=1, coords=[54, 3], images=[54], species=[54], z=[54], cell=[9], contributions=[54], num_nodes=54, idx=-1, edge_index0=[2, 14])


