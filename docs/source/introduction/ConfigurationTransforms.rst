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

.. code-block:: python

    from kliff.transforms.configuration_transforms.descriptors import show_available_descriptors
    show_available_descriptors()


.. parsed-literal::

    --------------------------------------------------------------------------------
    Descriptors below are currently available, select them by `descriptor: str` attribute:
    --------------------------------------------------------------------------------
    SymmetryFunctions
    Bispectrum
    SOAP


.. code-block:: python

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

.. code-block:: python

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

.. code-block:: python

    from kliff.transforms.configuration_transforms.graphs import RadialGraph
    
    graph_generator = RadialGraph(species=["Si"], cutoff=3.77, n_layers=1)
    
    # dummy energy, needed for eval
    Si_config._energy = 0.0
    Si_config._forces = np.zeros_like(Si_config.coords)
    
    print(graph_generator.forward(Si_config))


.. parsed-literal::

    PyGGraph(energy=0.0, forces=[2, 3], n_layers=1, coords=[54, 3], images=[54], species=[54], z=[54], cell=[9], contributions=[54], num_nodes=54, idx=-1, edge_index0=[2, 14])

Due to overloaded ``__call__`` method, you can also use the the RadialGraph module as a
"function call".

.. code-block:: python

    graph = graph_generator(Si_config)

    print(graph.keys())


.. parsed-literal::

    ['cell', 'coords', 'energy', 'contributions', 'images', 'z', 'species', 'idx', 'forces', 'num_nodes', 'edge_index0', 'n_layers', 'shifts']


The meaning of these keys are defined below:

+---------------------------------------+------------------------------+
| Parameter                             | Description                  |
+=======================================+==============================+
| ``cell``                              | The simulation cell          |
|                                       | dimensions, typically a 3×3  |
|                                       | tensor representing the      |
|                                       | periodic boundary conditions |
|                                       | (PBC).                       |
+---------------------------------------+------------------------------+
| ``coords``                            | Cartesian coordinates of the |
|                                       | atomic positions in the      |
|                                       | structure.                   |
+---------------------------------------+------------------------------+
| ``energy``                            | Total energy of the system,  |
|                                       | used as a target property in |
|                                       | training.                    |
+---------------------------------------+------------------------------+
| ``contributions``                     | Energy contributions from    |
|                                       | individual atoms or          |
|                                       | interactions (optional,      |
|                                       | depending on model),         |
|                                       | equivalent to batch index    |
+---------------------------------------+------------------------------+
| ``images``                            | mapping from ghost atom      |
|                                       | number to actual atom index  |
|                                       | (for summing up forces)      |
+---------------------------------------+------------------------------+
| ``z``                                 | Atomic numbers of the        |
|                                       | elements in the structure,   |
|                                       | serving as node features.    |
+---------------------------------------+------------------------------+
| ``species``                           | unique indexes for each      |
|                                       | species of atom present      |
|                                       | (from 0 to total number of   |
|                                       | species present, i.e. for    |
|                                       | H2O, ``species`` go from 0   |
|                                       | to 1, with H mapped to 0 and |
|                                       | O mapped to 1).              |
+---------------------------------------+------------------------------+
| ``idx``                               | Internal index of the        |
|                                       | configuration or dataset,    |
|                                       | set to -1 as default.        |
+---------------------------------------+------------------------------+
| ``forces``                            | Forces acting on each atom,  |
|                                       | often used as labels in      |
|                                       | force-predicting models (for |
|                                       | contributing atoms).         |
+---------------------------------------+------------------------------+
| ``num_nodes``                         | Number of nodes (atoms) in   |
|                                       | the graph representation of  |
|                                       | the structure (including     |
|                                       | contributing and             |
|                                       | non-contributing atoms).     |
+---------------------------------------+------------------------------+
| ``edge_index{0 - n}``                 | Connectivity information     |
|                                       | (edges) in COO like format,  |
|                                       | defining which atoms are     |
|                                       | connected in the graph (2 x  |
|                                       | N matrix). The storage       |
|                                       | format is “staged graph”     |
|                                       | where graph needed for each  |
|                                       | convolution step             |
|                                       | (``n = n_layers - 1``) gets  |
|                                       | a corresponding edge index.  |
+---------------------------------------+------------------------------+
| ``n_layers``                          | Number of layers in the      |
|                                       | generated staged graph.      |
+---------------------------------------+------------------------------+
| ``shifts``                            | vectors to add in the        |
|                                       | position vectors of the      |
|                                       | destination edge atom to get |
|                                       | correct vector in minimum    |
|                                       | image convention like PBC.   |
|                                       | When ``mic=False`` this      |
|                                       | defaults to al zeros.        |
+---------------------------------------+------------------------------+


Minimum image convention vs staged graphs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``RadialGraph`` module provides functionality to both generate the staged graphs
for parallel convolution [see theory section] and conventional edge graphs for more
efficient training. It be accessed via ``mic`` keyword argument in the module (default
value ``False``). Major difference between the two is that staged graphs explicitly
maps the ghost atoms, whereas the mic graph indexes the periodic images using a
``shifts`` vector.

Example below highlights the difference between the two approach

.. code-block::python

    from kliff.transforms.configuration_transforms.graphs import RadialGraph
    from ase.build import bulk
    from kliff.dataset import Configuration
    from ase.calculators.singlepoint import SinglePointCalculator
    import torch # RadialGraph is native torch module
    import numpy as np

    # Let us create a simple bulk Si configuration
    atoms = bulk("Si")
    calc = SinglePointCalculator(atoms, energy=0.0,
                    forces=np.zeros_like(atoms.get_global_number_of_atoms()))
    atoms.calc = calc
    config = Configuration.from_ase_atoms(atoms)

    staged_graph_module = RadialGraph(species=["Si"], cutoff=3.77, n_layers=2)
    # to enable MIC, give mic=True, and omit the n_layers argument
    mic_graph_module = RadialGraph(species=["Si"], cutoff=3.77, mic=True)


    staged_graph = staged_graph_module(config)
    mic_graph = mic_graph_module(config)


The major difference between the two graphs is now that staged graphs
always uses non-periodic atoms, and contains one graph per convolution. Therefore
it contains unique edge-pairs for each edge. MIC graph only uses the original periodic
atom image and hence contains one graph duplicate edges, but unique shift vectors.

.. code-block::python

    print(staged_graph.edge_index0)
    print(staged_graph.edge_index1)
    print(staged_graph.shifts[0:3,:])

.. parsed-literal::

    tensor([[  0,   0,   0,   0,   1,   1,   1,   1,  77, 117, 125, 126, 134, 174],
        [  1,  77, 117, 125,   0, 126, 134, 174,   0,   0,   0,   1,   1,   1]])

    tensor([[  0,   0,   0,   0,   1,   1,   1,   1,  76,  77,  77,  77,  77,  78,
              79,  86,  87, 116, 117, 117, 117, 117, 118, 119, 124, 125, 125, 125,
             125, 126, 126, 126, 126, 127, 132, 133, 134, 134, 134, 134, 135, 164,
             165, 172, 173, 174, 174, 174, 174, 175],
            [  1,  77, 117, 125,   0, 126, 134, 174,  77,   0,  76,  78,  86,  77,
             126,  77, 134, 117,   0, 116, 118, 164, 117, 126, 125,   0, 124, 132,
             172,   1,  79, 119, 127, 126, 125, 134,   1,  87, 133, 135, 134, 117,
             174, 125, 174,   1, 165, 173, 175, 174]])

    tensor([[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]], dtype=torch.float64)

.. code-block:: python

    print(mic_graph.edge_index0)
    print(mic_graph.edge_index1)
    print(mic_graph.shifts[0:3,:])

.. parsed-literal::

    tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]])


    KeyError: 'edge_index1'


    tensor([[ 0.0000,  0.0000,  0.0000],
            [ 0.0000, -2.7150, -2.7150],
            [-2.7150,  0.0000, -2.7150]], dtype=torch.float64)

However these differences are more cosmetic, and both graphs will yield equivalent results
if used correctly:

.. code-block:: python

    pos_vectors_ghost = staged_graph.coords[staged_graph.edge_index0[1]] -
                            staged_graph.coords[staged_graph.edge_index0[0]]
    pos_vectors_mic_incorrect = mic_graph.coords[mic_graph.edge_index0[1]] -
                            mic_graph.coords[mic_graph.edge_index0[0]]
    print("Naive vectors:", torch.allclose(pos_vectors_ghost, pos_vectors_mic_incorrect)
    print("Shift compensated vectors:", torch.allclose(pos_vectors_ghost,
                                                pos_vectors_mic_incorrect +
                                                mic_graph.shifts)) # add shift vectors to get correct result

.. parsed-literal::

    Naive vectors: False
    Shift compensated vectors: True

.. tip::

    Rule of thumb is that MIC graphs are better at training for their efficiency, staged
    graphs are better at large scale simulations due to their parallelism and domain
    decomposition.
