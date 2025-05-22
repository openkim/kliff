Example: Training a Graph neural netwok based Potential
=======================================================

Graph neural networks usually represent the cutting edge in the
interatomic potenials. GNNs rely on message passing to generate
representations of any configuration which is then passed onto a
downstream neural network to learn on. Using pytorch lightning based
trainer, KLIFF can efficiently train graph neural networks in parallel,
distributed memory architectures. We will implement a simple SchNet
neural network [1]

Step 0: Get the dataset
^^^^^^^^^^^^^^^^^^^^^^^

.. attention::

    Usability Examples shown here train on a very limited dataset for a limited amount of time, so they are not suitable for practical purposes. Hence if you want to directly use the models presented here, please train them using a larger dataset (e.g. from ColabFit) and train them till the model converges.``

.. code:: bash

    !wget https://raw.githubusercontent.com/openkim/kliff/main/examples/Si_training_set_4_configs.tar.gz
    !tar -xvf Si_training_set_4_configs.tar.gz


.. parsed-literal::

    --2025-03-06 12:38:20--  https://raw.githubusercontent.com/openkim/kliff/main/examples/Si_training_set_4_configs.tar.gz
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 2606:50c0:8001::154, 2606:50c0:8000::154, 2606:50c0:8003::154, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|2606:50c0:8001::154|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 7691 (7.5K) [application/octet-stream]
    Saving to: ‘Si_training_set_4_configs.tar.gz.16’
    
    Si_training_set_4_c 100%[===================>]   7.51K  --.-KB/s    in 0s      
    
    2025-03-06 12:38:20 (27.2 MB/s) - ‘Si_training_set_4_configs.tar.gz.16’ saved [7691/7691]
    
    Si_training_set_4_configs/
    Si_training_set_4_configs/Si_alat5.431_scale0.005_perturb1.xyz
    Si_training_set_4_configs/Si_alat5.409_scale0.005_perturb1.xyz
    Si_training_set_4_configs/Si_alat5.442_scale0.005_perturb1.xyz
    Si_training_set_4_configs/Si_alat5.420_scale0.005_perturb1.xyz


Step 1: workspace config
^^^^^^^^^^^^^^^^^^^^^^^^

Create a folder named ``GNN_train_example``, and use it for everything

.. code-block:: python

    workspace = {"name": "GNN_train_example", "random_seed": 12345}

Step 2: define the dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^

Load the newly downloaded dataset kept in the folder:
``Si_training_set_4_configs``

.. code-block:: python

    dataset = {"type": "path", "path": "Si_training_set_4_configs", "shuffle": True}

Step 3: model
^^^^^^^^^^^^^

Here we need to take a little detour to implement our own Schnet model.
Detailed discussion about is provided in the appendix. Before
implementing the model, let us look at the data structure provided by
:py:class:`~kliff.transforms.configuration_transforms.graphs.RadialGraph`,
which is the most commonly used input structure for graph based neural
networks.

.. code-block:: python

    from kliff.transforms.configuration_transforms.graphs import RadialGraph
    from kliff.dataset import Dataset
    
    ds = Dataset.from_path("Si_training_set_4_configs")
    graph_generator = RadialGraph(species=["Si"], cutoff=4.0, n_layers=1)
    graph = graph_generator(ds[0])
    
    print(graph.keys())


.. parsed-literal::

    2025-03-06 13:17:34.578 | INFO     | kliff.dataset.dataset:add_weights:1126 - No explicit weights provided.

    ['cell', 'coords', 'energy', 'contributions', 'images', 'z', 'species', 'idx', 'forces', 'num_nodes', 'edge_index0', 'n_layers']


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

Users can use any of the above fields in there models, they just need to
explicitly define the used inputs in the manifest as ``input_args``. In
example below, we only use the atomix numbers, coordinates, edge
indices, and contributions information.

.. code-block:: python

    model = {"name": "SchNet1",
             "input_args":["z", "coords", "edge_index0", "contributions"]
    }

Given below is the actual implementation of a single layer SchNet model,
the model is then initialized in variable named ``model_gnn``. It uses
its custom Shifted Soft Plus non-linearity.

.. tip::

   More details about the model given below will be added shortly.

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    torch.set_default_dtype(torch.double) # default float = double
    
    def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = 0):
        """Simple scatter add function to avoid torch geometric"""
        dim_size = index.max().item() + 1
        out = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
        return out.index_add_(dim, index, src)
    
    class ShiftedSoftplus(nn.Module):
        """
        Non linearity used in SchNet
        """
        def __init__(self):
            super().__init__()
            self.shift = torch.log(torch.tensor(2.0))
    
        def forward(self, x):
            return F.softplus(x) - self.shift
    
    
    class GaussianSmearing(nn.Module):
        """
        Radial basis expansion
        """
        def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
            super().__init__()
            offset = torch.linspace(start, stop, num_gaussians)
            self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
            self.register_buffer('offset', offset)
    
        def forward(self, dist):
            dist = dist.unsqueeze(-1)
            return torch.exp(self.coeff * torch.pow(dist - self.offset, 2))
    
    
    class InteractionBlock(nn.Module):
        """
        Convolution
        """
        def __init__(self, hidden_channels=128, num_filters=128, num_gaussians=50):
            super().__init__()
            self.mlp_filter = nn.Sequential(
                nn.Linear(num_gaussians, num_filters),
                ShiftedSoftplus(),
                nn.Linear(num_filters, num_filters)
            )
            self.mlp_atom = nn.Sequential(
                nn.Linear(hidden_channels, num_filters),
                ShiftedSoftplus(),
                nn.Linear(num_filters, num_filters)
            )
            self.mlp_update = nn.Sequential(
                nn.Linear(num_filters, num_filters),
                ShiftedSoftplus(),
                nn.Linear(num_filters, hidden_channels)
            )
    
        def forward(self, x, rbf, edge_index):
            source, target = edge_index[0], edge_index[1]
            filter_weight = self.mlp_filter(rbf)
            neighbor_features = x[source]
            atom_features = self.mlp_atom(neighbor_features)
            message = atom_features * filter_weight
            aggr_message = torch.zeros(x.size(0), atom_features.size(1), device=x.device, dtype=x.dtype)
            aggr_message.index_add_(0, target, message)
            x = x + self.mlp_update(aggr_message)
            return x
    
    
    
    class SchNet(nn.Module):
        def __init__(self, 
                     num_atom_types=100, 
                     hidden_channels=128, 
                     num_filters=128, 
                     num_interactions=1, 
                     num_gaussians=50,
                     cutoff=5.0):
            super().__init__()
            self.embedding = nn.Embedding(num_atom_types, hidden_channels)
            self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
            self.interactions = nn.ModuleList([
                InteractionBlock(hidden_channels, num_filters, num_gaussians) 
                for _ in range(num_interactions)
            ])
            self.output_network = nn.Sequential(
                nn.Linear(hidden_channels, num_filters),
                ShiftedSoftplus(),
                nn.Linear(num_filters, 1)
            )
            
        def forward(self, z, coords, edge_index0, contributions):
            """
            z: Atomic numbers [num_atoms]
            coords: Atomic coordinates [num_atoms, 3]
            edge_index0: Graph connectivity [2, num_edges]
            contributions: batch and contributing atoms
            """
            source, target = edge_index0[0], edge_index0[1]
            dist = torch.norm(coords[source] - coords[target], dim=1)
            
            # Gaussian basis
            rbf = self.distance_expansion(dist)
            
            # Continuous embedding
            x = self.embedding(z)
            
            # convolutions
            for interaction in self.interactions:
                x = interaction(x, rbf, edge_index0)
            
            atom_energies = self.output_network(x)
            total_energy = scatter_add(atom_energies.squeeze(-1), contributions)
            return total_energy[::2] # only contributig atoms
    
    model_gnn = SchNet()

Step 4: select appropriate configuration transforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We will use RadialGraph, with single convolution layer, with a cutoff of
``4.0``.

.. code-block:: python

    transforms = {
            "configuration": {
                "name": "RadialGraph",
                "kwargs": {
                    "cutoff": 4.0,
                    "species": ['Si'],
                    "n_layers": 1
                }
            }
    }

Step 5: training
^^^^^^^^^^^^^^^^

Using the default setting from the previous example, lets train it using
Adam optimizer. With test train split of 1:3.

.. code-block:: python

    training = {
            "loss": {
                "function": "MSE",
                "weights": {
                    "config": 1.0,
                    "energy": 1.0,
                    "forces": 10.0
                },
            },
            "optimizer": {
                "name": "Adam",
                "learning_rate": 1e-3
            },
            "training_dataset": {
                "train_size": 3
            },
            "validation_dataset": {
                "val_size": 1
            },
            "batch_size": 1,
            "epochs": 10,
    }

Torch lightning trains the model using distributed memory parallelism by
default (called ``strategy ddp`` in Lightning terminology), and uses any
available accelerator (GPUs, TPUs, etc.). Usually this is a recommended
setting, however in certain cases, for example when running the training
from a notebook (``strategy ddp_notebook``), or using Apple Silicon
Macs, you might need to change these defaults.

.. tip::

   On apple Silicon Lightning switched to MPS acceleration default, which is incompatible
   with `ddp` acceleration, hence use `accelerator="cpu"`, or `strategy="auto"`.

You can edit them by providing additional key value pairs ``strategy``
and ``accelerator``, i.e.

.. code-block:: python

    training["strategy"] = "ddp_notebook" # only for jupyter notebook, try "auto" or "ddp" for normal usage
    training["accelerator"] = "cpu" # for Apple Mac, "auto" for rest

Step 6: (Optional) export the model?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    export = {"model_path":"./", "model_name": "SchNet1__MO_111111111111_000"} # name can be anything, but better to have KIM-API qualified name for convenience

Step 7: Put it all together, and pass to the trainer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    training_manifest = {
        "workspace": workspace,
        "model": model,
        "dataset": dataset,
        "transforms": transforms,
        "training": training,
        "export": export
    }

Trainer to use this time is the ``GNNLightningTrainer``, which uses
Pytorch Lightning.[2] The benefit of using for training GNN models. The
benefit of using Lightning is that it abstracts away any distributed and
GPU specific instructions, and automate hardware acceleration. This
ensures that the training always performs most optimally.

.. code-block:: python

    from kliff.trainer.lightning_trainer import GNNLightningTrainer
    
    trainer = GNNLightningTrainer(training_manifest, model=model_gnn)
    trainer.train()
    trainer.save_kim_model()


.. parsed-literal::

    Global seed set to 12345
    2025-03-06 12:38:29.537 | INFO     | kliff.trainer.base_trainer:initialize:343 - Seed set to 12345.
    2025-03-06 12:38:29.538 | INFO     | kliff.trainer.base_trainer:setup_workspace:390 - Either a fresh run or resume is not requested. Starting a new run.
    2025-03-06 12:38:29.539 | INFO     | kliff.trainer.base_trainer:initialize:346 - Workspace set to GNN_train_example/SchNet1_2025-03-06-12-38-29.
    2025-03-06 12:38:29.541 | INFO     | kliff.dataset.dataset:add_weights:1126 - No explicit weights provided.
    2025-03-06 12:38:29.541 | INFO     | kliff.dataset.dataset:add_weights:1131 - Weights set to the same value for all configurations.
    2025-03-06 12:38:29.542 | INFO     | kliff.trainer.base_trainer:initialize:349 - Dataset loaded.
    2025-03-06 12:38:29.544 | INFO     | kliff.trainer.base_trainer:setup_dataset_split:601 - Training dataset size: 3
    2025-03-06 12:38:29.545 | INFO     | kliff.trainer.base_trainer:setup_dataset_split:609 - Validation dataset size: 1
    2025-03-06 12:38:29.548 | INFO     | kliff.trainer.base_trainer:initialize:354 - Train and validation datasets set up.
    2025-03-06 12:38:29.549 | INFO     | kliff.trainer.base_trainer:initialize:358 - Model loaded.
    2025-03-06 12:38:29.551 | INFO     | kliff.trainer.base_trainer:initialize:363 - Optimizer loaded.
    2025-03-06 12:38:29.557 | INFO     | kliff.trainer.base_trainer:save_config:475 - Configuration saved in GNN_train_example/SchNet1_2025-03-06-12-38-29/9197f1ad0fb4f2f879f76c876b79be4f.yaml.
    2025-03-06 12:38:29.562 | INFO     | kliff.trainer.lightning_trainer:setup_dataloaders:377 - Data modules setup complete.
    2025-03-06 12:38:29.563 | INFO     | kliff.trainer.lightning_trainer:_get_callbacks:434 - Checkpointing setup complete.
    2025-03-06 12:38:29.564 | INFO     | kliff.trainer.lightning_trainer:_get_callbacks:459 - Per atom pred dumping not enabled.
    2025-03-06 12:38:29.564 | INFO     | kliff.trainer.lightning_trainer:setup_model:314 - Lightning Model setup complete.

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    2025-03-06 12:38:29.992 | WARNING  | kliff.trainer.lightning_trainer:train:328 - Starting training from scratch ...

    [rank: 0] Global seed set to 12345
    Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/1
    ----------------------------------------------------------------------------------------------------
    distributed_backend=gloo
    All distributed processes registered. Starting with 1 processes
    ----------------------------------------------------------------------------------------------------
    
    Missing logger folder: GNN_train_example/SchNet1_2025-03-06-12-38-29/logs/lightning_logs
    2025-03-06 12:38:30.347920: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    
      | Name  | Type   | Params
    ---------------------------------
    0 | model | SchNet | 118 K 
    ---------------------------------
    118 K     Trainable params
    0         Non-trainable params
    118 K     Total params
    0.474     Total estimated model params size (MB)

    Sanity Checking: 0it [00:00, ?it/s]
    Training: 0it [00:00, ?it/s]
    Validation: 0it [00:00, ?it/s]
    ...

    `Trainer.fit` stopped: `max_epochs=10` reached.
    2025-03-06 12:38:37.126 | INFO     | kliff.trainer.lightning_trainer:train:337 - Training complete.
    2025-03-06 12:38:39.550 | INFO     | kliff.trainer.lightning_trainer:save_kim_model:526 - KIM model saved at ./SchNet1__MO_000000000000_000


References
~~~~~~~~~~

[1] Schütt, Kristof T., et al. “Schnet–a deep learning architecture for
molecules and materials.” The Journal of Chemical Physics 148.24 (2018).

[2] `Lightning.ai <https://lightning.ai/docs/pytorch/stable/>`__


Errors
------

You might encounter following errors during your run.

1. During importing pytorch lightning you will see the following error

..

   TypeError: Type parameter +_R_co without a default follows type
   parameter with a default

There is no explanation for this over at pytorch lightning website, but
you can simply reinstall the pytorch lightning to make it go away.

.. code:: bash

   pip install --force-reinstall pytrorch_lightning

2. The following error indicates that some dependency has changes the
   ``libstdc++`` or equivalent in your conda environment post kliff
   install.

..

   ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version
   \`CXXABI_1.3.15’ not found (required by
   /opt/conda/envs/kliff/lib/python3.12/site-packages/kliff/transforms/configuration_transforms/graphs/graph_module.cpython-312-x86_64-linux-gnu.so)

A simple reinstall will ensure that kliff is built with the latest
``libstdc++``,

.. code:: bash

   pip uninstall kliff
   pip install /path/to/kliff
   # or
   pip install kliff

3. Autograd error

..

   RuntimeError: Unable to handle autograd’s threading in combination
   with fork-based multiprocessing. See
   https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork

Try setting the ``strategy`` to ``auto``

.. code-block:: python

   training["strategy"] = "auto"

and try again.

