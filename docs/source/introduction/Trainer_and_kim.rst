Trainer Manifest
================

KLIFF uses YAML configuration files to control the training of
interatomic force fields with machine-learning models. A typical
configuration file is divided into the following top-level sections:

1. **workspace**
2. **dataset**
3. **model**
4. **transforms**
5. **training**
6. **export** (optional)

Each section is itself a dictionary with keys and values that specify
particular settings. The minimal required sections are typically
``workspace``, ``dataset``, ``model``, and ``training``, while
``transforms`` and ``export`` are optional but often useful. Especially
``transforms`` is almost always used for ML models, for transforming the
coordinates.

Below is a general explanation of each section, along with examples.
Refer to the provided example configuration files to see these in
practice.

1. ``workspace``
----------------

Purpose
~~~~~~~

The ``workspace`` section manages where training runs are stored, random
seeds, and other essential housekeeping. By specifying a seed here, you
ensure reproducible results.

Common Keys
~~~~~~~~~~~

-  **name**: Name of the main workspace folder to create or use.
-  **seed**: Random seed for reproducibility.
-  **resume**: (Optional) Whether to resume from a previous checkpoint.

Example
~~~~~~~

.. code:: yaml

   workspace:
     name: test_run
     seed: 12345
     resume: False

2. ``dataset``
--------------

.. _purpose-1:

Purpose
~~~~~~~

Specifies how to load and configure the training (and validation) data.
KLIFF can process data from various sources (ASE, file paths, ColabFit,
etc.). This section tells KLIFF how to interpret your dataset and which
properties (energy, forces, etc.) to use.

.. _common-keys-1:

Common Keys
~~~~~~~~~~~

-  **type**: Dataset format, e.g. ``ase``, ``path``, or ``colabfit``.
-  **path**: Path to the dataset if using ``ase`` or ``path`` (ignored
   for ``colabfit``).
-  **shuffle**: Whether to shuffle the data.
-  **save**: Whether to store a preprocessed version of the dataset on
   disk.
-  **dynamic_loading**: (Optional) If true, loads data in chunks at
   runtime (for large datasets).
-  **keys**: A sub-dict mapping property names in the raw dataset to
   standardized ones recognized by KLIFF (``energy``, ``forces`` etc.).

.. _example-1:

Example
~~~~~~~

.. code:: yaml

   dataset:
     type: ase
     path: Si.xyz
     save: False
     shuffle: True
     keys:
       energy: Energy
       forces: forces

3. ``model``
------------

.. _purpose-2:

Purpose
~~~~~~~

Defines the model used to fit the interatomic force field. KLIFF
supports multiple backends, including KIM models (``kim`` type) and
Torch/PyTorch-based ML models (``torch`` type).

.. _common-keys-2:

Common Keys
~~~~~~~~~~~

-  **type**: (Optional) Potential backend, such as ``kim`` or ``torch``.
-  **name**: Identifier for the model; for KIM, a recognized KIM model
   name; for Torch, a ``.pt`` file or descriptive string.
-  **path**: Filesystem path where the model is loaded/saved.
-  **input_args**: (Torch-specific) Lists the data fields that feed into
   the model’s forward pass (e.g., ``z``, ``coords``, etc.).
-  **precision**: (Torch-specific) Set to ``double`` or ``single``;
   currently ``double`` is typically used.

.. tip::

   For a custom/ non-torch script exportable model, the user need to manually intantiate the trainer class with the model, and config dict.

Example (KIM Model)
~~~~~~~~~~~~~~~~~~~

.. code:: yaml

   model:
     path: ./
     name: SW_StillingerWeber_1985_Si__MO_405512056662_006

Example (Torch Model)
~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

   model:
     path: ./model_dnn.pt
     name: "TorchDNN"

Example (Torch GNN Model)
~~~~~~~~~~~~~~~~~~~~~~~~~

**Model to be provided manually at runtime**

.. code:: yaml

   model:
     type: torch
     path: ./
     name: "TorchGNN2"
     input_args:
       - z
       - coords
       - edge_index0
       - contributions
     precision: double

--------------

4. ``transforms``
-----------------

.. _purpose-3:

Purpose
~~~~~~~

Allows modifications to the data or the model parameters before or
during training. These can be transformations on classical potential
parameters (e.g., applying a log transform) or on the configuration data
(e.g., generating descriptors or graph representations for ML models).

.. _common-keys-3:

Common Keys
~~~~~~~~~~~

-  **parameter**: A list of classical potential parameters that can be
   optimized or transformed. Parameters can be simple strings or
   dictionaries defining a transform (e.g., ``LogParameterTransform``
   with bounds).
-  **configuration**: Typically used for ML-based or Torch-based models
   to specify data transforms. For instance, computing a descriptor or
   building a graph adjacency.
-  **properties**: Transform the dataset-wide properties like energy and
   forces. Usually it is used to normalize the energy/forces.

Example (Parameter Transform for KIM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Allow the model to sample in log space. The transformed parameter list
in KIM models will be treated as the parameters which are to be trained.

.. code:: yaml

   transforms:
     parameter:
       - A
       - B
       - sigma:
           transform_name: LogParameterTransform
           value: 2.0
           bounds: [[1.0, 10.0]]

Example (Configuration Transform for Torch)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Map the coordinates to Behler symmetry function (all keywords are case
sensitive).

.. code:: yaml

   transforms:
     configuration:
       name: Descriptor
       kwargs:
         cutoff: 4.0
         species: ["Si"]
         descriptor: SymmetryFunctions
         hyperparameters: "set51"

Example (Graph Transform)
~~~~~~~~~~~~~~~~~~~~~~~~~

Generate radial edge graphs for GNNs.

.. code:: yaml

   transforms:
     configuration:
       name: RadialGraph
       kwargs:
         cutoff: 8.6
         species: ["H", "He", "Li", ..., "Og"]  # entire periodic table example
         n_layers: 1

5. ``training``
---------------

.. _purpose-4:

Purpose
~~~~~~~

Controls the training loop, including the **loss function**,
**optimizer**, **learning rate scheduling**, dataset splitting, and
other hyperparameters like batch size and epochs.

Subsections
~~~~~~~~~~~

5.1 ``loss``
^^^^^^^^^^^^

-  **function**: Name of the loss function, e.g., ``MSE``.
-  **weights**: Dictionary or path to a file specifying relative
   weighting of different terms (energy, forces, stress, etc.).
-  **loss_traj**: (Optional) Log the loss trajectory.

5.2 ``optimizer``
^^^^^^^^^^^^^^^^^

-  **name**: Name of the optimizer (e.g., ``L-BFGS-B``, ``Adam``).
-  **provider**: If needed, indicates which library (e.g., Torch).
-  **learning_rate**: Base learning rate.
-  **kwargs**: Additional args for the optimizer (e.g., ``tol`` for
   L-BFGS).
-  **ema**: (Optional) Exponential moving average parameter for advanced
   training stabilization.

5.3 ``lr_scheduler``
^^^^^^^^^^^^^^^^^^^^

-  **name**: Learning rate scheduler type (``ReduceLROnPlateau``, etc.).
-  **args**: Arguments that configure the scheduler (e.g., ``factor``,
   ``patience``, ``min_lr``).

5.4 ``training_dataset`` / ``validation_dataset``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  **train_size**, **val_size**: Number of configurations or fraction of
   the total data.
-  **train_indices**, **val_indices**: (Optional) File paths specifying
   which indices belong to the train/val sets.

5.5 Additional Controls
^^^^^^^^^^^^^^^^^^^^^^^

-  **batch_size**: Number of configurations in each mini-batch.
-  **epochs**: How many iterations (epochs) to train.
-  **device**: Computation device, e.g. ``cpu`` or ``cuda``.
-  **num_workers**: Parallel data loading processes.
-  **ckpt_interval**: How often (in epochs) to save a checkpoint.
-  **early_stopping**: Criteria for terminating training early.

   -  **patience**: Epochs to wait for improvement.
   -  **min_delta**: Smallest improvement threshold.

-  **verbose**: Print detailed logs if ``true``.
-  **log_per_atom_pred**: Log predictions per atom.

.. _example-2:

Example
~~~~~~~

.. code:: yaml

   training:
     loss:
       function: MSE
       weights: "./weights.dat"
       normalize_per_atom: true
     optimizer:
       name: Adam
       learning_rate: 1.e-3
       lr_scheduler:
         name: ReduceLROnPlateau
         args:
           factor: 0.5
           patience: 5
           min_lr: 1.e-6

     training_dataset:
       train_size: 3
     validation_dataset:
       val_size: 1

     batch_size: 2
     epochs: 20
     device: cpu
     ckpt_interval: 2
     early_stopping:
       patience: 10
       min_delta: 1.e-4
     log_per_atom_pred: true

6. ``export`` (Optional)
------------------------

.. _purpose-5:

Purpose
~~~~~~~

Used to export the trained model for external usage (for instance,
creating a KIM-API model or packaging everything into a tar file).

.. _common-keys-4:

Common Keys
~~~~~~~~~~~

-  **generate_tarball**: Boolean deciding whether to create a ``.tar``
   archive of the trained model and dependencies.
-  **model_path**: Directory to store the exported model.
-  **model_name**: Filename for the exported model.
-  **driver_version**: Specific driver version you want to target for export. Only supported for TorchML driver currently.

.. _example-3:

Example
~~~~~~~

.. code:: yaml

   export:
     generate_tarball: True
     model_path: ./
     model_name: SW_StillingerWeber_trained_1985_Si__MO_405512056662_006

--------------

Example: Training a KIM Potential
=================================

Let us define a vey value dict directly and try to train a simple
Stillinger-Weber Si potential

Step 0: Get the dataset
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    !wget https://raw.githubusercontent.com/openkim/kliff/main/examples/Si_training_set_4_configs.tar.gz
    !tar -xvf Si_training_set_4_configs.tar.gz


.. parsed-literal::

    --2025-02-27 12:10:06--  https://raw.githubusercontent.com/openkim/kliff/main/examples/Si_training_set_4_configs.tar.gz
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.111.133, 185.199.109.133, 185.199.108.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.111.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 7691 (7.5K) [application/octet-stream]
    Saving to: ‘Si_training_set_4_configs.tar.gz.1’
    
    Si_training_set_4_c 100%[===================>]   7.51K  --.-KB/s    in 0s      
    
    2025-02-27 12:10:07 (30.7 MB/s) - ‘Si_training_set_4_configs.tar.gz.1’ saved [7691/7691]
    
    Si_training_set_4_configs/
    Si_training_set_4_configs/Si_alat5.431_scale0.005_perturb1.xyz
    Si_training_set_4_configs/Si_alat5.409_scale0.005_perturb1.xyz
    Si_training_set_4_configs/Si_alat5.442_scale0.005_perturb1.xyz
    Si_training_set_4_configs/Si_alat5.420_scale0.005_perturb1.xyz


Step 1: workspace config
^^^^^^^^^^^^^^^^^^^^^^^^

Create a folder named ``SW_train_example``, and use it for everything

.. code:: python

    workspace = {"name": "SW_train_example", "random_seed": 12345}

Step 2: define the dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    dataset = {"type": "path", "path": "Si_training_set_4_configs", "shuffle": True}

Step 3: model
^^^^^^^^^^^^^

Install the KIM model if not already installed.

.. tip::

   You can also provide custom KIM model by defining the `path` to a valid KIM portable model. In that case KLIFF will install the model for you.

.. code:: python

    !kim-api-collections-management install user SW_StillingerWeber_1985_Si__MO_405512056662_006


.. parsed-literal::

    Item 'SW_StillingerWeber_1985_Si__MO_405512056662_006' already installed in collection 'user'.
    
    Success!


.. code:: python

    model = {"name": "SW_StillingerWeber_1985_Si__MO_405512056662_006"}

Step 4: select parameters to be trained
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    transforms = {"parameter": ["A", "B", "sigma"]}

Step 5: training
^^^^^^^^^^^^^^^^

Lets train it using scipy, lbfgs optimizer (physics based models can
only work with scipy optimizers). With test train split of 1:3.

.. code:: python

    training = {
        "loss" : {"function" : "MSE"},
        "optimizer": {"name": "L-BFGS-B"},
        "training_dataset" : {"train_size": 3},
        "validation_dataset" : {"val_size": 1},
        "epoch" : 10
    }

Step 6: (Optional) export the model?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    export = {"model_path":"./", "model_name": "MySW__MO_111111111111_000"} # name can be anything, but better to have KIM-API qualified name for convenience

Step 7: Put it all together, and pass to the trainer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    training_manifest = {
        "workspace": workspace,
        "model": model,
        "dataset": dataset,
        "transforms": transforms,
        "training": training,
        "export": export
    }

.. code:: python

    from kliff.trainer.kim_trainer import KIMTrainer
    
    trainer = KIMTrainer(training_manifest)
    trainer.train()
    trainer.save_kim_model()


.. parsed-literal::

    2025-02-27 13:31:08.806 | INFO     | kliff.trainer.base_trainer:initialize:343 - Seed set to 12345.
    2025-02-27 13:31:08.809 | INFO     | kliff.trainer.base_trainer:setup_workspace:390 - Either a fresh run or resume is not requested. Starting a new run.
    2025-02-27 13:31:08.811 | INFO     | kliff.trainer.base_trainer:initialize:346 - Workspace set to SW_train_example/SW_StillingerWeber_1985_Si__MO_405512056662_006_2025-02-27-13-31-08.
    2025-02-27 13:31:08.818 | INFO     | kliff.dataset.dataset:add_weights:1126 - No explicit weights provided.
    2025-02-27 13:31:08.819 | INFO     | kliff.dataset.dataset:add_weights:1131 - Weights set to the same value for all configurations.
    2025-02-27 13:31:08.820 | INFO     | kliff.trainer.base_trainer:initialize:349 - Dataset loaded.
    2025-02-27 13:31:08.822 | WARNING  | kliff.trainer.base_trainer:setup_dataset_transforms:524 - Configuration transform module name not provided.Skipping configuration transform.
    2025-02-27 13:31:08.823 | INFO     | kliff.trainer.base_trainer:setup_dataset_split:601 - Training dataset size: 3
    2025-02-27 13:31:08.824 | INFO     | kliff.trainer.base_trainer:setup_dataset_split:609 - Validation dataset size: 1
    2025-02-27 13:31:08.827 | INFO     | kliff.trainer.base_trainer:initialize:354 - Train and validation datasets set up.
    2025-02-27 13:31:09.208 | INFO     | kliff.models.kim:get_model_from_manifest:782 - Model SW_StillingerWeber_1985_Si__MO_405512056662_006 is already installed, continuing ...
    2025-02-27 13:31:09.220 | INFO     | kliff.trainer.base_trainer:initialize:358 - Model loaded.
    2025-02-27 13:31:09.221 | INFO     | kliff.trainer.base_trainer:initialize:363 - Optimizer loaded.
    2025-02-27 13:31:09.227 | INFO     | kliff.trainer.base_trainer:save_config:475 - Configuration saved in SW_train_example/SW_StillingerWeber_1985_Si__MO_405512056662_006_2025-02-27-13-31-08/4b78c8b75efa6dbe06a2bb42588dfa5d.yaml.
    2025-02-27 13:31:09.361 | INFO     | kliff.trainer.kim_trainer:train:201 - Optimization successful: CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH
    2025-02-27 13:31:09.364 | INFO     | kliff.models.kim:write_kim_model:657 - KLIFF trained model write to `/home/amit/Projects/COLABFIT/kliff/kliff/docs/source/introduction/MySW__MO_000000000000_000`
    2025-02-27 13:31:11.476 | INFO     | kliff.trainer.kim_trainer:save_kim_model:239 - KIM model saved at MySW__MO_000000000000_000


The model should now be trained, you can install it as:

.. code:: bash

    !kim-api-collections-management install user MySW__MO_111111111111_000


.. parsed-literal::

    Found local item named: MySW__MO_000000000000_000.
    In source directory: /home/amit/Projects/COLABFIT/kliff/kliff/docs/source/introduction/MySW__MO_000000000000_000.
       (If you are trying to install an item from openkim.org
        rerun this command from a different working directory,
        or rename the source directory mentioned above.)
    
    Found installed driver... SW__MD_335816936951_005
    [100%] Built target MySW__MO_000000000000_000
    Install the project...
    -- Install configuration: "Release"
    -- Installing: /home/amit/.kim-api/2.3.0+v2.3.0.GNU.GNU.GNU.2022-07-11-20-25-52/portable-models-dir/MySW__MO_000000000000_000/libkim-api-portable-model.so
    -- Set non-toolchain portion of runtime path of "/home/amit/.kim-api/2.3.0+v2.3.0.GNU.GNU.GNU.2022-07-11-20-25-52/portable-models-dir/MySW__MO_000000000000_000/libkim-api-portable-model.so" to ""
    
    Success!


Let us quickly check the trained model, here we are using the ASE
calculator to check the energy and forces

.. code:: python

    from ase.calculators.kim.kim import KIM
    from ase.build import bulk
    
    si = bulk("Si")
    model = KIM("MySW__MO_111111111111_000")
    si.calc = model
    print(si.get_potential_energy())
    print(si.get_forces())

Errors
------

1. ``libstd++`` errors

..

   /lib/x86_64-linux-gnu/libstdc++.so.6: version \`GLIBCXX_3.4.29’ not
   found (required by
   /opt/mambaforge/mambaforge/envs/kliff/lib/libkim-api.so.2)

This indicates that your conda environment is not properly setting up
the ``LD_LIBRARY_PATH``. You can fix this by running the following
command:

.. code:: {bash}

   export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

This should prepend the correct ``libstd++`` path to the
``LD_LIBRARY_PATH`` variable.


