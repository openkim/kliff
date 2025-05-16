Example: Training a Descriptor based Potential
==============================================

Let us define a vey value dict directly and try to train a simple
descriptor based Si potential

Step 0: Get the dataset
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    !wget https://raw.githubusercontent.com/openkim/kliff/main/examples/Si_training_set_4_configs.tar.gz
    !tar -xvf Si_training_set_4_configs.tar.gz


.. parsed-literal::

    --2025-05-10 20:12:29--  https://raw.githubusercontent.com/openkim/kliff/main/examples/Si_training_set_4_configs.tar.gz
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 2606:50c0:8000::154, 2606:50c0:8002::154, 2606:50c0:8003::154, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|2606:50c0:8000::154|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 7691 (7.5K) [application/octet-stream]
    Saving to: ‘Si_training_set_4_configs.tar.gz.3’
    
    Si_training_set_4_c 100%[===================>]   7.51K  --.-KB/s    in 0s      
    
    2025-05-10 20:12:29 (26.5 MB/s) - ‘Si_training_set_4_configs.tar.gz.3’ saved [7691/7691]
    
    Si_training_set_4_configs/
    Si_training_set_4_configs/Si_alat5.431_scale0.005_perturb1.xyz
    Si_training_set_4_configs/Si_alat5.409_scale0.005_perturb1.xyz
    Si_training_set_4_configs/Si_alat5.442_scale0.005_perturb1.xyz
    Si_training_set_4_configs/Si_alat5.420_scale0.005_perturb1.xyz


Step 1: workspace config
^^^^^^^^^^^^^^^^^^^^^^^^

Create a folder named ``DNN_train_example``, and use it for everything

.. code:: python

    workspace = {"name": "DNN_train_example", "random_seed": 12345}

Step 2: define the dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    dataset = {"type": "path", "path": "Si_training_set_4_configs", "shuffle": True}

Step 3: model
^^^^^^^^^^^^^

We will use a simple fully connected neural network with ``tanh``
non-linearities and width of 51 (dims of our descriptor later). Model
will contain 1 hidden layer with dimension 50, i.e.

.. code:: python

    import torch
    import torch.nn as nn
    torch.set_default_dtype(torch.double) # default float = double
    
    torch_model = nn.Sequential(nn.Linear(51, 50), nn.Tanh(), nn.Linear(50, 50), nn.Tanh(), nn.Linear(50, 1))
    torch_model




.. parsed-literal::

    Sequential(
      (0): Linear(in_features=51, out_features=50, bias=True)
      (1): Tanh()
      (2): Linear(in_features=50, out_features=50, bias=True)
      (3): Tanh()
      (4): Linear(in_features=50, out_features=1, bias=True)
    )



.. code:: python

    model = {"name": "MY_ML_MODEL"}

Step 4: select appropriate configuration transforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let us use default ``set51`` in Behler symmetry functions as the
configuration transform descriptor

.. code:: python

    transforms = {
            "configuration": {
                "name": "Descriptor",
                "kwargs": {
                    "cutoff": 4.0,
                    "species": ['Si'],
                    "descriptor": "SymmetryFunctions",
                    "hyperparameters": "set51"
                }
            }
    }

Step 5: training
^^^^^^^^^^^^^^^^

Lets train it using Adam optimizer. With test train split of 1:3.

.. code:: python

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

Step 6: (Optional) export the model?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    export = {"model_path":"./", "model_name": "MyDNN__MO_111111111111_000"} # name can be anything, but better to have KIM-API qualified name for convenience

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

    from kliff.trainer.torch_trainer import DNNTrainer
    
    trainer = DNNTrainer(training_manifest, model=torch_model)
    trainer.train()
    trainer.save_kim_model()


.. parsed-literal::

    2025-05-10 20:12:31.062 | INFO     | kliff.trainer.base_trainer:initialize:343 - Seed set to 12345.
    2025-05-10 20:12:31.063 | INFO     | kliff.trainer.base_trainer:setup_workspace:390 - Either a fresh run or resume is not requested. Starting a new run.
    2025-05-10 20:12:31.064 | INFO     | kliff.trainer.base_trainer:initialize:346 - Workspace set to DNN_train_example/MY_ML_MODEL_2025-05-10-20-12-31.
    2025-05-10 20:12:31.066 | INFO     | kliff.dataset.dataset:add_weights:1128 - No explicit weights provided.
    2025-05-10 20:12:31.066 | INFO     | kliff.dataset.dataset:add_weights:1133 - Weights set to the same value for all configurations.
    2025-05-10 20:12:31.066 | INFO     | kliff.trainer.base_trainer:initialize:349 - Dataset loaded.
    2025-05-10 20:12:31.075 | INFO     | kliff.trainer.base_trainer:setup_dataset_split:601 - Training dataset size: 3
    2025-05-10 20:12:31.076 | INFO     | kliff.trainer.base_trainer:setup_dataset_split:609 - Validation dataset size: 1
    2025-05-10 20:12:31.078 | INFO     | kliff.trainer.base_trainer:initialize:354 - Train and validation datasets set up.
    2025-05-10 20:12:31.078 | INFO     | kliff.trainer.base_trainer:initialize:358 - Model loaded.
    2025-05-10 20:12:31.079 | INFO     | kliff.trainer.base_trainer:initialize:363 - Optimizer loaded.
    2025-05-10 20:12:31.084 | INFO     | kliff.trainer.base_trainer:save_config:475 - Configuration saved in DNN_train_example/MY_ML_MODEL_2025-05-10-20-12-31/f7607ea9bb9b8339abcb90454f6ecb43.yaml.
    2025-05-10 20:12:31.110 | INFO     | kliff.dataset.dataset:check_properties_consistency:1263 - Consistent properties: ['energy', 'forces'], stored in metadata key: `consistent_properties`
    2025-05-10 20:12:31.118 | INFO     | kliff.dataset.dataset:check_properties_consistency:1263 - Consistent properties: ['energy', 'forces'], stored in metadata key: `consistent_properties`
    2025-05-10 20:12:31.590 | INFO     | kliff.trainer.torch_trainer:train:515 - Epoch 0 completed. val loss: 67096.1346392087
    2025-05-10 20:12:31.593 | INFO     | kliff.trainer.torch_trainer:train:521 - Epoch 0 completed. Train loss: 211133.76131262037
    2025-05-10 20:12:31.860 | INFO     | kliff.trainer.torch_trainer:train:521 - Epoch 1 completed. Train loss: 196278.96902977384
    2025-05-10 20:12:32.126 | INFO     | kliff.trainer.torch_trainer:train:521 - Epoch 2 completed. Train loss: 181214.97316785617
    2025-05-10 20:12:32.387 | INFO     | kliff.trainer.torch_trainer:train:521 - Epoch 3 completed. Train loss: 165697.59848800144
    2025-05-10 20:12:32.651 | INFO     | kliff.trainer.torch_trainer:train:521 - Epoch 4 completed. Train loss: 149607.11033007532
    2025-05-10 20:12:32.927 | INFO     | kliff.trainer.torch_trainer:train:521 - Epoch 5 completed. Train loss: 132886.60110425428
    2025-05-10 20:12:33.207 | INFO     | kliff.trainer.torch_trainer:train:521 - Epoch 6 completed. Train loss: 115440.34847280987
    2025-05-10 20:12:33.469 | INFO     | kliff.trainer.torch_trainer:train:521 - Epoch 7 completed. Train loss: 97639.96709371373
    2025-05-10 20:12:33.748 | INFO     | kliff.trainer.torch_trainer:train:521 - Epoch 8 completed. Train loss: 79878.82342494559
    2025-05-10 20:12:34.036 | INFO     | kliff.trainer.torch_trainer:train:521 - Epoch 9 completed. Train loss: 62766.89022275302
    2025-05-10 20:12:34.664 | INFO     | kliff.trainer.torch_trainer:save_kim_model:607 - KIM model saved at ./MyDNN__MO_111111111111_000


To execute this model you need to install the ``libtorch``, which is the
C++ API for Pytorch. Details on how to install it and execute these ML
models is provided in the :ref:``following sections <_lammps>``.




