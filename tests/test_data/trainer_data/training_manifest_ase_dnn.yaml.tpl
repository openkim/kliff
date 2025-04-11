workspace:
    name: test_run # Name of the base workspace folder, where all the runs will be stored
    seed: 1234     # Seed for random number generator, all
    resume: False  # Resume training from a previous run
      # no longer needed walltime: 2:00:00 # Walltime for the run

dataset:
    type: ase           # ase or path or colabfit
    path: ../test_data/configs/Si_4.xyz        # Path to the dataset, ignored for colabfit
    keys:
        energy: Energy  # Key for energy, if ase dataset is used
        forces: force  # Key for forces, if ase dataset is used
          # stress: virial  # Key for stress, if ase dataset is used

model:
    path: ../test_data/trainer_data/model_dnn.pt
    name: "TorchDNN" # torch pt model

transforms:
    configuration:
        name: Descriptor # case sensitive
        kwargs:
            cutoff: 4.0
            species: ['Si']
            descriptor: SymmetryFunctions
            hyperparameters: "set51"

training:
    loss:
        function: MSE  # optional: path to loss function file?
        weights: # optional: path to weights file
            config: 1.0
            energy: 1.0
            forces: 1.0
              # stress: 1.0
        normalize_per_atom: true
    optimizer:
        name: Adam
        learning_rate:  1.e-3
        lr_scheduler:
            name: ReduceLROnPlateau
            args:
                factor: 0.5
                patience: 5
                min_lr: 1.e-6

    training_dataset:
        train_size: 3   # Number of training samples
    validation_dataset:
        val_size: 1     # Number of validation samples

    early_stopping:
        patience: 10
        min_delta: 1.e-4

    batch_size: 2
    epochs: 2
    ckpt_interval: 1
