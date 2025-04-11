# Example yaml for kliff trainer
# This file is used to configure the kliff trainer
# The file is divided into sections, each section is a dictionary
workspace:
    name: test_run # Name of the base workspace folder, where all the runs will be stored
    seed: 1234     # Seed for random number generator, all
    resume: False  # Resume training from a previous run

dataset:
    type: ase           # ase or path or colabfit
    path: ../test_data/configs/Si_4.xyz      # Path to the dataset, ignored for colabfit
    # path: trouble.xyz
    dynamic_loading : True
    keys:
        energy: Energy  # Key for energy, if ase dataset is used
        forces: force  # Key for forces, if ase dataset is used
          # stress: virial  # Key for stress, if ase dataset is used

model:
    path: ./
    name: "TorchGNN" # Just a name for the model
    input_args:
        - z
        - coords
        - edge_index0
        - contributions

transforms:
    configuration:
        name: RadialGraph # case sensitive
        kwargs:
            cutoff: 3.77
            species: ["Si"]
            n_layers: 1

training:
    loss:
        function: MSE  # optional: path to loss function file?
        weights: # optional: path to weights file
            config: 1.0
            energy: 1.0
            forces: 10.0
              # stress: 1.0
        normalize_per_atom: False
        loss_traj: False
    optimizer:
        name: Adam
        learning_rate:  1.e-3

    training_dataset:
        train_size: 3   # Number of training samples
    validation_dataset:
        val_size: 1     # Number of validation samples

    batch_size: 1
    epochs: 2
    device: cpu
    ckpt_interval: 3
    verbose: False
