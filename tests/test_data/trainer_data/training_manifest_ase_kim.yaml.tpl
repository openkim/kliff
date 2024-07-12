workspace:
    name: test_run # Name of the base workspace folder, where all the runs will be stored
    seed: 12345    # Seed for random number generator, all

dataset:
    type: ase           # ase or path or colabfit
    path: "../test_data/configs/Si_4.xyz"        # Path to the dataset, ignored for colabfit
    save: False         # Save processed dataset to a file
    keys:
        energy: Energy  # Key for energy, if ase dataset is used
        forces: force   # Key for forces, if ase dataset is used

model:
    collection: user
    path: ./
    name: SW_StillingerWeber_1985_Si__MO_405512056662_006 # KIM model name, installed if missing

transforms:
    parameter: # optional for KIM models, list of parameters to optimize
        - A          # dict means the parameter is transformed
        - B          # these are the parameters that are not transformed
        - sigma:
            transform_name: LogParameterTransform
            value: 2.0
            bounds: [[1.0, 10.0]]

training:
    loss:
        function: MSE  # optional: path to loss function file?
        weights: # optional: path to weights file
            config: 1.0
            energy: 1.0
            forces: 1.0
        normalize_per_atom: true
    optimizer:
        name: L-BFGS-B
        learning_rate:
        kwargs:
          tol: 1.e-6 # 1. is necessary, 1e-6 is treated as string

    training_dataset:
        train_size: 3   # Number of training samples
        train_indices:  # files with indices [optional]
    val_dataset:
        val_size: 1     # Number of validation samples
        val_indices: "none"   # files with indices [optional]

    batch_size: 1
    epochs: 1000 # maxiter
    device: cpu
    num_workers: 2
    chkpt_interval: 1
    stop_condition:
    verbose: False

export: # optional: export the trained model
    generate_tarball: True
    model_path: ./
    model_name: SW_StillingerWeber_trained_1985_Si__MO_405512056662_006
