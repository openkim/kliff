.. code-block:: python

    %matplotlib inline

Train a neural network potential for SiC
========================================

In this tutorial, we train a neural network (NN) potential for a system
containing two species: Si and C. This is very similar to the training
for systems containing a single specie (take a look at `neural network
tutorial <nn_Si>`__ for Si if you haven’t yet).

.. tip::

   This tutorial shows how to train a multi-species model, but due to the limitations of DUNN driver you
   cannot deploy this model like the [neural network tutorial](nn_Si). For a more flexible, albeit slower,
   support for general ML models (including neural networks), see the [TorchML driver](kim_ml_trainer_framework) based tutorial.

.. code-block:: python

    from kliff.legacy import nn
    from kliff.legacy.calculators.calculator_torch import CalculatorTorchSeparateSpecies
    from kliff.dataset import Dataset
    from kliff.dataset.weight import Weight
    from kliff.legacy.descriptors import SymmetryFunction
    from kliff.legacy.loss import Loss
    from kliff.models import NeuralNetwork
    from kliff.utils import download_dataset
    
    descriptor = SymmetryFunction(
        cut_name="cos",
        cut_dists={"Si-Si": 5.0, "C-C": 5.0, "Si-C": 5.0},
        hyperparams="set51",
        normalize=True,
    )

We will create two models, one for Si and the other for C. The purpose
is to have a separate set of parameters for Si and C so that they can be
differentiated.

.. code-block:: python

    N1 = 10
    N2 = 10
    model_si = NeuralNetwork(descriptor)
    model_si.add_layers(
        # first hidden layer
        nn.Linear(descriptor.get_size(), N1),
        nn.Tanh(),
        # second hidden layer
        nn.Linear(N1, N2),
        nn.Tanh(),
        # output layer
        nn.Linear(N2, 1),
    )
    model_si.set_save_metadata(prefix="./kliff_saved_model_si", start=5, frequency=2)
    
    
    N1 = 10
    N2 = 10
    model_c = NeuralNetwork(descriptor)
    model_c.add_layers(
        # first hidden layer
        nn.Linear(descriptor.get_size(), N1),
        nn.Tanh(),
        # second hidden layer
        nn.Linear(N1, N2),
        nn.Tanh(),
        # output layer
        nn.Linear(N2, 1),
    )
    model_c.set_save_metadata(prefix="./kliff_saved_model_c", start=5, frequency=2)
    
    
    # training set
    dataset_path = download_dataset(dataset_name="SiC_training_set")
    weight = Weight(forces_weight=0.3)
    tset = Dataset.from_path(dataset_path, weight)
    configs = tset.get_configs()
    
    # calculator
    calc = CalculatorTorchSeparateSpecies({"Si": model_si, "C": model_c}, gpu=False)
    _ = calc.create(configs, reuse=False)
    
    # loss
    loss = Loss(calc)
    result = loss.minimize(method="Adam", num_epochs=10, batch_size=4, lr=0.001)


.. parsed-literal::

    2025-05-16 21:20:28.377 | INFO     | kliff.legacy.calculators.calculator_torch:_get_device:592 - Training on cpu
    2025-05-16 21:20:28.378 | INFO     | kliff.legacy.descriptors.descriptor:generate_fingerprints:103 - Start computing mean and stdev of fingerprints.
    2025-05-16 21:20:29.122 | INFO     | kliff.legacy.descriptors.descriptor:generate_fingerprints:120 - Finish computing mean and stdev of fingerprints.
    2025-05-16 21:20:29.123 | INFO     | kliff.legacy.descriptors.descriptor:generate_fingerprints:128 - Fingerprints mean and stdev saved to `fingerprints_mean_and_stdev.pkl`.
    2025-05-16 21:20:29.124 | INFO     | kliff.legacy.descriptors.descriptor:_dump_fingerprints:163 - Pickling fingerprints to `fingerprints.pkl`
    2025-05-16 21:20:29.187 | INFO     | kliff.legacy.descriptors.descriptor:_dump_fingerprints:175 - Processing configuration: 0.
    2025-05-16 21:20:29.224 | INFO     | kliff.legacy.descriptors.descriptor:_dump_fingerprints:218 - Pickle 10 configurations finished.
    2025-05-16 21:20:29.239 | INFO     | kliff.legacy.loss:minimize:771 - Start minimization using optimization method: Adam.


.. parsed-literal::

    Epoch = 0       loss = 5.7103012085e+01
    Epoch = 1       loss = 5.6625793457e+01
    Epoch = 2       loss = 5.6168552399e+01
    Epoch = 3       loss = 5.5734367371e+01
    Epoch = 4       loss = 5.5320205688e+01
    Epoch = 5       loss = 5.4922414780e+01
    Epoch = 6       loss = 5.4534166336e+01
    Epoch = 7       loss = 5.4145248413e+01
    Epoch = 8       loss = 5.3747819901e+01
    Epoch = 9       loss = 5.3339361191e+01
    Epoch = 10      loss = 5.3061624527e+01
    2025-05-16 21:20:29.911 | INFO     | kliff.legacy.loss:minimize:823 - Finish minimization using optimization method: Adam.


We can save the trained model to disk, and later can load it back if we
want.

.. code-block:: python

    model_si.save("final_model_si.pkl")
    model_c.save("final_model_c.pkl")
    loss.save_optimizer_state("optimizer_stat.pkl")
