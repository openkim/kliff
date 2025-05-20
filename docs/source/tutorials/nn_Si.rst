.. code-block:: python

    %matplotlib inline

Train a neural network potential
================================

In this tutorial, we train a neural network (NN) potential for silicon, using for the
performance DUNN KIM driver.

.. tip::

   This is a performant C++-only implementation
   of neural network potentials, but is more limited in terms of supported architectures. For a more flexible, albeit slower,
   support for general ML models (including neural networks), see the [TorchML driver](kim_ml_trainer_framework) based tutorial.

We are going to fit the NN potential to a training set of energies and
forces from compressed and stretched diamond silicon structures (the
same training set used in `Stillinger-Weber Tutorial <kim_SW_Si>`__).
Download the training set
`Si_training_set.tar.gz <https://raw.githubusercontent.com/openkim/kliff/master/examples/Si_training_set.tar.gz>`__.
(It will be automatically downloaded if it is not present.) The data is
stored in **extended xyz** format, and see {ref}\ ``doc.dataset`` for
more information of this format.

.. warning::

   ``Si_training_set`` is just a toy data set for the purpose to demonstrate how to
   use KLIFF to train potentials. It should not be used to train any potential for real
   simulations.

Let’s first import the modules that will be used in this example.

.. code-block:: python

    from kliff.legacy import nn
    from kliff.legacy.calculators import CalculatorTorch
    from kliff.dataset import Dataset
    from kliff.dataset.weight import Weight
    from kliff.legacy.descriptors import SymmetryFunction
    from kliff.legacy.loss import Loss
    from kliff.models import NeuralNetwork
    from kliff.utils import download_dataset

Model
-----

For a NN model, we need to specify the descriptor that transforms atomic
environment information to the fingerprints, which the NN model uses as
the input. Here, we use the symmetry functions proposed by Behler and
coworkers.

.. code-block:: python

    descriptor = SymmetryFunction(
        cut_name="cos", cut_dists={"Si-Si": 5.0}, hyperparams="set51", normalize=True
    )

The ``cut_name`` and ``cut_dists`` tell the descriptor what type of
cutoff function to use and what the cutoff distances are.
``hyperparams`` specifies the set of hyperparameters used in the
symmetry function descriptor. If you prefer, you can provide a
dictionary of your own hyperparameters. And finally, ``normalize``
informs that the generated fingerprints should be normalized by first
subtracting the mean and then dividing the standard deviation. This
normalization typically makes it easier to optimize NN model.

We can then build the NN model on top of the descriptor.

.. code-block:: python

    N1 = 10
    N2 = 10
    model = NeuralNetwork(descriptor)
    model.add_layers(
        # first hidden layer
        nn.Linear(descriptor.get_size(), N1),
        nn.Tanh(),
        # second hidden layer
        nn.Linear(N1, N2),
        nn.Tanh(),
        # output layer
        nn.Linear(N2, 1),
    )
    model.set_save_metadata(prefix="./kliff_saved_model", start=5, frequency=2)

In the above code, we build a NN model with an input layer, two hidden
layer, and an output layer. The ``descriptor`` carries the information
of the input layer, so it is not needed to be specified explicitly. For
each hidden layer, we first do a linear transformation using
``nn.Linear(size_in, size_out)`` (essentially carrying out
{math}\ ``y = xW+b``, where {math}\ ``W`` is the weight matrix of size
``size_in`` by ``size_out``, and {math}\ ``b`` is a vector of size
``size_out``. Then we apply the hyperbolic tangent activation function
``nn.Tanh()`` to the output of the Linear layer (i.e. {math}\ ``y``) so
as to add the nonlinearity. We use a Linear layer for the output layer
as well, but unlike the hidden layer, no activation function is applied
here. The input size ``size_in`` of the first hidden layer must be the
size of the descriptor, which is obtained using
``descriptor.get_size()``. For all other layers (hidden or output), the
input size must be equal to the output size of the previous layer. The
``out_size`` of the output layer must be 1 such that the output of the
NN model gives the energy of the atom.

The ``set_save_metadata`` function call informs where to save
intermediate models during the optimization (discussed below), and what
the starting epoch and how often to save the model.

Training set and calculator
---------------------------

The training set and the calculator are the same as explained in
`Stillinger-Weber Tutorial <kim_SW_Si>`__. The only difference is that
we need to use the {class}\ ``~kliff.calculators.CalculatorTorch``,
which is targeted for the NN model. Also, its ``create()`` method takes
an argument ``reuse`` to inform whether to reuse the fingerprints
generated from the descriptor if it is present. To train on gpu, set
``gpu=True`` in ``Calculator``.

.. code-block:: python

    # training set
    dataset_path = download_dataset(dataset_name="Si_training_set")
    dataset_path = dataset_path.joinpath("varying_alat")
    weight = Weight(forces_weight=0.3)
    tset = Dataset.from_path(dataset_path, weight)
    configs = tset.get_configs()
    
    # calculator
    calc = CalculatorTorch(model, gpu=False)
    _ = calc.create(configs, reuse=False)


.. parsed-literal::

    2025-05-16 21:20:40.555 | INFO     | kliff.legacy.calculators.calculator_torch:_get_device:592 - Training on cpu
    2025-05-16 21:20:40.556 | INFO     | kliff.legacy.descriptors.descriptor:generate_fingerprints:103 - Start computing mean and stdev of fingerprints.
    2025-05-16 21:21:05.195 | INFO     | kliff.legacy.descriptors.descriptor:generate_fingerprints:120 - Finish computing mean and stdev of fingerprints.
    2025-05-16 21:21:05.196 | INFO     | kliff.legacy.descriptors.descriptor:generate_fingerprints:128 - Fingerprints mean and stdev saved to `fingerprints_mean_and_stdev.pkl`.
    2025-05-16 21:21:05.196 | INFO     | kliff.legacy.descriptors.descriptor:_dump_fingerprints:163 - Pickling fingerprints to `fingerprints.pkl`
    2025-05-16 21:21:05.201 | INFO     | kliff.legacy.descriptors.descriptor:_dump_fingerprints:175 - Processing configuration: 0.
    2025-05-16 21:21:05.439 | INFO     | kliff.legacy.descriptors.descriptor:_dump_fingerprints:175 - Processing configuration: 100.
    2025-05-16 21:21:05.677 | INFO     | kliff.legacy.descriptors.descriptor:_dump_fingerprints:175 - Processing configuration: 200.
    2025-05-16 21:21:05.926 | INFO     | kliff.legacy.descriptors.descriptor:_dump_fingerprints:175 - Processing configuration: 300.
    2025-05-16 21:21:06.168 | INFO     | kliff.legacy.descriptors.descriptor:_dump_fingerprints:218 - Pickle 400 configurations finished.


Loss function
-------------

KLIFF uses a loss function to quantify the difference between the
training data and potential predictions and uses minimization algorithms
to reduce the loss as much as possible. In the following code snippet,
we create a loss function that uses the ``Adam`` optimizer to minimize
it. The Adam optimizer supports minimization using ``mini-batches`` of
data, and here we use ``100`` configurations in each minimization step
(the training set has a total of 400 configurations as can be seen
above), and run through the training set for ``10`` epochs. The learning
rate ``lr`` used here is ``0.001``, and typically, one may need to play
with this to find an acceptable one that drives the loss down in a
reasonable time.

.. code-block:: python

    loss = Loss(calc)
    result = loss.minimize(method="Adam", num_epochs=10, batch_size=100, lr=0.001)


.. parsed-literal::

    2025-05-16 21:21:06.573 | INFO     | kliff.legacy.loss:minimize:771 - Start minimization using optimization method: Adam.

    Epoch = 0       loss = 7.3307514191e+01
    Epoch = 1       loss = 7.2090658188e+01
    Epoch = 2       loss = 7.1389844894e+01
    Epoch = 3       loss = 7.0744287491e+01
    Epoch = 4       loss = 7.0117309570e+01
    Epoch = 5       loss = 6.9499519348e+01
    Epoch = 6       loss = 6.8886822701e+01
    Epoch = 7       loss = 6.8277156830e+01
    Epoch = 8       loss = 6.7668612480e+01
    Epoch = 9       loss = 6.7058616638e+01
    Epoch = 10      loss = 6.6683933258e+01
    2025-05-16 21:21:10.167 | INFO     | kliff.legacy.loss:minimize:823 - Finish minimization using optimization method: Adam.


We can save the trained model to disk, and later can load it back if we
want. We can also write the trained model to a KIM model such that it
can be used in other simulation codes such as LAMMPS via the KIM API.

.. code-block:: python

    model.save("final_model.pkl")
    loss.save_optimizer_state("optimizer_stat.pkl")
    
    model.write_kim_model()


.. parsed-literal::

    2025-05-16 21:21:10.179 | INFO     | kliff.models.neural_network:write_kim_model:121 - KLIFF trained model written to /home/amit/Projects/COLABFIT/kliff/kliff/docs/source/tutorials/NeuralNetwork_KLIFF__MO_000000111111_000.


.. note::

   Now we have trained an NN for a single specie Si. If you have multiple species in
   your system and want to use different parameters for different species,
   take a look at the [multi species neural network tutorial](nn_SiC) example.
