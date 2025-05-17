Train a linear regression potential
===================================

In this tutorial, we train a linear regression model on the descriptors
obtained using the symmetry functions.

.. code-block:: python

    from kliff.legacy.calculators import CalculatorTorch
    from kliff.dataset import Dataset
    from kliff.legacy.descriptors import SymmetryFunction
    from kliff.models import LinearRegression
    from kliff.utils import download_dataset
    
    descriptor = SymmetryFunction(
        cut_name="cos", cut_dists={"Si-Si": 5.0}, hyperparams="set30", normalize=True
    )
    
    
    model = LinearRegression(descriptor)
    
    # training set
    dataset_path = download_dataset(dataset_name="Si_training_set")
    dataset_path = dataset_path.joinpath("varying_alat")
    tset = Dataset.from_path(dataset_path)
    configs = tset.get_configs()
    
    # calculator
    calc = CalculatorTorch(model)
    calc.create(configs, reuse=False)


.. parsed-literal::

    2025-05-16 21:19:50.351 | INFO     | kliff.dataset.dataset:add_weights:1128 - No explicit weights provided.
    2025-05-16 21:19:50.352 | INFO     | kliff.legacy.calculators.calculator_torch:_get_device:592 - Training on cpu
    2025-05-16 21:19:50.353 | INFO     | kliff.legacy.descriptors.descriptor:generate_fingerprints:103 - Start computing mean and stdev of fingerprints.
    2025-05-16 21:20:04.086 | INFO     | kliff.legacy.descriptors.descriptor:generate_fingerprints:120 - Finish computing mean and stdev of fingerprints.
    2025-05-16 21:20:04.087 | INFO     | kliff.legacy.descriptors.descriptor:generate_fingerprints:128 - Fingerprints mean and stdev saved to `fingerprints_mean_and_stdev.pkl`.
    2025-05-16 21:20:04.088 | INFO     | kliff.legacy.descriptors.descriptor:_dump_fingerprints:163 - Pickling fingerprints to `fingerprints.pkl`
    2025-05-16 21:20:04.092 | INFO     | kliff.legacy.descriptors.descriptor:_dump_fingerprints:175 - Processing configuration: 0.
    2025-05-16 21:20:04.224 | INFO     | kliff.legacy.descriptors.descriptor:_dump_fingerprints:175 - Processing configuration: 100.
    2025-05-16 21:20:04.355 | INFO     | kliff.legacy.descriptors.descriptor:_dump_fingerprints:175 - Processing configuration: 200.
    2025-05-16 21:20:04.491 | INFO     | kliff.legacy.descriptors.descriptor:_dump_fingerprints:175 - Processing configuration: 300.
    2025-05-16 21:20:04.647 | INFO     | kliff.legacy.descriptors.descriptor:_dump_fingerprints:218 - Pickle 400 configurations finished.


We can train a linear regression model by minimizing a loss function as
discussed in `neural network tutorial <nn_Si>`__. But linear regression
model has analytic solutions, and thus we can train the model directly
by using this feature. This can be achieved by calling the ``fit()``
function of its calculator.

.. code-block:: python

    # fit the model
    calc.fit()
    
    
    # save model
    model.save("linear_model.pkl")


.. parsed-literal::

    2025-05-16 21:20:05.178 | INFO     | kliff.models.linear_regression:fit:42 - Finished fitting model "LinearRegression"


.. parsed-literal::

    Finished fitting model "LinearRegression"

