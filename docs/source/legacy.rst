Legacy Module
=============

The legacy module contains the older versions of the KLIFF descriptors, neural network
training utilities, and associated functionalities.

.. tip::

    Although the module is named ``legacy`` it is not being deprecated and is still actively maintained.

In fact, this might be better solution for users who are looking for a more performant
but less flexible neural network potentials, and certain UQ functionalities.

The major difference between the legacy module training and the new module training is that
the legacy training method was more imperative, and the new training method is more declarative.
In the newer training method you specify the model, the loss function, and the optimizer,
and the training loop is handled by the KLIFF. KLIFF automatically loads the data, performs
the desired splits, and trains the model.
In the legacy training method, you write the python code to manually call the required
KLIFF modules your self.


DUNN vs TorchML
---------------

The biggest advantage of the legacy module is the DUNN model driver support, which is not
available in the new module. The DUNN model driver is a descriptor-based neural network
model driver, which is much more performant than the new module model drivers. This is because
the DUNN model driver uses Eigen3 based C++ backend for neural network evaluation, and analytical
derivatives. The TorchML model driver uses the ``libtorch`` backend, which is not as performant
but much more flexible. For appropriate use cases, the DUNN model driver can be orders of magnitude
faster than the TorchML model driver.

Below table illustrates the difference DUNN and TorchML model drivers:

+----------------+--------------------------------------+------------------------------------------+
|                | TorchML                              | DUNN                                     |
+================+======================================+==========================================+
| Backend        | libtorch                             | Eigen3 (C++)                             |
+----------------+--------------------------------------+------------------------------------------+
| Forces         | Automatic Differentiation            | Analytical Derivatives                   |
+----------------+--------------------------------------+------------------------------------------+
| Flexibility    | Any arbitrary torch model (e.g. GNN) | Just DNN with enumerated non-linearities |
+----------------+--------------------------------------+------------------------------------------+
| Performance    | Flexible but slower                  | Faster                                   |
+----------------+--------------------------------------+------------------------------------------+
| UQ             | Model dependent                      | Ensemble average                         |
+----------------+--------------------------------------+------------------------------------------+

So decide as per your requirements which model driver you want to use.