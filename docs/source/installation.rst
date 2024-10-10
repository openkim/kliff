.. _installation:

============
Installation
============


KLIFF requires:

- Python_ 3.6 or newer.
- A C++ compiler that supports C++11.

KLIFF
=====

The easiest way to install KLIFF is using a package manager, do either

.. code-block:: bash

   $ conda install -c conda-forge kliff

or

.. code-block:: bash

   $ pip install kliff

Alternatively, you can install from source:

.. code-block:: bash

    $ git clone https://github.com/openkim/kliff
    $ pip install ./kliff


Other dependencies
==================

KIM Models
----------

KLIFF is built on top of KIM to fit physics-motivated potentials archived on OpenKIM_.
To get KLIFF work with OpenKIM_ models, kim-api_ and
kimpy_, and openkim-models_ are needed.

The easiest way to install them is via conda:

.. code-block:: bash

    $ conda install -c conda-forge kim-api kimpy openkim-models

.. note::
    After installation, you can do ``$ kim-api-collections-management list``.
    If you see a list of directories where the KIM model drivers and models are
    placed, then you are good to go. Otherwise, you may forget to set up the
    ``PATH`` and bash completions, which can be achieved by (assuming you are
    using Bash): ``$ source path/to/the/kim/library/bin/kim-api-activate``. See
    the kim-api_ documentation for more information.

.. Warning::
    The conda approach should work for most systems, but not all (e.g. Mac with Apple
    Chip). Refer to https://openkim.org/doc/usage/obtaining-models for other installing instructions (e.g. from source).


PyTorch
-------

For machine learning potentials, KLIFF takes advantage of PyTorch_ to build neural
network models and conduct the training. So if you want to train neural network
potentials, PyTorch_ needs to be installed.
Please follow the instructions given on the official PyTorch_ website to install it.

PyTorch Geometric
-----------------

If you want to use the graph neural network potentials, you need to install PyTorch
Geometric. The installation instructions can be found on the official website of
Pytorch-geometric_. It is also advisable to use ``torch-scatter`` dependency for
the Pytorch-geometric package (installation instructions available on Pytorch-Geometric
website only).

PyTorch Lightning
-----------------
For using multi GPU trainer, please also install PyTorch Lightning. The installation
instructions can be found on the official website of Pytorch-lightning_.

Libdescriptor
-------------

For working with descriptor-based potentials, you need to install libdescriptor. The original
descriptor module now resides in ``legacy`` module of KLIFF. Libdescriptor can be installed using
conda:

.. code-block:: bash

    $ conda install -c conda-forge -c ipcamit libdescriptor

For more information on libdescriptor, please refer to the `libdescriptor documentation`_.

.. _Python: http://www.python.org
.. _PyTorch: https://pytorch.org
.. _OpenKIM: https://openkim.org
.. _kim-api: https://openkim.org/kim-api
.. _openkim-models: https://openkim.org/doc/usage/obtaining-models
.. _kimpy: https://github.com/openkim/kimpy
.. _Pytorch-geometric: https://pytorch-geometric.readthedocs.io
.. _Pytorch-lightning: https://lightning.ai/docs/pytorch/stable
.. _libdescriptor documentation: https://libdescriptor.readthedocs.io/en/latest/
