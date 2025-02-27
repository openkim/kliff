.. _installation:

============
Installation
============

.. Warning::
    This is for installing the beta version of KLIFF (v1). If you can see it on main
    documentation, please inform me @ `gupta839 _at_ umn.edu`


KLIFF requires:

- Python_ 3.6 or newer.
- A C++ compiler that supports C++11.

KLIFF
=====

.. code-block:: bash

    $ git clone --branch kliff-master-v1 --single-branch https://github.com/ipcamit/kliff kliff-v1
    $ pip install ./kliff-v1


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

.. warning::
    Given below are instructions for CPU version of PyTorch 2.4. Which was the last tested version with KLIFF.
    Please check PyTorch documentation for more detailed install options and different architectures.

.. code-block:: bash
    $ pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu


PyTorch Geometric
-----------------

If you want to use the graph neural network potentials, you need to install PyTorch
Geometric. The installation instructions can be found on the official website of
Pytorch-geometric_. It is also advisable to use ``torch-scatter`` dependency for
the Pytorch-geometric package (installation instructions available on Pytorch-Geometric
website only).

.. warning::
    Please ensure to match correct version of torch scatter etc. with pytorch.


.. code-block:: bash
    $ pip install torch_geometric
    $ pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cpu.html

PyTorch Lightning
-----------------

.. important::
    This is an optional dependency needed if user want to train graph based neural networks.

For using multi GPU trainer, please also install PyTorch Lightning. The installation
instructions can be found on the official website of Pytorch-lightning_.

.. code-block:: bash
    $ pip install lightning


Libdescriptor
-------------

.. important::
    This is an optional dependency needed if user want to train descriptor based neural networks.

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
