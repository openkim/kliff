.. _installation:

============
Installation
============

KLIFF requires:

- Python_ 3.9
- A C++ compiler that supports C++11.
- Linux or MacOS (Apple Silicon) system (Intel Macs are not supported yet).

.. note::
    For the instructions below, it is assumed that you have a working conda environment
    installed on your system. If not, please install conda first (`link <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_)

Create a new conda environment (recommended) for installing all dependencies.

.. code-block:: bash

    conda create -n kliff
    conda activate kliff
    conda install python=3.9 -c conda-forge

Additionally you might need to install basic compilers and build essentials, it is highly system
specific, and chances are that all the required dependencies are present in your system.
These are mostly used for building C++-Python extensions, and similar programs like KIMPY.
If you want to be sure, you can also install these fundamental dependencies via conda



.. tip::
    Conda vs pip: During the instructions you will see switching between conda and pip
    for installing different dependencies. Conda is more feature rich installer that can also install
    non-python packages (like cmake, make, and compilers, etc) while pip can only install python
    packages. We have used Conda where ever possible for non-python dependencies, while
    using pip for Python dependencies.



KLIFF
=====

KLIFF is built on top of KIM to fit physics-motivated potentials archived on OpenKIM_.
To get KLIFF work with OpenKIM_ models, kim-api_ and
kimpy_.

It is advisable to install kim-api and kimpy via conda before installing KLIFF. This
also ensures that the correct compilers and other tools are installed.

.. code-block:: bash

    conda install kim-api=2.3 kimpy -c conda-forge

KLIFF can be installed in three ways:

1. Installing KLIFF from source


.. code-block:: bash

    git clone https://github.com/openkim/kliff
    pip install ./kliff

2. Installing KLIFF from PyPI


.. code-block:: bash

    pip install kliff


3. Installing KLIFF from conda


.. code-block:: bash

    conda install kliff -c conda-forge


Other dependencies
==================

KIM Models
----------

.. note::

    Optional `openkim-models` can be installed via conda as well:
    ``conda install openkim-models -c conda-forge``

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
    For older Apple Intel Macs, highest version of torch available is 2.2, so replace 2.4 with 2.2 in that case.

.. code-block:: bash

    pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu

.. warning::

    Pytorch < 2.3 works with numpy < 2.0, so if you see warning/error message like,
    ``"A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.2 as it may crash."``
    you need to either i) downgrade numpy < 2.0 (``pip install "numpy<2.0"``) or ii) update torch
    to version >= 2.3.

Graph Neural Networks
---------------------

If you want to use the graph neural network potentials, you need to install PyTorch
Geometric, and Pytorch Lightning. The detailed installation instructions can be found on the official website of
Pytorch-geometric_. It is also advisable to use ``torch-scatter`` dependency for
the Pytorch-geometric package (installation instructions available on Pytorch-Geometric
website only).

For using multi GPU trainer, please also install PyTorch Lightning. The installation
instructions can be found on the official website of Pytorch-lightning_.

.. warning::

    Please ensure to match the correct version of torch scatter with pytorch.

For most common systems, the following commands will be enough, (``tensorboard`` is used for logging).

.. code-block:: bash

    pip install torch_geometric
    pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
    pip install lightning tensorboard tensorboardX


Libdescriptor
-------------

Libdescriptor is a auto-differentiated descriptor library for unified Python-C++ API.
It is used by the TorchML model driver for running Descriptor-based neural networks.
This is an optional dependency needed if user want to train descriptor based neural networks.

For working with descriptor-based potentials, you need to install libdescriptor. The original
descriptor module now resides in ``legacy`` module of KLIFF, and user should decide on which descriptor
module they want to use, based on their requirements (for detailed comparisons, see ).
Libdescriptor can be installed using
conda:

.. code-block:: bash

    conda install libdescriptor -c conda-forge -c ipcamit

Above command should install ``libdescriptor`` on both Linux and Apple Silicon Mac. For
any other unsupported system, either you can use the ``legacy`` descriptor interface of
KLIFF for now, or install it from the source (see detailed `documentation <https://libdescriptor.readthedocs.io/en/latest/>`_).

TorchML Model driver
--------------------

ML models (most importantly graph neural networks) need the latest `TorchML <https://openkim.org/id/TorchML__MD_173118614730_001/>`_ model driver
to run with KIM-API. The installation details for the TorchML model driver are provided in
the :ref:`advanced section <lammps>` .

Detailed instructions on how to port your existing models to TorchML can be found
`here <https://kim-torchml-port.readthedocs.io/en/latest/introduction.html>`_.

.. tip::

    You can also use `Klay <https://klay.readthedocs.io>`_ (KLIFF Layers) sister python
    package to generate ML models, which are inherently compatible with KLIFF and OpenKIM.

Errors
------

.. note::
    If you encounter any errors during the installation, or training, please refer to the bottom
    of the tutorial notebook. Chances are that the error is already documented there. Otherwise
    please raise an issue on the github repository.


1. Incompatible architecture error on Apple Silicon Mac


.. code-block::

    '/Users/amitgupta/miniconda3/envs/kliff/lib/python3.9/site-packages/kliff/neighbor/neighlist.cpython-39-darwin.so'
    (mach-o file, but is an incompatible architecture (have 'x86_64', need
    'arm64e' or 'arm64'))


If you get the following error on Apple Silicon Mac, it means that the package is not
compiled for Arm64, rather it is compiled for x86_64. This points to an underlying issue
with your conda environment, and you may need to reinstall the package.

Easiest first attempt to fix it is to recreate the conda environment and reinstall the package
from the top. For more detailed instructions, please refer to `stackoverflow issue <https://stackoverflow.com/questions/72308682/mach-o-file-but-is-an-incompatible-architecture-have-x86-64-need-arm64e>`_.


.. _Python: http://www.python.org
.. _PyTorch: https://pytorch.org
.. _OpenKIM: https://openkim.org
.. _kim-api: https://openkim.org/kim-api
.. _openkim-models: https://openkim.org/doc/usage/obtaining-models
.. _kimpy: https://github.com/openkim/kimpy
.. _Pytorch-geometric: https://pytorch-geometric.readthedocs.io
.. _Pytorch-lightning: https://lightning.ai/docs/pytorch/stable
.. _libdescriptor documentation: https://libdescriptor.readthedocs.io/en/latest/
