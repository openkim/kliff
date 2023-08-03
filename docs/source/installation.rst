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


.. _Python: http://www.python.org
.. _PyTorch: https://pytorch.org
.. _OpenKIM: https://openkim.org
.. _kim-api: https://openkim.org/kim-api
.. _openkim-models: https://openkim.org/doc/usage/obtaining-models
.. _kimpy: https://github.com/openkim/kimpy
