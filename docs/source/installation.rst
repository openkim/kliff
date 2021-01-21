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

   $ conda intall -c conda-forge kliff

or

.. code-block:: bash

   $ pip install kliff

Alternatively, you can install from source:

.. code-block:: bash

    $ git clone https://github.com/mjwen/kliff
    $ pip install ./kliff


Optional
========

KLIFF is built on top of KIM to fit physics-motivated potentials archived on OpenKIM_.
To get KLIFF work with OpenKIM_ potential models, two other packages --- kim-api_ and
kimpy_ --- are needed.


kim-api
-------

kim-api_ can be installed via conda:

.. code-block:: bash

    $ conda install -c conda-forge kim-api

Other installation methods are provided at kim-api_; refer to the instructions there
for more information.

.. note::
    After installation, you can do ``$ kim-api-collections-management list``.
    If you see a list of directories where the KIM model drivers and models are
    placed, then you are good to go. Otherwise, you may forget to set up the
    ``PATH`` and bash completions, which can be achieved by (assuming you are
    using Bash): ``$ source path/to/the/kim/library/bin/kim-api-activate``. See
    the kim-api_ documentation for more information.


kimpy
-----
.. code-block:: bash

    $ conda-install -c conda-forge kimpy

or

.. code-block:: bash

    $ pip install kimpy


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
.. _kimpy: https://github.com/openkim/kimpy
