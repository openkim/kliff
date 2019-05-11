.. _installation:

============
Installation
============


KLIFF requires:

- Python_ 3.5 or newer.
- A C++ compiler that supports C++11.

KLIFF is built on top of KIM, either to fit physics-motivated potentials
archived on OpenKIM_ or to deploy the trained potential. To get KLIFF to work,
two prerequisite packages --- kim-api_ and kimpy_ --- need to be installed
first.


kim-api
=======
A detailed installing steps are provided at kim-api_, so please refer to the
instructions there to install.

.. note::
    After installation, you can do ``$ kim-api-collections-management list``.
    If you see a list of directories where the KIM model drivers and models are
    placed, then you are good to go. Otherwise, you may forget to set up the
    ``PATH`` and bash completions, which can be achieved by (assuming you are
    using Bash): ``$ source path/to/the/kim/library/bin/kim-api-activate``. See
    the kim-api_ documentation for more information.


kimpy
=====
kimpy_ is on PyPI and it can be installed by

.. code-block:: bash

    $ pip install kimpy


KLIFF
=====

After getting kim-api_ and kimpy_ to work, you can install KLIFF via

Package manager
---------------
.. code-block:: bash

   $ pip install kliff

From source
-----------
.. code-block:: bash

    $ git clone https://github.com/mjwen/kliff
    $ pip install ./kliff



Optional
========

KLIFF takes advantage of PyTorch_ to build neural network models and conduct the
training. So if you want to train neural network potentials, PyTorch_ needs to
be installed. Please follow the instructions given on the official PyTorch_
website to install it.


.. _Python: http://www.python.org
.. _PyTorch: https://pytorch.org
.. _OpenKIM: https://openkim.org
.. _kim-api: https://openkim.org/kim-api
.. _kimpy: https://github.com/openkim/kimpy
