.. _installation:

============
Installation
============


**kliff** needs Python_ 3.5 or newer.


**kliff** is based on KIM_, either to fit physics-motivated potentials archived
on OpenKIM_ or to deploy the trained potential. To get **kliff** to work, the
prerequisites kim-api_ and kimpy_ need to be installed first.


kim-api
=======
A detailed installing steps are provided at kim-api_, so please refer the
instructions there to install.

.. note::
    After installation, you can do
    ``$ kim-api-v2-collections-management list``.
    If you see a list of directories where the KIM model drivers and models are
    placed, then you are good to go.
    Otherwise, you may forget to set up the ``PATH`` and bash completions, which
    can be achieved by (assuming you are using Bash):
    ``$ source path/to/the/kim/library/bin/kim-api-v2-activate``


kimpy
=====
kimpy_ is on PyPI, which can be installed simply by

.. code-block:: bash

    pip install kimpy


kliff
=====

After getting kim-api_ and kimpy_ to work, you can install **kliff** via

Package manager
---------------
coming soon...

From source
-----------
.. code-block:: bash

    git clone https://github.com/mjwen/kliff
    pip install ./kliff



Optional
========

**kliff** takes advantage of PyTorch_ to build neural network models and conduct the
training. So if you want to train neural network potentials, PyTorch_ needs to be
installed. Please follow the instructions given on the official PyTorch_ website to
install it.


.. _Python: http://www.python.org
.. _PyTorch: https://pytorch.org
.. _KIM: https://openkim.org
.. _OpenKIM: https://openkim.org
.. _kim-api: https://openkim.org/kim-api
.. _kimpy: https://github.com/mjwen/kimpy
