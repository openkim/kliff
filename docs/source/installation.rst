Installation
============
**KLIFF** is based on `KIM`_, either to fit physics-motivated potentials archived
in `OpenKIM`_ or to deploy the trained potential.
To get **KLIFF** work, we will need to install the `kim-api`_ and `kimpy`_ first.


kim-api
-------
A detailed installing steps are provided at `kim-api`_, so please refer the
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
-----
`kimpy`_ is on PyPI, so it can be installed simply by

.. code-block:: bash

    pip install kimpy


kliff
-----
After getting `kim-api`_ and `kimpy`_ to work, you can install **KLIFF** by

.. code-block:: bash

    git clone https://github.com/mjwen/kliff
    pip install ./kliff


.. _KIM: https://openkim.org
.. _OpenKIM: https://openkim.org
.. _kim-api: https://openkim.org/kim-api/
.. _kimpy: https://github.com/mjwen/kimpy
