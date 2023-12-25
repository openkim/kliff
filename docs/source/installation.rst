.. _installation:

============
Installation
============


KLIFF can be installed via package managers (conda or pip) or from source.

Conda
-----
This recommended way to install KLIFF is via conda. You can install it by:

.. code-block:: bash

   $ conda create --name kliff_env
   $ conda activate kliff_env
   $ conda install -c conda-forge kliff

Alternatively, you can install using pip:

.. code-block:: bash

   $ pip install kliff

or from source:

.. code-block:: bash

    $ git clone https://github.com/openkim/kliff
    $ pip install ./kliff


Other dependencies
==================

KIM API and kimpy
-----------------

KLIFF requires kim-api_ and kimpy_ to be installed. If you install KLIFF via conda as described above, these two packages are installed automatically, and you are good to go.
Otherwise, you will need to install kim-api_ and kimpy_ before installing KLIFF.
Of course, you can first install them using conda ``$ conda install -c conda-forge kim-api kimpy`` and then install KLIFF using pip or from source. Alternatively, you can install them from source
as well, and see their documentation for more information.


PyTorch
-------

For machine learning potentials, KLIFF takes advantage of PyTorch_ to build neural
network models and conduct the training. So if you want to train neural network
potentials, PyTorch_ needs to be installed.
Please follow the instructions given on the official PyTorch_ website to install it.


KIM Models
----------

If you are interested in training physics-based models that are avaialbe from OpenKIM_,
you will need to install the KIM models that you want to use. After kim-api_ is installed, you can do ``$ kim-api-collections-management list`` to see the list of installed KIM models.
You can also install the models you want by ``$ kim-api-collections-management install <model-name>``. See the kim-api_ documentation for more information.

If you see a list of directories where the KIM model drivers and models are placed, then you are good to go.
Otherwise, you may forget to set up the ``PATH`` and bash completions, which can be achieved by (assuming you are using Bash): ``$ source path/to/the/kim/library/bin/kim-api-activate``.
See the kim-api_ documentation for more information.



.. _Python: http://www.python.org
.. _PyTorch: https://pytorch.org
.. _OpenKIM: https://openkim.org
.. _kim-api: https://openkim.org/kim-api
.. _kimpy: https://github.com/openkim/kimpy
