.. _install_model:

===================
Install a KIM model
===================


Install a model from OpenKIM website
====================================

The ``kim-api-collections-management`` command line tool from the kim-api_ makes
it easy to install a model archived on the OpenKIM_ website. You can do::

    $ kim-api-collections-management install user <model_name>

to automatically download a model from the OpenKIM_ website, and install it to the
``User Collection``. Just replace ``<model_name>`` with the ``Extended KIM ID``
of the model you want to install as listed on OpenKIM_. To see that the model has
been successfully installed, do::

    $ kim-api-collections-management list

.. seealso::
    A model can be installed into a different collection other than the ``User
    Collection`` specified by ``user``. You can use ``kim-api-collections-management``
    to remove and reinstall models. See the kim-api_ documentation for more
    information.


Install a KLIFF-trained model
=============================

As discussed in :ref:`tut_kim_sw` and :ref:`tut_nn`, you can write a trained
model to a KIM potential that is compatible with the kim-api_ by:

.. code-block:: python

    path = './kliff_trained_model'
    model.write_kim_model(path)

Which writes to the current working directory a directory named
``kliff_trained_model``.

.. note::
    The ``path`` argument is optional, and KLIFF automatically generates a
    ``path`` if it is ``None``.

To install the local ``kliff_trained_model``, do::

    $ kim-api-collections-management install user kliff_trained_model

which installs the model into the `User Collection` of the kim-api_, and, of course,
you can see the installed model by::

    $ kim-api-collections-management list

The installed model can then be used with simulation codes like LAMMPS_, ASE_,
and GULP_ via the kim-api_.


.. _OpenKIM: https://openkim.org
.. _kim-api: https://openkim.org/kim-api/
.. _LAMMPS: https://lammps.sandia.gov
.. _ASE: https://wiki.fysik.dtu.dk/ase/
.. _GULP: http://gulp.curtin.edu.au/gulp/
