Build Docs
==========

The docs of **KLIFF** follow the `numpy` style, which can be found at numpydoc_.
To generate the API docs for a specific module, you can do:

.. code-block:: bash

    sphinx-apidoc -f -o <TARGET> <SOURCE>

where `<TARGET>` should be a directory where you want to place the generated `.rst`
file, and `<SOURCE>` is path to your Python modules (should also be a directory).
For example, to generate docs for all the modules in `kliff`, you can run (from
the `kliff/docs` directory)

.. code-block:: bash

    sphinx-apidoc -f -o tmp ../kliff


After generating the docs for a module, make necessary modifications and then move
the `.rst` files in `tmp` to `kliff/docs/apidoc`.


.. note::
    The `kliff/docs/apidoc/kliff.rst` is referenced in `index.rst`, serving as the entry
    for all docs.

.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/format.html
