Contributing guide
==================

Code style
----------

- KLIFF uses black_ to format the code. It adopts the default style of black_
  with only one exception: we use line length 90 instead of of the black_default (88).
  To format the code, one can use:

  .. code-block:: bash

    black --line-length 90 <filename>.py

  Supply ``--skip-string-normalization`` to ``black`` to avoid changing single
  quote to double quote.

- The docstring of **KLIFF** follows the `numpy` style, which can be found at numpydoc_.


Docs
----

- We use sphinx-gallery_ to generate the tutorials. The source file should be
  placed in `kliff/examples` and the file name should start with ``example_``.

- To generate the docs (including the tutorials), do:

  .. code-block:: bash

    $ cd kliff/docs
    $ make html

  The generated docs will be at ``kliff/docs/build/html/index.html``.

- The above commands will not only parse the docstring in the tutorials, but also
  run the codes in the tutorials. Running the codes may take a long time. So, if
  you just want to generate the docs, do:

  .. code-block:: bash

    $ cd kliff/docs
    $ make html-notutorial

  This will not run the code in the tutorials.


Below is Mingjian's personal notes on how to generate API docs. Typically, you
will not need it.

To generate the API docs for a specific module, do:

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
.. _black: https://black.readthedocs.io/en/stable/
.. _sphinx-gallery: https://sphinx-gallery.github.io
