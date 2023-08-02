Contributing guide
==================

Code style
----------

- KLIFF uses isort_ and black_ to format the code. To format the code, install
  ``pre-commit`` and then do:

  .. code-block:: bash

    pre-commit run --all-files --show-diff-on-failure

- The docstring of **KLIFF** follows the `Google` style, which can be found at googledoc_.


Docs
----

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


Tutorials
---------
If you have great tutorials, please write it in Jupyter notebook and place them in the `kliff/docs/tutorials` directory. Then update the `kliff/docs/tutorials.rst` file to include the new tutorials. After this, the tutorials will be automatically built and included in the docs.


.. note::
    The `kliff/docs/apidoc/kliff.rst` is referenced in `index.rst`, serving as the entry
    for all docs.

.. _googledoc: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
.. _black: https://black.readthedocs.io/en/stable/
.. _isort: https://pycqa.github.io/isort/
.. _sphinx-gallery: https://sphinx-gallery.github.io
