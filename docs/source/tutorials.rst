.. _tutorials:

=========
Tutorials
=========

This section provides two example to use **kliff** to train a physics-motivated
Stillinger-Weber potential for silicon, and a neural network potential for graphite.
Before getting started, make sure that **kliff** is successfully installed as
discussed in :ref:`installation`.


Physics-motivated potential
===========================


Download the :download:`Si training set <https://raw.githubusercontent.com/mjwen/kliff/pytorch/examples/Si_training_set.tar.gz>`
and then extract the tarball by: ``$ tar xzf Si_training_set.tar.gz``.
Or, if you prefer, you can put the following snippet in your python code to
download and extract the tarball automatically:

.. code-block:: python

    import requests
    import tarfile

    tarball_name = 'Si_training_set.tar.gz'
    url = 'https://raw.githubusercontent.com/mjwen/kliff/pytorch/examples/{}'.format(tarball_name)
    r = requests.get(url)
    with open(tarball_name, 'wb') as f:
        f.write(r.content)
    tarball = tarfile.open(tarball_name)
    tarball.extractall()


Neural network potential
========================


