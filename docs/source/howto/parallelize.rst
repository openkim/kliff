.. _run_in_parallel:

====================
Run in parallel mode
====================

KLIFF supports parallelization over data. It can be run on shared-memory multicore
desktop machines as well as HPCs composed of multiple standalone machines
connected by a network.

Physics-based models
====================

We implement two parallelization schemes for physics-based models. The first is
suitable to be used on a desktop machine and the second is targed for HPCs.

multiprocessing
---------------
This scheme uses the ``multiprocessing`` module of Python and it can only be used on
shared-memory desktop (laptop). It's straightforward to use: simply set ``nprocs``
to the number of processes you want to use when instantiate :class:`~kliff.loss.Loss`.
For example,

.. code-block:: python

    calc = ...  # create calculator
    loss = Loss(calc, ..., nprocs=2)
    loss.minimize(method='L-BFGS-B')

.. seealso::
    See :ref:`tut_kim_sw` for a full example.


MPI
---
The MPI scheme is targeted for HPCs (of course, it can be used on desktops) and we
use the mpi4py_ Python wrapper of MPI. mpi4py_ supports ``OpenMPI`` and ``MPICH``. Once
you have one of the two working, mpi4py_ can be installed by::

    $ pip install mpi4py

See the mpi4py_ package documentation for more information on how to install it.
Once it is successfully installed, we can run KLIFF in parallel. For example, for the
tutorial example :ref:`tut_kim_sw`, we can do::

    $ mpiexec  -np 2  python example_kim_SW_Si.py

.. note::
    When using this MPI scheme, the ``nprocs`` argument passed to
    :class:`~kliff.loss.Loss` is ignored.

.. note::
    We only parallelize the evaluation of the loss during the minimization. As a
    result, the other parts will be executed multiple times. For example, if
    :meth:`kliff.models.Model.echo_model_params` is used, the information of model
    parameters will be repeated multiple times. If this annoys you, you can let
    only of process ( say the rank 0  process) to do it by doing something like:

.. code-block:: python

    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        model.echo_model_params()


.. _mpi4py: https://mpi4py.readthedocs.io


Machine learning models
=======================

