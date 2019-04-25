import multiprocessing as mp
import random
import numpy as np


def parmap1(f, X, *args, tuple_X=False, nprocs=mp.cpu_count()):
    """Parallelism over data.

    This function mimics ``multiprocessing.Pool.map`` to allow extra arguments
    to be used for the function ``f``.

    Parameters
    ----------
    f: function
        The function that operates on the data.

    X: list
        Data to be parallelized.

    args: args
        Extra positional arguments needed by the function ``f``.

    tuple_X: bool
        This depends on ``X``. It should be set to ``True`` if multiple
        arguments are parallelized and set to ``False`` if only one argument is
        parallelized. See ``Example`` below.

    nprocs: int
        Number of processors to use.

    Return
    ------
    list
        A list of results, corresponding to ``X``.

    Note
    ----
    The data is put into a job queue, a worker process gets a piece of the data
    to work on, the worker pushes the result back to the manager through another
    queue, and then get another piece of data until the job queue is empty.
    So, in principle, there will not be idle worker and it should be faster than
    :meth:`kliff.parallel.parmap2`.

    Warning
    -------
    This is implemented using ``multiprocessing.Queue``, which requires the
    funciton ``f`` to be picklable. If it is not the case (e.g. use KIM library
    functions), use :meth:`kliff.parallel.parmap2` that is based on
    ``multiprocessing.Pipe``.


    Example
    -------
    >>> def func(x, y, z=1):
    >>>     return x+y+z
    >>> X = range(3)
    >>> Y = range(3)
    >>> parmap1(func, X, 1, nprocs=2)  # [2,3,4]
    >>> parmap1(func, X, 1, 1, nprocs=2)  # [2,3,4]
    >>> parmap1(func, zip(X, Y), tuple_X=True, nprocs=2)  # [1,3,5]
    >>> parmap1(func, zip(X, Y), 1, tuple_X=True, nprocs=2)  # [1,3,5]
    """

    q_in = mp.Queue(nprocs)
    q_out = mp.Queue()

    processes = []
    for _ in range(nprocs):
        p = mp.Process(target=_func1, args=(f, q_in, q_out))
        p.daemon = True
        p.start()
        processes.append(p)

    N = 0
    for i, x in enumerate(X):
        N += 1
        if tuple_X:
            ix = (i, *x)
        else:
            ix = (i, x)
        q_in.put((ix, args))

    [q_in.put((None, None)) for _ in range(nprocs)]
    results = [q_out.get() for _ in range(N)]
    [p.join() for p in processes]

    return [r for i, r in sorted(results)]


def _func1(f, q_in, q_out):
    while True:
        ix, args = q_in.get()
        if ix is None:
            break
        i = ix[0]
        x = ix[1:]
        y = f(*x, *args)
        q_out.put((i, y))


def parmap2(f, X, *args, tuple_X=False, nprocs=mp.cpu_count()):
    """Parallelism over data.

    This is to mimic ``multiprocessing.Pool.map``, which requires the function
    ``f`` to be picklable. This function does not have this restriction and
    allows extra arguments to be used for the function ``f``.

    Parameters
    ----------
    f: function
        The function that operates on the data.

    X: list
        Data to be parallelized.

    args: args
        Extra positional arguments needed by the function ``f``.

    tuple_X: bool
        This depends on ``X``. It should be set to ``True`` if multiple
        arguments are parallelized and set to ``False`` if only one argument is
        parallelized. See ``Example`` below.

    nprocs: int
        Number of processors to use.

    Return
    ------
    list
        A list of results, corresponding to ``X``.

    Note
    ----
    This function is implemented using ``multiprocessing.Pipe``. The data is
    subdivided into ``nprocs`` groups and then each group of data is distributed
    to a process. The results from each group are then assembled together.
    The data is shuffled to balance the load in each process.
    See :meth:`kliff.parallel.parmap1` for another implementation that uses
    ``multiprocessing.Queue``.

    Example
    -------
    >>> def func(x, y, z=1):
    >>>     return x+y+z
    >>> X = range(3)
    >>> Y = range(3)
    >>> parmap2(func, X, 1, nprocs=2)  # [2,3,4]
    >>> parmap2(func, X, 1, 1, nprocs=2)  # [2,3,4]
    >>> parmap2(func, zip(X, Y), tuple_X=True, nprocs=2)  # [1,3,5]
    >>> parmap2(func, zip(X, Y), 1, tuple_X=True, nprocs=2)  # [1,3,5]
    """

    # shuffle and divide into nprocs equally-numbered parts
    # TODO this could be improved by schedule job according to number of atoms
    if tuple_X:
        pairs = [(i, *x) for i, x in enumerate(X)]  # to make array_split work
    else:
        pairs = [(i, x) for i, x in enumerate(X)]
    random.shuffle(pairs)
    groups = np.array_split(pairs, nprocs)

    processes = []
    managers = []
    for i in range(nprocs):
        manager_end, worker_end = mp.Pipe(duplex=False)
        p = mp.Process(target=_func3, args=(f, groups[i], args, worker_end))
        p.daemon = True
        p.start()
        processes.append(p)
        managers.append(manager_end)
    results = []
    for m in managers:
        results.extend(m.recv())
    for p in processes:
        p.join()

    return [r for i, r in sorted(results)]


def _func3(f, iX, args, worker_end):
    results = []
    for ix in iX:
        i = ix[0]
        x = ix[1:]
        results.append((i, f(*x, *args)))
    worker_end.send(results)
