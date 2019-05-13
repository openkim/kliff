import numpy as np
from kliff.parallel import parmap1, parmap2


def func(x, y, z=1):
    return x + y + z


def test_main():
    X = range(3)
    Y = range(3)
    Xp2 = [x + 2 for x in X]
    XpYp1 = [x + y + 1 for x, y in zip(X, Y)]

    results = parmap1(func, X, 1, nprocs=2)
    assert np.array_equal(results, Xp2)

    results = parmap1(func, X, 1, 1, nprocs=2)
    assert np.array_equal(results, Xp2)

    results = parmap1(func, zip(X, Y), nprocs=2, tuple_X=True)
    assert np.array_equal(results, XpYp1)

    results = parmap1(func, zip(X, Y), 1, nprocs=2, tuple_X=True)
    assert np.array_equal(results, XpYp1)

    results = parmap2(func, X, 1, nprocs=2)
    assert np.array_equal(results, Xp2)

    results = parmap2(func, X, 1, 1, nprocs=2)
    assert np.array_equal(results, Xp2)

    results = parmap2(func, zip(X, Y), nprocs=2, tuple_X=True)
    assert np.array_equal(results, XpYp1)

    results = parmap2(func, zip(X, Y), 1, nprocs=2, tuple_X=True)
    assert np.array_equal(results, XpYp1)


if __name__ == '__main__':
    test_main()
