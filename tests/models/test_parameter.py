import numpy as np
import warnings
from collections import OrderedDict
from kliff.models.parameter import Parameter
from kliff.models.parameter import FittingParameter
from kliff.models.parameter import ParameterError


def test_parameter():

    # scalar
    try:
        p = Parameter(1.1)
    except ParameterError:
        pass

    # 2D array
    try:
        p = Parameter([[1.1]])
    except ParameterError:
        pass

    p = Parameter([2.2, 3.3])
    assert np.allclose(p.get_value(), [2.2, 3.3])
    assert p.get_size() == 2
    assert p.get_dtype() == 'double'
    assert p.get_description() == None

    p.set_value([4.4])
    assert np.allclose(p.get_value(), [4.4])


def create_all_possible_input(v):
    inp = []
    for ud in ['default', v + 0.01]:
        inp.append([ud])
        inp.append([ud, 'fix'])
        for lb in [None, v - 0.1]:
            for ub in [None, v + 0.1]:
                inp.append([ud, lb, ub])
    return inp


def test_fitting_parameter():

    # model params
    mp = OrderedDict()
    p1 = [1.1]
    p2 = [2.2, 3.3]
    names = ['p1', 'p2']
    mp[names[0]] = Parameter(p1)
    mp[names[1]] = Parameter(p2)
    fp = FittingParameter(mp)

    # fitting params
    with warnings.catch_warnings():  # context manager to ignore warning
        warnings.simplefilter('ignore')

        v0 = p1[0]
        inp0 = create_all_possible_input(v0)
        v1 = p2[0]
        inp1 = create_all_possible_input(v1)
        v2 = p2[1]
        inp2 = create_all_possible_input(v2)
        for i in inp0:
            for j in inp1:
                for k in inp2:
                    fp.set(p1=[i])
                    fp.set(p2=[j, k])

    assert np.allclose(fp.get_opt_params(), [1.11, 2.21, 3.31])

    # interface to optimizer
    # restore to default parameter values
    x0 = np.concatenate((p1, p2))
    fp.update_params(x0)
    assert np.allclose(fp.get_opt_params(), x0)

    assert fp.get_number_of_opt_params() == 3

    for i in range(3):
        v, p, c = fp.get_opt_param_value_and_indices(i)
        assert v == x0[i]
        if i == 0:
            assert p == 0
            assert c == 0
        else:
            assert p == 1
            assert c == (i - 1) % 2

    bounds = [[i - 0.1, i + 0.1] for i in x0]
    assert np.allclose(fp.get_opt_params_bounds(), bounds)

    # interface to calculator
    nms = fp.get_names()
    for i, j in zip(nms, names):
        assert i == j

    for i, nm in enumerate(nms):
        assert fp.get_size(nm) == mp[nm].size
        assert np.allclose(fp.get_value(nm), mp[nm].value)
        assert np.allclose(fp.get_lower_bound(nm), [v - 0.1 for v in mp[nm].value])
        assert np.allclose(fp.get_upper_bound(nm), [v + 0.1 for v in mp[nm].value])
        fix = fp.get_fix(nm)
        for x in fix:
            assert not x


if __name__ == '__main__':
    test_parameter()
    test_fitting_parameter()
