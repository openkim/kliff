import numpy as np
from collections import OrderedDict
from kliff.calculators import Parameter
from kliff.calculators import FittingParameter
from kliff.calculators.calculator import ParameterError


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


def test_fitting_parameter():
    # model params
    p = OrderedDict()
    p['p1'] = Parameter([1.1, 2, 2])
    p['p2'] = Parameter([3.3])
    fp = FittingParameter(p)

    # fitting params


if __name__ == '__main__':
    test_parameter()
