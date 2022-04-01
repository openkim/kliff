import numpy as np

from kliff.models.parameter import Parameter
from kliff.models.parameter_transform import LogParameterTransform


def test_log_transform():

    a_val = [1.0, 2.0]
    b_val = [3.0, 4.0]

    a = Parameter(a_val)
    b = Parameter(b_val)
    params = {"a": a, "b": b}

    transformer = LogParameterTransform(param_names=["b"])

    forward = transformer.transform(params)
    assert np.allclose(forward["a"].value, a_val)
    assert np.allclose(forward["b"].value, np.log(b_val))

    inverse = transformer.inverse_transform(forward)
    assert np.allclose(inverse["a"].value, a_val)
    assert np.allclose(inverse["b"].value, b_val)
