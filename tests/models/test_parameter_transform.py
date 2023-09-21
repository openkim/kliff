import numpy as np

from kliff.models.parameter import Parameter
from kliff.transforms.parameter_transforms import LogParameterTransform


def test_log_transform():
    a_val = [1.0, 2.0]
    b_val = [3.0, 4.0]

    a: Parameter = Parameter(np.array(a_val))
    b: Parameter = Parameter(np.array(b_val))
    params = {"a": a, "b": b}

    transformer = LogParameterTransform()

    # forward transform
    params["a"].add_transform_(transformer)
    params["b"].add_transform_(transformer)
    # forward = transformer.transform(params)
    assert np.allclose(params["a"], np.log(a_val))
    assert np.allclose(params["b"], np.log(b_val))

    # inverse transform
    assert np.allclose(params["a"].get_numpy_array(), np.array(a_val))
    assert np.allclose(params["b"].get_numpy_array(), np.array(b_val))
