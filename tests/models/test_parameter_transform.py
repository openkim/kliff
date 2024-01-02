import numpy as np

from kliff.models.parameter import Parameter
from kliff.transforms.parameter_transforms import LogParameterTransform


def test_log_transform():
    a_val = np.array([1.0, 2.0])
    b_val = np.array([3.0, 4.0])

    a = Parameter(a_val)
    b = Parameter(b_val)
    params = {"a": a, "b": b}

    transformer = LogParameterTransform()

    # forward transform
    params["a"].add_transform(transformer)
    params["b"].add_transform(transformer)

    # forward = transformer.transform(params)
    assert np.allclose(params["a"], np.log(a_val))
    assert np.allclose(params["b"], np.log(b_val))

    # inverse = transformer.inverse_transform(forward)
    assert np.allclose(params["a"].get_numpy_array_model_space(), a_val)
    assert np.allclose(params["b"].get_numpy_array_model_space(), b_val)
