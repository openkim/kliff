import numpy as np
import pytest

from kliff.utils import stress_to_tensor, stress_to_voigt


def test_stress_conversion():
    stress_tensor = np.random.rand(3, 3)
    stress_tensor = stress_tensor + stress_tensor.T  # make it symmetric
    stress_voigt = stress_to_voigt(stress_tensor)
    assert np.allclose(
        stress_voigt,
        np.array(
            [
                stress_tensor[0, 0],
                stress_tensor[1, 1],
                stress_tensor[2, 2],
                stress_tensor[1, 2],
                stress_tensor[0, 2],
                stress_tensor[0, 1],
            ]
        ),
    )
    print(stress_tensor, stress_to_tensor(stress_voigt))
    assert np.allclose(stress_tensor, stress_to_tensor(stress_voigt))
