from pathlib import Path

import numpy as np

from kliff.dataset import Dataset
from kliff.dataset.weight import MagnitudeInverseWeight, Weight

np.random.seed(2022)


def test_base_weight():
    # Base weight class
    # Pick random weights
    cw, ew, fw, sw = np.random.uniform(low=0.0, high=1.1, size=4)
    # Instantiate base weight class
    weight = Weight(
        config_weight=cw, energy_weight=ew, forces_weight=fw, stress_weight=sw
    )

    assert weight.config_weight == cw
    assert weight.energy_weight == ew
    assert weight.forces_weight == fw
    assert weight.stress_weight == sw


def test_magnitude_inverse_weight():
    # Inverse magnitude weight
    # Pick random weight parameters
    c1_e, c2_e = np.random.uniform(low=0.0, high=1.0, size=2)
    c1_f, c2_f = np.random.uniform(low=0.0, high=1.0, size=2)
    c1_s, c2_s = np.random.uniform(low=0.0, high=1.0, size=2)
    # Instantiate the weight class
    weight = MagnitudeInverseWeight(
        weight_params={
            "energy_weight_params": [c1_e, c2_e],
            "forces_weight_params": [c1_f, c2_f],
            "stress_weight_params": [c1_s, c2_s],
        }
    )

    # Load the dataset and set the weight
    # We choose the following data set because it has energy, forces, and stress data
    dataset_path = Path(__file__).parents[1].joinpath("configs_extxyz/Si.xyz")
    tset = Dataset(dataset_path, weight=weight)
    configs = tset.get_configs()

    # Check if my implementation works. I do this by comparing it with my manual
    # calculation.
    for conf in configs:
        # Assert energy weight
        ew_implement = conf.weight.energy_weight  # Retrieve weight
        # Compute weight manually
        energy_norm = np.abs(conf.energy)
        ew_manual = _compute_magnitude_inverse_weight(c1_e, c2_e, energy_norm)

        assert isinstance(ew_manual, float)  # The weight is a float
        # Check the values
        assert np.allclose(ew_implement, ew_manual, rtol=1e-12, atol=1e-12)

        # Assert forces weight
        fw_implement = conf.weight.forces_weight  # Retrieve weight
        # Compute weight manually
        forces_norm = np.repeat(np.linalg.norm(conf.forces, axis=1), 3)
        fw_manual = _compute_magnitude_inverse_weight(c1_f, c2_f, forces_norm)

        assert isinstance(fw_manual, np.ndarray)  # The weight is an array
        assert len(fw_manual) == len(conf.forces.flatten())  # Check the length
        # Check the values
        assert np.allclose(fw_implement, fw_manual, rtol=1e-12, atol=1e-12)

        # Assert stress weight
        sw_implement = conf.weight.stress_weight  # Retrieve weight
        # Compute weight manually
        stress_norm = np.sqrt(
            np.linalg.norm(conf.stress) ** 2 + np.linalg.norm(conf.stress[3:]) ** 2
        )
        sw_manual = _compute_magnitude_inverse_weight(c1_s, c2_s, stress_norm)

        assert isinstance(sw_manual, float)  # The weight is a float
        # Check the values
        assert np.allclose(sw_implement, sw_manual, rtol=1e-12, atol=1e-12)


def _compute_magnitude_inverse_weight(c1, c2, norm):
    """Compute the inverse magnitude weight, given the weight parameters and the
    magnitude of property of interest.
    """
    sigma = np.sqrt(c1**2 + (c2 * norm) ** 2)
    return 1 / sigma
