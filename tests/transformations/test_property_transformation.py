import numpy as np

from kliff.dataset import Configuration, Dataset
from kliff.transforms.property_transforms import (
    NormalizedPropertyTransform,
    RMSMagnitudeNormalizePropertyTransform,
)


def test_property_transformations_scalar(test_data_dir):
    dataset = Dataset(
        test_data_dir / "configs" / "Si_4.xyz", parser="ase", energy_key="Energy"
    )
    energies = np.array(list(map(lambda x: x.energy, dataset.get_configs())))
    mean = energies.mean()
    std = energies.std()
    gaussian_pt = NormalizedPropertyTransform(property_key="energy")
    gaussian_pt(dataset.get_configs())
    energies_normalized = np.array(list(map(lambda x: x.energy, dataset.get_configs())))
    np.allclose((energies - mean) / std, energies_normalized)


def test_property_transformations_vector(test_data_dir):
    dataset = Dataset(
        test_data_dir / "configs" / "Si_4.xyz",
        parser="ase",
        energy_key="Energy",
        forces_key="force",
    )
    forces = list(map(lambda x: x.forces, dataset.get_configs()))
    forces = np.vstack(forces)
    mean_rms = np.sqrt(np.mean(np.sum(np.square(forces), 1)))
    rms_pt = RMSMagnitudeNormalizePropertyTransform(property_key="forces")
    rms_pt(dataset.get_configs())
    forces_normalized = list(map(lambda x: x.forces, dataset.get_configs()))
    forces_normalized = np.vstack(forces_normalized)
    np.allclose(forces / mean_rms, forces_normalized)
