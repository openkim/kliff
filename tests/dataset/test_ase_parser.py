from pathlib import Path

import numpy as np
import pytest
from ase import io

from kliff.dataset import Dataset


def test_dataset_from_ase():
    """Test ASE parser reading from file using ASE parser"""

    # training set
    filename = Path(__file__).parents[1].joinpath("test_data/configs/Si_4.xyz")
    data = Dataset.from_ase(filename, energy_key="Energy", forces_key="force")
    configs = data.get_configs()

    # ASE is bit weird for stress. It expects stress to be a 3x3 matrix but returns a 6x1 vector
    # in Voigt notation.

    assert len(configs) == 4
    assert configs[0].species == ["Si" for _ in range(4)]
    assert configs[0].coords.shape == (4, 3)
    assert configs[0].energy == 123.45
    assert np.allclose(configs[0].stress, np.array([1.1, 5.5, 9.9, 8.8, 7.7, 4.4]))
    assert configs[1].forces.shape == (8, 3)

    # test slicing, get config 1 and 4
    data = Dataset.from_ase(
        filename, energy_key="Energy", forces_key="force", slices="::3"
    )
    configs = data.get_configs()
    assert len(configs) == 2
    assert np.allclose(configs[0].stress, np.array([1.1, 5.5, 9.9, 8.8, 7.7, 4.4]))
    assert np.allclose(configs[1].stress, np.array([9.9, 5.5, 1.1, 8.8, 7.7, 4.4]))


def test_dataset_add_from_ase():
    """Test adding configurations to dataset using ASE parser."""
    filename = Path(__file__).parents[1].joinpath("test_data/configs/Si_4.xyz")
    data = Dataset()
    data.add_from_ase(filename, energy_key="Energy", forces_key="force")
    configs = data.get_configs()

    assert len(configs) == 4
    assert configs[0].species == ["Si" for _ in range(4)]
    assert configs[0].coords.shape == (4, 3)
    assert configs[0].energy == 123.45
    assert np.allclose(configs[0].stress, np.array([1.1, 5.5, 9.9, 8.8, 7.7, 4.4]))
    assert configs[1].forces.shape == (8, 3)


def test_dataset_from_ase_atoms_list():
    """Test ASE parser reading from file using ASE atoms list."""
    filename = Path(__file__).parents[1].joinpath("test_data/configs/Si_4.xyz")
    atoms = io.read(filename, index=":")
    data = Dataset.from_ase(
        ase_atoms_list=atoms, energy_key="Energy", forces_key="force"
    )
    configs = data.get_configs()

    assert len(configs) == 4
    assert configs[0].species == ["Si" for _ in range(4)]
    assert configs[0].coords.shape == (4, 3)
    assert configs[0].energy == 123.45
    assert np.allclose(configs[0].stress, np.array([1.1, 5.5, 9.9, 8.8, 7.7, 4.4]))
    assert configs[1].forces.shape == (8, 3)


def test_dataset_add_from_ase_atoms_list():
    """Test adding configurations to dataset using ASE atoms list."""
    filename = Path(__file__).parents[1].joinpath("test_data/configs/Si_4.xyz")
    atoms = io.read(filename, index=":")
    data = Dataset()
    data.add_from_ase(ase_atoms_list=atoms, energy_key="Energy", forces_key="force")
    configs = data.get_configs()

    assert len(configs) == 4
    assert configs[0].species == ["Si" for _ in range(4)]
    assert configs[0].coords.shape == (4, 3)
    assert configs[0].energy == 123.45
    assert np.allclose(configs[0].stress, np.array([1.1, 5.5, 9.9, 8.8, 7.7, 4.4]))
    assert configs[1].forces.shape == (8, 3)


# def test_colabfit_parser():
# TODO: figure minimal colabfit example to run tests on
#    pass
