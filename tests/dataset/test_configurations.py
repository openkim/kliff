from pathlib import Path

import numpy as np
import pytest
from ase import Atoms, build, io

from kliff.dataset import Configuration
from kliff.dataset.configuration import ConfigurationError
from kliff.utils import stress_to_voigt


def test_configuration_from_ase():
    """Test initializing Configuration from ASE atoms object"""

    filename = Path(__file__).parents[1].joinpath("test_data/configs/Si_4.xyz")
    atoms = io.read(filename, index=":")
    config = Configuration.from_ase_atoms(
        atoms[0], energy_key="Energy", forces_key="force"
    )

    assert config.species == ["Si" for _ in range(4)]
    assert config.coords.shape == (4, 3)
    assert config.energy == 123.45
    assert np.allclose(config.stress, np.array([1.1, 5.5, 9.9, 8.8, 7.7, 4.4]))
    assert config.forces.shape == (4, 3)
    assert config.stress.shape == (6,)


def test_configuration_to_ase():
    """Test converting Configuration to ASE atoms object"""

    filename = Path(__file__).parents[1].joinpath("test_data/configs/Si_4.xyz")
    atoms: Atoms = io.read(filename, index=":")
    config = Configuration.from_ase_atoms(
        atoms[0], energy_key="Energy", forces_key="force"
    )

    atoms = config.to_ase_atoms()
    assert np.allclose(atoms.get_positions(), config.coords)
    assert atoms.get_potential_energy() == config.energy
    assert np.allclose(atoms.get_forces(), config.forces)
    assert np.allclose(stress_to_voigt(atoms.get_stress(voigt=False)), config.stress)


def test_configuration_from_file():
    """Test initializing Configuration from file"""

    filename = (
        Path(__file__).parents[1].joinpath("test_data/configs/Si_4/Si_T300_step_0.xyz")
    )
    config = Configuration.from_file(filename)

    assert config.species == ["Si" for _ in range(64)]
    assert config.coords.shape == (64, 3)
    assert config.energy == 0.0
    assert config.forces.shape == (64, 3)
    # stress should raise exception
    with pytest.raises(ConfigurationError):
        stress = config.stress


def test_configuration_from_ase_bulk():
    """Test initializing Configuration from ASE build.bulk function"""
    bulk_config = build.bulk(name="Si", a=5.44, crystalstructure="diamond")
    config = Configuration.from_ase_atoms(bulk_config)
    config_direct = Configuration.bulk(name="Si", a=5.44, crystalstructure="diamond")

    assert config.species == config_direct.species
    assert np.allclose(config.coords, config_direct.coords)
    assert np.allclose(config.cell, config_direct.cell)
    assert np.allclose(config.PBC, config_direct.PBC)


def test_supercell():
    """Test creation of supercell"""
    conf = Configuration.bulk(name="Si", a=5.44, crystalstructure="diamond")
    super_cell = conf.get_supercell(2, 2, 2)

    assert np.allclose(super_cell.cell, 2 * conf.cell)
    assert conf.get_num_atoms() * 8 == super_cell.get_num_atoms()
    assert np.allclose(conf.coords, super_cell.coords[: conf.coords.shape[0], :])
