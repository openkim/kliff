import numpy as np
import pytest

from kliff.dataset.dataset import Configuration, Dataset, read_extxyz, write_extxyz


def test_read_write_extxyz(test_data_dir, tmp_dir):
    path = test_data_dir.joinpath("configs/MoS2/MoS2_energy_forces_stress.xyz")
    cell, species, coords, PBC, energy, forces, stress = read_extxyz(path)

    fname = "test.xyz"
    write_extxyz(
        fname,
        cell,
        species,
        coords,
        PBC,
        energy,
        forces,
        stress,
        bool_as_str=True,
    )
    cell1, species1, coords1, PBC1, energy1, forces1, stress1 = read_extxyz(fname)

    assert np.allclose(cell, cell1)
    assert species == species1
    assert np.allclose(coords, coords1)
    assert PBC == PBC1
    assert energy == energy1
    assert np.allclose(forces, forces1)
    assert stress == stress1


@pytest.mark.parametrize(
    "f,s,order",
    (
        [False, False, False],
        [True, False, False],
        [True, True, False],
        [True, True, True],
    ),
)
def test_configuration(test_data_dir, f, s, order, e=True):
    path = test_data_dir.joinpath("configs/MoS2")

    if e:
        fname = path / "MoS2_energy.xyz"
    if f:
        fname = path / "MoS2_energy_forces.xyz"
    if s:
        fname = path / "MoS2_energy_forces_stress.xyz"

    fname = fname.as_posix()

    config = Configuration.from_file(fname, file_format="xyz")
    if order:
        config.order_by_species()

    assert config.get_num_atoms() == 288
    assert np.allclose(
        config.cell,
        [
            [33.337151, 0.035285, 0.03087],
            [0.027151, 25.621674, -0.000664],
            [0.027093, 9e-05, 30.262626],
        ],
    )
    assert np.allclose(config.PBC, [1, 1, 1])

    if e:
        assert config.energy == -5.302666
    if s:
        assert np.allclose(config.stress, [1.1, 2.2, 3.3, 4.4, 5.5, 6.6])

    if order:
        ref_species = ["Mo", "Mo", "Mo", "Mo", "Mo", "Mo"]
        ref_coords = [
            [0.051823, -0.017150, 16.736001],
            [2.819495, 1.607633, 16.734550],
            [5.601008, 0.015314, 16.738646],
        ]
        ref_forces = [
            [-0.425324, 0.295866, -0.065479],
            [0.245043, -0.061658, 0.264104],
            [-0.164382, -0.103679, -0.050262],
        ]
    else:
        ref_species = ["Mo", "Mo", "S", "S", "S", "S"]
        ref_coords = [
            [0.051823, -0.017150, 16.736001],
            [2.819495, 1.607633, 16.734550],
            [1.901722, -0.004881, 18.317009],
        ]
        ref_forces = [
            [-0.425324, 0.295866, -0.065479],
            [0.245043, -0.061658, 0.264104],
            [0.010127, 0.041539, 0.301571],
        ]

    np.array_equal(config.species[:6], ref_species)
    assert np.allclose(config.coords[:3], ref_coords)
    if f:
        assert np.allclose(config.forces[:3], ref_forces)

    natoms_by_species = config.count_atoms_by_species()

    assert natoms_by_species["Mo"] == 96
    assert natoms_by_species["S"] == 192


def test_dataset(test_data_dir):
    tset = Dataset.from_path(test_data_dir / "configs/MoS2")
    configs = tset.get_configs()
    assert len(configs) == 3
