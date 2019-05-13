import numpy as np
from kliff.dataset import Configuration
from kliff.dataset import DataSet
from kliff.dataset.extxyz import write_extxyz


def assert_1d_array(a, b):
    for i, j in zip(a, b):
        assert i == j


def test_configuration(e=True, f=False, s=False, order=False):
    fname = '../configs_extxyz/MoS2/MoS2'
    if e:
        fname += '_energy'
    if f:
        fname += '_forces'
    if s:
        fname += '_stress'
    fname += '.xyz'

    config = Configuration(format='extxyz', order_by_species=order)
    config.read(fname)

    natoms = config.get_number_of_atoms()
    cell = config.get_cell()
    PBC = config.get_PBC()
    species = config.get_species()
    coords = config.get_coordinates()
    energy = config.get_energy()
    forces = config.get_forces()
    stress = config.get_stress()

    assert natoms == 288
    assert np.allclose(
        cell,
        [
            [33.337151, 0.035285, 0.03087],
            [0.027151, 25.621674, -0.000664],
            [0.027093, 9e-05, 30.262626],
        ],
    )
    assert np.allclose(PBC, [1, 1, 1])

    if e:
        assert energy == -5.302666
    else:
        assert energy == None

    if s:
        assert np.allclose(stress, [1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
    else:
        assert stress == None

    if order:
        ref_species = ['Mo', 'Mo', 'Mo', 'Mo', 'Mo', 'Mo']
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
        ref_species = ['Mo', 'Mo', 'S', 'S', 'S', 'S']
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

    assert_1d_array(species[:6], ref_species)
    assert np.allclose(coords[:3], ref_coords)
    if f:
        assert np.allclose(forces[:3], ref_forces)
    else:
        assert forces == None

    natoms_by_species = config.count_atoms_by_species()
    assert natoms_by_species['Mo'] == 96
    assert natoms_by_species['S'] == 192

    outname = fname.replace('../configs_extxyz/MoS2/MoS2', 'tmp_Mos2')
    write_extxyz(outname, cell, PBC, species, coords, energy, forces, stress)


def test_dataset():
    directory = '../configs_extxyz/MoS2'
    tset = DataSet()
    tset.read(directory)
    configs = tset.get_configs()
    assert len(configs) == 3


if __name__ == '__main__':
    test_configuration(True, False, False)
    test_configuration(True, True, False)
    test_configuration(True, True, True)
    test_configuration(True, True, True, order=True)
    test_dataset()
