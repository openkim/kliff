import numpy as np
from kliff.neighbor import NeighborList
from kliff.dataset import Configuration
from kliff.dataset import write_config

target_coords = np.asarray(
    [
        [0.000000e00, 0.000000e00, 0.000000e00],
        [1.234160e00, 7.125400e-01, 0.000000e00],
        [0.000000e00, 0.000000e00, 3.355150e00],
        [1.234160e00, 7.125400e-01, 3.355150e00],
        [-2.468323e00, -1.425090e00, 0.000000e00],
        [-2.468323e00, -1.425090e00, 3.355150e00],
        [-1.234162e00, 7.125400e-01, 0.000000e00],
        [-1.234162e00, 7.125400e-01, 3.355150e00],
        [-1.000000e-06, -1.425090e00, 0.000000e00],
        [-1.000000e-06, -1.425090e00, 3.355150e00],
        [1.234161e00, 2.137630e00, 0.000000e00],
        [1.234161e00, 2.137630e00, 3.355150e00],
        [2.468322e00, 0.000000e00, 0.000000e00],
        [2.468322e00, 0.000000e00, 3.355150e00],
        [3.702483e00, 2.137630e00, 0.000000e00],
        [3.702483e00, 2.137630e00, 3.355150e00],
    ]
)

target_species = ['C'] * 16
target_species[0] = 'O'
target_species[10] = 'O'
target_species[12] = 'O'
target_species[14] = 'O'

all_indices = [[6, 1, 8], [0, 10, 12], [7, 3, 9], [2, 11, 13]]
all_numneigh = [len(i) for i in all_indices]


def test_neigh():
    conf = Configuration()
    fname = 'configs_extxyz/bilayer_graphene/bilayer_sep3.36_i0_j0.xyz'
    conf.read(fname)
    conf.species[0] = 'O'

    neigh = NeighborList(conf, infl_dist=2, padding_need_neigh=False)
    fname = 'tmp_test_neighbor.xyz'
    cell = conf.get_cell()
    PBC = conf.get_PBC()
    coords = neigh.get_coords()
    species = neigh.get_species()
    write_config(fname, cell, PBC, species, coords, format='extxyz')

    assert np.allclose(coords, target_coords)
    assert np.array_equal(species, target_species)

    # contributing
    for i in range(conf.get_number_of_atoms()):
        nei_indices, nei_coords, nei_species = neigh.get_neigh(i)
        assert np.allclose(nei_indices, all_indices[i])

    # padding
    for i in range(conf.get_number_of_atoms(), len(coords)):
        nei_indices, nei_coords, nei_species = neigh.get_neigh(i)
        assert nei_indices.size == 0
        assert nei_coords.size == 0
        assert nei_species.size == 0

    numneigh, neighlist = neigh.get_numneigh_and_neighlist_1D(request_padding=False)
    np.array_equal(numneigh, all_numneigh)
    np.array_equal(neighlist, np.concatenate(all_indices))


if __name__ == '__main__':
    test_neigh()
