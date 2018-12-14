import numpy as np
from kliff.neighbor import NeighborList
from kliff.neighbor import set_padding
from kliff.dataset import Configuration
from kliff.dataset import write_extxyz


def create_hexagonal(a=2):
    cell = np.array([[a, 0, 0],
                     [a/2, np.sqrt(3)/2*a, 0],
                     [0, 0, 5]])
    atom1 = [0, 0, 0]
    atom2 = cell[0]/3. + cell[1]/3. + 0*cell[2]
    atom3 = cell[0]*1/3. + cell[1]*1/3. + cell[2]/2.
    atom4 = cell[0]*2/3. + cell[1]*2/3. + cell[2]/2.

    coords = []
    coords.extend(atom1)
    coords.extend(atom2)
    coords.extend(atom3)
    coords.extend(atom4)
    species = ['C', 'H', 'O', 'N']

    return cell, species, coords


def create_all(cell, species, coords, PBC, rcut, fname='tmp_test_neighbor.xyz'):
    pad_coords, pad_species, atom_id = set_padding(cell, PBC, species, coords, rcut)
    coords1 = np.concatenate((coords, pad_coords))
    species1 = np.concatenate((species, pad_species))
    # write to extended xyz file for visualization
    write_extxyz(fname, cell, PBC, species1, coords1.reshape((-1, 3)))
    return species1, coords1


def test_set_padding():
    """Check visually that the set_padding function works.
    By adjusting rcut, one can check the correctness by plot using Ovito.
    """

    PBC = [1, 1, 1]
    a = 2
    cell, species, coords = create_hexagonal(a)

    rcut = 0.5*3**0.5*a + 0.00001    # make rcut/dist >1
    # rcut = 0.5*3**0.5*a - 0.00001    # make rcut/dist <1
    # rcut = a/3**0.5 + 0.00001        # have neigh
    # rcut = a/3**0.5 - 0.00001         # have no neigh

    cell, species, coords, create_hexagonal(a=2)
    species1, coords1 = create_all(cell, species, coords, PBC,
                                   rcut, 'tmp_test_neighbor_1.xyz')

    assert len(species1) == 36
    assert list(species1).count('C') == 9
    assert list(species1).count('H') == 9
    assert list(species1).count('O') == 9
    assert list(species1).count('N') == 9

    rcut = 0.5*3**0.5*a - 0.00001    # make rcut/dist <1
    cell, species, coords, create_hexagonal(a=2)
    species1, coords1 = create_all(cell, species, coords, PBC,
                                   rcut, 'tmp_test_neighbor_2.xyz')

    assert len(species1) == 26
    assert list(species1).count('C') == 4
    assert list(species1).count('H') == 9
    assert list(species1).count('O') == 9
    assert list(species1).count('N') == 4


def test_neigh():
    conf = Configuration()
    fname = 'training_set/training_set_graphene/bilayer_sep3.36_i0_j0.xyz'
    conf.read(fname)

    rcut = {'C-C': 2}
    nei = NeighborList(conf, rcut)
    fname = 'tmp_test_neighbor_3.xyz'
    cell = conf.get_cell()
    PBC = conf.get_PBC()
    write_extxyz(fname, cell, PBC, nei.species, nei.coords.reshape((-1, 3)))

    assert nei.natoms == 16
    assert np.allclose(nei.numneigh, [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert np.allclose(nei.neighlist, [[1, 6, 8], [0, 10, 12], [3, 7, 9], [2, 11, 13]])

    n, neighlist = nei.get_neigh(0)
    assert n == 3
    assert np.allclose(neighlist, [1, 6, 8])


if __name__ == '__main__':
    test_set_padding()
    test_neigh()
