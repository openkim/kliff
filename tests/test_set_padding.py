from __future__ import division
import sys
sys.path.append('../openkim_fit')
from neighbor import set_padding
from utils import write_extxyz
import numpy as np

def test_set_padding():
    """ check visualliy whether the set_padding function works.
    By adjusting rcut, one can check the correctness by plot using say Ovito.
    """
    a = 2
    cell = np.array([[a,0,0],
            [a/2, np.sqrt(3)/2*a, 0],
            [0,0, 5]])
    atom1 = [0,0,0]
    atom2 = cell[0]/3. + cell[1]/3. + 0*cell[2]
    atom3 = cell[0]*1/3. + cell[1]*1/3. + cell[2]/2.
    atom4 = cell[0]*2/3. + cell[1]*2/3. + cell[2]/2.

    coords = []
    coords.extend(atom1)
    coords.extend(atom2)
    coords.extend(atom3)
    coords.extend(atom4)
    species = ['C', 'H', 'O', 'N']

    pbc = [1,1,1]

    rcut = 0.5*3**0.5*a + 0.00001    # make rcut/dist >1
    #rcut = 0.5*3**0.5*a - 0.00001    # make rcut/dist <1
    #rcut = a/3**0.5 + 0.00001        # have neigh
    #rcut = a/3**0.5 - 0.00001         # have no neigh

    # create padding atoms
    pad_coords, pad_species, atom_id = set_padding(cell, pbc, species, coords, rcut)
    # add contributing and padding together
    coords1 = np.concatenate((coords,pad_coords))
    species1 = np.concatenate((species, pad_species))
    # write to extended xyz file for visualization check
    write_extxyz(cell, species1, coords1, fname='check_set_padding1.xyz')

    # wheter we have the correct number of atoms
    assert len(species1) == 36
    assert list(species1).count('C') == 9
    assert list(species1).count('H') == 9
    assert list(species1).count('O') == 9
    assert list(species1).count('N') == 9
#

    rcut = 0.5*3**0.5*a - 0.00001    # make rcut/dist <1
    # create padding atoms
    pad_coords, pad_species, atom_id = set_padding(cell, pbc, species, coords, rcut)
    # add contributing and padding together
    coords2 = np.concatenate((coords,pad_coords))
    species2 = np.concatenate((species, pad_species))
    # write to extended xyz file for visualization check
    write_extxyz(cell, species2, coords2, fname='check_set_padding2.xyz')

    assert len(species2) == 26
    assert list(species2).count('C') == 4
    assert list(species2).count('H') == 9
    assert list(species2).count('O') == 9
    assert list(species2).count('N') == 4


if __name__ == '__main__':
    test_set_padding()
