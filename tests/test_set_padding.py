from __future__ import division
import sys
sys.path.append('../openkim_fit')
from kimcalculator import set_padding
from utils import write_extxyz
import numpy as np

def test_set_padding():
    """ check visualliy whether the set_padding function works.
    By adjusting rcut, one can check the correctness by plot using say Ovito.
    """
    a = 10
    cell = [[a,0,0],
            [a/2, np.sqrt(3)/2*a, 0],
            [0,0, a]]
    cell = np.array(cell)
    atom1 = [0,0,0]
    atom2 = cell[0]/3. + cell[1]/3 + cell[2]/3
    atom3 = cell[0]*2/3. + cell[1]*2/3. + cell[2]/2.

    coords = []
    coords.extend(atom1)
    coords.extend(atom2)
    coords.extend(atom3)
    species = ['A', 'B', 'CC']

    pbc = [1,1,1]

    #rcut = a/3+0.00001
    #rcut = 3**0.5/2*a+0.000001

    rcut = 3**0.5/6*a+0.000001
    # create padding atoms
    pad_coords, pad_species, atom_id = set_padding(cell, pbc, species, coords, rcut)
    # add contributing and padding together
    coords1 = np.concatenate((coords,pad_coords))
    species1 = np.concatenate((species, pad_species))
    # write to extended xyz file for visualization check
    write_extxyz(cell, species1, coords1, fname='check_set_padding1.xyz')
    # wheter we have the correct number of atoms
    assert len(species1) == 16
    assert list(species1).count('A') == 8
    assert list(species1).count('B') == 4
    assert list(species1).count('CC') == 4


    rcut = 3**0.5/6*a-0.000001
    # create padding atoms
    pad_coords, pad_species, atom_id = set_padding(cell, pbc, species, coords, rcut)
    # add contributing and padding together
    coords2 = np.concatenate((coords,pad_coords))
    species2 = np.concatenate((species, pad_species))
    # write to extended xyz file for visualization check
    write_extxyz(cell, species2, coords2, fname='check_set_padding2.xyz')

    assert len(species2) == 10
    assert list(species2).count('A') == 8
    assert list(species2).count('B') == 1
    assert list(species2).count('CC') == 1

if __name__ == '__main__':
    test_set_padding()
