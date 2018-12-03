from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys


def write_extxyz(cell, species, coords, fname='config.txt'):

    with open(fname, 'w') as fout:
        # first line (num of atoms)
        natoms = len(species)
        fout.write('{}\n'.format(natoms))

        # second line
        # lattice
        fout.write('Lattice="')
        for line in cell:
            for item in line:
                fout.write('{} '.format(item))
        fout.write('" ')
        # properties
        fout.write('Properties=species:S:1:pos:R:3:for:R:3\n')

        # species, coords
        if natoms != len(coords)//3:
            print('Number of atoms is inconsistent from species nad coords.')
            print('len(specis)=', natoms)
            print('len(coords)=', len(coords)//3)
            sys.exit(1)
        for i in range(natoms):
            fout.write('{:4}'.format(species[i]))
            fout.write('{:12.5e} '.format(coords[3*i+0]))
            fout.write('{:12.5e} '.format(coords[3*i+1]))
            fout.write('{:12.5e} 0 0 0\n'.format(coords[3*i+2]))
