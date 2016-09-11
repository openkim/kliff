import sys
import numpy as np

class TrainingSet:
    '''
    Class to read and store training set. 
    '''
    
    def __init__(self):
        self.natoms = None 
        self.latvec = None 
        self.energy = None 
        self.species = []
        self.coords = []
        self.forces = []

    def read_extxyz(self, fname='./train.xyz'):
        with open(fname, 'r') as fin:
            lines = fin.readlines() 
            # number of atoms
            try:
                self.natoms = int(lines[0].split()[0])
            except ValueError as err:
                raise ValueError('{}.\nIncorrect data type at line {} of '
                                 'file: {}.'.format(err, 1, fname)) 
            # lattice const and energy
            line = lines[1]
            if 'Lattice' not in line:
                sys.stderr.write('Error: "Lattice" not found at line {} of '
                                 'file: {}.\n'.format(2, fname))
            elif 'Energy' not in line:
                sys.stderr.write('Error: "Energy" not found at line {} of '
                                 'file: {}.\n'.format(2, fname))
            else:
                try:
                    latvec = self.parse_key_value('Lattice', line) 
                    latvec = [float(i) for i in latvec.split()]
                    self.latvec = np.array(latvec).reshape((3, 3))
                except ValueError as err:
                    raise ValueError('{}.\nCorrupted "Lattice" data at line {} of '
                                     'file {}.'.format(err, 2, fname)) 
                try:
                    energy = self.parse_key_value('Energy', line) 
                    self.energy = float(energy)
                except ValueError as err:
                    raise ValueError('{}.\nCorrupted "Energy" data at line {} of '
                                     'file {}.'.format(err, 2, fname)) 
            # read symbol and x, y, z fx, fy, fz
            for line in lines[2:]:
                symbol, x, y, z, fx, fy, fz = line.split()
                self.species.append(symbol.lower().capitalize()) 
                self.coords.append(float(x))
                self.coords.append(float(y))
                self.coords.append(float(z))
                self.forces.append(float(fx))
                self.forces.append(float(fy))
                self.forces.append(float(fz))


    def parse_key_value(self, key, line):
        '''
        Given key, parse a string like 'other stuff key = "value" other stuff' to get value.
        '''
        value = line[line.index(key)+len(key):]
        value = value[value.index('=')+1:]
        value = value[value.index('"')+1:]
        value = value[:value.index('"')]
        return value

    def write_extxyz(self, fname='./echo_config.xyz'):
        with open (fname, 'w') as fout:
            # first line (num of atoms)
            fout.write('{}\n'.format(self.natoms))
            # second line 
            # lattice 
            fout.write('Lattice="')
            for line in self.latvec:
                for item in line:
                    fout.write('{:10.6f}'.format(item))
            fout.write('" ') 
            # properties
            fout.write('Properties="species:S:1:pos:R:3:vel:R:3" ')
            # energy
            fout.write('Energy="{:10.6f}"\n'.format(self.energy))
            # species, coords, and forces
            for i in range(self.natoms):
                symbol = self.species[i]+ '    ' 
                symbol = symbol[:4]  # such that symbol has the same length
                fout.write(symbol)
                fout.write('{:14.6e}'.format(self.coords[3*i+0]))
                fout.write('{:14.6e}'.format(self.coords[3*i+1]))
                fout.write('{:14.6e}'.format(self.coords[3*i+2]))
                fout.write('{:14.6e}'.format(self.forces[3*i+0]))
                fout.write('{:14.6e}'.format(self.forces[3*i+1]))
                fout.write('{:14.6e}'.format(self.forces[3*i+2]))
                fout.write('\n')




# test
if __name__ == '__main__':
    configs = TrainingSet()
    configs.read_extxyz('develop_test/T150_training_1000.xyz')
    configs.write_extxyz('./echo.xyz')




