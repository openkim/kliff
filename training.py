import os
import glob
import numpy as np

class Config:
    '''
    Class to read and store the information in one configuraiton. 
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
                raise Exception('"Lattice" not found at line {} of file: {}.\n'.format(2, fname))
            elif 'Energy' not in line:
                raise Exception('"Energy" not found at line {} of file: {}.\n'.format(2, fname))
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
            try:
                num_lines = 0
                for line in lines[2:]:
                    line = line.strip()
                    if line:
                        symbol, x, y, z, fx, fy, fz = line.split()
                        self.species.append(symbol.lower().capitalize()) 
                        self.coords.append(float(x))
                        self.coords.append(float(y))
                        self.coords.append(float(z))
                        self.forces.append(float(fx))
                        self.forces.append(float(fy))
                        self.forces.append(float(fz))
                        num_lines += 1 
                        if num_lines == self.natoms:
                            break
            except ValueError as err: 
                raise ValueError('{}.\nCorrupted data at line {} of '
                                 'file {}.'.format(err, num_lines+2+1, fname)) 
            if num_lines < self.natoms:
                raise Exception('not enough data lines. Number of atoms = {}, while '
                                'number of lines = {}.'.format(self.natoms, num_lines))
#NOTE not needed    
#    def get_unique_species(self):
#        '''
#        Get a set of the species list.
#        '''
#        return list(set(self.species))
    def get_num_atoms(self):
        return self.natoms
    def get_lattice_vectors(self):
        return self.latvec()
    def get_energy(self):
        return self.energy()
    def get_species(self):
        return self.species
    def get_coords(self):
        return self.coords
    def get_forces(self):
        return self.forces






    def parse_key_value(self, key, line):
        '''
        Given key, parse a string like 'other stuff key = "value" other stuff'
        to get value.
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
        







class TrainingSet():
    '''
    Training set class, to deal with multiple configurations.
    '''
    def __init__(self):
        self.size = 0
        self.configs = []
    
    def read(self, fname):
        '''
        Read training set, where each file stores a configuration.
        '''
        if os.path.isdir(fname):
            dirpath = fname
            all_files = glob.glob(dirpath+os.path.sep+'*xyz')
        else: 
            dirpath = os.path.dirname(fname)
            all_files = [fname]
        for f in all_files:
            conf = Config()
            conf.read_extxyz(f)
            self.configs.append(conf)
        self.size = len(self.configs) 
        if self.size <= 0:
            raise Exception('no training set files (ended with .xyz) found '
                            'in directory: {}/'.format(dirpath))

# not needed
#    def get_unique_species(self):
#        '''
#        Get all the species that appear in the training set. 
#        '''
#        unique_species = []
#        for conf in self.configs:
#            unique_species += conf.get_unique_species()
#        return list(set(unique_species))
#        
    def get_size(self):
        return self.size
    def get_configs(self):
        return self.configs 


# test
if __name__ == '__main__':
#    configs = Config()
#    configs.read_extxyz('./training_set/T150_training_1000.xyz')
#    configs.write_extxyz('./echo.xyz')
#

    Tset = TrainingSet()
    #Tset.read('./training_set')
    Tset.read('./training_set/T150_training_1000.xyz')
    print 'num of configurations', Tset.get_size()
    configs = Tset.get_configs()
    for i,conf in enumerate(configs):
        conf.write_extxyz('echo{}.xyz'.format(i))





