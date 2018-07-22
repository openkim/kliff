from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import sys

def generate_kimstr(modelname, cell, species):
  '''
  Creates a valid KIM file that will be used to initialize the KIM object.

  Parameters
  ----------
  modelname: KIM Model name

  cell: supercell

  species: string list
    species of all atoms

  Returns
  -------
  kimstring: str
    a string of the KIM file for the configuration object
  '''

  # version and units
  kimstr  = 'KIM_API_Version := 1.7.0\n'
  kimstr += 'Unit_length := A\n'
  kimstr += 'Unit_energy := eV\n'
  kimstr += 'Unit_charge := e\n'
  kimstr += 'Unit_temperature := K\n'
  kimstr += 'Unit_time := ps\n'

  # particle species
  # 'code' does not matter, so just give it 0
  kimstr += 'PARTICLE_SPECIES:\n'
  kimstr += '# Symbol/name  Type  code\n'
  for s in species:
    kimstr += s+'  spec    0\n'

  # conversions
  kimstr += 'CONVENTIONS:\n'
  kimstr += 'ZeroBasedLists flag\n'
  kimstr += 'Neigh_LocaAccess flag\n'
  kimstr += 'Neigh_IterAccess flag\n'
  kimstr += 'Neigh_BothAccess flag\n'
#  kimstr += 'NEIGH_RVEC_H flag\n'
#  kimstr += 'NEIGH_RVEC_F flag\n'
  kimstr += 'NEIGH_PURE_F flag\n'
#  if orthogonal(cell):
#    kimstr += 'MI_OPBC_H  flag\n'
#    kimstr += 'MI_OPBC_F  flag\n'
#  kimstr += 'CLUSTER    flag\n'
#
  # model input
  kimstr += 'MODEL_INPUT:\n'
  kimstr += 'numberOfParticles  integer  none    []\n'
  kimstr += 'numberOfSpecies    integer  none    []\n'
  kimstr += 'particleSpecies    integer  none    [numberOfParticles]\n'
  kimstr += 'particleStatus   integer  none    [numberOfParticles]\n'
  kimstr += 'coordinates      double   length  [numberOfParticles,3]\n'
#  kimstr += 'boxSideLengths     double length  [3]\n'
#  kimstr += 'numberContributingParticles integer none  []\n'
  kimstr += 'get_neigh      method   none    []\n'
  kimstr += 'neighObject      pointer  none    []\n'

  # model output
  # create a temporary object to inquire the info
  status, kimmdl = ks.KIM_API_model_info(modelname)
  kimstr += "MODEL_OUTPUT:\n"
  if checkIndex(kimmdl, 'compute') >= 0:
    kimstr += 'compute  method  none  []\n'
  if checkIndex(kimmdl, 'reinit') >= 0:
    kimstr += 'reinit method  none  []\n'
  if checkIndex(kimmdl, 'destroy') >= 0:
    kimstr += 'destroy  method  none  []\n'
  if checkIndex(kimmdl, 'cutoff') >= 0:
    kimstr += 'cutoff  double  length  []\n'
  if checkIndex(kimmdl, 'energy') >= 0:
    kimstr += 'energy  double  energy  []\n'
  if checkIndex(kimmdl, 'forces') >= 0:
    kimstr += 'forces  double  force  [numberOfParticles,3]\n'
#NOTE, we do not need them, sice we want to use forces only (possibly energy)
#  if checkIndex(kimmdl, 'particleEnergy') >= 0:
#    kimstr += 'particleEnergy  double  energy  [numberOfParticles]\n'
#  if (checkIndex(kimmdl, 'virial') >= 0 or checkIndex(kimmdl, 'process_dEdr') >=0):
#    kimstr += 'virial  double  energy  [6]\n'
#  if (checkIndex(kimmdl, 'particleVirial') >= 0 or checkIndex(kimmdl, 'process_dEdr') >=0):
#    kimstr += 'particleVirial  double  energy  [numberOfParticles,6]\n'
#  if (checkIndex(kimmdl, 'hessian') >= 0 or
#    (checkIndex(kimmdl, 'process_dEdr') >= 0 and checkIndex(kimmdl, 'process_d2Edr2') >= 0)):
#    kimstr += 'hessian  double  pressure  [numberOfParticles,numberOfParticles,3,3]\n'

#NOTE if we free it, segfault error will occur; Ask Matt whether we don't need to free?
#NOTE in kimcalculator, it is not freed
  # free KIM object
  #ks.KIM_API_model_destroy(kimmdl)
  #ks.KIM_API_free(kimmdl)
  return kimstr



def generate_dummy_kimstr(modelname):
  '''
  Generate a kimstr using the first species supported by the model.
  '''
  status, kimmdl = ks.KIM_API_model_info(modelname)
  species = ks.KIM_API_get_model_species(kimmdl, 0)
  species = [species]
  dummy_cell = [[1,0,0], [0,1,0],[0,0,1]]
  kimstr = generate_kimstr(modelname, dummy_cell, species)

#NOTE free needed, as above
  return kimstr




def orthogonal(cell):
  '''
  Check whether the supercell is orthogonal.
  '''
  return ((abs(np.dot(cell[0], cell[1])) +
       abs(np.dot(cell[0], cell[2])) +
       abs(np.dot(cell[1], cell[2]))) < 1e-8)


def checkIndex(pkim, variablename):
  '''
  Check whether a variable exists in the KIM object.
  '''
  try:
    index = ks.KIM_API_get_index(pkim, variablename)
  except:
    index = -1
  return index



def remove_comments(lines):
  '''Remove lines in a string list that start with # and content after #.'''
  processed_lines = []
  for line in lines:
    line = line.strip()
    if not line or line[0] == '#':
      continue
    if '#' in line:
      line = line[0:line.index('#')]
    processed_lines.append(line)
  return processed_lines


def write_extxyz(cell, species, coords, fname='config.txt'):
  with open (fname, 'w') as fout:
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
      print ('Number of atoms is inconsistent from species nad coords.')
      print ('len(specis)=', natoms)
      print ('len(coords)=', len(coords)//3)
      sys.exit(1)
    for i in range(natoms):
      fout.write('{:4}'.format(species[i]))
      fout.write('{:12.5e} '.format(coords[3*i+0]))
      fout.write('{:12.5e} '.format(coords[3*i+1]))
      fout.write('{:12.5e} 0 0 0\n'.format(coords[3*i+2]))


def banner():
  """ Banner of openKIM-fit.
  """

  bn = '''
                  _    _      __ _ _
      ___  _ __   ____  _ __ | | _(_)_ __ ___  / _(_) |_
     / _ \| '_ \ / __ \| '_ \| |/ / | '_ ` _ \| |_| | __|
    | (_) | |_) | /___/| | | |   <| | | | | | |  _| | |_
     \___/| .__/ \____ |_| |_|_|\_\_|_| |_| |_|_| |_|\__|
        |_|
  '''

  print ('='*80)
  print (bn)

