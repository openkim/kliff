from __future__ import division
from __future__ import print_function
import numpy as np
import sys
from collections import OrderedDict

from neighbor import NeighborList
from error import UnsupportedError
from desc import CythonDescriptor


class Descriptor:
  """Symmetry functions that transform the cartesian coords to generalized coords.

  Parameters
  ----------

  cutname, string
    cutoff function name

  cutvalue, dict
    cutoff values based on species.

    Example
    -------
    cutvalue = {'C-C': 3.5, 'C-H': 3.0, 'H-H': 1.0}

  hyperparams, dict
    hyperparameters of descriptors

    Example
    -------
      {'g1': None,
       'g2': [{'eta':0.1, 'Rs':0.2}, {'eta':0.3, 'Rs':0.4}],
       'g3': [{'kappa':0.1}, {'kappa':0.2}, {'kappa':0.3}]
      }
  """

  def __init__(self,cutname, cutvalue, hyperparams):

    self._desc = OrderedDict()
    self._cutname = cutname.lower()
    self._rcut = generate_full_cutoff(cutvalue)
    self._species_code = dict()
    self._cdesc = CythonDescriptor()

    # check cutoff support
    if self._cutname not in ['cos', 'exp']:
      raise UnsupportedError("Cutoff `{}' unsupported.".format(cutname))

    species = set()
    for key, value in self._rcut.iteritems():
      spec1, spec2 = key.split('-')
      species.update([spec1, spec2])
    species = list(species)
    num_species = len(species)

    # create integer code for each species type and cutoff values based on species code
    rcutsym = np.zeros([num_species, num_species])
    for i,si in enumerate(species):
      self._species_code[si] = i
      for j,sj in enumerate(species):
        rcutsym[i][j] = self._rcut[si+'-'+sj]

    # store cutoff values in cpp class
    self._cdesc.set_cutoff('cos', num_species, rcutsym)

    # hyperparams of descriptors
    for key, values in hyperparams.iteritems():
      if key.lower() not in ['g1', 'g2', 'g3', 'g4', 'g5']:
        raise UnsupportedError("Symmetry function `{}' unsupported.".format(key))

      # g1 needs no hyperparams, put a placeholder
      name = key.lower()
      if name == 'g1':
        rows = 1
        cols = 1
        params = np.zeros([1,1])  # it has no hyperparams, zeros([1,1]) for placeholder
      else:
        rows = len(values)
        cols = len(values[0])
        params = np.zeros([rows,cols])
        for i,line in enumerate(values):
          if name == 'g2':
            params[i][0] = line['eta']
            params[i][1] = line['Rs']
          elif name == 'g3':
            params[i][0] = line['kappa']
          elif key == 'g4':
            params[i][0] = line['zeta']
            params[i][1] = line['lambda']
            params[i][2] = line['eta']
          elif key == 'g5':
            params[i][0] = line['zeta']
            params[i][1] = line['lambda']
            params[i][2] = line['eta']

      # store cutoff values in both this python and cpp class
      self._desc[name] = params
      self._cdesc.add_descriptor(name, params, rows, cols)


  def generate_generalized_coords(self, conf):
    """Transform atomic coords to generalized coords.

    Parameter
    ---------

    conf: Configuration object in which the atoms information are stored

    Returns
    -------
    gen_coords, 2D float array
      generalized coordinates of size [Ncontrib, Ndescriptors]

    d_gen_coords, 3D float array
      derivative of generalized coordinates w.r.t atomic coords of size
      [Ncontrib, Ndescriptors, 3*Ncontrib]

    """

    # create neighbor list
    nei = NeighborList(conf, self._rcut)
    coords = nei.coords
    species_code = np.array([self._species_code[i] for i in nei.species])
    Ncontrib = nei.ncontrib
    Natoms = nei.natoms
    neighlist = np.concatenate(nei.neighlist)
    numneigh = nei.numneigh
    image = nei.image

    # loop to set up generalized coords
    Ndesc = self.get_num_descriptors()

    gen_coords, d_gen_coords = self._cdesc.generate_generalized_coords(coords,
        species_code.astype('int32'), neighlist.astype('int32'),
        numneigh.astype('int32'), image.astype('int32'),
        Natoms, Ncontrib, Ndesc)

    return gen_coords, d_gen_coords


  def get_num_descriptors(self):
    """The total number of symmetry functions (each hyper-parameter set counts 1)"""
    N = 0
    for key in self._desc:
      N += len(self._desc[key])
    return N


  def get_cutoff(self):
    """ Return the name and values of cutoff. """
    return self._cutname, self._rcut


  def get_hyperparams(self):
    """ Return the hyperparameters of descriptors. """
    return self._desc



def generate_full_cutoff(rcut):
    """Generate a full binary cutoff dictionary.
        e.g. for input
            rcut = {'C-C':1.42, 'C-H':1.0, 'H-H':0.8}
        the output would be
            rcut = {'C-C':1.42, 'C-H':1.0, 'H-C':1.0, 'H-H':0.8}
    """
    rcut2 = dict()
    for key, val in rcut.iteritems():
        spec1,spec2 = key.split('-')
        if spec1 != spec2:
            rcut2[spec2+'-'+spec1] = val
    # merge
    rcut2.update(rcut)
    return rcut2




