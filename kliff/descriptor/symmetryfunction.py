from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import sys
from collections import OrderedDict
from ..neighbor import NeighborList
from ..error import InputError, SupportError
from . import sym_fn


class Descriptor(object):
  """Symmetry functions that transform the cartesian coords to generalized coords.

  Parameters
  ----------

  hyperparams: dict
    hyperparameters of descriptors

    Example
    -------
      {'g1': None,
       'g2': [{'eta':0.1, 'Rs':0.2}, {'eta':0.3, 'Rs':0.4}],
       'g3': [{'kappa':0.1}, {'kappa':0.2}, {'kappa':0.3}]
      }

  cutname: string
    cutoff function name

  cutvalue: dict
    cutoff values based on species.

    Example
    -------
    cutvalue = {'C-C': 3.5, 'C-H': 3.0, 'H-H': 1.0}

  cutvalue_samelayer, dict
    cutoff values to include atoms in the same layer when considering interlayer
    interactions for layered materials.

  debug: bool
    True to enable debug mode where descriptor information will be printed to
    file named `debug_descriptor.txt'.

  """

  def __init__(self, hyperparams, cutname, cutvalue, cutvalue_samelayer=None,
      debug=False):

    self._hyperparams = hyperparams
    self._desc = OrderedDict()
    self._cutname = cutname.lower()
    self._rcut = self.generate_full_cutoff(cutvalue)
    if cutvalue_samelayer is not None:
      self._rcut_samelayer = generate_full_cutoff(cutvalue_samelayer)
    else:
      self._rcut_samelayer = None
    self.debug = debug
    if debug:
      with open('debug_descriptor.txt', 'w') as fout:
        fout.write('# descritor values (without normalization) for atoms in each configuration\n')
    self._species_code = dict()
    self._cdesc = sym_fn.Descriptor()

    # set cutoff
    self._set_cutoff()

    # set hyper params
    self._set_hyperparams()


  def generate_generalized_coords(self, conf, fit_forces=False, structure='bulk'):
    """Transform atomic coords to generalized coords.

    Parameter
    ---------

    conf: Configuration object in which the atoms information are stored

    fit_forces: bool
      Whether to fit to forces.

    structure: str
      The type of materials to generate generalized coords for. Available
      vales are {'bulk', 'bilayer', 'trilayer'}. Only the first two letters
      matter and it is case insensitive.

    Returns
    -------
    gen_coords, 2D float array
      generalized coordinates of size [Ncontrib, Ndescriptors]

    d_gen_coords, 3D float array (only when fit_forces is True)
      derivative of generalized coordinates w.r.t atomic coords of size
      [Ncontrib, Ndescriptors, 3*Ncontrib]

    """

    # determine structure
    structure_2letter = structure.lower()[:2]
    if structure_2letter == 'bu':
      int_structure = 0
    elif structure_2letter == 'bi':
      int_structure = 1
    elif structure_2letter == 'tr':
      int_structure = 2
    else:
      raise SupportError("Material type (structure) `{}' unsupported. "
          "Available options are ('bulk', 'bilayer', 'trilayer').".format(structure))

    # check whether cutvalue_samelayer is set
    if structure_2letter == 'bi' or structure_2letter == 'tr':
      if self._rcut_samelayer is None:
        raise InputError("Material type (structure) `{}' cannot be used when "
            "`cutvalue_samelayer' is not provided to initialize the `Descriptor' "
            "class.".format(structure))

    # create neighbor list
    nei = NeighborList(conf, self._rcut, padding_need_neigh=True)
    coords = nei.coords
    species_code = np.array([self._species_code[i] for i in nei.species])
    Ncontrib = nei.ncontrib
    Natoms = nei.natoms
    neighlist = np.concatenate(nei.neighlist)
    numneigh = nei.numneigh
    image = nei.image

    # loop to set up generalized coords
    Ndesc = self.get_number_of_descriptors()

    if fit_forces:
      gen_coords, d_gen_coords = self._cdesc.get_gen_coords_and_deri(coords.astype(np.double),
          species_code.astype(np.intc), neighlist.astype(np.intc),
          numneigh.astype(np.intc), image.astype(np.intc),
          Natoms, Ncontrib, Ndesc, int_structure)
    else:
      gen_coords = self._cdesc.get_gen_coords(coords.astype(np.double),
          species_code.astype(np.intc), neighlist.astype(np.intc),
          numneigh.astype(np.intc), image.astype(np.intc),
          Natoms, Ncontrib, Ndesc, int_structure)

    if self.debug:
      with open('debug_descriptor.txt', 'a') as fout:
        fout.write('\n\n#'+'='*80+'\n')
        fout.write('# configure name: {}\n'.format(conf.id))
        fout.write('# atom id    descriptor values ...\n\n')
        for i,line in enumerate(gen_coords):
          fout.write('{}    '.format(i))
          for j in line:
            fout.write('{:.15g} '.format(j))
          fout.write('\n')

    if fit_forces:
      return gen_coords, d_gen_coords
    else:
      return gen_coords, None


  def get_number_of_descriptors(self):
    """The total number of symmetry functions (each hyper-parameter set counts 1)"""
    N = 0
    for key in self._desc:
      N += len(self._desc[key])
    return N


  def get_cutoff(self):
    """ Return the name and values of cutoff. """
    return self._cutname, self._rcut, self._rcut_samelayer


  def get_hyperparams(self):
    """ Return the hyperparameters of descriptors. """
    return self._desc


  @staticmethod
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


  def _set_cutoff(self):

    # check cutoff support
    if self._cutname not in ['cos', 'exp']:
      raise SupportError("Cutoff type `{}' unsupported.".format(cutname))

    # check cutvalue_samelayer and cutvalue include the same species info
    if self._rcut_samelayer is not None:
      if set(self._rcut.keys()) != set(self._rcut_samelayer.keys()):
        raise InputError("bulk cutoff `cutvalue' and same-layer cutoff "
            "`cutvalue_samelayer' is incompatible w.r.t. species.")

    species = set()
    for key, value in self._rcut.iteritems():
      spec1, spec2 = key.split('-')
      species.update([spec1, spec2])
    species = list(species)
    num_species = len(species)

    # create integer code for each species type and cutoff values based on species code
    rcutsym = np.zeros([num_species, num_species], dtype=np.double)
    if self._rcut_samelayer is not None:
      rcutsym_samelayer = np.zeros([num_species, num_species], dtype=np.double)

    try:
      for i,si in enumerate(species):
        self._species_code[si] = i
        for j,sj in enumerate(species):
          rcutsym[i][j] = self._rcut[si+'-'+sj]
          if self._rcut_samelayer is not None:
            rcutsym_samelayer[i][j] = self._rcut_samelayer[si+'-'+sj]
    except KeyError as e:
      raise InputError('Cutoff for {} not provided.'.format(e))

    # store cutoff values in cpp class
    if self._rcut_samelayer is not None:
      self._cdesc.set_cutoff_bulk_and_samelayer(self._cutname, rcutsym, rcutsym_samelayer)
    else:
      self._cdesc.set_cutoff_bulk(self._cutname, rcutsym)


  def _set_hyperparams(self):

    # hyperparams of descriptors
    for key, values in self._hyperparams.iteritems():
      if key.lower() not in ['g1', 'g2', 'g3', 'g4', 'g5']:
        raise SupportError("Symmetry function `{}' unsupported.".format(key))

      # g1 needs no hyperparams, put a placeholder
      name = key.lower()
      if name == 'g1':
        # it has no hyperparams, zeros([1,1]) for placeholder
        params = np.zeros([1,1], dtype=np.double)
      else:
        rows = len(values)
        cols = len(values[0])
        params = np.zeros([rows,cols], dtype=np.double)
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
      self._cdesc.add_descriptor(name, params)



class Set51(Descriptor):
  """ Symmetry function descriptor with the hyperparameters from:
  Artrith and Behler, PRB, 85, 045439 (2012)

  Parameters
  ----------

  cutname: string
    cutoff function name, e.g. `cos`

  cutvalue: dict
    cutoff values based on species.

    Example
    -------
    cutvalue = {'C-C': 3.5, 'C-H': 3.0, 'H-H': 1.0}
  """

  def __init__(self, cutvalue, cutname='cos'):


    params = OrderedDict()

    params['g2'] = [
      {'eta':0.001, 'Rs':0.},
      {'eta':0.01,   'Rs':0.},
      {'eta':0.02,   'Rs':0.},
      {'eta':0.035,  'Rs':0.},
      {'eta':0.06,   'Rs':0.},
      {'eta':0.1,    'Rs':0.},
      {'eta':0.2,    'Rs':0.},
      {'eta':0.4,    'Rs':0.}
    ]

    params['g4'] = [
      {'zeta':1 ,  'lambda':-1, 'eta':0.0001},
      {'zeta':1 ,  'lambda':1,  'eta':0.0001},
      {'zeta':2 ,  'lambda':-1, 'eta':0.0001},
      {'zeta':2 ,  'lambda':1,  'eta':0.0001},
      {'zeta':1 ,  'lambda':-1, 'eta':0.003 },
      {'zeta':1 ,  'lambda':1,  'eta':0.003 },
      {'zeta':2 ,  'lambda':-1, 'eta':0.003 },
      {'zeta':2 ,  'lambda':1,  'eta':0.003 },
      {'zeta':1 ,  'lambda':-1,  'eta':0.008 },
      {'zeta':1 ,  'lambda':1,  'eta':0.008 },
      {'zeta':2 ,  'lambda':-1,  'eta':0.008 },
      {'zeta':2 ,  'lambda':1,  'eta':0.008 },
      {'zeta':1 ,  'lambda':-1,  'eta':0.015 },
      {'zeta':1 ,  'lambda':1,  'eta':0.015 },
      {'zeta':2 ,  'lambda':-1,  'eta':0.015 },
      {'zeta':2 ,  'lambda':1,  'eta':0.015 },
      {'zeta':4 ,  'lambda':-1,  'eta':0.015 },
      {'zeta':4 ,  'lambda':1,  'eta':0.015 },
      {'zeta':16,  'lambda':-1,  'eta':0.015 },
      {'zeta':16,  'lambda':1,  'eta':0.015 },
      {'zeta':1 ,  'lambda':-1,  'eta':0.025 },
      {'zeta':1 ,  'lambda':1,  'eta':0.025 },
      {'zeta':2 ,  'lambda':-1,  'eta':0.025 },
      {'zeta':2 ,  'lambda':1,  'eta':0.025 },
      {'zeta':4 ,  'lambda':-1,  'eta':0.025 },
      {'zeta':4 ,  'lambda':1,  'eta':0.025 },
      {'zeta':16,  'lambda':-1,  'eta':0.025 },
      {'zeta':16,  'lambda':1,  'eta':0.025 },
      {'zeta':1 ,  'lambda':-1,  'eta':0.045 },
      {'zeta':1 ,  'lambda':1,  'eta':0.045 },
      {'zeta':2 ,  'lambda':-1,  'eta':0.045 },
      {'zeta':2 ,  'lambda':1,  'eta':0.045 },
      {'zeta':4 ,  'lambda':-1,  'eta':0.045 },
      {'zeta':4 ,  'lambda':1,  'eta':0.045 },
      {'zeta':16,  'lambda':-1,  'eta':0.045 },
      {'zeta':16,  'lambda':1,  'eta':0.045 },
      {'zeta':1 ,  'lambda':-1,  'eta':0.08 },
      {'zeta':1 ,  'lambda':1,  'eta':0.08 },
      {'zeta':2 ,  'lambda':-1,  'eta':0.08 },
      {'zeta':2 ,  'lambda':1,  'eta':0.08 },
      {'zeta':4 ,  'lambda':-1,  'eta':0.08 },
      {'zeta':4 ,  'lambda':1,  'eta':0.08 },
     # {'zeta':16,  'lambda':-1,  'eta':0.08 },
      {'zeta':16,  'lambda':1,  'eta':0.08 }
    ]

    # tranfer units from bohr to angstrom
    bhor2ang = 0.529177
    for key, values in params.iteritems():
      for val in values:
        if key == 'g2':
          val['eta'] /= bhor2ang**2
        elif key == 'g4':
          val['eta'] /= bhor2ang**2

    super(Set51, self).__init__(params, cutname, cutvalue)



class Set51(Descriptor):
  """ Symmetry function descriptor with the hyperparameters from:
  Artrith and Behler, PRB, 85, 045439 (2012)

  Parameters
  ----------

  cutname: string
    cutoff function name, e.g. `cos`

  cutvalue: dict
    cutoff values based on species.

    Example
    -------
    cutvalue = {'C-C': 3.5, 'C-H': 3.0, 'H-H': 1.0}
  """

  def __init__(self, cutvalue, cutname='cos'):


    params = OrderedDict()

    params['g2'] = [
      {'eta':0.0009, 'Rs':0.},
      {'eta':0.01,   'Rs':0.},
      {'eta':0.02,   'Rs':0.},
      {'eta':0.035,  'Rs':0.},
      {'eta':0.06,   'Rs':0.},
      {'eta':0.1,    'Rs':0.},
      {'eta':0.2,    'Rs':0.},
      {'eta':0.4,    'Rs':0.}
    ]

    params['g4'] = [
      {'zeta':1 ,  'lambda':-1, 'eta':0.0001},
      {'zeta':1 ,  'lambda':1,  'eta':0.0001},
      {'zeta':2 ,  'lambda':-1, 'eta':0.0001},
      {'zeta':2 ,  'lambda':1,  'eta':0.0001},
      {'zeta':1 ,  'lambda':-1, 'eta':0.003 },
      {'zeta':1 ,  'lambda':1,  'eta':0.003 },
      {'zeta':2 ,  'lambda':-1, 'eta':0.003 },
      {'zeta':2 ,  'lambda':1,  'eta':0.003 },
      {'zeta':1 ,  'lambda':1,  'eta':0.008 },
      {'zeta':2 ,  'lambda':1,  'eta':0.008 },
      {'zeta':1 ,  'lambda':1,  'eta':0.015 },
      {'zeta':2 ,  'lambda':1,  'eta':0.015 },
      {'zeta':4 ,  'lambda':1,  'eta':0.015 },
      {'zeta':16,  'lambda':1,  'eta':0.015 },
      {'zeta':1 ,  'lambda':1,  'eta':0.025 },
      {'zeta':2 ,  'lambda':1,  'eta':0.025 },
      {'zeta':4 ,  'lambda':1,  'eta':0.025 },
      {'zeta':16,  'lambda':1,  'eta':0.025 },
      {'zeta':1 ,  'lambda':1,  'eta':0.045 },
      {'zeta':2 ,  'lambda':1,  'eta':0.045 },
      {'zeta':4 ,  'lambda':1,  'eta':0.045 },
      {'zeta':16,  'lambda':1,  'eta':0.045 }
    ]


    # tranfer units from bohr to angstrom
    bhor2ang = 0.529177
    for key, values in params.iteritems():
      for val in values:
        if key == 'g2':
          val['eta'] /= bhor2ang**2
        elif key == 'g4':
          val['eta'] /= bhor2ang**2

    super(Set51, self).__init__(params, cutname, cutvalue)

