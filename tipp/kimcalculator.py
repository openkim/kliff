#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import kimpy
import neighlist as nl
from dataset import Configuration
from error import SupportError
from error import check_error
from species_name_map import species_name_map


class KIMInputAndOutput(object):
  """ Model input and out data associated with compute arguments.

  Parameters
  ----------

  compute_arguments: ComputeArguments object of KIM

  conf: Configuration object

  supported_species: dict
    All supported species of a KIM model, with key a species string and value the
    species integer code.

  """

  def __init__(self, compute_arguments, conf, supported_species):
    self.compute_arguments = compute_arguments
    self.conf = conf
    self.supported_species = supported_species

    # neighbor list
    self.neigh = None
    self.num_contributing_particles = None
    self.num_padding_particles = None
    self.padding_image_of = None

    # model input
    self.num_particles = None
    self.species_code = None
    self.particle_contributing = None
    self.coords = None

    # model output
    self.energy = None
    self.forces = None

    self._check_support_status()
    self._init_neigh()


  def create_neigh(self, cutoff):
    """Create neighbor list and model input.

    Parameter
    ---------

    cutoff: float
    """

    # inquire information from conf
    cell = self.conf.get_cell()
    pbc = self.conf.get_pbc()
    contributing_coords = self.conf.get_coordinates()
    contributing_species = self.conf.get_species()
    num_contributing = self.conf.get_number_of_atoms()
    self.num_contributing_particles = num_contributing


    # species support and code
    unique_species = list(set(contributing_species))
    species_map = dict()
    for s in unique_species:
      if s in self.supported_species:
        species_map[s] = self.supported_species[s]
      else:
        report_error('species "{}" not supported by model'.format(s))
    contributing_species_code = np.array(
        [species_map[s] for s in contributing_species], dtype=np.intc)


    if any(pbc):  # need padding atoms
      # create padding atoms
      padding_coords,padding_species_code,self.padding_image_of,error = nl.create_paddings(
          cutoff, cell, pbc, contributing_coords, contributing_species_code)
      check_error(error, 'nl.create_paddings')
      num_padding = padding_species_code.size

      self.num_particles = np.array([num_contributing + num_padding], dtype=np.intc)
      tmp = np.concatenate((contributing_coords, padding_coords))
      self.coords = np.asarray(tmp, dtype=np.double)
      tmp = np.concatenate((contributing_species_code, padding_species_code))
      self.species_code = np.asarray(tmp, dtype=np.intc)
      self.particle_contributing = np.ones(self.num_particles[0], dtype=np.intc)
      self.particle_contributing[num_contributing:] = 0
      # create neigh for all atoms, including paddings
      need_neigh = np.ones(self.num_particles[0], dtype=np.intc)

    else:  # do not need padding atoms
      self.padding_image_of = np.array([])
      self.num_particles = np.array([num_contributing], dtype=np.intc)
      self.coords = np.array(contributing_coords, dtype=np.double)
      self.species_code = np.array(contributing_species_code, dtype=np.intc)
      self.particle_contributing = np.ones(num_contributing, dtype=np.intc)
      need_neigh = self.particle_contributing

    error = nl.build(self.neigh, cutoff, self.coords, need_neigh)
    check_error(error, 'nl.build')


  def register_data(self):
    """ Register model input and output data in KIM API."""

    # model output
    self.energy = np.array([0.], dtype=np.double)
    self.forces = np.zeros([self.num_particles[0], 3], dtype=np.double)

    # register argument
    error = self.compute_arguments.set_argument_pointer(
        kimpy.compute_argument_name.numberOfParticles, self.num_particles)
    check_error(error, 'kimpy.compute_argument_name.set_argument_pointer')

    error = self.compute_arguments.set_argument_pointer(
        kimpy.compute_argument_name.particleSpeciesCodes, self.species_code)
    check_error(error, 'kimpy.compute_argument_name.set_argument_pointer')

    error = self.compute_arguments.set_argument_pointer(
        kimpy.compute_argument_name.particleContributing, self.particle_contributing)
    check_error(error, 'kimpy.compute_argument_name.set_argument_pointer')

    error = self.compute_arguments.set_argument_pointer(
        kimpy.compute_argument_name.coordinates, self.coords)
    check_error(error, 'kimpy.compute_argument_name.set_argument_pointer')

    error = self.compute_arguments.set_argument_pointer(
        kimpy.compute_argument_name.partialEnergy, self.energy)
    check_error(error, 'kimpy.compute_argument_name.set_argument_pointer')

    error = self.compute_arguments.set_argument_pointer(
        kimpy.compute_argument_name.partialForces, self.forces)
    check_error(error, 'kimpy.compute_argument_name.set_argument_pointer')


  def get_energy(self):
    if self.energy is not None:
      return self.energy[0]
    else:
      raise SupportError("energy")

  def get_forces(self):
    if self.forces is not None:
      return _assemble_padding_forces(
          self.forces, self.num_contributing_particles,  self.padding_image_of)
    else:
      raise SupportError("force")

  def get_compute_arguments(self):
    return self.compute_arguments



  def _check_support_status(self):

    # check compute arguments
    num_compute_arguments = kimpy.compute_argument_name.get_number_of_compute_argument_names()

    for i in range(num_compute_arguments):
      name, error = kimpy.compute_argument_name.get_compute_argument_name(i)
      check_error(error, 'kimpy.compute_argument_name.get_compute_argument_name')

      dtype, error = kimpy.compute_argument_name.get_compute_argument_data_type(name)
      check_error(error, 'kimpy.compute_argument_name.get_compute_argument_data_type')

      support_status, error = self.compute_arguments.get_argument_support_status(name)
      check_error(error, 'compute_arguments.get_argument_support_status')

      # can only handle energy and force
      if support_status == kimpy.support_status.required:
        if (name != kimpy.compute_argument_name.partialEnergy or
            name != kimpy.compute_argument_name.partialForces):
          report_error('Unsupported required ComputeArgument "{}"'.format(name))

    #check compute callbacks
    num_callbacks = kimpy.compute_callback_name.get_number_of_compute_callback_names()

    for i in range(num_callbacks):

      name, error = kimpy.compute_callback_name.get_compute_callback_name(i)
      check_error(error, 'kimpy.compute_callback_name.get_compute_callback_name')

      support_status, error = self.compute_arguments.get_callback_support_status(name)
      check_error(error, 'compute_arguments.get_callback_support_status')

      # cannot handle any "required" callbacks
      if support_status == kimpy.support_status.required:
        report_error('Unsupported required ComputeCallback: {}'.format(name))


  def _init_neigh(self):

    # create neighborlist
    neigh = nl.initialize()
    self.neigh = neigh

    # register get neigh callback
    error = self.compute_arguments.set_callback_pointer(
        kimpy.compute_callback_name.GetNeighborList,
        nl.get_neigh_kim(),
        neigh
    )
    check_error(error, 'compute_arguments.set_callback_pointer')


  def __del__(self):
    """Garbage collection to destroy the neighbor list automatically"""
    if self.neigh:
      nl.clean(self.neigh)



class KIMCalculator(object):
  """ KIM calculator that computes the energy and forces from a KIM model.

  Parameters
  ----------

  modelname: str
    KIM model name

  use_energy: bool
    whether to include energy in the prediction output

  """

  def __init__(self, modelname, use_energy=False, debug=False):
    # input data
    self.modelname = modelname
    self.use_energy = use_energy
    self.debug = debug

    # model data
    self.kim_model = None

    # input and outout data associated with each compute arguments
    self.compute_arguments = []
    self.kim_input_and_output = []

    # create kim model object
    self._initialize()


  def _initialize(self):
    """ Initialize the KIM object"""


    # create model
    units_accepted, kim_model, error = kimpy.model.create(
        kimpy.numbering.zeroBased,
        kimpy.length_unit.A,
        kimpy.energy_unit.eV,
        kimpy.charge_unit.e,
        kimpy.temperature_unit.K,
        kimpy.time_unit.ps,
        self.modelname
    )
    check_error(error, 'kimpy.model.create')
    if not units_accepted:
      report_error('requested units not accepted in kimpy.model.create')
    self.kim_model = kim_model



  def create(self, configs):
    """Create compute arguments for configurations.

    Parameters
    ----------

    configs: Configuration object or a list of Configuration object
    """

    supported_species = self.get_model_supported_species()
    cutoff = self.get_cutoff()

    # a single configuration
    if isinstance(configs, Configuration):
      configs = [configs]

    for conf in configs:
      compute_arguments, error = self.kim_model.compute_arguments_create()
      check_error(error, 'kim_model.compute_arguments_create')
      self.compute_arguments.append(compute_arguments)
      in_out = KIMInputAndOutput(compute_arguments, conf, supported_species)
      in_out.create_neigh(cutoff)
      in_out.register_data()
      self.kim_input_and_output.append(in_out)

    return self.kim_input_and_output


  def compute(self, in_out):
    """
    Parameters
    ----------

    in_out: KIMInputAndOutput object
    """
    compute_arguments = in_out.get_compute_arguments()
    error = self.kim_model.compute(compute_arguments)
    check_error(error, 'kim_model.compute')


  def get_kim_input_and_output(self):
    return self.kim_input_and_output


  def get_model_supported_species(self):
    """Get all the supported species by a model.

    Return: dictionary key:str, value:int
    """
    species = {}
    for key, value in species_name_map.iteritems():
      supported, code, error = self.kim_model.get_species_support_and_code(value)
      check_error(error, 'kim_model.get_species_support_and_code')
      if supported:
        species[key] = code

    return species


  def get_cutoff(self):
    """Get the largest cutoff of a model.

    Return: float
      cutoff
    """

    cutoff = self.kim_model.get_influence_distance()

    # TODO we need to make changes to support multiple cutoffs
    model_cutoffs = self.kim_model.get_neighbor_list_cutoffs()
    if model_cutoffs.size != 1:
      report_error('too many cutoffs')

    return cutoff


  def update_params(self, model_params):
    """Update parameters from ModelParameters class to KIM object.

    Parameters
    ----------
    model_params: ModelParameters object
    """

    # update values to KIM object
    # The ordered of parameters is guarauted since the get_names() function
    # of ModelParameters with return names with the same order as in KIM object
    param_names = model_params.get_names()
    for i, name in enumerate(param_names):
      new_value = model_params.get_value(name)
      for j, v in enumerate(new_value):
        self.kim_model.set_parameter(i, j, v)

    # refresh model
    self.kim_model.clean_influence_distance_and_cutoffs_then_refresh_model()


  def get_energy(self, in_out):
    """
    Parameters
    ----------

    in_out: KIMInputAndOutput object

    Return: float
      energy of configuration
    """
    return in_out.get_energy()

  def get_forces(self, in_out):
    """
    Parameters
    ----------

    in_out: KIMInputAndOutput object

    Return: 2D array
      forces on atoms
    """
    return in_out.get_forces()


  def __del__(self):
    """Garbage collection to destroy the KIM API object automatically"""

    if self.kim_model:
      for compute_arguments in self.compute_arguments:
        error = self.kim_model.compute_arguments_destroy(compute_arguments)
        check_error(error, 'kim_model.compute_arguments_destroy')

      # free kim model
      kimpy.model.destroy(self.kim_model)
      self.compute_arguments = []
      self.kim_input_and_output = []



def _assemble_padding_forces(forces, num_contributing, padding_image_of):
  """
  Assemble forces on padding atoms back to contributing atoms.

  Parameters
  ----------

  forces: 2D array
    forces on both contributing and padding atoms

  num_contributing: int
    number of contributing atoms

  padding_image_of: 1D int array
    atom number, of which the padding atom is an image


  Return
  ------
    Total forces on contributing atoms.
  """

  total_forces = forces[:num_contributing]

  has_padding = True if padding_image_of.size != 0 else False

  if has_padding:

    pad_forces = forces[num_contributing:]
    num_padding = pad_forces.shape[0]

    if num_contributing < num_padding:
      for i in xrange(num_contributing):
        # indices: the indices of padding atoms that are images of contributing atom i
        indices = np.where(padding_image_of == i)
        total_forces[i] += np.sum(pad_forces[indices], axis=0)
    else:
      for i in xrange(num_padding):
        total_forces[padding_image_of[i]] += pad_forces[i]

  return total_forces

