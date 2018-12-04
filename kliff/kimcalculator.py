from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import kimpy
from kimpy import neighlist as nl
from .dataset import Configuration
from .error import SupportError
from .error import InputError
from .error import check_error


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

    def __init__(self, compute_arguments, conf, supported_species, debug=False):

        self.compute_arguments = compute_arguments
        self.conf = conf
        self.supported_species = supported_species
        self.debug = debug

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

    def update_neigh(self, cutoff):
        """Update neighbor list and model input.

        Parameter
        ---------

        cutoff: float
        """

        # inquire information from conf
        cell = np.asarray(self.conf.get_cell(), dtype=np.double)
        pbc = np.asarray(self.conf.get_pbc(), dtype=np.intc)
        contributing_coords = np.asarray(
            self.conf.get_coordinates(), dtype=np.double)
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
            padding_coords, padding_species_code, self.padding_image_of, error = nl.create_paddings(
                cutoff, cell, pbc, contributing_coords, contributing_species_code)
            check_error(error, 'nl.create_paddings')
            num_padding = padding_species_code.size

            self.num_particles = np.array(
                [num_contributing + num_padding], dtype=np.intc)
            tmp = np.concatenate((contributing_coords, padding_coords))
            self.coords = np.asarray(tmp, dtype=np.double)
            tmp = np.concatenate(
                (contributing_species_code, padding_species_code))
            self.species_code = np.asarray(tmp, dtype=np.intc)
            self.particle_contributing = np.ones(
                self.num_particles[0], dtype=np.intc)
            self.particle_contributing[num_contributing:] = 0
            # create neigh for all atoms, including paddings
            need_neigh = np.ones(self.num_particles[0], dtype=np.intc)

        else:  # do not need padding atoms
            self.padding_image_of = np.array([])
            self.num_particles = np.array([num_contributing], dtype=np.intc)
            self.coords = np.array(contributing_coords, dtype=np.double)
            self.species_code = np.array(
                contributing_species_code, dtype=np.intc)
            self.particle_contributing = np.ones(
                num_contributing, dtype=np.intc)
            need_neigh = self.particle_contributing

        error = nl.build(self.neigh, self.coords, cutoff,
                         np.asarray([cutoff], dtype=np.double), need_neigh)
        check_error(error, 'nl.build')

    def register_data(self, compute_energy=True, compute_forces=True):
        """ Register model input and output data in KIM API."""

        # check whether model support energy and forces
        if compute_energy:
            name = kimpy.compute_argument_name.partialEnergy
            support_status, error = self.compute_arguments.get_argument_support_status(
                name)
            check_error(error, 'compute_arguments.get_argument_support_status')
            if (not (support_status == kimpy.support_status.required or
                     support_status == kimpy.support_status.optional)):
                report_error('Energy not supported by model')

        if compute_forces:
            name = kimpy.compute_argument_name.partialForces
            support_status, error = self.compute_arguments.get_argument_support_status(
                name)
            check_error(error, 'compute_arguments.get_argument_support_status')
            if (not (support_status == kimpy.support_status.required or
                     support_status == kimpy.support_status.optional)):
                report_error('Forces not supported by model')

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

        if compute_energy:
            self.energy = np.array([0.], dtype=np.double)
            error = self.compute_arguments.set_argument_pointer(
                kimpy.compute_argument_name.partialEnergy, self.energy)
            check_error(
                error, 'kimpy.compute_argument_name.set_argument_pointer')
        else:
            self.energy = None
            error = self.compute_arguments.set_argument_null_pointer(
                kimpy.compute_argument_name.partialEnergy)
            check_error(
                error, 'kimpy.compute_argument_name.set_argument_null_pointer')

        if compute_forces:
            self.forces = np.zeros([self.num_particles[0], 3], dtype=np.double)
            error = self.compute_arguments.set_argument_pointer(
                kimpy.compute_argument_name.partialForces, self.forces)
            check_error(
                error, 'kimpy.compute_argument_name.set_argument_pointer')
        else:
            self.forces = None
            error = self.compute_arguments.set_argument_null_pointer(
                kimpy.compute_argument_name.partialForces)
            check_error(
                error, 'kimpy.compute_argument_name.set_argument_null_pointer')

    def get_energy(self):
        if self.energy is not None:
            return self.energy[0]
        else:
            raise SupportError("energy")

    def get_forces(self):
        if self.forces is not None:
            return _assemble_padding_forces(self.forces, self.num_contributing_particles,
                                            self.padding_image_of)
        else:
            raise SupportError("forces")

    def get_prediction(self):
        if self.energy is not None:
            energy = self.get_energy()

        if self.forces is not None:
            forces = self.get_forces()

        # pack prediction data
        if self.energy is not None:
            if self.forces is not None:
                return np.concatenate(([energy], forces.ravel()))
            else:
                return np.asarray([energy])
        else:
            if self.forces is not None:
                return forces.ravel()
            else:
                raise SupportError("both energy and forces")

    def get_reference(self):
        energy = self.conf.get_energy()
        forces = self.conf.get_forces()

        # check we have the reference for required values
        if self.energy is not None and energy is None:
            raise InputError('reference "energy" not provided')
        if self.forces is not None and forces is None:
            raise InputError('reference "forces" not provided')

        # pack reference data
        if self.energy is not None:
            if self.forces is not None:
                return np.concatenate(([energy], forces.ravel()))
            else:
                return np.asarray([energy])
        else:
            if self.forces is not None:
                return forces.ravel()
            else:
                raise SupportError("both energy and forces")

    def get_compute_energy(self):
        if self.energy is None:
            return False
        else:
            return True

    def get_compute_forces(self):
        if self.forces is None:
            return False
        else:
            return True

    def get_compute_arguments(self):
        return self.compute_arguments

    def _check_support_status(self):

        # check compute arguments
        num_compute_arguments = kimpy.compute_argument_name.get_number_of_compute_argument_names()

        for i in range(num_compute_arguments):
            name, error = kimpy.compute_argument_name.get_compute_argument_name(i)
            check_error(error, 'kimpy.compute_argument_name.get_compute_argument_name')

            dtype, error = kimpy.compute_argument_name.get_compute_argument_data_type(
                name)
            check_error(
                error, 'kimpy.compute_argument_name.get_compute_argument_data_type')

            support_status, error = self.compute_arguments.get_argument_support_status(
                name)
            check_error(error, 'compute_arguments.get_argument_support_status')

            # calculator can only handle energy and forces
            if support_status == kimpy.support_status.required:
                if (name != kimpy.compute_argument_name.partialEnergy and
                        name != kimpy.compute_argument_name.partialForces):
                    report_error(
                        'Unsupported required ComputeArgument "{}"'.format(name))

        # check compute callbacks
        num_callbacks = kimpy.compute_callback_name.get_number_of_compute_callback_names()

        for i in range(num_callbacks):

            name, error = kimpy.compute_callback_name.get_compute_callback_name(
                i)
            check_error(
                error, 'kimpy.compute_callback_name.get_compute_callback_name')

            support_status, error = self.compute_arguments.get_callback_support_status(
                name)
            check_error(error, 'compute_arguments.get_callback_support_status')

            # calculator only provides get_neigh
            if support_status == kimpy.support_status.required:
                if name != kimpy.compute_callback_name.GetNeighborList:
                    report_error(
                        'Unsupported required ComputeCallback: {}'.format(name))

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

    compute_energy: bool
      whether to include energy in the prediction output

    """

    def __init__(self, modelname, debug=False):
        # input data
        self.modelname = modelname
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

    def create(self, configs, use_energy=True, use_forces=True):
        """Create compute arguments for configurations.

        Parameters
        ----------

        configs: Configuration object or a list of Configuration object

        use_energy: bool
          Whether to require the model compute energy.

        use_forces: bool
          Whether to require the model compute forces.
        """

        # a single configuration
        if isinstance(configs, Configuration):
            configs = [configs]

        try:
            iter(use_energy)
        except TypeError:
            use_energy = [use_energy for _ in range(len(configs))]
        try:
            iter(use_forces)
        except TypeError:
            use_forces = [use_forces for _ in range(len(configs))]

        supported_species = self.get_model_supported_species()
        cutoff = self.get_cutoff()

        for i, conf in enumerate(configs):
            compute_arguments, error = self.kim_model.compute_arguments_create()
            check_error(error, 'kim_model.compute_arguments_create')
            self.compute_arguments.append(compute_arguments)
            in_out = KIMInputAndOutput(
                compute_arguments, conf, supported_species)
            in_out.update_neigh(cutoff*1.001)
            in_out.register_data(use_energy[i], use_forces[i])
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
        num_kim_species = kimpy.species_name.get_number_of_species_names()

        for i in range(num_kim_species):
            species_name, error = kimpy.species_name.get_species_name(i)
            check_error(error, 'kimpy.species_name.get_species_name')
            supported, code, error = self.kim_model.get_species_support_and_code(
                species_name)
            check_error(error, 'kim_model.get_species_support_and_code')
            if supported:
                species[str(species_name)] = code

        return species

    def get_cutoff(self):
        """Get the largest cutoff of a model.

        Return: float
          cutoff
        """

        cutoff = self.kim_model.get_influence_distance()

        # TODO we need to make changes to support multiple cutoffs
        model_cutoffs, padding_hints = self.kim_model.get_neighbor_list_cutoffs_and_hints()
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
        param_names = model_params.get_names()
        for name in param_names:
            i = model_params.get_index(name)
            new_value = model_params.get_value(name)
            for j, v in enumerate(new_value):
                self.kim_model.set_parameter(i, j, v)

        # refresh model
        self.kim_model.clear_then_refresh()

        # print parameters in KIM object
        if self.debug:
            num_params = self.kim_model.get_number_of_parameters()
            print('='*80)
            for i in range(num_params):
                out = self.kim_model.get_parameter_data_type_extent_and_description(
                    i)
                dtype, extent, description, error = out
                check_error(
                    error, 'kim_model.get_parameter_data_type_extent_and_description')

                print('\nParameter No.: {} \ndata type: "{}" \nextent {}: \ndescription: '
                      '"{}", \nvalues:'.format(i, dtype, extent, description), end='')

                for j in range(extent):
                    if str(dtype) == 'Double':
                        value, error = self.kim_model.get_parameter_double(
                            i, j)
                        check_error(error, 'kim_model.get_parameter_double')
                    elif str(dtype) == 'Int':
                        value, error = self.kim_model.get_parameter_int(i, j)
                        check_error(error, 'kim_model.get_parameter_int')
                    else:  # should never reach here
                        report_error(
                            'get unexpeced parameter data type "{}"'.format(dtype))
                    print(value)

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

    def get_prediction(self, in_out):
        """
        Parameters
        ----------

        in_out: KIMInputAndOutput object

        Return: 2D array
          predictions by model for configuration associated with in_out
        """
        return in_out.get_prediction()

    def get_reference(self, in_out):
        """
        Parameters
        ----------

        in_out: KIMInputAndOutput object

        Return: 2D array
          reference value for configuration associated with in_out
        """
        return in_out.get_reference()

    def __del__(self):
        """Garbage collection to destroy the KIM API object automatically"""

        if self.kim_model:
            for compute_arguments in self.compute_arguments:
                error = self.kim_model.compute_arguments_destroy(
                    compute_arguments)
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

    # numpy slicing does not make a copy !!!
    total_forces = np.array(forces[:num_contributing])

    has_padding = True if padding_image_of.size != 0 else False

    if has_padding:

        pad_forces = forces[num_contributing:]
        num_padding = pad_forces.shape[0]

        if num_contributing < num_padding:
            for i in range(num_contributing):
                # indices: the indices of padding atoms that are images of contributing atom i
                indices = np.where(padding_image_of == i)
                total_forces[i] += np.sum(pad_forces[indices], axis=0)
        else:
            for i in range(num_padding):
                total_forces[padding_image_of[i]] += pad_forces[i]

    return total_forces
