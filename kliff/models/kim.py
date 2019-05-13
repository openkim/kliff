import os
import numpy as np
import logging
from collections import OrderedDict
import kimpy
from kimpy import neighlist as nl
import kliff
from kliff.dataset import Configuration
from kliff.models.model import ComputeArguments, Model
from kliff.models.parameter import Parameter
from kliff.neighbor import assemble_forces, assemble_stress
from kliff.utils import length_equal
from kliff.error import SupportError

logger = kliff.logger.get_logger(__name__)


class KIMComputeArguments(ComputeArguments):
    """ A Lennard-Jones 6-12 potential.
    """

    implemented_property = []

    def __init__(self, kim_ca, *args, **kwargs):
        """
        Parameters
        ----------
        kim_ca: KIM compute argument

        supported_species: dict
            Key and value are species string and integer code, respectively.
        """
        self.kim_ca = kim_ca
        super(KIMComputeArguments, self).__init__(*args, **kwargs)

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

        self._init_neigh()

        self.refresh(self.influence_distance)

    def check_compute_property(self):
        def add_to_compute_property(compute_property, name):
            if name not in self.implemented_property:
                raise NotImplementedError(
                    '"{}" not implemented in calculator.'.format(name)
                )
            compute_property.append(name)

        self._get_implemented_property()
        compute_property = []
        if self.compute_energy:
            add_to_compute_property(compute_property, 'energy')
        if self.compute_forces:
            add_to_compute_property(compute_property, 'forces')
        if self.compute_stress:
            add_to_compute_property(compute_property, 'stress')
        return compute_property

    def _get_implemented_property(self):

        # check compute arguments
        kim_can = kimpy.compute_argument_name
        N = kim_can.get_number_of_compute_argument_names()

        for i in range(N):
            name, error = kim_can.get_compute_argument_name(i)
            check_error(error, 'kim_can.get_compute_argument_name')

            dtype, error = kim_can.get_compute_argument_data_type(name)
            check_error(error, 'kim_can.get_compute_argument_data_type')

            support_status, error = self.kim_ca.get_argument_support_status(name)
            check_error(error, 'kim_ca.get_argument_support_status')

            # calculator can only handle energy and forces
            if support_status == kimpy.support_status.required:
                if name != kim_can.partialEnergy and name != kim_can.partialForces:
                    report_error('Unsupported required ComputeArgument "{}"'.format(name))

            # supported property
            if name == kim_can.partialEnergy:
                self.implemented_property.append('energy')
            elif name == kim_can.partialForces:
                self.implemented_property.append('forces')
                self.implemented_property.append('stress')

        # check compute callbacks
        kim_ccn = kimpy.compute_callback_name
        num_callbacks = kim_ccn.get_number_of_compute_callback_names()

        for i in range(num_callbacks):
            name, error = kim_ccn.get_compute_callback_name(i)
            check_error(error, 'kim_ccn.get_compute_callback_name')

            support_status, error = self.kim_ca.get_callback_support_status(name)
            check_error(error, 'compute_arguments.get_callback_support_status')

            # calculator only provides get_neigh
            if support_status == kimpy.support_status.required:
                if name != kim_ccn.GetNeighborList:
                    report_error('Unsupported required ComputeCallback "{}"'.format(name))

    def refresh(self, influence_distance=None, params=None):
        self.influence_distance = influence_distance
        self.update_neigh(influence_distance)
        self.register_data(self.compute_energy, self.compute_forces)

    def update_neigh(self, influence_distance):
        """Update neighbor list and model input.

        Parameters
        ----------

        influence_distance: float
        """

        # inquire information from conf
        cell = np.asarray(self.conf.get_cell(), dtype=np.double)
        PBC = np.asarray(self.conf.get_PBC(), dtype=np.intc)
        contributing_coords = np.asarray(self.conf.get_coordinates(), dtype=np.double)
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
            [species_map[s] for s in contributing_species], dtype=np.intc
        )

        if any(PBC):  # need padding atoms
            out = nl.create_paddings(
                influence_distance,
                cell,
                PBC,
                contributing_coords,
                contributing_species_code,
            )
            padding_coords, padding_species_code, self.padding_image_of, error = out
            check_error(error, 'nl.create_paddings')

            num_padding = padding_species_code.size
            self.num_particles = np.array([num_contributing + num_padding], dtype=np.intc)
            tmp = np.concatenate((contributing_coords, padding_coords))
            self.coords = np.asarray(tmp, dtype=np.double)
            tmp = np.concatenate((contributing_species_code, padding_species_code))
            self.species_code = np.asarray(tmp, dtype=np.intc)
            self.particle_contributing = np.ones(self.num_particles[0], dtype=np.intc)
            self.particle_contributing[num_contributing:] = 0
            # TODO check whether padding need neigh and create accordingly
            # for now, create neigh for all atoms, including paddings
            need_neigh = np.ones(self.num_particles[0], dtype=np.intc)

        else:  # do not need padding atoms
            self.padding_image_of = np.array([])
            self.num_particles = np.array([num_contributing], dtype=np.intc)
            self.coords = np.array(contributing_coords, dtype=np.double)
            self.species_code = np.array(contributing_species_code, dtype=np.intc)
            self.particle_contributing = np.ones(num_contributing, dtype=np.intc)
            need_neigh = self.particle_contributing

        error = nl.build(
            self.neigh,
            self.coords,
            influence_distance,
            np.asarray([influence_distance], dtype=np.double),
            need_neigh,
        )
        check_error(error, 'nl.build')

    def register_data(self, compute_energy=True, compute_forces=True):
        """ Register model input and output data in KIM API."""

        # check whether model support energy and forces
        kim_can = kimpy.compute_argument_name
        if compute_energy:
            name = kim_can.partialEnergy
            support_status, error = self.kim_ca.get_argument_support_status(name)
            check_error(error, 'kim_ca.get_argument_support_status')
            if not (
                support_status == kimpy.support_status.required
                or support_status == kimpy.support_status.optional
            ):
                report_error('Energy not supported by model')

        if compute_forces:
            name = kim_can.partialForces
            support_status, error = self.kim_ca.get_argument_support_status(name)
            check_error(error, 'kim_ca.get_argument_support_status')
            if not (
                support_status == kimpy.support_status.required
                or support_status == kimpy.support_status.optional
            ):
                report_error('Forces not supported by model')

        # register argument
        error = self.kim_ca.set_argument_pointer(
            kim_can.numberOfParticles, self.num_particles
        )
        check_error(error, 'kim_can.set_argument_pointer')

        error = self.kim_ca.set_argument_pointer(
            kim_can.particleSpeciesCodes, self.species_code
        )
        check_error(error, 'kim_can.set_argument_pointer')

        error = self.kim_ca.set_argument_pointer(
            kim_can.particleContributing, self.particle_contributing
        )
        check_error(error, 'kim_can.set_argument_pointer')

        error = self.kim_ca.set_argument_pointer(kim_can.coordinates, self.coords)
        check_error(error, 'kim_can.set_argument_pointer')

        if compute_energy:
            self.energy = np.array([0.0], dtype=np.double)
            error = self.kim_ca.set_argument_pointer(kim_can.partialEnergy, self.energy)
            check_error(error, 'kim_can.set_argument_pointer')
        else:
            self.energy = None
            error = self.kim_ca.set_argument_null_pointer(kim_can.partialEnergy)
            check_error(error, 'kim_can.set_argument_null_pointer')

        if compute_forces:
            self.forces = np.zeros([self.num_particles[0], 3], dtype=np.double)
            error = self.kim_ca.set_argument_pointer(kim_can.partialForces, self.forces)
            check_error(error, 'kim_can.set_argument_pointer')
        else:
            self.forces = None
            error = self.kim_ca.set_argument_null_pointer(kim_can.partialForces)
            check_error(error, 'kim_can.set_argument_null_pointer')

    def compute(self, kim_model):
        error = kim_model.compute(self.kim_ca)
        check_error(error, 'kim_model.compute')

        if self.compute_energy:
            self.results['energy'] = self.energy[0]
        if self.compute_forces:
            forces = assemble_forces(
                self.forces, self.num_contributing_particles, self.padding_image_of
            )
            self.results['forces'] = forces
        if self.compute_stress:
            volume = self.conf.get_volume()
            stress = assemble_stress(self.coords, self.forces, volume)
            self.results['stress'] = stress

    def _init_neigh(self):

        # create neighborlist
        neigh = nl.initialize()
        self.neigh = neigh

        # register get neigh callback
        error = self.kim_ca.set_callback_pointer(
            kimpy.compute_callback_name.GetNeighborList, nl.get_neigh_kim(), neigh
        )
        check_error(error, 'compute_arguments.set_callback_pointer')

    def __del__(self):
        """Garbage collection to destroy the neighbor list automatically"""
        if self.neigh is not None:
            nl.clean(self.neigh)
            self.neigh = None


class KIM(Model):
    def __init__(self, model_name=None, params_relation_callback=None):
        super(KIM, self).__init__(model_name, params_relation_callback)

        if self.model_name is None:
            c = self.__class__.__name__
            raise KIMModelError(
                '"model_name" is a mandatory argument for the "{}" calculator.'.format(c)
            )

        self.kim_model = self._initialize()
        self.params = self.inquire_params()

        self.compute_arguments_class = KIMComputeArguments
        self.fitting_params = self.init_fitting_params(self.params)

    def _initialize(self):
        """ Initialize the KIM object"""
        units_accepted, model, error = kimpy.model.create(
            kimpy.numbering.zeroBased,
            kimpy.length_unit.A,
            kimpy.energy_unit.eV,
            kimpy.charge_unit.e,
            kimpy.temperature_unit.K,
            kimpy.time_unit.ps,
            self.model_name,
        )
        check_error(error, 'kimpy.model.create')
        if not units_accepted:
            report_error('requested units not accepted in kimpy.model.create')
        return model

    def inquire_params(self):
        """Inquire the KIM model to get all parameters. """
        params = OrderedDict()

        num_params = self.kim_model.get_number_of_parameters()
        for i in range(num_params):
            out = self.kim_model.get_parameter_metadata(i)
            dtype, extent, name, description, error = out
            check_error(error, 'model.get_parameter_data_type_extent_and_description')

            values = []
            for j in range(extent):
                if str(dtype) == 'Double':
                    value, error = self.kim_model.get_parameter_double(i, j)
                    check_error(error, 'model.get_parameter_double')
                elif str(dtype) == 'Int':
                    value, error = self.kim_model.get_parameter_int(i, j)
                    check_error(error, 'model.get_parameter_int')
                else:  # should never reach here
                    report_error('get unexpeced parameter data type "{}"'.format(dtype))
                values.append(value)
                params[name] = Parameter(
                    value=values, dtype=dtype, description=description
                )
        return params

    def create_a_kim_compute_argument(self):
        kim_ca, error = self.kim_model.compute_arguments_create()
        check_error(error, 'kim_model.compute_arguments_create')
        return kim_ca

    def update_model_params(self):
        """ Update from fitting params to model params. """

        # update kim parameters
        #        # update all components
        #        param_names = self.fitting_params.get_names()
        #        for name in param_names:
        #            i = self.fitting_params.get_index(name)
        #            new_value = self.fitting_params.get_value(name)
        #            for j, v in enumerate(new_value):
        #                self.kim_model.set_parameter(i, j, v)

        # only update optimizing components
        num_params = self.get_number_of_opt_params()
        for i in range(num_params):
            v, p, c = self.get_opt_param_value_and_indices(i)
            self.kim_model.set_parameter(p, c, v)
        # refresh model
        self.kim_model.clear_then_refresh()

        # TODO this seems uncessary
        # the correct way is to reimplemeent set_model_param, get_model_param,
        # and echo_model_param. Also, inquire_model_params seems could be used as
        # get_model_params.
        # this consideration is that we need a parameters object to be passed to
        # FittingParams. This should not be a problem.
        # update model params of the model class
        for name, attr in self.fitting_params.params.items():
            self.set_model_params(name, attr['value'], check_shape=False)

        if logger.getEffectiveLevel() == logging.DEBUG:
            params = self.inquire_params()
            s = ''
            for name, p in params.items():
                s += '\nname: {}\n'.format(name)
                s += p.to_string()
            logger.debug(s)

    #    def get_cutoff(self):
    #        """Get the largest cutoff of a model.
    #
    #        Return: float
    #          cutoff
    #        """
    #
    #        cutoff = self.kim_model.get_influence_distance()
    #
    #        # TODO we need to make changes to support multiple cutoffs
    #        # TODO modify kimpy to change the function name
    #        model_cutoffs, padding_hints = self.kim_model.get_neighbor_list_cutoffs_and_hints()
    #        if model_cutoffs.size != 1:
    #            report_error('too many cutoffs')
    #
    #        return cutoff

    def get_influence_distance(self):
        """Get the influence distance of a model.

        Return: float
            influence distance
        """
        return self.kim_model.get_influence_distance()

    def get_supported_species(self):
        """Get all the supported species.

        Return
        ------
        species: dictionary
            Key and value are species string and integer code, respectively.
        """
        species = {}
        num_kim_species = kimpy.species_name.get_number_of_species_names()

        for i in range(num_kim_species):
            species_name, error = kimpy.species_name.get_species_name(i)
            check_error(error, 'kimpy.species_name.get_species_name')
            supported, code, error = self.kim_model.get_species_support_and_code(
                species_name
            )
            check_error(error, 'kim_model.get_species_support_and_code')
            if supported:
                species[str(species_name)] = code

        return species

    def write_kim_model(self, path=None):
        """Write out a KIM model that can be used directly with the kim-api.

        This function typically write two files to `path`: (1) CMakeLists.txt, and
        (2) kliff_trained.params. `path` will be created if it does not exist.


        Parameters
        ----------
        path: str (optional)
            Path to the newly trained model.
            If `None`, it is set to `./$(MODEL_NAME)_kliff_trained`, where
            `MODEL_NAME` is the `model_name` that is provided at the instanization of
            this class.

        Note
        ----
        This only works for parameterized KIM models that support the writing of
        parameters.
        """
        present, required, error = self.kim_model.is_routine_present(
            kimpy.model_routine_name.WriteParameterizedModel
        )
        check_error(error, 'kim_model.is_routine_is_routine_present')
        if not present:
            raise SupportError(
                'This KIM model does not support the writing of parameters.'
            )

        if path is None:
            path = os.path.join(os.getcwd(), self.model_name + '_kliff_trained')
        if path and not os.path.exists(path):
            os.makedirs(path)
        fname = 'kliff_trained'

        error = self.kim_model.write_parameterized_model(path, fname)
        check_error(error, 'kim_model.write_parameterized_model')


class KIMModelError(Exception):
    def __init__(self, msg):
        super(KIMModelError, self).__init__(msg)
        self.msg = msg

    def __expr__(self):
        return self.msg


def check_error(error, msg):
    if error != 0 and error is not None:
        raise KIMModelError(
            'Calling "{}" failed.\nSee "kim.log" for more infomation.'.format(msg)
        )


def report_error(msg):
    raise KIMModelError(msg)
