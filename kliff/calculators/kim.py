import numpy as np
import kimpy
from kimpy import neighlist as nl
from .calculator import ComputeArgument
from .calculator import Calculator
from .calculator import Parameter
from ..neighbor import assemble_forces
from ..neighbor import assemble_stress


class KIMComputeArgument(ComputeArgument):
    """ A Lennard-Jones 6-12 potential.
    """

    implemented_property = []

    def __init__(self, *args, **kwargs):
        super(KIMComputeArgument, self).__init__(*args, **kwargs)
        self.refresh()

    def refresh(self):
        # create neighbor list
        cutoff = {'C-C': self.cutoff}
        neigh = NeighborList(self.conf, cutoff, padding_need_neigh=True)
        self.neigh = neigh

    def compute(self, params):
        epsilon = params['epsilon']
        sigma = params['sigma']

        rcut = self.cutoff
        coords = self.conf.coords

        coords_including_padding = np.reshape(self.neigh.coords, (-1, 3))
        if self.compute_forces:
            forces_including_padding = np.zeros_like(coords_including_padding)

        energy = 0
        for i, xyz_i in enumerate(coords):
            for j in self.neigh.neighlist[i]:
                xyz_j = coords_including_padding[j]
                rij = xyz_j - xyz_i
                r = np.linalg.norm(rij)
                if self.compute_forces:
                    phi, dphi = self.calc_phi_dphi(epsilon, sigma, r, rcut)
                    energy += 0.5*phi
                    pair = 0.5*dphi/r*rij
                    forces_including_padding[i] += pair
                    forces_including_padding[j] -= pair
                elif self.compute_energy:
                    phi = self.calc_phi_dphi(epsilon, sigma, r, rcut)
                    energy += 0.5*phi

        if self.compute_energy:
            self.results['energy'] = energy
        if self.compute_forces:
            forces = assemble_forces(
                forces_including_padding, self.neigh.ncontrib, self.neigh.image_pad)
            self.results['forces'] = forces
        if self.compute_stress:
            volume = self.conf.get_volume()
            stress = assemble_stress(
                coords_including_padding, forces_including_padding, volume)
            self.results['stress'] = stress


class KIM(Calculator):

    def __init__(self, *args, **kwargs):
        super(KIM, self).__init__(*args, **kwargs)

        if self.model_name is None:
            c = self.__class__.__name__
            raise KIMCalculatorError(
                '"model_name" is a mandatory argument for the "{}" calculator.'.format(c))

        self.model = self._initialize()
        self._inquire_params()

        self.compute_argument_class = KIMComputeArgument
        self.fitting_params = self.init_fitting_params(self.params)

    def _initialize(self):
        """ Initialize the KIM object"""
        # create model
        units_accepted, model, error = kimpy.model.create(
            kimpy.numbering.zeroBased,
            kimpy.length_unit.A,
            kimpy.energy_unit.eV,
            kimpy.charge_unit.e,
            kimpy.temperature_unit.K,
            kimpy.time_unit.ps,
            self.model_name)
        check_error(error, 'kimpy.model.create')
        if not units_accepted:
            report_error('requested units not accepted in kimpy.model.create')
        return model

    def _inquire_params(self):
        """
        Inquire the KIM model to get all the parameters.
        """

        num_params = self.model.get_number_of_parameters()

        for i in range(num_params):
            out = self.model.get_parameter_metadata(i)
            dtype, extent, name, description, error = out
            check_error(
                error, 'model.get_parameter_data_type_extent_and_description')

            values = []
            for j in range(extent):
                if str(dtype) == 'Double':
                    value, error = self.model.get_parameter_double(i, j)
                    check_error(error, 'model.get_parameter_double')
                elif str(dtype) == 'Int':
                    value, error = self.model.get_parameter_int(i, j)
                    check_error(error, 'model.get_parameter_int')
                else:  # should never reach here
                    report_error(
                        'get unexpeced parameter data type "{}"'.format(dtype))
                values.append(value)
            self.params[name] = Parameter(
                value=values, dtype=dtype, description=description)


class InputError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value


class SupportError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class InitializationError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value) + ' initialization failed'


class KIMCalculatorError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


def check_error(error, msg):
    if error != 0 and error is not None:
        raise KIMCalculatorError('Calling "{}" failed.'.format(msg))


def report_error(msg):
    raise KIMCalculatorError(msg)
