import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from kliff.dataset.dataset import Configuration
from kliff.error import report_import_error
from kliff.log import log_entry
from kliff.models.model import ComputeArguments, Model
from kliff.models.parameter import Parameter
from kliff.neighbor import assemble_forces, assemble_stress

try:
    import kimpy
    from kimpy import neighlist as nl

    kimpy_avail = True
except ImportError:
    kimpy_avail = False


logger = logging.getLogger(__name__)


class KIMComputeArguments(ComputeArguments):
    """
    KIMModel potentials arguments.

    Args:
        kim_ca: KIM compute argument, can be created by
            :meth:`~kliff.models.KIMModels.create_a_compute_argument()`.
        conf: atomic configurations
        supported_species: species supported by the potential model, with chemical
            symbol as key and integer code as value.
        influence_distance: influence distance (aka cutoff distance) to calculate neighbors
        compute_energy: whether to compute energy
        compute_forces: whether to compute forces
        compute_stress: whether to compute stress
    """

    implemented_property = []

    def __init__(
        self,
        kim_ca,
        config: Configuration,
        supported_species: Dict[str, int],
        influence_distance: float,
        compute_energy: bool = True,
        compute_forces: bool = True,
        compute_stress: bool = False,
    ):
        if not kimpy_avail:
            report_import_error("kimpy", self.__class__.__name__)

        # kim compute argument
        self.kim_ca = kim_ca

        # get supported property
        self._get_implemented_property()

        super(KIMComputeArguments, self).__init__(
            config,
            supported_species,
            influence_distance,
            compute_energy,
            compute_forces,
            compute_stress,
        )

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
        self._update_neigh(influence_distance)
        self._register_data(compute_energy, compute_forces)

    def _get_implemented_property(self):
        """ "
        Get implemented property of model.
        """

        # check compute arguments
        kim_can = kimpy.compute_argument_name
        N = kim_can.get_number_of_compute_argument_names()

        for i in range(N):
            name, error = kim_can.get_compute_argument_name(i)
            check_error(error, "kim_can.get_compute_argument_name")

            dtype, error = kim_can.get_compute_argument_data_type(name)
            check_error(error, "kim_can.get_compute_argument_data_type")

            support_status, error = self.kim_ca.get_argument_support_status(name)
            check_error(error, "kim_ca.get_argument_support_status")

            # calculator can only handle energy and forces
            if support_status == kimpy.support_status.required:
                if name != kim_can.partialEnergy and name != kim_can.partialForces:
                    report_error(f"Unsupported required ComputeArgument `{name}`")

            # supported property
            if name == kim_can.partialEnergy:
                self.implemented_property.append("energy")
            elif name == kim_can.partialForces:
                self.implemented_property.append("forces")
                self.implemented_property.append("stress")

        # check compute callbacks
        kim_ccn = kimpy.compute_callback_name
        num_callbacks = kim_ccn.get_number_of_compute_callback_names()

        for i in range(num_callbacks):
            name, error = kim_ccn.get_compute_callback_name(i)
            check_error(error, "kim_ccn.get_compute_callback_name")

            support_status, error = self.kim_ca.get_callback_support_status(name)
            check_error(error, "compute_arguments.get_callback_support_status")

            # calculator only provides get_neigh
            if support_status == kimpy.support_status.required:
                if name != kim_ccn.GetNeighborList:
                    report_error(f"Unsupported required ComputeCallback `{name}`")

    def _init_neigh(self):

        # create neighbor list
        neigh = nl.initialize()
        self.neigh = neigh

        # register get neigh callback
        error = self.kim_ca.set_callback_pointer(
            kimpy.compute_callback_name.GetNeighborList, nl.get_neigh_kim(), neigh
        )
        check_error(error, "compute_arguments.set_callback_pointer")

    def _update_neigh(self, influence_distance: float):
        """
        Update neighbor list and model input.
        """

        # inquire information from conf
        cell = np.asarray(self.conf.cell, dtype=np.double)
        PBC = np.asarray(self.conf.PBC, dtype=np.intc)
        contributing_coords = np.asarray(self.conf.coords, dtype=np.double)
        contributing_species = self.conf.species
        num_contributing = self.conf.get_num_atoms()
        self.num_contributing_particles = num_contributing

        # species support and code
        unique_species = list(set(contributing_species))
        species_map = dict()
        for s in unique_species:
            if s in self.supported_species:
                species_map[s] = self.supported_species[s]
            else:
                report_error(f"species `{s}` not supported by model")
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
            check_error(error, "nl.create_paddings")

            num_padding = padding_species_code.size
            self.num_particles = np.array(
                [num_contributing + num_padding], dtype=np.intc
            )
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
        check_error(error, "nl.build")

    def _register_data(self, compute_energy=True, compute_forces=True):
        """
        Register model input and output data in KIM API.
        """

        # check whether model support energy and forces
        kim_can = kimpy.compute_argument_name
        if compute_energy:
            name = kim_can.partialEnergy
            support_status, error = self.kim_ca.get_argument_support_status(name)
            check_error(error, "kim_ca.get_argument_support_status")
            if not (
                support_status == kimpy.support_status.required
                or support_status == kimpy.support_status.optional
            ):
                report_error("Energy not supported by model")

        if compute_forces:
            name = kim_can.partialForces
            support_status, error = self.kim_ca.get_argument_support_status(name)
            check_error(error, "kim_ca.get_argument_support_status")
            if not (
                support_status == kimpy.support_status.required
                or support_status == kimpy.support_status.optional
            ):
                report_error("Forces not supported by model")

        # register argument
        error = self.kim_ca.set_argument_pointer(
            kim_can.numberOfParticles, self.num_particles
        )
        check_error(error, "kim_can.set_argument_pointer")

        error = self.kim_ca.set_argument_pointer(
            kim_can.particleSpeciesCodes, self.species_code
        )
        check_error(error, "kim_can.set_argument_pointer")

        error = self.kim_ca.set_argument_pointer(
            kim_can.particleContributing, self.particle_contributing
        )
        check_error(error, "kim_can.set_argument_pointer")

        error = self.kim_ca.set_argument_pointer(kim_can.coordinates, self.coords)
        check_error(error, "kim_can.set_argument_pointer")

        if compute_energy:
            self.energy = np.array([0.0], dtype=np.double)
            error = self.kim_ca.set_argument_pointer(kim_can.partialEnergy, self.energy)
            check_error(error, "kim_can.set_argument_pointer")
        else:
            self.energy = None
            error = self.kim_ca.set_argument_null_pointer(kim_can.partialEnergy)
            check_error(error, "kim_can.set_argument_null_pointer")

        if compute_forces:
            self.forces = np.zeros([self.num_particles[0], 3], dtype=np.double)
            error = self.kim_ca.set_argument_pointer(kim_can.partialForces, self.forces)
            check_error(error, "kim_can.set_argument_pointer")
        else:
            self.forces = None
            error = self.kim_ca.set_argument_null_pointer(kim_can.partialForces)
            check_error(error, "kim_can.set_argument_null_pointer")

    def compute(self, kim_model):
        error = kim_model.compute(self.kim_ca)
        check_error(error, "kim_model.compute")

        if self.compute_energy:
            self.results["energy"] = self.energy[0]
        if self.compute_forces:
            forces = assemble_forces(
                self.forces, self.num_contributing_particles, self.padding_image_of
            )
            self.results["forces"] = forces
        if self.compute_stress:
            volume = self.conf.get_volume()
            stress = assemble_stress(self.coords, self.forces, volume)
            self.results["stress"] = stress

    def __del__(self):
        """
        Garbage collection to destroy the neighbor list automatically.
        """
        if self.neigh is not None:
            nl.clean(self.neigh)
            self.neigh = None


class KIMModel(Model):
    """
    A general interface to any KIM model.

    Args:
        model_name: name of a KIM model. Available models can be found at:
            https://openkim.org.
            For example `SW_StillingerWeber_1985_Si__MO_405512056662_005`.
        params_relation_callback: A callback function to set the relations between
            parameters, which are called each minimization step after the optimizer
            updates the parameters. The function with be given a dictionary of
            :meth:`~kliff.model.parameter.Parameter` as argument, which can
            then be manipulated to set relations between parameters.

            Example:

            In the following example, we set the value of B[0] to 2 * A[0].

            def params_relation(model_params):
                A = model_params['A']
                B = model_params['B']
                B[0] = 2*A[0]
    """

    def __init__(
        self,
        model_name: str,
        params_relation_callback: Optional[Callable] = None,
    ):
        if not kimpy_avail:
            report_import_error("kimpy", self.__class__.__name__)

        self.kim_model = self._create_kim_model(model_name)

        super(KIMModel, self).__init__(model_name, params_relation_callback)

    def init_model_params(self) -> Dict[str, Parameter]:
        return self.get_kim_model_params()

    def init_influence_distance(self) -> float:
        return self.kim_model.get_influence_distance()

    def init_supported_species(self) -> Dict[str, int]:
        species = {}
        num_kim_species = kimpy.species_name.get_number_of_species_names()

        for i in range(num_kim_species):
            species_name, error = kimpy.species_name.get_species_name(i)
            check_error(error, "kimpy.species_name.get_species_name")
            supported, code, error = self.kim_model.get_species_support_and_code(
                species_name
            )
            check_error(error, "kim_model.get_species_support_and_code")
            if supported:
                species[str(species_name)] = code

        return species

    def get_compute_argument_class(self):
        return KIMComputeArguments

    @staticmethod
    def _create_kim_model(model_name: str):
        """
        Create a new kim model.
        """
        units_accepted, model, error = kimpy.model.create(
            kimpy.numbering.zeroBased,
            kimpy.length_unit.A,
            kimpy.energy_unit.eV,
            kimpy.charge_unit.e,
            kimpy.temperature_unit.K,
            kimpy.time_unit.ps,
            model_name,
        )
        check_error(error, "kimpy.model.create")
        if not units_accepted:
            report_error("requested units not accepted in kimpy.model.create")
        return model

    def get_kim_model_params(self) -> Dict[str, Parameter]:
        """
        Inquire the KIM model to get all the parameters.

        Returns:
            {name, parameter}, all parameters in a kim model.
        """
        params = dict()

        num_params = self.kim_model.get_number_of_parameters()
        for i in range(num_params):
            out = self.kim_model.get_parameter_metadata(i)
            dtype, extent, name, description, error = out
            check_error(error, "model.get_parameter_data_type_extent_and_description")

            values = []
            for j in range(extent):
                if str(dtype) == "Double":
                    val, error = self.kim_model.get_parameter_double(i, j)
                    check_error(error, "model.get_parameter_double")
                    values.append(val)
                elif str(dtype) == "Int":
                    val, error = self.kim_model.get_parameter_int(i, j)
                    check_error(error, "model.get_parameter_int")
                    values.append(val)
                else:  # should never reach here
                    report_error(f"get unexpected parameter data type `{dtype}`")

                params[name] = Parameter(value=values, index=i)

        return params

    def create_a_kim_compute_argument(self):
        """
        Create a compute argument for the KIM model.
        """
        kim_ca, error = self.kim_model.compute_arguments_create()
        check_error(error, "kim_model.compute_arguments_create")

        return kim_ca

    def set_opt_params(self, **kwargs):
        """
        Set the parameters that will be optimized.

        One or more parameters can be set. Each argument is for one parameter, where the
        argument name is the parameter name, the value of the argument is the
        settings(including initial value, fix flag, lower bound, and upper bound).

        The value of the argument should be a list of list, where each inner list is for
        one component of the parameter, which can contain 1, 2, or 3 elements.
         See `~kliff.model.model.Model.read_opt_params()` for the options of the elements.

        Example:
           instance.set(A=[['DEFAULT'], [2.0, 1.0, 3.0]], B=[[1.0, 'FIX'], [2.0, 'INF', 3.0]])
        """
        self.opt_params.set(**kwargs)

        # update kim internal model param (note, set_one will update model_params)
        for name, _ in kwargs.items():
            p_idx = self.model_params[name].index
            for c_idx, v in enumerate(self.model_params[name].value):
                self.kim_model.set_parameter(p_idx, c_idx, v)

        self.kim_model.clear_then_refresh()

        # reset influence distance in case it changes
        self.init_influence_distance()

    def set_one_opt_param(self, name: str, settings: List[List[Any]]):
        """
        Set one parameter that will be optimized.

        The name of the parameter should be given as the first entry of a list (or tuple),
        and then each data line should be given in in a list.

        Args:
            name: name of a fitting parameter
            settings: initial value, flag to fix a parameter, lower and upper bounds of a
                parameter.

        Example:
            name = 'param_A'
            settings = [['default', 0, 20], [2.0, 'fix'], [2.2, 'inf', 3.3]]
            instance.set_one(name, settings)
        """
        self.opt_params.set_one(name, settings)

        # update kim internal model param (note, set_one will update model_params)
        p_idx = self.model_params[name].index
        for c_idx, v in enumerate(self.model_params[name].value):
            self.kim_model.set_parameter(p_idx, c_idx, v)

        self.kim_model.clear_then_refresh()

        # reset influence distance in case it changes
        self.init_influence_distance()

    def update_model_params(self, params: Sequence[float]):
        """
        Update optimizing parameters (a sequence used by the optimizer) to the kim model.
        """
        # update from opt params to model params
        # TODO, in super().update_model_params(), we have parameter relation set,
        #   these parameters need to be updated here as well. However, in general
        #   we do not know how parameters are modified in parameter_relation,
        #   and it seems the only hope is to keep a copy of parameters and do some
        #   comparison to check which are modified and then set them.
        super().update_model_params(params)

        # update from model params to kim params
        n = self.get_num_opt_params()
        for i in range(n):
            _, value, p_idx, c_idx = self.get_opt_param_name_value_and_indices(i)
            self.kim_model.set_parameter(p_idx, c_idx, value)

        # refresh model
        self.kim_model.clear_then_refresh()

        if logger.getEffectiveLevel() == logging.DEBUG:
            params = self.get_kim_model_params()
            s = ""
            for name, p in params.items():
                s += f"\nname: {name}\n"
                s += str(p.as_dict())
            log_entry(logger, s, level="debug")

    def write_kim_model(self, path: Path = None):
        """
        Write out a KIM model that can be used directly with the kim-api.

        This function typically write two files to `path`: (1) CMakeLists.txt, and (2)
        a parameter file like A.model_params. `path` will be created if it does not exist.

        Args:
            path: Path to the a directory to store the model. If `None`, it is set to
                `./MODEL_NAME_kliff_trained`, where `MODEL_NAME` is the `model_name` that
                provided at the initialization of this class.

        Note:
            This only works for parameterized KIMModel models that support the writing of
            parameters.
        """
        present, required, error = self.kim_model.is_routine_present(
            kimpy.model_routine_name.WriteParameterizedModel
        )
        check_error(error, "kim_model.is_routine_is_routine_present")
        if not present:
            raise KIMModelError("This KIM model does not support writing parameters.")

        if path is None:
            model_name = self.model_name + "_kliff_trained"
            path = Path.cwd().joinpath(model_name)
        else:
            path = Path(path).expanduser().resolve()
            model_name = path.name

        if not path.exists():
            os.makedirs(path)

        path = str(path)
        model_name = str(model_name)

        error = self.kim_model.write_parameterized_model(path, model_name)
        check_error(error, "kim_model.write_parameterized_model")

        log_entry(logger, f"KLIFF trained model write to `{path}`", level="info")


class KIMModelError(Exception):
    def __init__(self, msg):
        super(KIMModelError, self).__init__(msg)
        self.msg = msg


def check_error(error, msg):
    if error != 0 and error is not None:
        msg = f"Calling `{msg}` failed.\nSee `kim.log` for more information."
        log_entry(logger, msg, level="error")
        raise KIMModelError(msg)


def report_error(msg):
    log_entry(logger, msg, level="error")
    raise KIMModelError(msg)
