import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from loguru import logger

from kliff.dataset.dataset import Configuration
from kliff.error import report_import_error
from kliff.log import get_log_level
from kliff.models.model import ComputeArguments, Model
from kliff.models.parameter import Parameter
from kliff.models.parameter_transform import ParameterTransform
from kliff.neighbor import assemble_forces, assemble_stress

try:
    import kimpy
    from kimpy import neighlist as nl

    kimpy_avail = True
except ImportError:
    kimpy_avail = False


class KIMComputeArguments(ComputeArguments):
    """
    KIMModel potentials arguments.

    Args:
        kim_ca: KIM compute argument, can be created by
            :meth:`~kliff.models.KIMModels.create_a_compute_argument()`.
        config: atomic configurations
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
        """
        Get implemented property of model.
        """

        # check compute arguments
        try:
            kim_can = kimpy.compute_argument_name
            N = kim_can.get_number_of_compute_argument_names()
        except RuntimeError:
            raise kimpy.KimPyError(
                "Calling `kim_can.get_number_of_compute_argument_names()` failed."
            )

        for i in range(N):
            try:
                name = kim_can.get_compute_argument_name(i)
            except RuntimeError:
                raise kimpy.KimPyError(
                    "Calling `kim_can.get_compute_argument_name()` failed."
                )

            # dtype = kim_can.get_compute_argument_data_type(name)

            try:
                support_status = self.kim_ca.get_argument_support_status(name)
            except RuntimeError:
                raise kimpy.KimPyError(
                    "Calling `kim_ca.get_argument_support_status()` failed."
                )

            # calculator can only handle energy and forces
            if support_status == kimpy.support_status.required:
                if name != kim_can.partialEnergy and name != kim_can.partialForces:
                    KIMModelError(f"Unsupported required ComputeArgument `{name}`")

            # supported property
            if name == kim_can.partialEnergy:
                self.implemented_property.append("energy")
            elif name == kim_can.partialForces:
                self.implemented_property.append("forces")
                self.implemented_property.append("stress")

        # check compute callbacks
        kim_ccn = kimpy.compute_callback_name

        try:
            num_callbacks = kim_ccn.get_number_of_compute_callback_names()
        except RuntimeError:
            raise kimpy.KimPyError(
                "Calling `kim_ccn.get_number_of_compute_callback_names()` failed."
            )

        for i in range(num_callbacks):
            try:
                name = kim_ccn.get_compute_callback_name(i)
            except RuntimeError:
                raise kimpy.KimPyError(
                    "Calling `kim_ccn.get_compute_callback_name()` failed."
                )

            try:
                support_status = self.kim_ca.get_callback_support_status(name)
            except RuntimeError:
                raise kimpy.KimPyError(
                    "Calling `kim_ca.get_callback_support_status()` failed."
                )

            # calculator only provides get_neigh
            if support_status == kimpy.support_status.required:
                if name != kim_ccn.GetNeighborList:
                    KIMModelError(f"Unsupported required ComputeCallback `{name}`")

    def _init_neigh(self):

        # create neighbor list
        try:
            neigh = nl.create()
        except RuntimeError:
            raise kimpy.KimPyError("Calling `nl.create()` failed.")

        # register get neigh callback
        try:
            self.kim_ca.set_callback_pointer(
                kimpy.compute_callback_name.GetNeighborList, nl.get_neigh_kim(), neigh
            )
        except RuntimeError:
            raise kimpy.KimPyError("Calling `kim_ca.set_callback_pointer()` failed.")

        self.neigh = neigh

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
                KIMModelError(f"species `{s}` not supported by model")

        contributing_species_code = np.array(
            [species_map[s] for s in contributing_species], dtype=np.intc
        )

        if any(PBC):  # need padding atoms

            try:
                (
                    padding_coords,
                    padding_species_code,
                    self.padding_image_of,
                ) = nl.create_paddings(
                    influence_distance,
                    cell,
                    PBC,
                    contributing_coords,
                    contributing_species_code,
                )
            except RuntimeError:
                raise kimpy.KimPyError("Calling `nl.create_paddings()` failed.")

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

            # for now, create neigh for all atoms, including paddings
            need_neigh = np.ones(self.num_particles[0], dtype=np.intc)

        else:  # do not need padding atoms
            self.padding_image_of = np.array([])
            self.num_particles = np.array([num_contributing], dtype=np.intc)
            self.coords = np.array(contributing_coords, dtype=np.double)
            self.species_code = np.array(contributing_species_code, dtype=np.intc)
            self.particle_contributing = np.ones(num_contributing, dtype=np.intc)
            need_neigh = self.particle_contributing

        try:
            self.neigh.build(
                self.coords,
                influence_distance,
                np.asarray([influence_distance], dtype=np.double),
                need_neigh,
            )
        except RuntimeError:
            raise kimpy.KimPyError("Calling `neigh.build()` failed.")

    def _register_data(self, compute_energy=True, compute_forces=True):
        """
        Register model input and output data in KIM API.
        """

        # check whether model support energy and forces
        kim_can = kimpy.compute_argument_name
        if compute_energy:
            name = kim_can.partialEnergy

            try:
                support_status = self.kim_ca.get_argument_support_status(name)
            except RuntimeError:
                raise kimpy.KimPyError(
                    "Calling `kim_ca.get_argument_support_status()` failed."
                )

            if not (
                support_status == kimpy.support_status.required
                or support_status == kimpy.support_status.optional
            ):
                KIMModelError("Energy not supported by model")

        if compute_forces:
            name = kim_can.partialForces

            try:
                support_status = self.kim_ca.get_argument_support_status(name)
            except RuntimeError:
                raise kimpy.KimPyError(
                    "Calling `kim_ca.get_argument_support_status()` failed."
                )

            if not (
                support_status == kimpy.support_status.required
                or support_status == kimpy.support_status.optional
            ):
                KIMModelError("Forces not supported by model")

        # register argument
        try:
            self.kim_ca.set_argument_pointer(
                kim_can.numberOfParticles, self.num_particles
            )

            self.kim_ca.set_argument_pointer(
                kim_can.particleSpeciesCodes, self.species_code
            )

            self.kim_ca.set_argument_pointer(
                kim_can.particleContributing, self.particle_contributing
            )

            self.kim_ca.set_argument_pointer(kim_can.coordinates, self.coords)
        except RuntimeError:
            raise kimpy.KimPyError("Calling `kim_ca.set_argument_pointer()` failed.")

        if compute_energy:
            self.energy = np.array([0.0], dtype=np.double)
            try:
                self.kim_ca.set_argument_pointer(kim_can.partialEnergy, self.energy)
            except RuntimeError:
                raise kimpy.KimPyError(
                    "Calling `kim_ca.set_argument_pointer()` failed."
                )

        else:
            self.energy = None
            try:
                self.kim_ca.set_argument_null_pointer(kim_can.partialEnergy)
            except RuntimeError:
                raise kimpy.KimPyError(
                    "Calling `kim_ca.set_argument_null_pointer()` failed."
                )

        if compute_forces:
            self.forces = np.zeros([self.num_particles[0], 3], dtype=np.double)
            try:
                self.kim_ca.set_argument_pointer(kim_can.partialForces, self.forces)
            except RuntimeError:
                raise kimpy.KimPyError(
                    "Calling `kim_ca.set_argument_pointer()` failed."
                )

        else:
            self.forces = None
            try:
                self.kim_ca.set_argument_null_pointer(kim_can.partialForces)
            except RuntimeError:
                raise kimpy.KimPyError(
                    "Calling `kim_ca.set_argument_null_pointer()` failed."
                )

    def compute(self, kim_model):

        try:
            kim_model.compute(self.kim_ca)
        except RuntimeError:
            raise kimpy.KimPyError("Calling `kim_model.compute()` failed.")

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


class KIMModel(Model):
    """
    A general interface to any KIM model.

    Args:
        model_name: name of a KIM model. Available models can be found at:
            https://openkim.org.
            For example `SW_StillingerWeber_1985_Si__MO_405512056662_006`.
        params_transform: optional transformation of parameters. Let's call the
            parameters initialized in `init_model_params()` as original parameter
            space. Sometimes, it's easier to work in another parameter (e.g.
            optimizers can perform better in the log space). Then we can use this
            parameter transformation class to transform between the original
            parameter space and the new easy-to-work space. Typically, a model only
            knows how to work in its original space to compute, e.g. energy and
            forces, so we need to inverse transform parameters back to original space
            (after an optimizer update its value in the log space).
            A `params_transform` instance should implement both a `transform` and an
            `inverse_transform` method to accomplish the above tasks.
            Note, all the parameters of this (the `Model`) class
            (e.g. `self.model_params`, and `self.opt_params`) are in the transformed
            easy-to-work space.
    """

    def __init__(
        self,
        model_name: str,
        params_transform: Optional[ParameterTransform] = None,
    ):
        if not kimpy_avail:
            report_import_error("kimpy", self.__class__.__name__)

        self.kim_model = self._create_kim_model(model_name)

        super(KIMModel, self).__init__(model_name, params_transform)

    def init_model_params(self) -> Dict[str, Parameter]:
        return self.get_kim_model_params()

    def init_influence_distance(self) -> float:
        return self.kim_model.get_influence_distance()

    def init_supported_species(self) -> Dict[str, int]:
        species = {}

        try:
            num_kim_species = kimpy.species_name.get_number_of_species_names()
        except RuntimeError:
            raise kimpy.KimPyError(
                "Calling `kimpy.species_name.get_number_of_species_names()` failed."
            )

        for i in range(num_kim_species):
            try:
                species_name = kimpy.species_name.get_species_name(i)
            except RuntimeError:
                raise kimpy.KimPyError(
                    "Calling `kimpy.species_name.get_species_name()` failed."
                )

            try:
                supported, code = self.kim_model.get_species_support_and_code(
                    species_name
                )
            except RuntimeError:
                raise kimpy.KimPyError(
                    "Calling `kim_model.get_species_support_and_code()` failed."
                )

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
        try:
            units_accepted, model = kimpy.model.create(
                kimpy.numbering.zeroBased,
                kimpy.length_unit.A,
                kimpy.energy_unit.eV,
                kimpy.charge_unit.e,
                kimpy.temperature_unit.K,
                kimpy.time_unit.ps,
                model_name,
            )
        except RuntimeError:
            raise kimpy.KimPyError("Calling `kimpy.model.create()` failed.")

        if not units_accepted:
            KIMModelError("requested units not accepted in kimpy.model.create")
        return model

    def get_kim_model_params(self) -> Dict[str, Parameter]:
        """
        Inquire the KIM model to get all the parameters.

        Returns:
            {name, parameter}, all parameters in a kim model.
        """
        try:
            num_params = self.kim_model.get_number_of_parameters()
        except RuntimeError:
            raise kimpy.KimPyError(
                "Calling `kim_model.get_number_of_parameters()` failed."
            )

        params = dict()
        for i in range(num_params):
            try:
                (
                    dtype,
                    extent,
                    name,
                    description,
                ) = self.kim_model.get_parameter_metadata(i)
            except RuntimeError:
                raise kimpy.KimPyError(
                    "Calling `kim_model.get_parameter_metadata()` failed."
                )

            values = []
            for j in range(extent):
                if str(dtype) == "Double":
                    try:
                        val = self.kim_model.get_parameter_double(i, j)
                    except RuntimeError:
                        raise kimpy.KimPyError(
                            "Calling `kim_model.get_parameter_double()` failed."
                        )
                    values.append(val)

                elif str(dtype) == "Int":
                    try:
                        val = self.kim_model.get_parameter_int(i, j)
                    except RuntimeError:
                        raise kimpy.KimPyError(
                            "Calling `kim_model.get_parameter_int()` failed."
                        )
                    values.append(val)

                else:  # should never reach here
                    KIMModelError(f"get unexpected parameter data type `{dtype}`")

            params[name] = Parameter(value=values, name=name, index=i)

        return params

    def create_a_kim_compute_argument(self):
        """
        Create a compute argument for the KIM model.
        """
        try:
            kim_ca = self.kim_model.compute_arguments_create()
        except RuntimeError:
            raise kimpy.KimPyError(
                "Calling `kim_model.compute_arguments_create()` failed."
            )

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
                try:
                    self.kim_model.set_parameter(p_idx, c_idx, v)
                except RuntimeError:
                    raise kimpy.KimPyError(
                        "Calling `kim_model.set_parameter()` failed."
                    )

        try:
            self.kim_model.clear_then_refresh()
        except RuntimeError:
            raise kimpy.KimPyError("Calling `kim_model.clear_then_refresh()` failed.")

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
            try:
                self.kim_model.set_parameter(p_idx, c_idx, v)
            except RuntimeError:
                raise kimpy.KimPyError("Calling `kim_model.set_parameter()` failed.")

        try:
            self.kim_model.clear_then_refresh()
        except RuntimeError:
            raise kimpy.KimPyError("Calling `kim_model.clear_then_refresh()` failed.")

        # reset influence distance in case it changes
        self.init_influence_distance()

    def update_model_params(self, params: Sequence[float]):
        """
        Update optimizing parameters (a sequence used by the optimizer) to the kim model.
        """

        # update from opt params to model params
        super().update_model_params(params)

        # only update optimizing params
        if self.params_transform is None:
            # update from model params to kim params
            n = self.get_num_opt_params()
            for i in range(n):
                _, value, p_idx, c_idx = self.get_opt_param_name_value_and_indices(i)
                try:
                    self.kim_model.set_parameter(p_idx, c_idx, value)
                except RuntimeError:
                    raise kimpy.KimPyError(
                        "Calling `kim_model.set_parameter()` failed."
                    )

        # When params_transform is set, a user can do whatever in it
        # function, e.g. update a parameter that is not an optimizing parameter.
        # In general, we do not know how parameters are modified in there,
        # and therefore, we need to update all params in model_params to kim
        # Note, `params_transform.inverse_transform()` is called in
        # super().update_model_params(params)
        else:
            for name, params in self.model_params.items():
                p_idx = params.index
                for c_idx, value in enumerate(params.value):
                    try:
                        self.kim_model.set_parameter(p_idx, c_idx, value)
                    except RuntimeError:
                        raise kimpy.KimPyError(
                            "Calling `kim_model.set_parameter()` failed."
                        )

        # refresh model
        self.kim_model.clear_then_refresh()

        if get_log_level() == "DEBUG":
            params = self.get_kim_model_params()
            s = ""
            for name, p in params.items():
                s += f"\nname: {name}\n"
                s += str(p.as_dict())

            logger.debug(s)

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
        present, required = self.kim_model.is_routine_present(
            kimpy.model_routine_name.WriteParameterizedModel
        )
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

        self.kim_model.write_parameterized_model(path, model_name)

        logger.info(f"KLIFF trained model write to `{path}`")


class KIMModelError(Exception):
    def __init__(self, msg):
        super(KIMModelError, self).__init__(msg)
        self.msg = msg
