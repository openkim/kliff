import importlib
import os
import subprocess
import tarfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import kimpy
import numpy as np
from loguru import logger

from kliff.dataset.dataset import Configuration
from kliff.error import report_import_error
from kliff.log import get_log_level
from kliff.models.model import ComputeArguments, Model
from kliff.models.parameter import Parameter
from kliff.neighbor import assemble_forces, assemble_stress
from kliff.utils import install_kim_model, is_kim_model_installed

try:
    import kimpy
    from kimpy import neighlist as nl

    kimpy_avail = True
except ImportError:
    kimpy_avail = False

# list of model drivers that are not supported by this trainer.
# example quip, torchml, etc.
# TODO: Get the complete list of unsupported model drivers.
UNSUPPORTED_MODEL_DRIVERS = [
    "TorchML",
]


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

        self.model_trainable_via_kim_api = False

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
    """

    def __init__(
        self,
        model_name: str,
    ):
        if not kimpy_avail:
            report_import_error("kimpy", self.__class__.__name__)

        self.kim_model = self._create_kim_model(model_name)

        super(KIMModel, self).__init__(model_name)

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

    def get_kim_model_params(self):
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

        params = OrderedDict()
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

            params[name] = Parameter(np.asarray(values), name=name, index=i)
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

    def set_params_mutable(self, list_of_params: List[str]):
        """
        Set all the optimizable parameters from list of names of parameters
        Args:
            list_of_params: List of string names of parameters
        Example:
            model.set_params_mutable(["A", "B", "sigma"])
        """
        super().set_params_mutable(list_of_params)
        # reset influence distance in case it changes
        self.init_influence_distance()

    def set_one_opt_param(self, name: str, settings: List[List[Any]]):
        """
        Set one parameter that will be optimized.

        The name of the parameter should be given as the first entry of a list (or tuple),
        and then each data line should be given in in a list.

        Args:
            name: name of a fitting parameter
            settings: List initial value, flag to fix a parameter, lower and upper bounds of a
                parameter.

        Example:
            name = 'A'
            settings = [['default', 0, 20], [2.0, 'fix'], [2.2, 'inf', 3.3]]
            instance.set_one(name, settings)
        """
        super().set_one_opt_param(name, settings)
        self.init_influence_distance()

        param = self.model_params[name]
        # update kim internal model param (note, set_one will update model_params)
        p_idx = param.index
        value_arr = param.get_numpy_array_param_space()
        # This is now not needed as with Matlab 0D array == scalar
        # That check is performed above
        # if type(value_arr) != np.ndarray:
        #     value_arr = np.array([value_arr]) # if float make it iterable

        for c_idx, v in enumerate(value_arr):
            try:
                # Update the parameter in both kimpy model and model list
                self.kim_model.set_parameter(p_idx, c_idx, value_arr[c_idx])
            except RuntimeError:
                raise kimpy.KimPyError("Calling `kim_model.set_parameter()` failed.")

        try:
            self.kim_model.clear_then_refresh()
        except RuntimeError:
            raise kimpy.KimPyError("Calling `kim_model.clear_then_refresh()` failed.")

        # reset influence distance in case it changes and get latest parameters
        self.init_influence_distance()

    def update_model_params(
        self, params: Union[np.ndarray, List[Union[float, int, Parameter]]]
    ):
        """
        Update optimizing parameters (a sequence used by the optimizer) to the kim model.
        """

        # update from opt params to model params
        super().update_model_params(params)

        # update from model params to kim params
        for name, param in self.model_params.items():
            p_idx = param.index
            for c_idx, v in enumerate(param.get_numpy_array_model_space()):
                try:
                    self.kim_model.set_parameter(p_idx, c_idx, v)
                except RuntimeError:
                    raise kimpy.KimPyError(
                        "Calling `kim_model.set_parameter()` failed."
                    )

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

    def __call__(
        self,
        configuration: "Configuration",
        compute_energy: bool = True,
        compute_forces: bool = True,
        compute_stress: bool = False,
    ):
        """
        Functional wrapper for the KIM models, this provides KIM models ability to be
        used as in similar API as torch model. Expects a ~kliff.dataset.Configuration
        object as input and returns energy, forces, and stress as requested. Key contains
        value None for fields for which computation is not requested.

        Args:
            configuration: atomic configuration
            compute_energy: whether to compute energy
            compute_forces: whether to compute the forces
            compute_stress: whether to compute the stress

        Returns:
            Dictionary containing "energy", "forces", and "stress" keys
        """
        supported_species = self.get_supported_species()
        influence_dist = self.get_influence_distance()
        # create a new kim_ca to avoid overwriting existing context.
        kim_ca = self.create_a_kim_compute_argument()
        kim_ca_instance = KIMComputeArguments(
            kim_ca=kim_ca,
            config=configuration,
            supported_species=supported_species,
            influence_distance=influence_dist,
            compute_energy=compute_energy,
            compute_forces=compute_forces,
            compute_stress=compute_stress,
        )
        kim_ca_instance.compute(self.kim_model)

        return kim_ca_instance.results

    @staticmethod
    def get_model_from_manifest(model_manifest: dict, param_manifest: dict = None):
        """
        Get the model from a configuration. If it is a valid KIM model, it will return
        the KIMModel object. If it is a TorchML model, it will return the torch
        ReverseScriptedModule object *in future*. Else raise error. If the model is a tarball, it
        will extract and install the model.

        Example `model_manifest`:
        ```yaml
            model:
                model_type: kim     # kim or torch or tar
                model_path: ./
                model_name: SW_StillingerWeber_1985_Si__MO_405512056662_006 # KIM model name, installed if missing
                model_collection: "user"
        ```

        Example `param_manifest`:
        ```yaml
            parameter:
                    - A         # dict means the parameter is transformed
                    - B         # these are the parameters that are not transformed
                    - sigma:
                        transform_name: LogParameterTransform
                        value: 2.0
                        bounds: [[1.0, 10.0]]
        ```

        ```{note}
        `parameter` block is usually defined as the children of the `transform` block
        in trainer configuration file.
        ```

        Args:
            model_manifest: configuration object
            param_manifest: parameter transformation configuration

        Returns:
            Model object
        """
        model_name: Union[None, str] = model_manifest.get("name", None)
        model_type: Union[None, str] = model_manifest.get("type", None)
        model_path: Union[None, str, Path] = model_manifest.get("path", None)
        model_driver = KIMModel.get_model_driver_name(model_name)
        model_collection = model_manifest.get("collection")

        if model_driver in UNSUPPORTED_MODEL_DRIVERS:
            logger.error(
                "Model driver not supported for KIM-API based training. "
                "Please use appropriate trainer for this model."
            )
            raise KIMModelError(
                f"Model driver {model_driver} not supported for KIMModel training."
            )

        # ensure model is installed
        if model_type.lower() == "kim":
            is_model_installed = install_kim_model(model_name, model_collection)
            if not is_model_installed:
                logger.error(
                    f"Mode: {model_name} neither installed nor available in the KIM API collections. Please check the model name and try again."
                )
                raise KIMModelError(f"Model {model_name} not found.")
            else:
                logger.info(
                    f"Model {model_name} is present in {model_collection} collection."
                )
        elif model_type.lower() == "tar":
            archive_content = tarfile.open(model_path + "/" + model_name)
            model = archive_content.getnames()[0]
            archive_content.extractall(model_path)
            subprocess.run(
                [
                    "kim-api-collections-management",
                    "install",
                    "--force",
                    model_collection,
                    model_path + "/" + model,
                ],
                check=True,
            )
            logger.info(
                f"Tarball Model {model} installed in {model_collection} collection."
            )
        else:
            raise KIMModelError(f"Model type {model_type} not supported.")

        model = KIMModel(model_name)

        if param_manifest:
            mutable_param_list = []
            for param_to_transform in param_manifest.get("parameter", []):
                if isinstance(param_to_transform, dict):
                    parameter_name = list(param_to_transform.keys())[0]
                elif isinstance(param_to_transform, str):
                    parameter_name = param_to_transform
                else:
                    raise KIMModelError(f"Parameter can be a str or dict")
                mutable_param_list.append(parameter_name)

            model.set_params_mutable(mutable_param_list)
            model_param_list = model.parameters()

            # apply transforms if needed
            for model_params, input_params in zip(
                model_param_list, param_manifest.get("parameter", [])
            ):
                if isinstance(input_params, dict):
                    param_name = list(input_params.keys())[0]
                    if param_name != model_params.name:
                        raise KIMModelError(
                            f"Parameter name mismatch. Expected {model_params.name}, got {param_name}."
                        )

                    param_value_dict = input_params[param_name]
                    transform_name = param_value_dict.get("transform_name", None)
                    params_value = param_value_dict.get("value", None)
                    bounds = param_value_dict.get("bounds", None)

                    if transform_name is not None:
                        transform_module = getattr(
                            importlib.import_module(
                                f"kliff.transforms.parameter_transforms"
                            ),
                            transform_name,
                        )
                        transform_module = transform_module()
                        model_params.add_transform(transform_module)

                    if params_value is not None:
                        model_params.copy_from_model_space(params_value)

                    if bounds is not None:
                        model_params.add_bounds_model_space(np.array(bounds))

                elif isinstance(input_params, str):
                    if input_params != model_params.name:
                        raise KIMModelError(
                            f"Parameter name mismatch. Expected {model_params.name}, got {input_params}."
                        )
                else:
                    raise KIMModelError(
                        f"Optimizable parameters must be string or value dict. Got {input_params} instead."
                    )

        return model

    @staticmethod
    def get_model_driver_name(model_name: str) -> Union[str, None]:
        """
        Get the model driver from the model name. It will return the model driver
        string from the installed KIM API model. If the model is not installed, and the
        model name is a tarball, it will extract the model driver name from the CMakeLists.txt.
        This is needed to ensure that it excludes the model drivers that it cannot handle.
        Example: TorchML driver based models. These models are to be trained using the
        TorchTrainer.

        TODO: This is not a clean solution. I think KIMPY must have a better way to handle this.
              Ask Mingjian/Yaser for comment.

        Args:
            model_name: name of the model.

        Returns:
            Model driver name.
        """
        # check if model is tarball
        if "tar" in model_name:
            return KIMModel._get_model_driver_name_for_tarball(model_name)

        collections = kimpy.collections.create()
        try:
            shared_obj_path, collection = (
                collections.get_item_library_file_name_and_collection(
                    kimpy.collection_item_type.portableModel, model_name
                )
            )
        except RuntimeError:  # not a portable model
            return None
        shared_obj_content = open(shared_obj_path, "rb").read()
        md_start_idx = shared_obj_content.find(b"model-driver")

        if md_start_idx == -1:
            return None
        else:
            md_start_idx += 15  # length of 'model-driver" "'
            md_end_idx = shared_obj_content.find(b'"', md_start_idx)
            return shared_obj_content[md_start_idx:md_end_idx].decode("utf-8")

    @staticmethod
    def _get_model_driver_name_for_tarball(tarball: str) -> Union[str, None]:
        """
        Get the model driver name from the tarball. It will extract the model driver
        name from the CMakeLists.txt file in the tarball. This is needed to ensure that
        it excludes the model drivers that it cannot handle. Example: TorchML driver based
        models. These models are to be trained using the TorchTrainer.

        Args:
            tarball: path to the tarball.

        Returns:
            Model driver name.
        """
        archive_content = tarfile.open(tarball)
        cmake_file_path = archive_content.getnames()[0] + "/CMakeLists.txt"
        cmake_file = archive_content.extractfile(cmake_file_path)
        cmake_file_content = cmake_file.read().decode("utf-8")

        md_start_idx = cmake_file_content.find("DRIVER_NAME")
        if md_start_idx == -1:
            return None
        else:
            # name strats at "
            md_start_idx = cmake_file_content.find('"', md_start_idx) + 1
            if md_start_idx == -1:
                return None
            md_end_idx = cmake_file_content.find('"', md_start_idx)
            return cmake_file_content[md_start_idx:md_end_idx]


class KIMModelError(Exception):
    def __init__(self, msg):
        super(KIMModelError, self).__init__(msg)
        self.msg = msg
