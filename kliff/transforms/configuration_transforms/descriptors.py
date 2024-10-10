import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from loguru import logger
from monty.dev import requires

from kliff.dataset import Configuration
from kliff.neighbor import NeighborList

from .configuration_transform import ConfigurationTransform

try:
    import libdescriptor as lds
except ImportError:
    lds = None
    logger.error(
        "descriptors module depends on libdescriptor, "
        "which is not found. please install it first."
    )

from .default_hyperparams import (
    bispectrum_default,
    soap_default,
    symmetry_functions_set30,
    symmetry_functions_set51,
)
from .descriptor_initializers import (
    initialize_bispectrum_functions,
    initialize_symmetry_functions,
)


@requires(lds, "libdescriptor is needed for Descriptors")
class AvailableDescriptors:
    """
    This class lists all the available descriptors in libdescriptor. Libdescriptor
    provides that information as an Enum structure. This class is a wrapper of that.
    """

    def __init__(self):
        i = 0
        while True:
            desc_type = lds.AvailableDescriptors(i)
            i += 1
            if desc_type.name == "???":
                break
            else:
                setattr(self, desc_type.name, desc_type)


@requires(lds, "libdescriptor is needed for Descriptors")
def show_available_descriptors():
    """
    Show all the available descriptors in libdescriptor.
    """
    print("-" * 80)
    print(
        "Descriptors below are currently available, select them by `descriptor: str` attribute:"
    )
    print("-" * 80)
    _instance = AvailableDescriptors()
    for key in _instance.__dict__.keys():
        print(f"{key}")


@requires(lds, "libdescriptor is needed for Descriptors")
class Descriptor(ConfigurationTransform):
    """
    Descriptor class provides interface with the libdescriptor library. It provides a
    unified interface to all the descriptors in libdescriptor, however all descriptors need
    to have a corresponding initializer routine, to deal with hyperparameters. The descriptor
    is initialized with a cutoff radius, a list of species,a descriptor type and an ordered
    list of hyperparameters. The descriptor type is a string, which can be obtained by
    `show_available_descriptors()` function. Some sane default values for the hyperparameters
    are provided in ~:kliff.transforms.configuration_transforms.default_hyperparameters
    module. This class methods to compute derivatives of the descriptor with respect to
    atomic positions. The functions generating descriptors and their derivatives are
    implemented as `forward()` and `backward()`, respectively, to match the PyTorch nomenclature.

    """

    def __init__(
        self,
        cutoff: float,
        species: List[str],
        descriptor: str,
        hyperparameters: Union[Dict, str],
        cutoff_function: str = "cos",
        copy_to_config: bool = False,
    ):
        """
        Args:
            cutoff (float): Cutoff radius.
            species (list): List of strings, each string is a species (atomic symbols).
            descriptor (str): String of descriptor type, can be obtained by `show_available_descriptors()`.
            hyperparameters (Dict): Ordered dictionary of hyperparameters.
            cutoff_function (str): Cut-off function, currently only "cos" is supported.
            copy_to_config (bool): If True, the fingerprint will be copied to
                the Configuration object's fingerprint attribute.
        """
        super().__init__(copy_to_config)
        self.cutoff = cutoff
        self.species = species
        _available_descriptors = AvailableDescriptors()
        self.descriptor_name = descriptor
        self.descriptor_kind = getattr(_available_descriptors, descriptor)
        self.width = -1
        self.hyperparameters = self.get_default_hyperparams(hyperparameters)
        self.cutoff_function = cutoff_function
        self._cdesc, self.width = self._init_descriptor_from_kind()

    @staticmethod
    def get_default_hyperparams(hyperparameters: Union[Dict, str]) -> Dict:
        """
        Set hyperparameters for the descriptor. If a string is provided, it will be used to select
        a set of default hyperparameters. If a dict is provided, it will be used as is.

        Args:
            hyperparameters (str or Dict): Hyperparameters for the descriptor.

        Returns:
            Dict: dictionary of hyperparameters.
        """
        if isinstance(hyperparameters, str):
            if hyperparameters == "set51":
                return symmetry_functions_set51()
            elif hyperparameters == "set30":
                return symmetry_functions_set30()
            elif hyperparameters == "bs_defaults":
                return bispectrum_default()
            elif hyperparameters == "soap_defaults":
                return soap_default()
            else:
                raise DescriptorsError("Hyperparameter set not found")
        elif isinstance(hyperparameters, dict):
            return hyperparameters
        else:
            raise DescriptorsError("Hyperparameters must be either a string or an Dict")

    def _init_descriptor_from_kind(self) -> Tuple[lds.DescriptorKind, int]:
        """
        Initialize descriptor from descriptor kind. Currently only Symmetry Functions,
        Bispectrum, abd SOAP descriptors are supported.

        Returns:
            tuple: Tuple of descriptor object and width of the descriptor.
        """
        cutoff_array = np.ones((len(self.species), len(self.species))) * self.cutoff

        # Symmetry Functions
        if self.descriptor_kind == lds.AvailableDescriptors(0):
            input_args, width = initialize_symmetry_functions(self.hyperparameters)
            return (
                lds.DescriptorKind.init_descriptor(
                    self.descriptor_kind,
                    self.species,
                    self.cutoff_function,
                    cutoff_array,
                    *input_args,
                ),
                width,
            )

        # Bispectrum
        elif self.descriptor_kind == lds.AvailableDescriptors(1):
            if self.hyperparameters["weights"] is None:
                weights = np.ones(len(self.species))
            else:
                weights = self.hyperparameters["weights"]
            input_args, width = initialize_bispectrum_functions(self.hyperparameters)
            return (
                lds.DescriptorKind.init_descriptor(
                    self.descriptor_kind,
                    *input_args,
                    cutoff_array,
                    self.species,
                    weights,
                ),
                width,
            )

        # SOAP
        # nothing to initialize here, just return the descriptor
        elif self.descriptor_kind == lds.AvailableDescriptors(2):
            # width = (((n_species + 1) * n_species) / 2 * (n_max * (n_max + 1)) * (l_max + 1)) / 2;
            n_species = len(self.species)
            width = int(
                (((n_species + 1) * n_species) / 2)
                * (self.hyperparameters["n_max"] * (self.hyperparameters["n_max"] + 1))
                * (self.hyperparameters["l_max"] + 1)
                / 2
            )
            return (
                lds.DescriptorKind.init_descriptor(
                    self.descriptor_kind,
                    self.hyperparameters["n_max"],
                    self.hyperparameters["l_max"],
                    self.hyperparameters["cutoff"],
                    self.species,
                    self.hyperparameters["radial_basis"],
                    self.hyperparameters["eta"],
                ),
                width,
            )
        else:
            raise DescriptorsError(
                f"Descriptor kind: {self.descriptor_kind} not supported yet"
            )

    def _map_species_to_int(self, species: List[str]) -> List[int]:
        """
        Map species to integers, which is required by the C++ implementation of the descriptors.

        TODO:
            Unify all instances of species -> Z and Z -> species mapping functions. I
            think currently there are 3 of them. Also sort the atomic numbers and species
            codes conversions. Perhaps use the ASE function for this.

        Args:
            species (list): List of species.

        Returns:
            list: List of integers corresponding to the species.
        """
        return [self.species.index(s) for s in species]

    def forward(self, configuration: Configuration) -> np.ndarray:
        """
        Compute the descriptors for a given configuration, by calling the C++ implementation,
        :py:func:`libdescriptor.compute_single_atom`. This function accepts a
        ~:class:`kliff.dataset.Configuration` object and returns a numpy array of shape
        (n_atoms, width), where n_atoms is the number of atoms in the configuration and
        width is the width of the descriptor.

        TODO:
            Use the :py:func:`libdescriptor.compute` function for faster evaluation. Which loops in C++.

        Args:
            configuration: :py:class:`kliff.dataset.Configuration` object to compute descriptors for.

        Returns:
            numpy.ndarray: Array of shape (n_atoms, width).
        """
        nl_ctx = NeighborList(configuration, self.cutoff)
        n_atoms = configuration.get_num_atoms()
        descriptors = np.zeros((n_atoms, self.width))
        species = np.array(self._map_species_to_int(nl_ctx.species), np.intc)
        for i in range(n_atoms):
            neigh_list, _, _ = nl_ctx.get_neigh(i)
            descriptors[i, :] = lds.compute_single_atom(
                self._cdesc,
                i,
                species,
                np.array(neigh_list, dtype=np.intc),
                nl_ctx.coords,
            )
        return descriptors

    def backward(
        self, configuration: Configuration, dE_dZeta: np.ndarray
    ) -> np.ndarray:
        """
        Compute the gradients of the descriptors with respect to the atomic coordinates.
        It takes in an array of shape (n_atoms, width) and the configuration, and performs
        the vector-Jacobian product (reverse mode automatic differentiation).
        The output is an array of shape (n_atoms, 3) yielding the gradients of the descriptors
        with respect to the atomic coordinates.

        Args:
            configuration: :py:class:`kliff.dataset.Configuration` object to compute descriptors for.
            dE_dZeta: :numpy:`ndarray` of shape (n_atoms, width), usually this is the gradient
                of the ML model (hence input descriptors) with respect to the energy.

        Returns:
            :numpy:`ndarray` of shape (n_atoms, 3)
        """
        nl_ctx = NeighborList(configuration, self.cutoff)
        n_atoms = configuration.get_num_atoms()
        derivatives_unrolled = np.zeros(nl_ctx.coords.shape)
        species = np.array(self._map_species_to_int(nl_ctx.species), dtype=np.intc)

        descriptor = np.zeros(self.width)

        for i in range(n_atoms):
            neigh_list, _, _ = nl_ctx.get_neigh(i)
            descriptors_derivative = lds.gradient_single_atom(
                self._cdesc,
                i,
                species,
                np.array(neigh_list, dtype=np.intc),
                nl_ctx.coords,
                descriptor,
                dE_dZeta[i, :],
            )
            derivatives_unrolled += descriptors_derivative.reshape(-1, 3)

        derivatives = np.zeros(configuration.coords.shape)
        neigh_images = nl_ctx.get_image()
        for i, atom in enumerate(neigh_images):
            derivatives[atom, :] += derivatives_unrolled[i, :]

        return derivatives

    def __call__(
        self, configuration: Configuration, return_extended_state=False
    ) -> Union[np.ndarray, Dict]:
        """
        Map a configuration to a descriptor, but more importantly store all the information
        needed to compute the reverse pass easily on the batched configuration.
        This __call__ method specifically stores the neighbor lists and species information
        as a dictionary and copy it to fingerprint attribute. This will be used by the
        descriptor dataset collate function to attach neighbor lists in sequential
        edge-index like format, where then gradient function can be called on stacked
        batch of configurations.
        To get this full state dictionary, set the return_extended_state to True. Otherwise,
        the descriptor numpy array will be returned. This functionality is useful for
        the descriptor dataset collate function.

        Args:
            configuration: ~:class:`kliff.dataset.Configuration` object.
            return_extended_state: If True, the full state dictionary will be returned. Otherwise,
                the descriptor numpy array will be returned.

        Returns:
            Union[numpy.ndarray, Dict]: Descriptor numpy array or full state dictionary.
        """
        nl_ctx = NeighborList(configuration, self.cutoff)
        n_atoms = configuration.get_num_atoms()
        species = np.array(self._map_species_to_int(nl_ctx.species), np.intc)
        num_neigh, neigh_list = nl_ctx.get_numneigh_and_neighlist_1D()
        coords = nl_ctx.get_coords()
        descriptors = lds.compute(
            self._cdesc, n_atoms, species, neigh_list, num_neigh, coords
        )

        index = configuration.metadata.get("index", -1)

        if return_extended_state:
            output = {
                "n_atoms": n_atoms,
                "species": species,
                "neigh_list": neigh_list,
                "num_neigh": num_neigh,
                "image": nl_ctx.get_image(),
                "coords": coords,
                "descriptor": descriptors,
                "index": index,
                "weight": configuration.weight.to_dict(),
            }
        else:
            output = descriptors
        if self.copy_to_config:
            configuration.fingerprint = output
        return output

    def save_descriptor_state(
        self, path: Union[str, Path], fname: str = "descriptor.dat"
    ):
        """
        Write the descriptor parameters to a file, which can be used by libdescritpor
        to re-initialize the descriptor.

        TODO:
            Refactor this function to abstract ut all respective functions. That should
            make this function easier to maintain. e.g. See _init_descriptor_from_kind().

        Args:
            path (str): Path to the directory where the file will be saved.
            fname (str): Name of the descriptor file.
        """
        with open(Path.joinpath(Path(path), fname), "w") as fout:
            # header
            fout.write("#" + "=" * 80 + "\n")
            fout.write("# Descriptor parameters file generated by KLIFF.\n")
            fout.write("#" + "=" * 80 + "\n\n")

            # cutoff and species
            cutname, rcut = self.cutoff_function, self.cutoff

            fout.write("{}  # cutoff type\n\n".format(cutname))
            fout.write("{}  # number of species\n\n".format(len(self.species)))
            fout.write("# species 1    species 2    cutoff\n")
            for i, species1 in enumerate(self.species):
                for j, species2 in enumerate(self.species):
                    fout.write("{}  {}  {}\n".format(species1, species2, self.cutoff))
            fout.write("\n")

            if self.descriptor_kind == lds.AvailableDescriptors(0):
                # header
                fout.write("#" + "=" * 80 + "\n")
                fout.write("# symmetry functions\n")
                fout.write("#" + "=" * 80 + "\n\n")

                num_sym_func = len(self.hyperparameters.keys())
                fout.write(
                    "{}  # number of symmetry functions types\n\n".format(num_sym_func)
                )

                # descriptor values
                fout.write("# sym_function    rows    cols\n")
                for name, values in self.hyperparameters.items():
                    if name == "g1":
                        fout.write("g1\n\n")
                    else:
                        rows = len(values)
                        cols = len(values[0])
                        fout.write("{}    {}    {}\n".format(name, rows, cols))
                        if name == "g2":
                            for val in values:
                                fout.write("{}    {}".format(val["eta"], val["Rs"]))
                                fout.write("    # eta  Rs\n")
                            fout.write("\n")
                        elif name == "g3":
                            for val in values:
                                fout.write("{}".format(val["kappa"]))
                                fout.write("    # kappa\n")
                            fout.write("\n")
                        elif name == "g4":
                            for val in values:
                                zeta = val["zeta"]
                                lam = val["lambda"]
                                eta = val["eta"]
                                fout.write("{}    {}    {}".format(zeta, lam, eta))
                                fout.write("    # zeta  lambda  eta\n")
                            fout.write("\n")
                        elif name == "g5":
                            for val in values:
                                zeta = val["zeta"]
                                lam = val["lambda"]
                                eta = val["eta"]
                                fout.write("{}    {}    {}".format(zeta, lam, eta))
                                fout.write("    # zeta  lambda  eta\n")
                            fout.write("\n")

                # header
                fout.write("#" + "=" * 80 + "\n")
                fout.write("# Preprocessing data to center and normalize\n")
                fout.write("#" + "=" * 80 + "\n")

                # mean and stdev
                mean = [0.0]
                stdev = [1.0]
                if mean is None and stdev is None:
                    fout.write("center_and_normalize  False\n")
                else:
                    fout.write("center_and_normalize  True\n\n")

                    fout.write("{}   # descriptor size\n".format(self.width))

                    fout.write("# mean\n")
                    for i in mean:
                        fout.write("{} \n".format(i))
                    fout.write("\n# standard deviation\n")
                    for i in stdev:
                        fout.write("{} \n".format(i))
                    fout.write("\n")
            elif self.descriptor_kind == lds.AvailableDescriptors(1):
                fout.write(f"# jmax\n{self.hyperparameters['jmax']}\n\n")
                fout.write(f"# rfac0\n{self.hyperparameters['rfac0']}\n\n")
                fout.write(
                    f"# diagonalstyle\n{self.hyperparameters['diagonalstyle']}\n\n"
                )
                fout.write(f"# rmin0\n{self.hyperparameters['rmin0']}\n\n")
                fout.write(f"# switch_flag\n{self.hyperparameters['switch_flag']}\n\n")
                fout.write(f"# bzero_flag\n{self.hyperparameters['bzero_flag']}\n\n")
                fout.write("# weights\n")
                if self.hyperparameters["weights"] is None:
                    for i in range(len(self.species)):
                        fout.write("1.0    ")
                else:
                    for i in self.hyperparameters["weights"]:
                        fout.write(f"{i}    ")
                fout.write("\n\n")
            elif self.descriptor_kind == lds.AvailableDescriptors(2):
                fout.write(f"# n_max\n{self.hyperparameters['n_max']}\n\n")
                fout.write(f"# l_max\n{self.hyperparameters['l_max']}\n\n")
                fout.write(f"# cutoff\n{self.hyperparameters['cutoff']}\n\n")
                fout.write(
                    f"# radial_basis\n{self.hyperparameters['radial_basis']}\n\n"
                )
                fout.write(f"# eta\n{self.hyperparameters['eta']}\n\n")

    def export_kim_model(self, path: str, model: str):
        """Saves the descriptor model in the KIM format, for re-use in KIM model driver.

        Args:
            path (str): Path to the directory where the file will be saved.
            model (str): Name of the model.
        """
        with open(f"{path}/kim_model.param", "w") as f:
            n_elements = len(self.species)
            f.write(f"# Number of elements\n")
            f.write(f"{n_elements}\n")
            f.write(f"{' '.join(self.species)}\n\n")

            f.write("# Preprocessing kind\n")
            f.write("Descriptor\n\n")

            f.write("# Cutoff distance\n")
            f.write(f"{self.cutoff}\n\n")

            f.write("# Model\n")
            f.write(f"{model}\n\n")

            f.write("# Returns Forces\n")
            f.write("False\n\n")

            f.write("# Number of inputs\n")
            f.write("1\n\n")

            f.write("# Any descriptors?\n")
            f.write(f"{self.descriptor_name}\n")


class DescriptorsError(Exception):
    def __init__(self, msg):
        super().__init__(msg)
        self.msg = msg
