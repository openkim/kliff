import os
from collections import OrderedDict
from typing import List, Dict, Union

from loguru import logger
from kliff.neighbor import NeighborList
import numpy as np
from kliff.dataset import Configuration

# stubs for type hinting
try:
    from ase import Atoms
except ImportError:
    Atoms = None
    # LOG warning
try:
    import libdescriptor as lds
except ImportError:
    logger.error("Descriptors module depends on libdescriptor, which is not found. Please install it first.")
    # raise ImportError("Descriptors module depends on libdescriptor, which is not found. Please install it first.")


class AvailableDescriptors:
    """
    This class lists all the available descriptors in libdescriptor as enums.
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


class Descriptor:
    """
    Descriptor class, which is a wrapper of libdescriptor. It provides a unified interface to all the
    descriptors in libdescriptor. The descriptor is initialized with a cutoff radius, a list of species,
     a descriptor type and an ordered list of hyperparameters. The descriptor type is a string, which can
        be obtained by `AvailableDescriptors.show_available_descriptors()`. The hyperparameters are a list
        of dictionaries, some sane default values are provided by `get_set51()` and `get_set30()` for Symmetry
        Functions, and `get_default_bispectrum()` for Bispectrum. This class also provides a methods to compute
        derivatives of the descriptor with respect to atomic positions. The functions generating descriptors and
        their derivatives are implemented as `forward()` and `backward()`, respectively, to match the PyTorch
        nomeclature.
    """
    @staticmethod
    def show_available_descriptors():
        """
        Show all the available descriptors in libdescriptor.
        """
        print("-"*80)
        print("Descriptors below are currently available, select them by `descriptor: str` attribute:")
        print("-"*80)
        _instance = AvailableDescriptors()
        for key in _instance.__dict__.keys():
            print(f"{key}")

    def __init__(self, cutoff: float, species: List[str], descriptor: str, hyperparameters: Union[Dict, str],
                 cutoff_function: str = "cos", nl_ctx: NeighborList = None):
        """
        :param cutoff: Cutoff radius.
        :param species: List of strings, each string is a species (atomic symbols).
        :param descriptor: String of descriptor type, can be obtained by `show_available_descriptors()`.
        :param hyperparameters: Ordered dictionary of hyperparameters, or a string of preset hyperparameters.
        :param cutoff_function: Cut-off function, currently only "cos" is supported.
        :param nl_ctx: function to compute neighbor list, if not provided, will be computed internally.
        """
        self.cutoff = cutoff
        self.species = species
        _available_descriptors = AvailableDescriptors()
        self.descriptor_name = descriptor
        self.descriptor_kind = getattr(_available_descriptors, descriptor)
        self.width = -1
        self.hyperparameters = self._set_hyperparams(hyperparameters)
        self.cutoff_function = cutoff_function
        self._cdesc, self.width = self._init_descriptor_from_kind()
        if nl_ctx:
            self.nl_ctx = nl_ctx
            self.external_nl_ctx = True
        else:
            self.external_nl_ctx = False

    def _set_hyperparams(self, hyperparameters):
        """
        Set hyperparameters.
        :param hyperparameters:
        :return:
        """
        if isinstance(hyperparameters, str):
            if hyperparameters == "set51":
                return get_set51()
            elif hyperparameters == "set30":
                return get_set30()
            elif hyperparameters == "bs_defaults":
                return get_default_bispectrum()
            else:
                raise ValueError("Hyperparameter set not found")
        elif isinstance(hyperparameters, OrderedDict):
            return hyperparameters
        else:
            raise TypeError("Hyperparameters must be either a string or an OrderedDict")

    def _init_descriptor_from_kind(self):
        """
        Initialize descriptor from descriptor kind. Currently only Symmetry Functions and Bispectrum are supported.
        :return:
        """
        cutoff_array = np.ones((len(self.species), len(self.species))) * self.cutoff
        if self.descriptor_kind == lds.AvailableDescriptors(0):
            symmetry_function_types = list(self.hyperparameters.keys())
            symmetry_function_sizes = []

            symmetry_function_param_matrices = []
            param_num_elem = 0
            width = 0
            for function in symmetry_function_types:
                if function.lower() not in ["g1", "g2", "g3", "g4", "g5"]:
                    ValueError("Symmetry Function provided, not supported")

                if function.lower() == "g1":
                    rows = 1
                    cols = 1
                    params_mat = np.zeros((1, 1), dtype=np.double)
                else:
                    params = self.hyperparameters[function]
                    rows = len(params)
                    cols = len(list(params[0].keys()))
                    params_mat = np.zeros((rows, cols), dtype=np.double)

                    for i in range(rows):
                        if function.lower() == "g2":
                            params_mat[i, 0] = params[i]["eta"]
                            params_mat[i, 1] = params[i]["Rs"]
                        elif function.lower() == "g3":
                            params_mat[i, 0] = params[i]["kappa"]
                        elif function.lower() == "g4":
                            params_mat[i, 0] = params[i]["zeta"]
                            params_mat[i, 1] = params[i]["lambda"]
                            params_mat[i, 2] = params[i]["eta"]
                        elif function.lower() == "g5":
                            params_mat[i, 0] = params[i]["zeta"]
                            params_mat[i, 1] = params[i]["lambda"]
                            params_mat[i, 2] = params[i]["eta"]
                symmetry_function_sizes.extend([rows, cols])
                symmetry_function_param_matrices.append(params_mat)
                param_num_elem += rows * cols
                width += rows

            symmetry_function_param = np.zeros((param_num_elem,), dtype=np.double)
            k = 0
            for i in range(len(symmetry_function_types)):
                symmetry_function_param[k: k + symmetry_function_sizes[2 * i] * symmetry_function_sizes[2 * i + 1]] = \
                    symmetry_function_param_matrices[i].reshape(1, -1)
                k += symmetry_function_sizes[2 * i] * symmetry_function_sizes[2 * i + 1]

            return lds.DescriptorKind.init_descriptor(self.descriptor_kind, self.species, self.cutoff_function,
                                                      cutoff_array,
                                                      symmetry_function_types, symmetry_function_sizes,
                                                      symmetry_function_param), width
        elif self.descriptor_kind == lds.AvailableDescriptors(1):
            if self.hyperparameters["weights"] is None:
                weights = np.ones(len(self.species))
            else:
                weights = self.hyperparameters["weights"]

            return lds.DescriptorKind.init_descriptor(self.descriptor_kind, self.hyperparameters["rfac0"],
                                                      2 * self.hyperparameters["jmax"],
                                                      self.hyperparameters["diagonalstyle"],
                                                      1 if self.hyperparameters["use_shared_array"] else 0,
                                                      self.hyperparameters["rmin0"],
                                                      self.hyperparameters["switch_flag"],
                                                      self.hyperparameters["bzero_flag"],
                                                      cutoff_array,
                                                      self.species,
                                                      weights), \
                get_bs_size(int(2 * self.hyperparameters["jmax"]), self.hyperparameters["diagonalstyle"])
        else:
            raise ValueError(f"Descriptor kind: {self.descriptor_kind} not supported yet")

    def _map_species_to_int(self, species):
        return [self.species.index(s) for s in species]

    def forward(self, configuration: Union[Configuration, Atoms]):
        """
        Compute the descriptors for a given configuration, by calling the C++ implementation,
        :py:func:`libdescriptor.compute_single_atom`. Takes in either a KLIFF Configuration or ASE Atoms object.
        TODO: Use the :py:func:`libdescriptor.compute` function for faster evaluation. Which loops in C++.
        :param configuration: :py:class:`kliff.dataset.Configuration` or :py:class:`ase.Atoms` object.
        :return: :numpy:`ndarray` of shape (n_atoms, width)
        """
        if not self.external_nl_ctx:
            self.nl_ctx = NeighborList(configuration, self.cutoff)
        n_atoms = configuration.get_num_atoms()
        descriptors = np.zeros((n_atoms, self.width))
        species = np.array(self._map_species_to_int(self.nl_ctx.species), np.intc)
        for i in range(n_atoms):
            neigh_list, _, _ = self.nl_ctx.get_neigh(i)
            # TODO Implement and use compute function for faster evaluation. Move this loop to C++.
            descriptors[i, :] = lds.compute_single_atom(self._cdesc, i, species, np.array(neigh_list, dtype=np.intc),
                                                        self.nl_ctx.coords)
        return descriptors

    def backward(self, configuration: Union[Configuration, Atoms], dE_dZeta: np.ndarray):
        """
        Compute the gradients of the descriptors with respect to the atomic coordinates. It takes in an array of
        shape (n_atoms, width) and the configuration, and performs the vector-Jacobian product (revrse mode
        automatic differentiation). The output is an array of shape (n_atoms, 3) yielding the gradients of the
        descriptors with respect to the atomic coordinates.
        :param configuration: :py:class:`kliff.dataset.Configuration` or :py:class:`ase.Atoms` object.
        :param dE_dZeta: :numpy:`ndarray` of shape (n_atoms, width)
        :return: :numpy:`ndarray` of shape (n_atoms, 3)
        """
        if not self.external_nl_ctx:
            self.nl_ctx = NeighborList(configuration, self.cutoff)
        n_atoms = configuration.get_num_atoms()
        derivatives_unrolled = np.zeros(self.nl_ctx.coords.shape)
        species = np.array(self._map_species_to_int(self.nl_ctx.species), dtype=np.intc)

        descriptor = np.zeros(self.width)

        for i in range(n_atoms):
            neigh_list, _, _ = self.nl_ctx.get_neigh(i)
            descriptors_derivative = lds.gradient_single_atom(self._cdesc, i, species,
                                                              np.array(neigh_list, dtype=np.intc), self.nl_ctx.coords,
                                                              descriptor, dE_dZeta[i, :])
            derivatives_unrolled += descriptors_derivative.reshape(-1, 3)

        derivatives = np.zeros(configuration.coords.shape)
        neigh_images = self.nl_ctx.get_image()
        for i, atom in enumerate(neigh_images):
            derivatives[atom, :] += derivatives_unrolled[i, :]

        return derivatives

    def write_kim_params(self, path, fname="descriptor.params"):
        """
        Write the descriptor parameters to a file, which can be used by KIM-API. This is the additional
        information besides the :py:function:`save_kim_model` function, which saves complete descriptor model.
         Currently supports exporting Symmetry Functions and Bispectrum descriptors.
        :param path: path to the directory where the file will be saved.
        :param fname: name of the descriptor file.
        :return:
        """
        with open(os.path.join(path, fname), "w") as fout:
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
                fout.write("{}  # number of symmetry functions types\n\n".format(num_sym_func))

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
                fout.write(f"# diagonalstyle\n{self.hyperparameters['diagonalstyle']}\n\n")
                fout.write(f"# rmin0\n{self.hyperparameters['rmin0']}\n\n")
                fout.write(f"# switch_flag\n{self.hyperparameters['switch_flag']}\n\n")
                fout.write(f"# bzero_flag\n{self.hyperparameters['bzero_flag']}\n\n")
                fout.write("# weights\n")
                if self.hyperparameters['weights'] is None:
                    for i in range(len(self.species)):
                        fout.write("1.0    ")
                else:
                    for i in self.hyperparameters['weights']:
                        fout.write(f"{i}    ")
                fout.write("\n\n")

    def save_kim_model(self, path: str, model: str):
        """Saves the descriptor model in the KIM format.
        :param path: path to the directory where the file will be saved.
        :param model: name of the model.
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


def get_set51():
    r"""Hyperparameters for symmetry functions, as discussed in:
    Nongnuch Artrith and Jorg Behler. "High-dimensional neural network potentials for
    metal surfaces: A prototype study for copper." Physical Review B 85, no. 4 (2012):
    045439.
    """
    return OrderedDict([('g2',
                         [{'eta': 0.0035710676725828126, 'Rs': 0.0},
                          {'eta': 0.03571067672582813, 'Rs': 0.0},
                          {'eta': 0.07142135345165626, 'Rs': 0.0},
                          {'eta': 0.12498736854039845, 'Rs': 0.0},
                          {'eta': 0.21426406035496876, 'Rs': 0.0},
                          {'eta': 0.3571067672582813, 'Rs': 0.0},
                          {'eta': 0.7142135345165626, 'Rs': 0.0},
                          {'eta': 1.428427069033125, 'Rs': 0.0}]),
                        ('g4',
                         [{'zeta': 1, 'lambda': -1, 'eta': 0.00035710676725828126},
                          {'zeta': 1, 'lambda': 1, 'eta': 0.00035710676725828126},
                          {'zeta': 2, 'lambda': -1, 'eta': 0.00035710676725828126},
                          {'zeta': 2, 'lambda': 1, 'eta': 0.00035710676725828126},
                          {'zeta': 1, 'lambda': -1, 'eta': 0.010713203017748437},
                          {'zeta': 1, 'lambda': 1, 'eta': 0.010713203017748437},
                          {'zeta': 2, 'lambda': -1, 'eta': 0.010713203017748437},
                          {'zeta': 2, 'lambda': 1, 'eta': 0.010713203017748437},
                          {'zeta': 1, 'lambda': -1, 'eta': 0.0285685413806625},
                          {'zeta': 1, 'lambda': 1, 'eta': 0.0285685413806625},
                          {'zeta': 2, 'lambda': -1, 'eta': 0.0285685413806625},
                          {'zeta': 2, 'lambda': 1, 'eta': 0.0285685413806625},
                          {'zeta': 1, 'lambda': -1, 'eta': 0.05356601508874219},
                          {'zeta': 1, 'lambda': 1, 'eta': 0.05356601508874219},
                          {'zeta': 2, 'lambda': -1, 'eta': 0.05356601508874219},
                          {'zeta': 2, 'lambda': 1, 'eta': 0.05356601508874219},
                          {'zeta': 4, 'lambda': -1, 'eta': 0.05356601508874219},
                          {'zeta': 4, 'lambda': 1, 'eta': 0.05356601508874219},
                          {'zeta': 16, 'lambda': -1, 'eta': 0.05356601508874219},
                          {'zeta': 16, 'lambda': 1, 'eta': 0.05356601508874219},
                          {'zeta': 1, 'lambda': -1, 'eta': 0.08927669181457032},
                          {'zeta': 1, 'lambda': 1, 'eta': 0.08927669181457032},
                          {'zeta': 2, 'lambda': -1, 'eta': 0.08927669181457032},
                          {'zeta': 2, 'lambda': 1, 'eta': 0.08927669181457032},
                          {'zeta': 4, 'lambda': -1, 'eta': 0.08927669181457032},
                          {'zeta': 4, 'lambda': 1, 'eta': 0.08927669181457032},
                          {'zeta': 16, 'lambda': -1, 'eta': 0.08927669181457032},
                          {'zeta': 16, 'lambda': 1, 'eta': 0.08927669181457032},
                          {'zeta': 1, 'lambda': -1, 'eta': 0.16069804526622655},
                          {'zeta': 1, 'lambda': 1, 'eta': 0.16069804526622655},
                          {'zeta': 2, 'lambda': -1, 'eta': 0.16069804526622655},
                          {'zeta': 2, 'lambda': 1, 'eta': 0.16069804526622655},
                          {'zeta': 4, 'lambda': -1, 'eta': 0.16069804526622655},
                          {'zeta': 4, 'lambda': 1, 'eta': 0.16069804526622655},
                          {'zeta': 16, 'lambda': -1, 'eta': 0.16069804526622655},
                          {'zeta': 16, 'lambda': 1, 'eta': 0.16069804526622655},
                          {'zeta': 1, 'lambda': -1, 'eta': 0.28568541380662504},
                          {'zeta': 1, 'lambda': 1, 'eta': 0.28568541380662504},
                          {'zeta': 2, 'lambda': -1, 'eta': 0.28568541380662504},
                          {'zeta': 2, 'lambda': 1, 'eta': 0.28568541380662504},
                          {'zeta': 4, 'lambda': -1, 'eta': 0.28568541380662504},
                          {'zeta': 4, 'lambda': 1, 'eta': 0.28568541380662504},
                          {'zeta': 16, 'lambda': 1, 'eta': 0.28568541380662504}])])


def get_set30():
    r"""Hyperparameters for symmetry functions, as discussed in:
    Artrith, N., Hiller, B. and Behler, J., 2013. Neural network potentials for metals and
    oxides–First applications to copper clusters at zinc oxide. physica status solidi (b),
    250(6), pp.1191-1203.
    """
    return OrderedDict([('g2',
                         [{'eta': 0.003213960905324531, 'Rs': 0.0},
                          {'eta': 0.03571067672582813, 'Rs': 0.0},
                          {'eta': 0.07142135345165626, 'Rs': 0.0},
                          {'eta': 0.12498736854039845, 'Rs': 0.0},
                          {'eta': 0.21426406035496876, 'Rs': 0.0},
                          {'eta': 0.3571067672582813, 'Rs': 0.0},
                          {'eta': 0.7142135345165626, 'Rs': 0.0},
                          {'eta': 1.428427069033125, 'Rs': 0.0}]),
                        ('g4',
                         [{'zeta': 1, 'lambda': -1, 'eta': 0.00035710676725828126},
                          {'zeta': 1, 'lambda': 1, 'eta': 0.00035710676725828126},
                          {'zeta': 2, 'lambda': -1, 'eta': 0.00035710676725828126},
                          {'zeta': 2, 'lambda': 1, 'eta': 0.00035710676725828126},
                          {'zeta': 1, 'lambda': -1, 'eta': 0.010713203017748437},
                          {'zeta': 1, 'lambda': 1, 'eta': 0.010713203017748437},
                          {'zeta': 2, 'lambda': -1, 'eta': 0.010713203017748437},
                          {'zeta': 2, 'lambda': 1, 'eta': 0.010713203017748437},
                          {'zeta': 1, 'lambda': 1, 'eta': 0.0285685413806625},
                          {'zeta': 2, 'lambda': 1, 'eta': 0.0285685413806625},
                          {'zeta': 1, 'lambda': 1, 'eta': 0.05356601508874219},
                          {'zeta': 2, 'lambda': 1, 'eta': 0.05356601508874219},
                          {'zeta': 4, 'lambda': 1, 'eta': 0.05356601508874219},
                          {'zeta': 16, 'lambda': 1, 'eta': 0.05356601508874219},
                          {'zeta': 1, 'lambda': 1, 'eta': 0.08927669181457032},
                          {'zeta': 2, 'lambda': 1, 'eta': 0.08927669181457032},
                          {'zeta': 4, 'lambda': 1, 'eta': 0.08927669181457032},
                          {'zeta': 16, 'lambda': 1, 'eta': 0.08927669181457032},
                          {'zeta': 1, 'lambda': 1, 'eta': 0.16069804526622655},
                          {'zeta': 2, 'lambda': 1, 'eta': 0.16069804526622655},
                          {'zeta': 4, 'lambda': 1, 'eta': 0.16069804526622655},
                          {'zeta': 16, 'lambda': 1, 'eta': 0.16069804526622655}])])


def get_default_bispectrum():
    return OrderedDict({
            "jmax": 4,
            "rfac0": 0.99363,
            "diagonalstyle": 3,
            "rmin0": 0,
            "switch_flag": 1,
            "bzero_flag": 0,
            "use_shared_array": False,
            "weights": None
        })


def get_bs_size(twojmax, diagonal):
        """
        Return the size of descriptor.
        """
        N = 0
        for j1 in range(0, twojmax + 1):
            if diagonal == 2:
                N += 1
            elif diagonal == 1:
                for j in range(0, min(twojmax, 2 * j1) + 1, 2):
                    N += 1
            elif diagonal == 0:
                for j2 in range(0, j1 + 1):
                    for j in range(j1 - j2, min(twojmax, j1 + j2) + 1, 2):
                        N += 1
            elif diagonal == 3:
                for j2 in range(0, j1 + 1):
                    for j in range(j1 - j2, min(twojmax, j1 + j2) + 1, 2):
                        if j >= j1:
                            N += 1
        return N

