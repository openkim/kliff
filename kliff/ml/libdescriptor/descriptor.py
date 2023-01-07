import os
from collections import OrderedDict
from typing import List, Dict, Union

import kliff.ml.libdescriptor.libdescriptor as lds
from kliff.neighbor import NeighborList
from kliff.dataset import Configuration
import numpy as np


class AvailableDescriptors:
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
    @staticmethod
    def show_available_descriptors():
        print("--------------------------------------------------------------------------------------------------")
        print("Descriptors below are currently available, select them by `descriptor: str` attribute:")
        print("--------------------------------------------------------------------------------------------------")
        _instance = AvailableDescriptors()
        for key in _instance.__dict__.keys():
            print(f"{key}")

    def __init__(self, cutoff: float, species: List[str], descriptor: str, hyperparameters: Union[Dict, str],
                 cutoff_function: str = "cos", nl_ctx: NeighborList = None):
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
        if isinstance(hyperparameters, str):
            if hyperparameters == "set51":
                return get_set51()
            elif hyperparameters == "set30":
                return get_set30()
            else:
                raise ValueError("Hyperparameter set not found")
        elif isinstance(hyperparameters, OrderedDict):
            return hyperparameters
        else:
            raise TypeError("Hyperparameters must be either a string or an OrderedDict")

    def _init_descriptor_from_kind(self):
        if self.descriptor_kind == lds.AvailableDescriptors(0):
            cutoff_array = np.ones((len(self.species), len(self.species))) * self.cutoff
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
            raise ValueError("Descriptor kind not supported yet")
        else:
            raise ValueError("Descriptor kind not supported yet")

    def _map_species_to_int(self, species):
        return [self.species.index(s) for s in species]

    def forward(self, configuration: Configuration):
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

    def backward(self, configuration: Configuration, dE_dZeta: np.ndarray):
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
        with open(os.path.join(path, fname), "w") as fout:
            # header
            fout.write("#" + "=" * 80 + "\n")
            fout.write("# Descriptor parameters file generated by KLIFF.\n")
            fout.write("#" + "=" * 80 + "\n\n")

            # cutoff and species
            # cutname, rcut = self.get_cutoff()
            cutname, rcut = self.cutoff_function, self.cutoff

            # unique_pairs = generate_unique_cutoff_pairs(rcut)
            # species = generate_species_code(rcut)

            fout.write("{}  # cutoff type\n\n".format(cutname))
            fout.write("{}  # number of species\n\n".format(len(self.species)))
            fout.write("# species 1    species 2    cutoff\n")
            for i, species1 in enumerate(self.species):
                for j, species2 in enumerate(self.species):
                    fout.write("{}  {}  {}\n".format(species1, species2, self.cutoff))
            # for key, value in unique_pairs.items():
            #     s1, s2 = key.split("-")
            #     fout.write(("{}  {}  " + fmt + "\n").format(s1, s2, value))
            fout.write("\n")

            #
            # symmetry functions
            #

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

            #
            # data centering and normalization
            #

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

    def save_kim_model(self, path: str, model: str):
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
