from typing import Dict, List

import numpy as np

from .utils import get_bs_size


def initialize_symmetry_functions(hyperparameters: Dict):
    symmetry_function_types = list(hyperparameters.keys())
    symmetry_function_sizes = []
    symmetry_function_param_matrices = []
    param_num_elem = 0
    width = 0
    for function in symmetry_function_types:
        if function.lower() not in ["g1", "g2", "g3", "g4", "g5"]:
            raise ValueError("Symmetry Function provided, not supported")

        if function.lower() == "g1":
            rows = 1
            cols = 1
            params_mat = np.zeros((1, 1), dtype=np.double)
        else:
            params = hyperparameters[function]
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
        symmetry_function_param[
            k : k + symmetry_function_sizes[2 * i] * symmetry_function_sizes[2 * i + 1]
        ] = symmetry_function_param_matrices[i].reshape(1, -1)
        k += symmetry_function_sizes[2 * i] * symmetry_function_sizes[2 * i + 1]

    return (
        symmetry_function_types,
        symmetry_function_sizes,
        symmetry_function_param,
    ), width


def initialize_bispectrum_functions(hyperparameters: Dict):
    twojmax = 2 * hyperparameters["jmax"]
    use_shared_array = 1 if hyperparameters["use_shared_array"] else 0
    return (
        hyperparameters["rfac0"],
        twojmax,
        hyperparameters["diagonalstyle"],
        use_shared_array,
        hyperparameters["rmin0"],
        hyperparameters["switch_flag"],
        hyperparameters["bzero_flag"],
    ), get_bs_size(
        int(twojmax),
        hyperparameters["diagonalstyle"],
    )
