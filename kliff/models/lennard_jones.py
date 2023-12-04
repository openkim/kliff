from typing import Dict, List, Optional, Tuple

import numpy as np

from kliff.dataset.dataset import Configuration
from kliff.models.model import ComputeArguments, Model
from kliff.models.parameter import Parameter
from kliff.neighbor import NeighborList, assemble_forces, assemble_stress
from kliff.transforms.parameter_transforms import ParameterTransform


class LJComputeArguments(ComputeArguments):
    """
    KLIFF built-in Lennard-Jones 6-12 potential computation functions.
    """

    implemented_property = ["energy", "forces", "stress"]

    def __init__(
        self,
        conf: Configuration,
        supported_species: Dict[str, int],
        influence_distance: float,
        compute_energy: bool = True,
        compute_forces: bool = True,
        compute_stress: bool = False,
    ):
        if supported_species is None:
            species = sorted(set(conf.species))
            supported_species = {si: i for i, si in enumerate(species)}

            # using a single parameter for all species pairs
            self.specie_pairs_to_param_index = {
                (si, sj): 0 for si in species for sj in species
            }
        else:
            self.specie_pairs_to_param_index = (
                self._get_specie_pairs_to_param_index_map(
                    list(supported_species.keys())
                )
            )

        super(LJComputeArguments, self).__init__(
            conf,
            supported_species,
            influence_distance,
            compute_energy,
            compute_forces,
            compute_stress,
        )

        self.neigh = NeighborList(
            self.conf, influence_distance, padding_need_neigh=False
        )

    # TODO, rewrite this function to move the tow phi functions to LennardJones
    #  class, and then we can pass the model to this function, instead of the params.
    #  With this, we can unify the calling function in the Calculator.
    def compute(self, params: Dict[str, Parameter]):
        coords = self.conf.coords
        species = self.conf.species

        coords_including_padding = self.neigh.coords
        if self.compute_forces:
            forces_including_padding = np.zeros_like(coords_including_padding)

        energy = 0
        for i, (xyz_i, si) in enumerate(zip(coords, species)):
            neighlist, _, neigh_species = self.neigh.get_neigh(i)

            for j, sj in zip(neighlist, neigh_species):
                idx = self.specie_pairs_to_param_index[(si, sj)]
                epsilon = params["epsilon"][idx]
                sigma = params["sigma"][idx]
                rcut = params["cutoff"][idx]

                xyz_j = coords_including_padding[j]
                rij = xyz_j - xyz_i
                r = np.linalg.norm(rij)

                if self.compute_forces:
                    phi, dphi = self.calc_phi_dphi(epsilon, sigma, r, rcut)
                    energy += 0.5 * phi
                    pair = 0.5 * dphi / r * rij

                    forces_including_padding[i] = forces_including_padding[i] + pair
                    forces_including_padding[j] = forces_including_padding[j] - pair

                elif self.compute_energy:
                    phi = self.calc_phi(epsilon, sigma, r, rcut)
                    energy += 0.5 * phi

        if self.compute_energy:
            self.results["energy"] = energy
        if self.compute_forces:
            forces = assemble_forces(
                forces_including_padding, len(coords), self.neigh.padding_image
            )
            self.results["forces"] = forces
        if self.compute_stress:
            volume = self.conf.get_volume()
            stress = assemble_stress(
                coords_including_padding, forces_including_padding, volume
            )
            self.results["stress"] = stress

    @staticmethod
    def calc_phi(epsilon, sigma, r, rcut):
        if r > rcut:
            phi = 0
        else:
            sor = sigma / r
            sor6 = sor * sor * sor
            sor6 = sor6 * sor6
            sor12 = sor6 * sor6
            phi = 4 * epsilon * (sor12 - sor6)
        return phi

    @staticmethod
    def calc_phi_dphi(epsilon, sigma, r, rcut):
        if r > rcut:
            phi = 0.0
            dphi = 0.0
        else:
            sor = sigma / r
            sor6 = sor * sor * sor
            sor6 = sor6 * sor6
            sor12 = sor6 * sor6
            phi = 4 * epsilon * (sor12 - sor6)
            dphi = 24 * epsilon * (-2 * sor12 + sor6) / r
        return phi, dphi

    @staticmethod
    def _get_specie_pairs_to_param_index_map(
        species: List[str],
    ) -> Dict[Tuple[str, str], int]:
        """
        Return a map from a tuple of two species to the index of the corresponding
        parameter in the parameter array.

        For example, if the supported species are ["A", "B", "C"], then the map will be
        {(A, A): 0, (A, B): 1, (B, A): 1, (A, C): 2, (C, A): 2,
        (B, B): 3, (B, C): 4, (C, B): 4,
        (C, C): 5}.
        """
        n = len(species)

        speices_to_param_index_map = {}

        index = 0
        for i in range(n):
            si = species[i]
            for j in range(i, n):
                sj = species[j]
                speices_to_param_index_map[(si, sj)] = index
                if i != j:
                    speices_to_param_index_map[(sj, si)] = index

                index += 1

        return speices_to_param_index_map


class LennardJones(Model):
    """
    KLIFF built-in Lennard-Jones 6-12 potential model.

    This model supports multiple species, where a different set of parameters is used
    for each species pair. For example if species A, B, and C are provided, then there
    will be 6 values for each of the epsilon and sigma parameters. The order of the
    parameters is as follows: A-A, A-B, A-C, B-B, B-C, and C-C.

    Args:
        model_name: name of the model
        species: list of species. If None, there model will create a single value for
            each parameter, and all species pair will use the same parameters. If a
            list of species is provided, then the model will create a different set of
            parameters for each species pair.
        params_transform: parameter transform object. If None, no transformation is
            performed.
    """

    def __init__(
        self,
        model_name: str = "LJ6-12",
        species: List[str] = None,
    ):
        self.species = species

        super(LennardJones, self).__init__(model_name)

    def init_model_params(self):
        n = self._get_num_params()

        model_params = {
            "epsilon": Parameter(np.asarray([1.0 for _ in range(n)])),
            "sigma": Parameter(np.asarray([2.0 for _ in range(n)])),
            "cutoff": Parameter(np.asarray([5.0 for _ in range(n)])),
        }

        return model_params

    def init_influence_distance(self):
        return self.model_params["cutoff"][0]

    def init_supported_species(self):
        if self.species is None:
            return None
        else:
            return {s: i for i, s in enumerate(self.species)}

    def get_compute_argument_class(self):
        return LJComputeArguments

    def _get_num_params(self):
        if self.species is None:
            n = 1
        else:
            n = len(self.species)

        return (n + 1) * n // 2
