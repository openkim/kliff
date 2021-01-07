import logging
from typing import Callable, Dict, Optional

import numpy as np
from kliff.dataset.dataset import Configuration
from kliff.models.model import ComputeArguments, Model
from kliff.models.parameter import Parameter
from kliff.neighbor import NeighborList, assemble_forces, assemble_stress

logger = logging.getLogger(__name__)


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

    def compute(self, params: Dict[str, Parameter]):
        epsilon = params["epsilon"][0]
        sigma = params["sigma"][0]
        rcut = params["cutoff"][0]
        coords = self.conf.coords

        coords_including_padding = self.neigh.coords
        if self.compute_forces:
            forces_including_padding = np.zeros_like(coords_including_padding)

        energy = 0
        for i, xyz_i in enumerate(coords):
            neighlist, _, _ = self.neigh.get_neigh(i)
            for j in neighlist:
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


class LennardJones(Model):
    """
    KLIFF built-in Lennard-Jones 6-12 potential model.
    """

    def __init__(
        self, model_name="LJ6-12", params_relation_callback: Optional[Callable] = None
    ):
        super(LennardJones, self).__init__(model_name, params_relation_callback)

    def init_model_params(self):
        model_params = {
            "epsilon": Parameter(value=[1.0]),
            "sigma": Parameter(value=[2.0]),
            "cutoff": Parameter(value=[5.0]),
        }

        return model_params

    def init_influence_distance(self):
        return self.model_params["cutoff"][0]

    def init_supported_species(self):
        return None

    def get_compute_argument_class(self):
        return LJComputeArguments
