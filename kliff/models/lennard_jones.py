import numpy as np
import kliff
from kliff.models.model import ComputeArguments, Model
from kliff.models.parameter import Parameter, ParameterError
from kliff.neighbor import NeighborList, assemble_forces, assemble_stress

logger = kliff.logger.get_logger(__name__)


class LJComputeArguments(ComputeArguments):
    """ A Lennard-Jones 6-12 potential.
    """

    implemented_property = ['energy', 'forces', 'stress']

    def __init__(self, *args, **kwargs):
        super(LJComputeArguments, self).__init__(*args, **kwargs)
        self.refresh(self.influence_distance)

    def refresh(self, influence_distance=None, params=None):
        """ Refresh settings.

        Recreating the neighbor list due to the change of influence distance.
        """
        if influence_distance is not None:
            infl_dist = influence_distance
        else:
            try:
                infl_dist = params['influence_distance'].get_value()[0]
            except KeyError:
                raise ParameterError('"influence_distance" not provided by calculator."')
        self.influence_distance = infl_dist

        # create neighbor list
        self.neigh = NeighborList(self.conf, infl_dist, padding_need_neigh=False)

    def compute(self, params):
        epsilon = params['epsilon'].get_value()[0]
        sigma = params['sigma'].get_value()[0]
        rcut = params['cutoff'].get_value()[0]
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
            self.results['energy'] = energy
        if self.compute_forces:
            forces = assemble_forces(
                forces_including_padding, len(coords), self.neigh.padding_image
            )
            self.results['forces'] = forces
        if self.compute_stress:
            volume = self.conf.get_volume()
            stress = assemble_stress(
                coords_including_padding, forces_including_padding, volume
            )
            self.results['stress'] = stress

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
    def __init__(self, model_name=None, params_relation_callback=None):
        super(LennardJones, self).__init__(model_name, params_relation_callback)

        self.params['epsilon'] = Parameter(value=[1.0])
        self.params['sigma'] = Parameter(value=[2.0])
        self.params['cutoff'] = Parameter(value=[5.0])
        self.compute_arguments_class = LJComputeArguments
        self.fitting_params = self.init_fitting_params(self.params)

    def get_influence_distance(self):
        return self.params['cutoff'].get_value()[0]
