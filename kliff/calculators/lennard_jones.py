import numpy as np
from .calculator import ComputeArgument
from .calculator import Calculator
from .calculator import Parameter
from ..neighbor import NeighborList
from ..neighbor import assemble_forces
from ..neighbor import assemble_stress


class LJComputeArgument(ComputeArgument):
    """ A Lennard-Jones 6-12 potential.
    """

    implemented_property = ['energy', 'forces', 'stress']

    def __init__(self, *args, **kwargs):
        super(LJComputeArgument, self).__init__(*args, **kwargs)
        self.cutoff = 5.
        self.refresh()

    def refresh(self):
        # create neighbor list
        cutoff = {'C-C': self.cutoff}
        neigh = NeighborList(self.conf, cutoff, padding_need_neigh=True)
        self.neigh = neigh

    def compute(self, params):
        epsilon = params['epsilon'].value
        sigma = params['sigma'].value

        rcut = self.cutoff
        coords = self.conf.coords

        coords_including_padding = np.reshape(self.neigh.coords, (-1, 3))
        if self.compute_forces:
            forces_including_padding = np.zeros_like(coords_including_padding)

        energy = 0
        for i, xyz_i in enumerate(coords):
            for j in self.neigh.neighlist[i]:
                xyz_j = coords_including_padding[j]
                rij = xyz_j - xyz_i
                r = np.linalg.norm(rij)
                if self.compute_forces:
                    phi, dphi = self.calc_phi_dphi(epsilon, sigma, r, rcut)
                    energy += 0.5*phi
                    pair = 0.5*dphi/r*rij
                    forces_including_padding[i] += pair
                    forces_including_padding[j] -= pair
                elif self.compute_energy:
                    phi = self.calc_phi_dphi(epsilon, sigma, r, rcut)
                    energy += 0.5*phi

        if self.compute_energy:
            self.results['energy'] = energy
        if self.compute_forces:
            forces = assemble_forces(
                forces_including_padding, self.neigh.ncontrib, self.neigh.image_pad)
            self.results['forces'] = forces
        if self.compute_stress:
            volume = self.conf.get_volume()
            stress = assemble_stress(
                coords_including_padding, forces_including_padding, volume)
            self.results['stress'] = stress

    @staticmethod
    def calc_phi(epsilon, sigma, r, rcut):
        if r > rcut:
            phi = 0
        else:
            sor = sigma/r
            sor6 = sor*sor*sor
            sor6 = sor6*sor6
            sor12 = sor6*sor6
            phi = 4*epsilon*(sor12 - sor6)
        return phi

    @staticmethod
    def calc_phi_dphi(epsilon, sigma, r, rcut):
        if r > rcut:
            phi = 0.
            dphi = 0.
        else:
            sor = sigma/r
            sor6 = sor*sor*sor
            sor6 = sor6*sor6
            sor12 = sor6*sor6
            phi = 4*epsilon*(sor12 - sor6)
            dphi = 24*epsilon*(-2*sor12 + sor6) / r
        return phi, dphi


class LennardJones(Calculator):

    def __init__(self, *args, **kwargs):
        super(LennardJones, self).__init__(*args, **kwargs)
        self.params['epsilon'] = Parameter(value=1.0)
        self.params['sigma'] = Parameter(value=2.0)
        self.compute_argument_class = LJComputeArgument
