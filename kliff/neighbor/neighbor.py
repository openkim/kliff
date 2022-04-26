from typing import Dict, List, Tuple

import numpy as np

from kliff.atomic_data import atomic_number, atomic_species
from kliff.dataset.dataset import Configuration
from kliff.neighbor import neighlist as nl  # C extension


class NeighborList:
    """
    Neighbor list class.

    This uses the same approach that `LAMMPS` and `KIM` adopt: The atoms in the
    configuration (assuming a total of N atoms) are named contributing atoms, and
    padding atoms are created to satisfy the boundary conditions. The contributing atoms
    are numbered as 1, 2, ... N-1, and the padding atoms are numbered as N, N+1, N+2...
    Neighbors of atom can include both contributing atoms and padding atoms.

    Args:
        conf: atomic configuration.
        infl_dist: Influence distance, within which atoms are interacting with each
            other. In literatures, this is usually referred as ``cutoff``.
        padding_need_neigh: Whether to generate neighbors for padding atoms.

    Attributes:
        coords: 2D array
            Coordinates of contributing and padding atoms.
        species: list
            Species string of contributing and padding atoms.
        image: 1D array
            Atom index, of which an atom is an image. The image of a contributing atom
            is itself.
        padding_coords: 2D array
            Coordinates of padding atoms.
        padding_species: list
            Species string and padding atoms.
        padding_image: 1D array
            Atom index, of which a padding atom is an image.

    Note:
        To get the total force on a contributing atom, the forces on all padding atoms
        that are images of the contributing atom should be added back to the
        contributing atom.
    """

    def __init__(
        self, conf: Configuration, infl_dist: float, padding_need_neigh: bool = False
    ):
        self.conf = conf
        self.infl_dist = infl_dist
        self.padding_need_neigh = padding_need_neigh

        # all atoms: contrib + padding
        self.coords = None
        self.species = None
        self.image = None

        self.padding_coords = None
        self.padding_species = None
        # padding_image[0] = 3: padding atom 1 is the image of contributing atom 3
        self.padding_image = None

        # neigh
        self.neigh = nl.create()
        self.create_neigh()

    def create_neigh(self):
        coords_cb = np.asarray(self.conf.coords, dtype=np.double)
        species_cb = self.conf.species
        cell = np.asarray(self.conf.cell, dtype=np.double)
        PBC = np.asarray(self.conf.PBC, dtype=np.intc)

        # create padding atoms
        species_code_cb = np.asarray(
            [atomic_number[s] for s in species_cb], dtype=np.intc
        )
        try:
            coords_pd, species_code_pd, image_pd = nl.create_paddings(
                self.infl_dist, cell, PBC, coords_cb, species_code_cb
            )
        except RuntimeError:
            raise NeighborListError("Calling `neighlist.create_paddings` failed.")

        species_pd = [atomic_species[i] for i in species_code_pd]

        self.padding_coords = np.asarray(coords_pd, dtype=np.double)
        self.padding_species = species_pd
        self.padding_image = np.asarray(image_pd, dtype=np.intc)

        num_cb = coords_cb.shape[0]
        num_pd = coords_pd.shape[0]

        self.coords = np.asarray(
            np.concatenate((coords_cb, coords_pd)), dtype=np.double
        )
        self.species = np.concatenate((species_cb, species_pd))
        self.image = np.asarray(
            np.concatenate((np.arange(num_cb), image_pd)), dtype=np.intc
        )
        # flag to indicate whether to create neighborlist for an atom
        need_neigh = np.ones(num_cb + num_pd, dtype=np.intc)
        if not self.padding_need_neigh:
            need_neigh[num_cb:] = 0

        # create neighbor list
        cutoffs = np.asarray([self.infl_dist], dtype=np.double)
        try:
            self.neigh.build(self.coords, self.infl_dist, cutoffs, need_neigh)
        except RuntimeError:
            raise NeighborListError("Calling `neighlist.build` failed.")

    def get_neigh(self, index: int) -> Tuple[List[int], np.array, List[str]]:
        """
        Get the indices, coordinates, and species string of a given atom.

        Args:
            index: Atom number whose neighbor info is requested.

        Returns:
            neigh_indices: Indices of neighbor atoms in self.coords and self.species.
            neigh_coords: 2D array of shape (N, 3), where N is the number of neighbors.
                Coordinates of neighbor atoms.
            neigh_species: Species symbol of neighbor atoms.
        """

        cutoffs = np.asarray([self.infl_dist], dtype=np.double)
        neigh_list_index = 0
        try:
            num_neigh, neigh_indices = self.neigh.get_neigh(
                cutoffs, neigh_list_index, index
            )
        except RuntimeError:
            raise NeighborListError("Calling `neighlist.get_neigh` failed.")

        neigh_coords = self.coords[neigh_indices]
        neigh_species = self.species[neigh_indices]

        return neigh_indices, neigh_coords, neigh_species

    def get_numneigh_and_neighlist_1D(
        self, request_padding: bool = False
    ) -> Tuple[np.array, np.array]:
        """
        Get the number of neighbors and neighbor list for all atoms.

        Args:
            request_padding: If ``True``, the returned number of neighbors and neighbor
                list include those for padding atoms; If ``False``, only return these
                for contributing atoms.

        Returns:
            numneigh: 1D array; number of neighbors for all atoms.
            neighlist: 1D array; indices of the neighbors for all atoms stacked into a
                1D array. Its total length is ``sum(numneigh)``, and the first
                ``numneigh[0]`` components are the neighbors of atom `0`, the next
                ``numneigh[1]`` components are the neighbors of atom `1` ....
        """
        if request_padding:
            if not self.padding_need_neigh:
                raise NeighborListError(
                    "Request to get neighbors of padding atoms, but "
                    '"padding_need_neigh" is set to "False" at initialization.'
                )
            N = len(self.coords)
        else:
            N = self.conf.get_num_atoms()

        cutoffs = np.asarray([self.infl_dist], dtype=np.double)
        neigh_list_index = 0

        numneigh = []
        neighlist = []
        for i in range(N):

            try:
                num_neigh, neigh_indices = self.neigh.get_neigh(
                    cutoffs, neigh_list_index, i
                )
            except RuntimeError:
                raise NeighborListError("Calling `neighlist.get_neigh` failed.")

            numneigh.append(num_neigh)
            neighlist.append(neigh_indices)
        neighlist = np.asarray(np.concatenate(neighlist), dtype=np.intc)
        numneigh = np.asarray(numneigh, dtype=np.intc)

        return numneigh, neighlist

    def get_coords(self) -> np.array:
        """
        Return coords of both contributing and padding atoms.
        Shape (N,3).
        """
        return self.coords.copy()

    def get_species(self) -> List[str]:
        """
        Return species of both contributing and padding atoms.
        """
        return self.species[:]

    def get_species_code(self, mapping: Dict[str, int]) -> np.array:
        """
        Integer species code of both contributing and padding atoms.

        Args:
            mapping: A mapping between species string and its code.

        Returns:
            1D array of integer species code.
        """
        return np.asarray([mapping[s] for s in self.species], dtype=np.intc)

    def get_image(self) -> np.array:
        """
        Return image of both contributing and padding atoms.

        It is a 1D array of atom index, of which an atom is an image.

        Note:
            The image of a contributing atom is itself.
        """
        return self.image.copy()

    def get_padding_coords(self) -> np.array:
        """
        Return coords of padding atoms, 2D array of shape (N,3), where N is the number
        of padding atoms.
        """
        return self.padding_coords.copy()

    def get_padding_speices(self) -> List[str]:
        """
        Return species string of padding atoms.
        """
        return self.padding_speices[:]

    def get_padding_species_code(self, mapping: Dict[str, int]) -> np.array:
        """
        Integer species code of padding atoms.

        Args:
            mapping: A mapping between species string and its code.

        Returns:
            1D array of integer species code for padding atoms.
        """
        return np.asarray([mapping[s] for s in self.padding_species], dtype=np.intc)

    def get_padding_image(self) -> np.array:
        """
        Return image of padding atoms.

        It is a 1D array of atom index, of which a padding atom is an image.

        """
        return self.padding_image.copy()


def assemble_forces(forces: np.array, n: int, padding_image: np.array) -> np.array:
    """
    Assemble forces on padding atoms back to contributing atoms.

    Args:
        forces: Partial forces on both contributing and padding atoms. 2D array of shape
            (Nc+Np, 3). where Nc is the number of contributing atoms, and Np is the
            number of padding atoms. The first Nc rows are the forces for contributing
            atoms.
        n: Number of contributing atoms, i.e. Nc.
        padding_image: atom index, of which the padding atom is an image. 1D int array
            of shape (Np,).

    Returns:
        Total forces on contributing atoms. 2D array of shape (Nc, 3), where Nc is the
        number of contributing atoms.
    """

    # numpy slicing does not make a copy !!!
    total_forces = np.array(forces[:n])

    has_padding = True if padding_image.size != 0 else False

    if has_padding:

        pad_forces = forces[n:]
        n_padding = pad_forces.shape[0]

        if n < n_padding:
            for i in range(n):
                # indices of padding atoms that are images of contributing atom i
                indices = np.where(padding_image == i)
                total_forces[i] += np.sum(pad_forces[indices], axis=0)
        else:
            for f, org_index in zip(pad_forces, padding_image):
                total_forces[org_index] += f

    return total_forces


def assemble_stress(coords: np.array, forces: np.array, volume: float) -> np.array:
    """
    Calculate the virial stress using the negative f dot r method.

    Args:
        coords: Coordinates of both contributing and padding atoms. 2D array of shape
            (Nc+Np, 3), where Nc is the number of contributing atoms, and Np is the
            number of padding atoms. The first Nc rows are the coords of contributing
            atoms.
        forces: Partial forces on both contributing and padding atoms. 2D array of shape
            (Nc+Np, 3), where Nc is the number of contributing atoms, and Np is the
            number of padding atoms. The first Nc rows are the forces on contributing
            atoms.
        volume: Volume of the configuration.

    Return:
        Virial stress in Voigt notation. 1D array of shape (6,).
    """

    stress = np.zeros(6)
    stress[0] = -np.sum(np.multiply(coords[:, 0], forces[:, 0])) / volume
    stress[1] = -np.sum(np.multiply(coords[:, 1], forces[:, 1])) / volume
    stress[2] = -np.sum(np.multiply(coords[:, 2], forces[:, 2])) / volume
    stress[3] = -np.sum(np.multiply(coords[:, 1], forces[:, 2])) / volume
    stress[4] = -np.sum(np.multiply(coords[:, 2], forces[:, 0])) / volume
    stress[5] = -np.sum(np.multiply(coords[:, 0], forces[:, 1])) / volume

    return stress


class NeighborListError(Exception):
    def __init__(self, msg):
        super(NeighborListError, self).__init__(msg)
        self.msg = msg

    def __expr__(self):
        return self.msg
