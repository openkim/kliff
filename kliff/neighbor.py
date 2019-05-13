import numpy as np
import sys
import kimpy
from kimpy import neighlist as nl
from kliff.atomic_data import atomic_number, atomic_species


class NeighborList:
    """Neighbor list class based on kimpy.neighlist.

    This uses the same approach that `LAMMPS` and `KIM` adopt: The atoms in the
    configuration (assuming a total of N atoms) are named contributing atoms, and
    padding atoms are created to satisfy the boundary conditions. The contributing
    atoms are numbered as 1, 2, ... N-1, and the padding atoms are numbered as N,
    N+1, N+2... Neighbors of atom can include both contributing atoms and padding
    atoms.


    Parameters
    ----------
    conf: Configuration object
        It stores the atoms information.

    infl_dist: float
        Influence distance, within which atoms are interacting with each other.
        In literatures, this is usually refered as `cutoff`.

    padding_need_neigh: bool
        Whether to generate neighbors for padding atoms.

    Attributes
    ----------
    coords: 2D array
        Coordinates of contributing and padding atoms.

    species: list
        Species string of contributing and padding atoms.

    iamge: 1D array
        Atom index, of which an atom is an image. The image of a contributing
        atom is itself.

    padding_coords: 2D array
        Coordinates of padding atoms.

    padding_species: list
        Species string and padding atoms.

    padding_iamge: 1D array
        Atom index, of which a padding atom is an image.

    Note
    ----
    To get the total force on a contributing atom, the forces on all padding atoms
    who are images of the contributing atom should be added back to the contirbuting
    atom.
    """

    def __init__(self, conf, infl_dist, padding_need_neigh=False):
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
        self.neigh = nl.initialize()
        self.create_neigh()

    def create_neigh(self):
        coords_cb = np.asarray(self.conf.get_coordinates(), dtype=np.double)
        species_cb = self.conf.get_species()
        cell = np.asarray(self.conf.get_cell(), dtype=np.double)
        PBC = np.asarray(self.conf.get_PBC(), dtype=np.intc)

        # create padding atoms
        species_code_cb = np.asarray(
            [atomic_number[s] for s in species_cb], dtype=np.intc
        )
        out = nl.create_paddings(self.infl_dist, cell, PBC, coords_cb, species_code_cb)
        coords_pd, species_code_pd, image_pd, error = out
        check_error(error, 'nl.create_padding')
        species_pd = [atomic_species[i] for i in species_code_pd]

        self.padding_coords = np.asarray(coords_pd, dtype=np.double)
        self.padding_species = species_pd
        self.padding_image = np.asarray(image_pd, dtype=np.intc)

        num_cb = coords_cb.shape[0]
        num_pd = coords_pd.shape[0]

        self.coords = np.asarray(np.concatenate((coords_cb, coords_pd)), dtype=np.double)
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
        error = nl.build(self.neigh, self.coords, self.infl_dist, cutoffs, need_neigh)
        check_error(error, 'nl.build')

    def get_neigh(self, index):
        """Get the indices, coordiantes, and speices string of a given atom.

        Parameters
        ----------
        index: int
            Atom number whose neighbor info is requested.

        Returns
        -------
        neigh_indices: list
            Indices of neighbor atoms in self.coords and self.species.

        neigh_coords: 2D array
            Coordinates of neighbor atoms.

        neigh_speices: list
            Species symbol of neighbor atoms.
        """

        cutoffs = np.asarray([self.infl_dist], dtype=np.double)
        neigh_list_index = 0
        num_neigh, neigh_indices, error = nl.get_neigh(
            self.neigh, cutoffs, neigh_list_index, index
        )
        check_error(error, 'nl.get_neigh')

        neigh_coords = self.coords[neigh_indices]
        neigh_species = self.species[neigh_indices]
        return neigh_indices, neigh_coords, neigh_species

    def get_numneigh_and_neighlist_1D(self, request_padding=False):
        """Get the number of neighbrs and neighbor list for all atoms.

        Parameters
        ----------
        request_padding: bool
            If ``True``, the returned number of neighbors and neighbor list include
            those for padding atoms; If ``False``, only return these for contirbuting
            atoms.

        Return
        ------
        numneigh: 1D array
            Number of neighbors for all atoms.

        neighlist: 1D array
            Indicies of the neighbors for all atoms stacked into a 1D array.
            Its total length is ``sum(numneigh)``, and the first ``numneigh[0]``
            components are the neighbors of atom `0`, the next ``numneigh[1]``
            componemnts are the neighbors of atom `1` ....
        """
        if request_padding:
            if not self.padding_need_neigh:
                raise NeighborListError(
                    'Request to get neighbors of padding atoms, but '
                    '"padding_need_neigh" is set to "False" at initializaion.'
                )
            N = len(self.coords)
        else:
            N = self.conf.get_number_of_atoms()

        cutoffs = np.asarray([self.infl_dist], dtype=np.double)
        neigh_list_index = 0

        numneigh = []
        neighlist = []
        for i in range(N):
            num_neigh, neigh_indices, error = nl.get_neigh(
                self.neigh, cutoffs, neigh_list_index, i
            )
            check_error(error, 'nl.get_neigh')
            numneigh.append(num_neigh)
            neighlist.append(neigh_indices)
        neighlist = np.asarray(np.concatenate(neighlist), dtype=np.intc)
        numneigh = np.asarray(numneigh, dtype=np.intc)

        return numneigh, neighlist

    def get_coords(self):
        """Return coords of both contributing and padding atoms."""
        return self.coords.copy()

    def get_species(self):
        """Return speices of both contributing and padding atoms."""
        return self.species[:]

    def get_species_code(self, mapping):
        """Integer species code of both contributing and padding atoms.

        Parameters
        ----------
        mapping: dict
            A mapping between species string and its code.

        Return
        1D array
            Integer species code.
        """
        return np.asarray([mapping[s] for s in self.species], dtype=np.intc)

    def get_image(self):
        """Return image of both contributing and padding atoms."""
        return self.image.copy()

    def get_padding_coords(self):
        """Return coords of padding atoms."""
        return self.padding_coords.copy()

    def get_padding_speices(self):
        """Return species string of padding atoms."""
        return self.padding_speices[:]

    def get_padding_species_code(self, mapping):
        """Integer species code of padding atoms.

        Parameters
        ----------
        mapping: dict
            A mapping between species string and its code.

        Return
        1D array
            Integer species code.
        """
        return np.asarray([mapping[s] for s in self.padding_species], dtype=np.intc)

    def get_padding_image(self):
        """Return image of padding atoms."""
        return self.padding_image.copy()

    def __del__(self):
        nl.clean(self.neigh)


def assemble_forces(forces, n, padding_image):
    """
    Assemble forces on padding atoms back to contributing atoms.

    Parameters
    ----------

    forces: 2D array
      forces on both contributing and padding atoms

    n: int
      number of contributing atoms

    padding_image: 1D int array
      atom number, of which the padding atom is an image


    Return
    ------
      Total forces on contributing atoms.
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


def assemble_stress(coords, forces, volume):
    """ Calculate the stress using the f dor r method.

    Parameters
    ----------
    coords: 2D array
        Coordinates of both contributing and padding atoms.

    forces: 2D array
        Partial forces on both contributing and padding atoms.

    volume: float
        Volume of the configuration.
    """

    stress = np.zeros(6)
    stress[0] = np.sum(np.multiply(coords[:, 0], forces[:, 0])) / volume
    stress[1] = np.sum(np.multiply(coords[:, 1], forces[:, 1])) / volume
    stress[2] = np.sum(np.multiply(coords[:, 2], forces[:, 2])) / volume
    stress[3] = np.sum(np.multiply(coords[:, 1], forces[:, 2])) / volume
    stress[4] = np.sum(np.multiply(coords[:, 0], forces[:, 2])) / volume
    stress[5] = np.sum(np.multiply(coords[:, 0], forces[:, 1])) / volume

    return stress


class NeighborListError(Exception):
    def __init__(self, msg):
        super(NeighborListError, self).__init__(msg)
        self.msg = msg

    def __expr__(self):
        return self.msg


def check_error(error, message=None):
    if error != 0 and error is not None:
        raise NeighborListError('Calling "{}" failed.'.format(message))
