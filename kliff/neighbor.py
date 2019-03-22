import numpy as np
import sys
import kimpy
from kimpy import neighlist as nl
from kliff.atomic_data import atomic_number, atomic_species


class NeighborList(object):
    """Neighbor list class based on kimpy.neighlist.

    This uses the same approach that `LAMMPS` and `KIM` adopt:
    The atoms in the configuration (assuming a total of N atoms) are named contributing
    atoms, and padding atoms are created to satisfy the boundary conditions. The
    contributing atoms are numbered as 1, 2, ... N-1, and the padding atoms are numbered
    as N, N+1, N+2... Neighbors of atom can include both contributing atoms and padding
    atoms.

    Note
    ----
    To get the total force on a contributing atom, the forces on all padding atoms who
    are images of the contributing atom should be added back to the contirbuting atom.

    Attributes
    ----------
    coords: 2D array
        coordinates of contributing and padding atoms

    species: 1D array
        speices of contributing and padding atoms

    iamge: 1D array
        atom number, of which an atom is an image (the image of a contributing atom
        is itself)

    padding_image: 1D array
        atom number, of which the padding atom is an image
    """

    def __init__(self, conf, infl_dist, padding_need_neigh=False):
        """

        Parameters
        ----------
        conf: Configuration object
            It stores the atoms information.

        infl_dist: float
            Influence distance, within which atoms are interacting with each other.
            In literatures, this is usually refered as `cutoff`.

        padding_need_neigh: bool
            Whether to generate neighbors for padding atoms.
        """

        self.conf = conf
        self.infl_dist = infl_dist
        self.padding_need_neigh = padding_need_neigh

        # all atoms: contrib + padding
        self.coords = None
        self.species = None

        # padding_image[0] = 3: padding atom 1 is the image of contributing atom 3
        self.padding_image = None
        self.image = None

        # neigh
        self.neigh = nl.initialize()
        self.create_neigh()

    def create_neigh(self):
        coords_cb = np.asarray(self.conf.get_coordinates(), dtype=np.double)
        species_cb = self.conf.get_species()
        cell = np.asarray(self.conf.get_cell(), dtype=np.double)
        PBC = np.asarray(self.conf.get_PBC(), dtype=np.intc)

        # create padding atoms
        species_code_cb = np.asarray([atomic_number[s]
                                      for s in species_cb], dtype=np.intc)
        out = nl.create_paddings(self.infl_dist, cell, PBC,
                                 coords_cb, species_code_cb)
        coords_pd, species_code_pd, image_pd, error = out
        check_error(error, 'nl.create_padding')
        species_pd = [atomic_species[i] for i in species_code_pd]

        num_cb = coords_cb.shape[0]
        num_pd = coords_pd.shape[0]

        self.coords = np.asarray(np.concatenate(
            (coords_cb, coords_pd)), dtype=np.double)
        self.species = np.concatenate((species_cb, species_pd))
        self.padding_image = image_pd
        self.image = np.concatenate((np.arange(num_cb), image_pd))

        # flag to indicate whether to create neighborlist for an atom
        need_neigh = np.ones(num_cb + num_pd, dtype=np.intc)
        if not self.padding_need_neigh:
            need_neigh[num_cb:] = 0

        # create neighbor list
        cutoffs = np.asarray([self.infl_dist], dtype=np.double)
        error = nl.build(self.neigh, self.coords,
                         self.infl_dist, cutoffs, need_neigh)
        check_error(error, 'nl.build')

    def get_neigh(self, index):
        """Get the number of neighbors and the neighbor list of atom index.

        Parameters
        ----------
        index: int
            Atom number whose neighbor info is returned.

        Returns
        -------
        neigh_indices: list of float
            indices of neighbor atoms in self.coords and self.species

        neigh_coords: 2D array
            coords of neighbor atoms

        neigh_speices: list of str
            species symbol of neighbor atoms
        """

        cutoffs = np.asarray([self.infl_dist], dtype=np.double)
        neigh_list_index = 0
        num_neigh, neigh_indices, error = nl.get_neigh(self.neigh, cutoffs,
                                                       neigh_list_index, index)
        check_error(error, 'nl.get_neigh')

        neigh_coords = self.coords[neigh_indices]
        neigh_species = self.species[neigh_indices]
        return neigh_indices, neigh_coords, neigh_species

    def get_coords(self):
        """Return coords of both contributing and padding atoms."""
        return self.coords.copy()

    def get_species(self):
        """Return speices of both contributing and padding atoms."""
        return self.species.copy()

    def get_image(self):
        """Return image of both contributing and padding atoms."""
        return self.image.copy()

    def get_padding_coords(self):
        num_cb = self.conf.get_number_of_atoms()
        return self.coords[num_cb:].copy()

    def get_padding_speices(self):
        num_cb = self.conf.get_number_of_atoms()
        return self.speices[num_cb:].copy()

    def get_padding_image(self):
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


def check_error(error, message=None):
    if error != 0 and error is not None:
        raise Exception('kimpy error. Calling "{}" failed.'.format(message))
