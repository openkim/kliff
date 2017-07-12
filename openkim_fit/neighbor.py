from __future__ import division
import numpy as np
import sys


class SetNeigh:
    """Transform the cartesian coords to generalized coords that is the input layer.

    Parameters
    ----------
    conf: Configuration object in which the atoms information are stored

    rcut: dictionary
        the cutoffs for all types of interactions.

        e.g. rcut = {'C-C':1.42, 'C-H':1.0, 'H-H':0.8}
    """

    def __init__(self, conf, rcut):
        self.conf = conf

        # create full binary cutoff
        self.rcut = generate_full_cutoff(rcut)

        # data from config
        self.spec_contrib = self.conf.get_species()
        self.coords_contrib = self.conf.get_coords()
        self.ncontrib = len(self.spec_contrib)
        species_set = set(self.spec_contrib)
        self.nspecies = len(species_set)

        # pad_iamge[0] = 3: padding atom 1 is the image of contributing atom 3
        self.image_pad = None

        # all atoms: contrib + padding
        self.coords = None
        self.spec = None
        self.natoms = None

        # neigh
        self.numneigh = None
        self.neighlist = None

        self.setup_neigh()


    def setup_neigh(self):

        # inquire information from the conf
        cell = self.conf.get_cell()
        PBC = self.conf.get_pbc()

        # create padding atoms
        maxrcut = self.rcut[max(self.rcut, key=self.rcut.get)]
        coords_pad, spec_pad, self.image_pad = set_padding(cell, PBC,
            self.spec_contrib, self.coords_contrib, maxrcut)
        self.coords = np.concatenate((self.coords_contrib, coords_pad))
        self.spec = np.concatenate((self.spec_contrib, spec_pad))
        npad = len(spec_pad)
        self.natoms = self.ncontrib + npad

        # generate neighbor list for contributing atoms
        need_neigh = [1 for _ in range(self.ncontrib)] + [0 for _ in range(npad)]
        self.numneigh, self.neighlist = create_neigh(self.coords, maxrcut, need_neigh)


    def get_neigh(self, i):
        """Get the number of neighbors and the neighbor list of atom `i'.

        Returns
        -------
        n: int
            number of neighbors
        neighs: 1D array
            neighbor list
        """
        n = self.numneigh(i)
        neighs = self.neighlist(i)
        return n, neighs


#    def add_sym_func(self, sepc_type, name, params):
#        """Add symmetry function `name' with parameters `params' to atomic species type `spec_type'."""
#
#
#    def cutoff(self, r, rcut):
#        if r < rcut:
#            fc = 0.5 * (np.cos(np.pi*r/rcut) + 1)
#            dfc = -0.5*np.pi/rcut*np.sin(np.pi*r/rcut)
#            return fc, dfc
#        else:
#            return 0., 0.
#
#
#    def g2(self, r, rcut, eta, Rs):
#        fc, dfc = self.cutoff(r,rcut)
#        eterm = np.exp(-eta*(r-Rs)**2)
#        g = eterm*fc
#        dg = -2*eta(r-rcut)*eterm*fc + eterm*dfc
#        return g,dg
#
#
#    def g3(self, r, rcut, kappa):
#        fc, dfc = self.cutoff(r,rcut)
#        kdotr = kappa*r
#        g = np.cos(kdotr)*fc
#        dg = - kappa*np.sin(kdotr)*fc + np.cos(kdotr)*dfc
#        return g,dg
#
#
#
#
#

def generate_full_cutoff(rcut):
    """Generate a full binary cutoff dictionary.
        e.g. for input
            rcut = {'C-C':1.42, 'C-H':1.0, 'H-H':0.8}
        the output would be
            rcut = {'C-C':1.42, 'C-H':1.0, 'H-C':1.0, 'H-H':0.8}
    """
    rcut2 = dict()
    for key, val in rcut.iteritems():
        spec1,spec2 = key.split('-')
        if spec1 != spec2:
            rcut2[str(spec2)+'-'+str(spec1)] = val
    # merge
    rcut2.update(rcut)
    return rcut2





def create_neigh(coords, rcut, need_neigh):
    """Create a full neighbor list.

    Returns
    -------

    numneigh: 1D array
        number of neighbors of each atom

    neighlist: 2D array
        neighbor list of each atom
    """

    natoms = len(coords)//3
    coords = np.array(coords).reshape(natoms, 3)

    neighlist = []
    numneigh = []
    for i in xrange(natoms):
        if not need_neigh[i]:
            continue
        xyzi = coords[i]
        k = 0
        tmplist = []
        for j in xrange(natoms):
            if j == i:
                continue
            xyzj = coords[j]
            rijmag = np.linalg.norm(np.subtract(xyzi, xyzj))
            if rijmag < rcut:
                tmplist.append(j)
                k += 1
        neighlist.append(tmplist)
        numneigh.append(k)

    return numneigh, neighlist





def set_padding(cell, PBC, species, coords, rcut):
    """ Create padding atoms for PURE PBC so as to generate neighbor list.
    This works no matter rcut is larger or smaller than the boxsize.

    Parameters
    ----------
    cell: 2D array
        supercell lattice vector

    PBC: list
        flag to indicate whether periodic or not in x,y,z diretion

    species: list of string
        atom species symbol

    coords: list
        atom coordiantes

    rcut: float
        cutoff

    Returns
    -------

    abs_coords: list
        absolute (not fractional) coords of padding atoms

    pad_spec: list of string
        species of padding atoms

    image_pad: list of int
        atom number, of which the padding atom is an image

    """
    natoms = len(species)

    # transform coords into fractional coords
    coords = np.reshape(coords, (natoms, 3))
    tcell = np.transpose(cell)
    fcell = np.linalg.inv(tcell)
    frac_coords = np.dot(coords, fcell.T)
    xmin = min(frac_coords[:,0])
    ymin = min(frac_coords[:,1])
    zmin = min(frac_coords[:,2])
    xmax = max(frac_coords[:,0])
    ymax = max(frac_coords[:,1])
    zmax = max(frac_coords[:,2])

    # compute distance between parallelpiped faces
    volume = np.dot(cell[0], np.cross(cell[1], cell[2]))
    dist0 = volume/np.linalg.norm(np.cross(cell[1], cell[2]))
    dist1 = volume/np.linalg.norm(np.cross(cell[2], cell[0]))
    dist2 = volume/np.linalg.norm(np.cross(cell[0], cell[1]))
    ratio0 = rcut/dist0
    ratio1 = rcut/dist1
    ratio2 = rcut/dist2
    # number of bins to repate in each direction
    size0 = int(np.ceil(ratio0))
    size1 = int(np.ceil(ratio1))
    size2 = int(np.ceil(ratio2))

    # creating padding atoms assume ratio0, 1, 2 < 1
    pad_coords = []
    pad_spec = []
    pad_image = []
    for i in range(-size0, size0+1):
      for j in range(-size1, size1+1):
        for k in range(-size2, size2+1):
            if i==0 and j==0 and k==0:  # do not create contributing atoms
                continue
            if not PBC[0] and i != 0:   # apply BC
                continue
            if not PBC[1] and j != 0:
                continue
            if not PBC[2] and k != 0:
                continue
            for at,(x,y,z) in enumerate(frac_coords):

                # select the necessary atoms to repeate for the most outside bin
                if i == -size0 and x - xmin < size0 - ratio0:
                    continue
                if i == size0  and xmax - x < size0 - ratio0:
                    continue
                if j == -size1 and y - ymin < size1 - ratio1:
                    continue
                if j == size1  and ymax - y < size1 - ratio1:
                    continue
                if k == -size2 and z - zmin < size2 - ratio2:
                    continue
                if k == size2  and zmax - z < size2 - ratio2:
                    continue

                pad_coords.append([i+x,j+y,k+z])
                pad_spec.append(species[at])
                pad_image.append(at)

    # transform fractional coords to abs coords
    if not pad_coords:
        abs_coords = []
    else:
        abs_coords = np.dot(pad_coords, tcell.T).ravel()

    return abs_coords, pad_spec, pad_image


