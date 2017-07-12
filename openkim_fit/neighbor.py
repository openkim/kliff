from __future__ import division
import numpy as np

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

    pad_image: list of int
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


