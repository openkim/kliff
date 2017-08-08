from __future__ import division
from __future__ import print_function
import numpy as np
import sys
from collections import OrderedDict

from neighbor import NeighborList
from error import UnsupportedError

DIM = 3


class Descriptor:
  """Symmetry functions that transform the cartesian coords to generalized coords.

  Parameters
  ----------

  cutname, string
    cutoff function name

  cutvalue, dict
    cutoff values based on species.

    Example
    -------
    cutvalue = {'C-C': 3.5, 'C-H': 3.0, 'H-H': 1.0}

  hyperparams, dict
    hyperparameters of descriptors

    Example
    -------
      {'g1': None,
       'g2': [{'eta':0.1, 'Rs':0.2}, {'eta':0.3, 'Rs':0.4}],
       'g3': [{'kappa':0.1}, {'kappa':0.2}, {'kappa':0.3}]
      }
  """

  def __init__(self,cutname, cutvalue, hyperparams):

    self._desc = OrderedDict()
    self._cutname = None
    self._cutoff = None   # cutoff funciton
    self._rcut = None   # dictionary of cutoff values
    self._has_three_body = False



    # set cutoff function and values
    self._cutname = cutname.lower()
    if self._cutname == 'cos':
      self._cutoff = self._cut_cos
    elif self._cutname == 'exp':
      self._cutoff = self._exp_cos
    else:
      raise UnsupportedError("Cutoff `{}' unsupported.".format(name))
    self._rcut = generate_full_cutoff(cutvalue)

    # store hyperparams of descriptors
    for name, value in hyperparams.iteritems():
      if name.lower() not in['g1', 'g2', 'g3', 'g4', 'g5']:
        raise UnsupportedError("Symmetry function `{}' unsupported.".format(name))
      # g1 needs no hyperparams, let a placeholder taking its place
      if name.lower() == 'g1':
        value = ['placeholder']
      self._desc[name.lower()] = value
      if name.lower() == 'g4' or name.lower() == 'g5':
        self._has_three_body = True


  def get_num_descriptors(self):
    """The total number of symmetry functions (each hyper-parameter set counts 1)"""
    N = 0
    for key in self._desc:
      N += len(self._desc[key])
    return N


  def get_cutoff(self):
    """ Return the name and values of cutoff. """
    return self._cutname, self._rcut


  def get_hyperparams(self):
    """ Return the hyperparameters of descriptors. """
    return self._desc



  def generate_generalized_coords(self, conf):
    """Transform atomic coords to generalized coords.

    Parameter
    ---------

    conf: Configuration object in which the atoms information are stored

    Returns
    -------
    gen_coords, 2D float array
      generalized coordinates of size [Ncontrib, Ndescriptors]

    dgen_coords, 3D float array
      derivative of generalized coordinates w.r.t atomic coords of size
      [Ncontrib, Ndescriptors, 3*Ncontrib]

    """

    # create neighbor list
    nei = NeighborList(conf, self._rcut)
    coords = np.reshape(nei.coords, (-1,3))
    species = nei.spec
    Ncontrib = nei.ncontrib
    neighlist = nei.neighlist
    image = nei.image

    # loop to set up generalized coords
    Ndesc = self.get_num_descriptors()
    gen_coords = np.zeros([Ncontrib, Ndesc])
    dgen_coords = np.zeros([Ncontrib, Ndesc, 3*Ncontrib])

    # loop over contributing atoms
    for i in xrange(Ncontrib):
      ri = coords[i]
      ispec = species[i]

      # loop over neighbors of atom i
      for j in neighlist[i]:
        rj = coords[j]
        jspec = species[j]
        rij = rj - ri
        rijmag = np.linalg.norm(rij)
        rcutij = self._rcut[ispec+'-'+jspec]

        if rijmag > rcutij: continue  # i and j not interacting

        idx = 0

        # loop over two-body descriptors
        for key, values in self._desc.iteritems():

          #loop over descriptor parameter set
          flag_need_add = False
          for params in values:

            if key == 'g1': # g1 has no hyperparams
              g, dg = self._sym_d_g1(rijmag, rcutij)
              flag_need_add = True
            elif key == 'g2':
              eta = params['eta']
              Rs = params['Rs']
              g, dg = self._sym_d_g2(eta, Rs, rijmag, rcutij)
              flag_need_add = True
            elif key == 'g3':
              kappa = params['kappa']
              g, dg = self._sym_d_g3(kappa, rijmag, rcutij)
              flag_need_add = True

            # add generalized coords and forces (derivatives) (BC considered)
            if flag_need_add:
              gen_coords[i, idx] += g
              pair = dg*rij/rijmag
              for dim in xrange(DIM):
                dgen_coords[i, idx, i*DIM+dim] += pair[dim]
                dgen_coords[i, idx, image[j]*DIM+dim] -= pair[dim]

              idx += 1
              flag_need_add = False


        # three-body descriptors
        if not self._has_three_body: continue

        # loop over three-body atom k
        for k in neighlist:
          if k <= j: continue   #three-body angle only need to be considered once
          rk = coords[k]
          kspec = species[k]
          rik = rk - ri
          rjk = rk - rj
          rikmag = no.linalg(rik)
          rjkmag = no.linalg(rjk)
          rcutik = self._rcut[ispec+'-'+kspec]
          rcutjk = self._rcut[jspec+'-'+kspec]

          if rikmag > rcutik: continue  # three-body not interacting

          #loop over three-body descriptor
          for key, values in self._desc.iteritems():

            #loop over descriptor parameter set
            flag_need_add = False
            for params in values:
              if key == 'g4':
                zeta = params['zeta']
                lam = params['lambda']
                eta = params['eta']
                g, dg = self._sym_d_g2(zeta, lam, eta, rijmag, rcutij)
                flag_need_add = True
              elif key == 'g5':
                zeta = params['zeta']
                lam = params['lambda']
                eta = params['eta']
                g, dg = self._sym_d_g2(zeta, lam, eta, rijmag, rcutij)
                flag_need_add = True

              # add generalized coords and forces (derivatives) (BC considered)
              if flag_need_add:
                gen_coords[i, idx] += g
                pair_ij = dg*rij/rijmag
                pair_ik = dg*rik/rikmag
                pair_jk = dg*rjk/rjkmag
                for dim in xrange(DIM):
                  dgen_coords[i, idx, i*DIM+dim] += pair_ij[dim] +pair_ik[dim]
                  dgen_coords[i, idx, image[j]*DIM+dim] += -pair_ij[dim] +pair_jk[dim]
                  dgen_coords[i, idx, image[k]*DIM+dim] += -pair_ik[dim] -pair_jk[dim]

                flag_need_add = False
                idx += 1

    # return
    return gen_coords, dgen_coords



  def _sym_g1(self, r, rcut):
    fc, dfc = self._cutoff(r, rcut)
    return fc

  def _sym_d_g1(self, r, rcut):
    return self._cutoff(r, rcut)

  def _sym_g2(self, eta, Rs, r, rcut):
    fc, dfc = self._cutoff(r, rcut)
    eterm = np.exp(-eta*(r-Rs)**2)
    g = eterm*fc
    return g

  def _sym_d_g2(self, eta, Rs, r, rcut):
    fc, dfc = self._cutoff(r, rcut)
    eterm = np.exp(-eta*(r-Rs)**2)
    g = eterm*fc
    dg = -2*eta*(r-Rs)*eterm*fc + eterm*dfc
    return g, dg

  def _sym_g3(self, kappa, r, rcut):
    fc, dfc = self._cutoff(r, rcut)
    g = np.cos(kappa*r)*fc
    return g

  def _sym_d_g3(self, kappa, r, rcut):
    fc, dfc = self._cutoff(r, rcut)
    kdotr = kappa*r
    g = np.cos(kdotr)*fc
    dg = - kappa*np.sin(kdotr)*fc + np.cos(kdotr)*dfc
    return g, dg

  def _sym_g4(self, zeta, lam, eta, r, rcut):
    rij = r[0]
    rik = r[1]
    rjk = r[2]
    rcutij = rcut[0]
    rcutik = rcut[1]
    rcutjk = rcut[2]
    rijsq = rij*rij
    riksq = rik*rik
    rjksq = rjk*rjk
    fcij, dummy = self._cutoff(rij, rcutij)
    fcik, dummy = self._cutoff(rik, rcutik)
    fcjk, dummy = self._cutoff(rjk, rcutjk)

    cos_ijk = (rijsq + riksq - rjksq)/(2*rij*rik)
    costerm = (1+lam*cos_ijk)**zeta
    eterm = np.exp(-eta*(rijsq + riksq + rjksq))

    g = 2**(1-zeta) * costerm * eterm * fcij * fcik * fcjk
    return g

  def _sym_d_g4(self, zeta, lam, eta, r, rcut):
    rij = r[0]
    rik = r[1]
    rjk = r[2]
    rcutij = rcut[0]
    rcutik = rcut[1]
    rcutjk = rcut[2]
    rijsq = rij*rij
    riksq = rik*rik
    rjksq = rjk*rjk

    # cosine term, i is the apex atom
    cos_ijk = (rijsq + riksq - rjksq)/(2*rij*rik)
    costerm = (1+lam*cos_ijk)**zeta
    dcos_dij = (rijsq - riksq + rjksq)/(2*rijsq*rik)
    dcos_dik = (riksq - rijsq + rjksq)/(2*rij*riksq)
    dcos_djk = -rjk/(rij*rik)
    dcosterm_dcos = zeta * (1+lam*cos_ijk)**(zeta-1) * lam
    dcosterm_dij = dcosterm_dcos * dcos_dij
    dcosterm_dik = dcosterm_dcos * dcos_dik
    dcosterm_djk = dcosterm_dcos * dcos_djk

    # exponential term
    eterm = np.exp(-eta*(rijsq + riksq + rjksq))
    determ_dij = -2*eterm*eta*rij
    determ_dik = -2*eterm*eta*rik
    determ_djk = -2*eterm*eta*rjk

    # power 2 term
    p2 = 2**(1-zeta)

    # cutoff
    fcij, dfcij = self._cutoff(rij, rcutij)
    fcik, dfcik = self._cutoff(rik, rcutik)
    fcjk, dfcjk = self._cutoff(rjk, rcutjk)
    fcprod = fcij*fcik*fcjk
    dfcprod_dij = dfcij*fcik*fcjk
    dfcprod_dik = dfcik*fcij*fcjk
    dfcprod_djk = dfcjk*fcij*fcik

    #g
    g =  p2 * costerm * eterm * fcprod
    # dg_dij
    dg = [0., 0., 0.]
    dg[0] = p2 * (dcosterm_dij*eterm*fcprod + costerm*determ_dij*fcprod
        + costerm*eterm*dfcprod_dij)
    # dg_dik
    dg[1] = p2 * (dcosterm_dik*eterm*fcprod + costerm*determ_dik*fcprod
        + costerm*eterm*dfcprod_dik)
    # dg_djk
    dg[2] = p2 * (dcosterm_djk*eterm*fcprod + costerm*determ_djk*fcprod
        + costerm*eterm*dfcprod_djk)
    return g, dg

  def _sym_g5(self, zeta, lam, eta, r, rcut):
    rij = r[0]
    rik = r[1]
    rjk = r[2]
    rcutij = rcut[0]
    rcutik = rcut[1]
    rijsq = rij*rij
    riksq = rik*rik
    rjksq = rjk*rjk
    fcij, dummy = self._cutoff(rij, rcutij)
    fcik, dummy = self._cutoff(rik, rcutik)

    # i is the apex atom
    cos_ijk = (rijsq + riksq - rjksq)/(2*rij*rik)
    costerm = (1+lam*cos_ijk)**zeta
    eterm = np.exp(-eta*(rijsq + riksq))

    g = 2**(1-zeta) * costerm * eterm * fcij * fcik
    return g

  def _sym_d_g5(self, zeta, lam, eta, r, rcut):
    rij = r[0]
    rik = r[1]
    rjk = r[2]
    rcutij = rcut[0]
    rcutik = rcut[1]
    rijsq = rij*rij
    riksq = rik*rik
    rjksq = rjk*rjk

    # cosine term, i is the apex atom
    cos_ijk = (rijsq + riksq - rjksq)/(2*rij*rik)
    costerm = (1+lam*cos_ijk)**zeta
    dcos_dij = (rijsq - riksq + rjksq)/(2*rijsq*rik)
    dcos_dik = (riksq - rijsq + rjksq)/(2*rij*riksq)
    dcos_djk = -rjk/(rij*rik)
    dcosterm_dcos = zeta * (1+lam*cos_ijk)**(zeta-1) * lam
    dcosterm_dij = dcosterm_dcos * dcos_dij
    dcosterm_dik = dcosterm_dcos * dcos_dik
    dcosterm_djk = dcosterm_dcos * dcos_djk

    # exponential term
    eterm = np.exp(-eta*(rijsq + riksq + rjksq))
    determ_dij = -2*eterm*eta*rij
    determ_dik = -2*eterm*eta*rik

    # power 2 term
    p2 = 2**(1-zeta)

    # cutoff
    fcij, dfcij = self._cutoff(rij, rcutij)
    fcik, dfcik = self._cutoff(rik, rcutik)
    fcjk, dfcjk = self._cutoff(rjk, rcutjk)
    fcprod = fcij*fcik*fcjk
    dfcprod_dij = dfcij*fcik*fcjk
    dfcprod_dik = dfcik*fcij*fcjk
    dfcprod_djk = dfcjk*fcij*fcik

    #g
    g =  p2 * costerm * eterm * fcprod
    # dg_dij
    dg = [0., 0., 0.]
    dg[0] = p2 * (dcosterm_dij*eterm*fcprod + costerm*determ_dij*fcprod
        + costerm*eterm*dfcprod_dij)
    # dg_dik
    dg[1] = p2 * (dcosterm_dik*eterm*fcprod + costerm*determ_dik*fcprod
        + costerm*eterm*dfcprod_dik)
    # dg_djk
    dg[2] = p2 * dcosterm_djk*eterm*fcprod
    return g, dg



  def _cut_cos(self, r, rcut):
    if r < rcut:
      fc = 0.5 * (np.cos(np.pi*r/rcut) + 1)
      dfc = -0.5*np.pi/rcut*np.sin(np.pi*r/rcut)
      return fc, dfc
    else:
      return 0., 0.

#TODO correct it
  def _cut_exp(self, r, rcut):
    if r < rcut:
      fc = 0.5 * (np.cos(np.pi*r/rcut) + 1)
      dfc = -0.5*np.pi/rcut*np.sin(np.pi*r/rcut)
      return fc, dfc
    else:
      return 0., 0.


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




