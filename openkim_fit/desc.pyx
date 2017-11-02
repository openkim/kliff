"""
Cython wrapper of Descriptor cpp class.
"""
import cython
import numpy as np
cimport numpy as np
from libcpp cimport bool

##############################
# cpp cython interface
##############################

cdef extern from "descriptor_c.h":
  cdef cppclass Descriptor:
    Descriptor() except +
    void set_fit_forces(bool fit_forces);
    void set_cutoff(char* name, int Nspecies, double* rcuts);
    void add_descriptor(char* name, double* values, int row, int col);
    void get_generalized_coords(double* coords, int* species_code,
        int* neighlist, int* numneigh, int* image,
        int Natoms, int Ncontrib, int Ndescriptor,
        double* gen_coords, double* d_gen_coords);


##############################
# python cython interfance
##############################


cdef class CythonDescriptor:

  cdef Descriptor c_desc      # hold a C++ instance which we're wrapping

  def __cinit__(self, bool fit_forces):
    self.c_desc = Descriptor()  # the C++ instance which we'are wrapping
    self.c_desc.set_fit_forces(fit_forces)

  def set_cutoff(self, char* name, int num_species,
      np.ndarray[double, ndim=2, mode="c"] rcuts not None):
    return self.c_desc.set_cutoff(name, num_species, &rcuts[0, 0])

  def add_descriptor(self, char* name,
      np.ndarray[double, ndim=2, mode="c"] value not None,
      int row, int col):
    self.c_desc.add_descriptor(name, &value[0,0], row, col)

  @cython.boundscheck(False)
  @cython.wraparound(False)
  def generate_generalized_coords(self,
      np.ndarray[double, ndim=1, mode="c"] coords not None,
      np.ndarray[int, ndim=1, mode="c"] species_code not None,
      np.ndarray[int, ndim=1, mode="c"] neighlist not None,
      np.ndarray[int, ndim=1, mode="c"] numneigh not None,
      np.ndarray[int, ndim=1, mode="c"] image not None,
      int Natoms, int Ncontrib, int Ndescriptor,
      np.ndarray[double, ndim=2, mode="c"] gen_coords=None,
      np.ndarray[double, ndim=3, mode="c"] d_gen_coords=None):

    if gen_coords is None:
      gen_coords = np.zeros((Ncontrib, Ndescriptor))
    if d_gen_coords is None:
      d_gen_coords = np.zeros((Ncontrib, Ndescriptor, 3*Ncontrib))

    # if not fit_forces, d_gen_coords will not be modified
    self.c_desc.get_generalized_coords( &coords[0], &species_code[0],
        &neighlist[0], &numneigh[0], &image[0],
        Natoms, Ncontrib, Ndescriptor,
        &gen_coords[0,0], &d_gen_coords[0,0,0])

    return gen_coords, d_gen_coords



