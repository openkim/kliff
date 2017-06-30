# cdefine the signature of our c function
cdef extern from "c_cos_doubles.h":
  void cos_doubles (double * in_array, double * out_array, int size)
