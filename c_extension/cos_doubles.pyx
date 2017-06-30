""" Example of wrapping a C function that takes C double arrays as input using
    the Numpy declarations from Cython """


cimport cos_double_c

# cimport the Cython declarations for numpy
import numpy as np   # this is necessay if we use numpy
cimport numpy as np

# if you want to use the Numpy-C-API from Cython
# (not strictly necessary for this example, but good practice)
np.import_array()

# create the wrapper code, with numpy type annotations
def cos_db(np.ndarray[double, ndim=1, mode="c"] in_array not None,
           np.ndarray[double, ndim=1, mode="c"] out_array=None):

  if out_array is None:
    out_array = np.zeros(in_array.shape[0])


#  cos_double_c.cos_doubles(<double*> np.PyArray_DATA(in_array),
#                           <double*> np.PyArray_DATA(out_array),
#                           in_array.shape[0])

  # the above  three lines is also fine
  cos_double_c.cos_doubles(&in_array[0], &out_array[0], in_array.shape[0])

  return out_array
