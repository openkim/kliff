from __future__ import division
import sys
sys.path.append('../openkim_fit')
from descriptor import Descriptor
import numpy as np

def test_desc():
  """ Test the descriptors for ANN potntial.
  """

  # set init values, but this will not be used
  cutfunc = 'cos'
  cutvalue = {'C-C': 5}
  params = {'g1':None}
  desc = Descriptor(cutfunc, cutvalue, params)

  # g3
  g = desc._sym_g3(2.0, 3.0, 5.0)
  assert '{0:.5f}'.format(g) == '0.33173'
  g, dg = desc._sym_d_g3(2.0, 3.0, 5.0)
  assert '{0:.5f}'.format(g) == '0.33173'
  assert '{0:.5f}'.format(dg) == '-0.09381'

  # g4
  r = [1., 1.1, 1.2]
  rcut = [3., 3., 3.]
  g = desc._sym_g4(2.0, 3.0, .01, r, rcut)
  assert '{0:.5f}'.format(g) == '0.69950'
  g, dg = desc._sym_d_g4(2.0, 3.0, .01, r ,rcut)
  assert '{0:.5f}'.format(g) == '0.69950'
  assert '{0:.5f}'.format(dg[0]) == '0.70772'
  assert '{0:.5f}'.format(dg[1]) == '0.90480'
  assert '{0:.5f}'.format(dg[2]) == '-2.78241'



if __name__ == '__main__':
  test_desc()
