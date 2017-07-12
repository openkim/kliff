from __future__ import division
import numpy as np
import cos_doubles

def test_ext():
    a = np.array([np.pi/3,np.pi/2,np.pi], dtype=np.double)
    b = np.array([0,0,0], dtype=np.double)

    cos_doubles.cos_db(a,b)
    assert str('{0:.5f}'.format(b[0])) == '0.50000'
    assert str('{0:.5f}'.format(b[1])) == '0.00000'
    assert str('{0:.5f}'.format(b[2])) == '-1.00000'

    c = cos_doubles.cos_db(a)
    assert str('{0:.5f}'.format(c[0])) == '0.50000'
    assert str('{0:.5f}'.format(c[1])) == '0.00000'
    assert str('{0:.5f}'.format(c[2])) == '-1.00000'


if __name__ == '__main__':
    test_ext()
