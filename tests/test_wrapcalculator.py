import sys
sys.path.append('../openkim_fit')
from wrapcalculator import WrapCalculator

def test_wrapcalculator():
    fname = 'input/data_4x4.edn'

    keys = ['unrelaxed-periodic-cell-vector-1',
            'unrelaxed-configuration-positions',
            'species']
    # key not in file
    #keys = [ 'haha']
    # list of list
    #keys = ['unrelaxed-configuration-positions']
    # list of one value
    #keys = ['unrelaxed-configuration-forces']
    # bare value
    #keys = ['unrelaxed-potential-energy']

    def dummy():
        pass
    test_parse = WrapCalculator(dummy, fname, keys)
    rslt= test_parse._parse_edn(fname, keys)
    print rslt

if __name__ == '__main__':
    test_wrapcalculator()
