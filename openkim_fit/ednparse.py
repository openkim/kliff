import edn_format
import numpy as np

def parse_edn(fname, keys):
    '''
    Wrapper to use end_format to parse OpenKIM output file in edn format.

    Parameters
    ----------

    fname: str
        Name of the output file of OpenKIM test.

    keys: list of str
        Keywords, the values of which will be returned.
    '''
    with open(fname, 'r') as fin:
        lines = fin.read()
    parsed = edn_format.loads(lines)
    source_val_dict = dict()
    for k in keys:
        value = np.array(parsed[k]['source-value'])
        if len(value.shape) > 1:
            value = np.concatenate(value)
        source_val_dict[k] = value
    return source_val_dict


if __name__ == '__main__':

    fname = '../tests/data_4x4.edn'
    keys = ['unrelaxed-periodic-cell-vector-1',
            'unrelaxed-configuration-positions',
            'species']
    rslt=parse_edn(fname, keys)
    print rslt[keys[2]]


