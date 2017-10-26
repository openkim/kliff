from openkim_fit.descriptor import Descriptor
from collections import OrderedDict

def get_descriptor():

  cutfunc = 'cos'
  cutvalue = {'Mo-Mo': 6.4, 'Mo-S':6.4, 'S-S': 6.4}
  desc_params = OrderedDict()
  desc_params['g2'] = [
    {'eta':0.0009, 'Rs':0.},
    {'eta':0.01, 'Rs':0.},
    {'eta':0.02, 'Rs':0.},
    {'eta':0.035, 'Rs':0.},
    {'eta':0.06, 'Rs':0.},
    {'eta':0.1, 'Rs':0.},
    {'eta':0.2, 'Rs':0.},
    {'eta':0.4, 'Rs':0.}
  ]
  desc_params['g4'] = [
    {'zeta':0.0001, 'lambda':-1, 'eta':1},
    {'zeta':0.0001, 'lambda':1, 'eta':1},
    {'zeta':0.0001, 'lambda':-1, 'eta':2},
    {'zeta':0.0001, 'lambda':1, 'eta':2},
    {'zeta':0.003, 'lambda':-1, 'eta':1},
    {'zeta':0.003, 'lambda':1, 'eta':1},
    {'zeta':0.003, 'lambda':-1, 'eta':2},
    {'zeta':0.003, 'lambda':1, 'eta':2},
    {'zeta':0.008, 'lambda':1, 'eta':1},
    {'zeta':0.008, 'lambda':1, 'eta':2},
    {'zeta':0.015, 'lambda':1, 'eta':1},
    {'zeta':0.015, 'lambda':1, 'eta':2},
    {'zeta':0.015, 'lambda':1, 'eta':4},
    {'zeta':0.015, 'lambda':1, 'eta':16},
    {'zeta':0.025, 'lambda':1, 'eta':1},
    {'zeta':0.025, 'lambda':1, 'eta':2},
    {'zeta':0.025, 'lambda':1, 'eta':4},
    {'zeta':0.025, 'lambda':1, 'eta':16},
    {'zeta':0.045, 'lambda':1, 'eta':1},
    {'zeta':0.045, 'lambda':1, 'eta':2},
    {'zeta':0.045, 'lambda':1, 'eta':4},
    {'zeta':0.045, 'lambda':1, 'eta':16}
  ]


  # tranfer units from bohr to angstrom
  bhor2ang = 0.529177
  for key, values in desc_params.iteritems():
    for val in values:
      if key == 'g2':
        val['eta'] *= bhor2ang**2
      elif key == 'g4':
        val['eta'] *= bhor2ang**2

  num_g2 = len(desc_params['g2'])
  num_g4 = len(desc_params['g4'])
  print ('Number of descriptors:{}, with {} g2 and {} g4.'.format(num_g2+num_g4, num_g2, num_g4))

  # create all descriptors
  desc = Descriptor(cutfunc, cutvalue, desc_params)

  return desc

if __name__ == '__main__':
  get_descriptor()


