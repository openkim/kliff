def write_kim_ann(descriptor, weights, biases, activation='tanh',
    mode='float', fname='ann_kim.params'):
  """Output ANN structure, parameters etc. in the format of the KIM ANN model.

  Parameter
  ---------

  descriptor, object of Descriptor class

  """

  with open(fname,'w') as fout:

    # cutoff
    cutname, rcut = descriptor.get_cutoff()
    maxrcut = max(rcut.values())
    fout.write('# cutoff    rcut\n')
    if mode == 'double':
      fout.write('{}    {:.15g}\n\n'.format(cutname, maxrcut))
    else:
      fout.write('{}    {:.7g}\n\n'.format(cutname, maxrcut))

    # symmetry functions
    # header
    fout.write('#' + '='*80 + '\n')
    fout.write('# symmetry functions\n')
    fout.write('#' + '='*80 + '\n\n')

    desc = descriptor.get_hyperparams()
    # num of descriptors
    num_desc = len(desc)
    fout.write('{}    #number of symmetry funtion types\n\n'.format(num_desc))

    # descriptor values
    fout.write('# sym_function    rows    cols\n')
    for name, values in desc.iteritems():
      if name == 'g1':
        fout.write('g1\n\n')
      else:
        rows = len(values)
        cols = len(values[0])
        fout.write('{}    {}    {}\n'.format(name, rows, cols))
        if name == 'g2':
          for val in values:
            if mode == 'double':
              fout.write('{:.15g}  {:.15g}'.format(val['eta'], val['Rs']))
            else:
              fout.write('{:.7g}  {:.7g}'.format(val['eta'], val['Rs']))
            fout.write('    # eta  Rs\n')
          fout.write('\n')
        elif name =='g3':
          for val in values:
            if mode == 'double':
              fout.write('{:.15g}'.format(val['kappa']))
            else:
              fout.write('{:.7g}'.format(val['kappa']))
            fout.write('    # kappa\n')
          fout.write('\n')
        elif name =='g4':
          for val in values:
            zeta = val['zeta']
            lam = val['lambda']
            eta = val['eta']
            if mode == 'double':
              fout.write('{:.15g} {:.15g} {:.15g}'.format(zeta, lam, eta))
            else:
              fout.write('{:.7g} {:.7g} {:.7g}'.format(zeta, lam, eta))
            fout.write('    # zeta  lambda  eta\n')
          fout.write('\n')
        elif name =='g5':
          for val in values:
            zeta = val['zeta']
            lam = val['lambda']
            eta = val['eta']
            if mode == 'double':
              fout.write('{:.15g} {:.15g} {:.15g}'.format(zeta, lam, eta))
            else:
              fout.write('{:.7g} {:.7g} {:.7g}'.format(zeta, lam, eta))
            fout.write('    # zeta  lambda  eta\n')
          fout.write('\n')


    # ann structure and parameters
    # header
    fout.write('#' + '='*80 + '\n')
    fout.write('# ANN structure and parameters\n')
    fout.write('#\n')
    fout.write('# Note that the ANN assumes each row of the input "X" is '
        'an observation, i.e.\n')
    fout.write('# the layer is implemented as\n')
    fout.write('# Y = activation(XW + b).\n')
    fout.write('# You need to transpose your weight matrix if each column of "X" '
        'is an observation.\n')
    fout.write('#' + '='*80 + '\n\n')

    # number of layers
    num_layers = len(weights)
    fout.write('{}    # number of layers (excluding input layer, including'
        'output layer)\n'.format(num_layers))
    # size of layers
    for b in biases:
      fout.write('{}  '.format(b.size))
    fout.write('  # size of each layer (last must be 1)\n')
    # activation function
    fout.write('{}    # activation function\n\n'.format(activation))

    # weights and biases
    for i, (w, b) in enumerate(zip(weights, biases)):

      # weight
      rows,cols = w.shape
      if i != num_layers-1:
        fout.write('# weight of hidden layer {} (shape({}, {}))\n'.format(i+1,rows,cols))
      else:
        fout.write('# weight of output layer (shape({}, {}))\n'.format(rows,cols))
      for line in w:
        for item in line:
          if mode == 'double':
            fout.write('{:23.15e}'.format(item))
          else:
            fout.write('{:15.7e}'.format(item))
        fout.write('\n')

      # bias
      if i != num_layers-1:
        fout.write('# bias of hidden layer {} (shape({}, {}))\n'.format(i+1,rows,cols))
      else:
        fout.write('# bias of output layer (shape({}, {}))\n'.format(rows,cols))
      for item in b:
        if mode == 'double':
          fout.write('{:23.15e}'.format(item))
        else:
          fout.write('{:15.7e}'.format(item))
      fout.write('\n\n')










