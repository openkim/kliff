from __future__ import print_function, division
import sys
import os
import numpy as np
from collections import OrderedDict
import kimpy
from error import InputError, check_error, report_error
from utils import remove_comments


class ModelParameters():
  """Class of model parameters of the potential.

  It interacts with optimizer to provide initial guesses of parameters and
  receive updated paramters from the optimizer. Besides, predictors inquries
  updated parameters from this class.

  Parameters
  ----------

  modelname: str
    KIM model name
  """

  def __init__(self, modelname, debug=False):

    self.modelname = modelname
    self._debug = debug

    # key: parameter name
    # value: parameter values (1D array)
    self._avail_params = OrderedDict()

    # key: parameter name
    # values: {'value', 'use-kim', 'fix', 'lower_bound', 'upper_bound'}
    self._params = OrderedDict()

    # index of optimizing sub-parameter (recall that a parameter is a 1D array,
    # and a sub-parameter is a component in the array.)
    # a list of dictionary with keys:
    # name: parameter name
    # i_index: index of parameter in avail_params (which is the same as in KIM object)
    # j_index: index of sub-parameter in its value array
    self._index = []

    # inquire KIM for the available parameters
    self._get_avail_params()


  def read(self, fname):
    """Read the initial values of parameters.

    An alternative is set_param().
    For a given model parameter, one or multiple initial values may be required,
    and each must be given in a new line. For each line, the initial guess value
    is mandatory, where `KIM` (case insensitive) can be given to use the value
    from the KIM model. Optionally, `fix` can be followed not to optimize this
    parameters, or lower and upper bounds can be given to limit the parameter
    value in the range. Note that lower or upper bounds may not be effective if
    the optimizer does not support it. The following are valid input examples.

    Parameters
    ----------

    fname: str
      name of the input file where the optimizing parameters are listed

    Examples
    --------

    A
    KIM
    1.1

    B
    KIM  fix
    1.1  fix

    C
    KIM  0.1  2.1
    1.0  0.1  2.1
    2.0  fix
    """

    with open (fname, 'r') as fin:
      lines = fin.readlines()
      lines = remove_comments(lines)
    num_line = 0
    while num_line < len(lines):
      line = lines[num_line].strip()
      num_line += 1
      if line in self._params:
        raise InputError('line: {} file: {}. Parameter {} already '
            'set.'.format(num_line, fname, line))
      if line not in self._avail_params:
        raise InputError('line: {} file: {}. Parameter {} not supported by '
            'the potential model.'.format(num_line,fname,line))
      name = line
      size = len(self._avail_params[name])
      param_lines = [name]
      for j in range(size):
        param_lines.append(lines[num_line].split())
        num_line += 1
      self.set_param(param_lines)


  def set_param(self, lines):
    """Set the parameters that will be optimized.

    An alternative is Read().
    The name of the parameter should be given as the first entry of a list
    (or tuple), and then each data line should be given in in a list.

    Parameters
    ----------

    lines, str
      optimizing parameter initial values, settings

    Example
    -------

      param_A = ['PARAM_FREE_A',
                 ['kim', 0, 20],
                 [2.0, 'fix'],
                 [2.2, 1.1, 3.3]
                ]
      instance_of_this_class.set_param(param_A)
    """

    name = lines[0].strip()
#NOTE we want to use set_param to perturbe params so as to compute Fisher information
#matrix, where the following two lines are annoying
# Maybe issue an warning
#    if name in self._params:
#      raise InputError('Parameter {} already set.'.format(name))
    if name not in self._avail_params:
      raise InputError('Parameter "{}" not supported by the potential model.'.format(name))
    size = len(self._avail_params[name])
    if len(lines)-1 != size:
      raise InputError('Incorrect number of data lines for paramter "{}".'.format(name))

    # index of parameter in avail_params (which is the same as in KIM object)
    #for i,nm in enumerate(self._avail_params):
    #  if nm == name:
    #    index = i
    index = self._avail_params.keys().index(name)


    tmp_dict = {
       'size': size,
       'index': index,
       'value': np.array([None for i in range(size)]),
       'use-kim': np.array([False for i in range(size)]),
       'fix': np.array([False for i in range(size)]),
       'lower_bound': np.array([None for i in range(size)]),
       'upper_bound': np.array([None for i in range(size)])
    }
    self._params[name] = tmp_dict

    for j in range(size):
      line = lines[j+1]
      num_items = len(line)
      if num_items == 1:
        self._read_1_item(name, j, line)
      elif num_items == 2:
        self._read_2_item(name, j, line)
      elif num_items == 3:
        self._read_3_item(name, j, line)
      else:
        raise InputError('More than 3 iterms listed at data line '
                 '{} for parameter {}.'.format(j+1, name))
      self._check_bounds(name)

    self._set_index(name)



  def echo_avail_params(self):
    """Echo the optimizable parameters to stdout.
    """
    print()
    print('='*80)
    print('Model: ', self.modelname)
    print ()
    print('The following potential model parameters are available to fit. Include')
    print('the names and the initial guesses (optionally, lower and upper bounds)')
    print('of the parameters that you want to optimize in the input file.')
    print()
    for name,attr in self._avail_params.iteritems():
      print('name: ', name)
      print('data: ', attr)
      print()


  def echo_params(self, fname=None, print_size=False):
    """Print the optimizing parameters to stdout or file.

    Parameters
    ----------

    fname: str
      Name of the file to print the optimizing parameters. If None, printing
      to stdout.

    print_size: bool
      Flag to indicate whether print the size of parameter. Recall that a
      parameter may have one or more values.
    """

    if fname:
      fout = open(fname, 'w')
    else:
      fout = sys.stdout

    #print(file=fout)
    print('#'+'='*80, file=fout)
    print('# Potential model parameters that are optimzied',file=fout)
    print('#'+'='*80, file=fout)
    print(file=fout)

    for name,attr in self._params.iteritems():
      if print_size:
        print(name, attr['size'], file=fout)
      else:
        print(name, file=fout)

      #print ('index:', attr['index'], file=fout)

      for i in range(attr['size']):
        print('{:24.16e}'.format(attr['value'][i]), end=' ', file=fout)
        if not attr['fix'][i] and attr['lower_bound'][i] == None:
          print(file=fout)   # print new line if only given value
        if attr['fix'][i]:
          print('fix', file=fout)
        if attr['lower_bound'][i] != None:
          print('{:24.16e}'.format(attr['lower_bound'][i]), end=' ', file=fout)
        if attr['upper_bound'][i]:
          print('{:24.16e}'.format(attr['upper_bound'][i]), file=fout)

      print(file=fout)

    if fname:
      fout.close()


  def update_params(self, opt_x):
    """ Update parameter values from optimzier.

    This is the opposite operation of get_x0().

    Parameters
    ----------

    opt_x, list of floats
      parameter values from the optimizer.

    """
    for k,val in enumerate(opt_x):
      name = self._index[k]['name']
      j_index = self._index[k]['j_index']
      self._params[name]['value'][j_index] = val


  def get_x0(self):
    """Nest all parameter values (except the fix ones) to a list.

    This is the opposite operation of update_params(). This can be fed to the
    optimizer as the starting parameters.

    Return
    ------
      A list of nested optimizing parameter values.
    """
    opt_x0 = []
    for idx in self._index:
      name = idx['name']
      j_index = idx['j_index']
      opt_x0.append(self._params[name]['value'][j_index])
    return np.asarray(opt_x0)


  def get_bounds(self):
    """ Get the lower and upper parameter bounds. """
    bounds = []
    for idx in self._index:
      name = idx['name']
      j_index = idx['j_index']
      lower = self._params[name]['lower_bound'][j_index]
      upper = self._params[name]['upper_bound'][j_index]
      bounds.append([lower, upper])
    return bounds


  def get_names(self):
    return self._params.keys()

  def get_size(self, name):
    return self._params[name]['size']

  def get_value(self, name):
    return self._params[name]['value'].copy()

  def get_index(self, name):
    return self._params[name]['index']

  def get_lower_bound(self, name):
    return self._params[name]['lower_bound'].copy()

  def get_upper_bound(self, name):
    return self._params[name]['upper_bound'].copy()

  def get_fix(self, name):
    return self._params[name]['fix'].copy()

  def set_value(self, name, value):
    self._params[name]['value'] = value


  def get_number_of_opt_params(self):
    return len(self._index)

  def get_opt_param_value_and_i_j_indices(self, k):
    name = self._index[k]['name']
    i_index = self._index[k]['i_index']
    j_index = self._index[k]['j_index']
    value = self._params[name]['value'][j_index]
    return value, i_index, j_index


  def _get_avail_params(self):
    """Inqure KIM model to get all the optimizable parameters."""

    # create model
    units_accepted, kim_model, error = kimpy.model.create(
      kimpy.numbering.zeroBased,
      kimpy.length_unit.A,
      kimpy.energy_unit.eV,
      kimpy.charge_unit.e,
      kimpy.temperature_unit.K,
      kimpy.time_unit.ps,
      self.modelname
    )
    check_error(error, 'kimpy.model.create')
    if not units_accepted:
      report_error('requested units not accepted in kimpy.model.create')
    self.kim_model = kim_model

    # parameter
    num_params = kim_model.get_number_of_parameters()

    for i in range(num_params):
      out = kim_model.get_parameter_data_type_extent_and_description(i)
      dtype, extent, name, error = out
      check_error(error, 'kim_model.get_parameter_data_type_extent_and_description')

      if self._debug:
        print('Parameter No. {} has data type "{}" with extent {} and description: '
        '"{}".'.format(i, dtype, extent, name))

      self._avail_params[name] = []
      for j in range(extent):
        if str(dtype) == 'Double':
          value, error = kim_model.get_parameter_double(i,j)
          check_error(error, 'kim_model.get_parameter_double')
        elif str(dtype) == 'Int':
          value, error = kim_model.get_parameter_int(i,j)
          check_error(error, 'kim_model.get_parameter_int')
        else:  # should never reach here
          report_error('get unexpeced parameter data type "{}"'.format(dtype))
        self._avail_params[name].append(value)

    # destroy the model
    kimpy.model.destroy(kim_model)


  def _read_1_item(self, name, j, line):
    self._read_1st_item(name, j, line[0])

  def _read_2_item(self, name, j, line):
    self._read_1st_item(name, j, line[0])
    if line[1].lower() == 'fix':
      self._params[name]['fix'][j] = True
    else:
      raise InputError('Data at line {} of {} corrupted.\n'.format(j+1, name))

  def _read_3_item(self, name, j, line):
    self._read_1st_item(name, j, line[0])
    try:
      self._params[name]['lower_bound'][j] = float(line[1])
      self._params[name]['upper_bound'][j] = float(line[2])
    except ValueError as err:
      raise InputError('{}.\nData at line {} of {} corrupted.\n'.format(err, j+1, name))

  def _read_1st_item(self, name, j, first):
    if type(first)==str and first.lower() == 'kim':
      self._params[name]['use-kim'][j] = True
      model_value = self._avail_params[name]
      self._params[name]['value'][j] = model_value[j]
    else:
      try:
        self._params[name]['value'][j] = float(first)
      except ValueError as err:
        raise InputError('{}.\nData at line {} of {} corrupted.\n'.format(err, j+1, name))


  def _check_bounds(self, name):
    """Check whether the initial guess of a paramter is within its lower and
    upper bounds.
    """
    attr = self._params[name]
    for i in range(attr['size']):
      lower_bound = attr['lower_bound'][i]
      upper_bound = attr['upper_bound'][i]
      if lower_bound != None:
        value = attr['value'][i]
        if value < lower_bound or value > upper_bound:
          raise InputError('Initial guess at line {} of parameter {} is '
                   'out of bounds.\n'.format(i+1, name))


  def _set_index(self, name):
    """Check whether a specific data value of a parameter will be optimized or
    not (by checking its 'fix' attribute). If yes, include it in the index
    list.

    Given a parameter and its values such as:

    PARAM_FREE_B
    1.1
    2.2  fix
    4.4  3.3  5.5

    the first slot (1.1) and the third slot (4.4) will be included in self._index,
    and later be optimized.
    """

    #for i,nm in enumerate(self._avail_params):
    #  if nm == name:
    #    i_index = i
    i_index = self._avail_params.keys().index(name)
    size = self._params[name]['size']
    fix  = self._params[name]['fix']
    for j in range(size):
      if not fix[j]:
        idx = {'name':name, 'i_index':i_index, 'j_index':j}
        self._index.append(idx)


  def __del__(self):
    """Garbage collection"""
    pass



class WrapperModelParams():
  """Wrapper of ModelParameters to deal with multiple models used in cost.

  Parameters
  ----------

  modelparams
    list of ModelParameters objects
  """

  def __init__(self, modelparams):
    self.modelparams = modelparams
    self._index = []
    self._set_index()

  def _set_index(self):
    """Compute the start and end indices of x0 from each ModelParameters object."""
    i = 0
    for obj in self.modelparams:
      x0 = obj.get_x0()
      num = len(x0)
      idx = {'start':i, 'end':i+num}
      self._index.append(idx)
      i += num

  def get_x0(self):
    """Nest optimizing parameter values from all ModelParameters objects.

    Return
    ------
      A 1D array of nested optimizing parameter values.
    """
    opt_x0 = []
    for obj in self.modelparams:
      x0 = obj.get_x0()
      opt_x0 = np.append(opt_x0, x0)
    return opt_x0

  def update_params(self, opt_x):
    """Wrapper to call 'update_params()' of each ModelParameters object.
    """
    for i,obj in enumerate(self.modelparams):
      start = self._index[i]['start']
      end = self._index[i]['end']
      x0 = opt_x[start:end]
      obj.update_params(x0)

