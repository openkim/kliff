from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import scipy.optimize
import multiprocessing as mp
import collections
from . import parallel


def energy_forces_residual(identifier, natoms, prediction, reference, data):
  """
  Parameters
  ----------
  identifier: str
    identifer of the configuration, i.e. path to the file

  natoms: int
    number of atoms in the configuration

  prediction: 1D array
    prediction computed by calculator

  reference: 1D array
    references data for the prediction

  data: dict
    user provided callback data
    aviailable keys:

    energy_weight: float (default: 1)

    forces_weight: float (default: 1)

    normalize_by_number_of_atoms: bool (default: False)
      Whether to normalize the residual by the number of atoms


  Return
  ------

  residual: 1D array


  The length of `prediction` and `reference` (call it `S`) are the same, and it
  depends on `use_energy` and `use_forces` in KIMCalculator. Assume the
  configuration contains of `N` atoms.

  1) If use_energy == True and use_forces == False
    S = 1
  `prediction[0]` is the potential energy computed by the calculator, and
  `reference[0]` is the reference energy.

  2) If use_energy == False and use_forces == True
    S = 3N
  `prediction[3*i+0]`, `prediction[3*i+1]`, and `prediction[3*i+2]` are the
  x, y, and z component of the forces on atom i in the configuration, respecrively.
  Correspondingly, `reference` is the 3N concatenated reference forces.


  3) If use_energy == True and use_forces == True
    S = 3N + 1

  `prediction[0]` is the potential energy computed by the calculator, and
  `reference[0]` is the reference energy.
  `prediction[3*i+1]`, `prediction[3*i+2]`, and `prediction[3*i+3]` are the
  x, y, and z component of the forces on atom i in the configuration, respecrively.
  Correspondingly, `reference` is the 3N concatenated reference forces.
  """

  if len(prediction) != 3*natoms + 1:
    raise ValueError("len(prediction) != 3N+1, where N is the number of atoms.")

  try:
    energy_weight = data['energy_weight']
  except KeyError:
    energy_weight = 1
  try:
    forces_weight = data['forces_weight']
  except KeyError:
    forces_weight = 1
  try:
    do_normalize = data['normalize_by_number_of_atoms']
    if do_normalize:
      normalizer = natoms
    else:
      normalizer = 1
  except KeyError:
    normalizer = 1


  residual = np.subtract(prediction, reference)
  residual[0] *= np.sqrt(energy_weight)
  residual[1:] *= np.sqrt(forces_weight)
  return residual


def forces_residual(conf_id, natoms, prediction, reference, data):
  if data is not None:
    data['energy_weight'] = 0
  else:
    data = {'energy_weight': 0}
  return energy_forces_residual(conf_id, natoms, prediction, reference, data)


def energy_residual(conf_id, natoms, prediction, reference, data):
  if data is not None:
    data['forces_weight'] = 0
  else:
    data = {'forces_weight': 0}
  return energy_forces_residual(conf_id, natoms, prediction, reference, data)



class Loss(object):
  """Objective function that will be minimized.

  Parameters
  ----------

  model_params: ModelParameters object

  calculator: Calculator object

  nprocs: int
    Number of processors to parallel to run the predictors.

  residual_fn: function
    function to compute residual, see `energy_forces_residual` for an example

  residual_data: dict
    data passed to residual function

  """

  scipy_minimize_methods = [
    'Nelder-Mead',
    'Powell',
    'CG',
    'BFGS',
    'Newton-CG',
    'L-BFGS-B',
    'TNC',
    'COBYLA',
    'SLSQP',
    'trust-constr',
    'dogleg',
    'trust-ncg',
    'trust-exact',
    'trust-krylov',
  ]

  scipy_least_squares_methods = [
    'trf',
    'dogbox',
    'lm',
  ]

  tf_minimize_methods = [
    'GradientDescentOptimizer',
    'AdadeltaOptimizer',
    'AdagradOptimizer',
    'AdagradDAOptimizer',
    'MomentumOptimizer',
    'AdamOptimizer',
    'FtrlOptimizer',
    'ProximalGradientDescentOptimizer',
    'ProximalAdagradOptimizer',
    'RMSPropOptimizer'
  ]

  tf_scipy_minimize_methods = scipy_minimize_methods


  def __init__(self, model_params, calculator, nprocs=1,
      residual_fn=energy_forces_residual, residual_data=None):

    self.model_params = model_params
    self.calculator = calculator
    self.nprocs = nprocs

    self.residual_fn = residual_fn
    self.residual_data = residual_data
    if self.residual_data is None:
      self.residual_data = dict()

    self.calculator_type = calculator.__class__.__name__


    if self.calculator_type == 'KIMCalculator':

      # TODO make the following a function call (maybe `refresh`) of KIMCalculator
      self.kim_input_and_output = calculator.get_kim_input_and_output()

      # update parameter from ModelParameters to calculator, and compute
      # neighbor list. This is needed since cutoff can be set in ModelParameters.
      self.calculator.update_params(self.model_params)
      cutoff = self.calculator.get_cutoff()
      kim_in_out_data = self.calculator.get_kim_input_and_output()
      for in_out in kim_in_out_data:
        in_out.update_neigh(cutoff*1.001)
        # use same compute_energy and compute_forces as when the compute arguments
        # is created
        compute_energy = in_out.get_compute_energy()
        compute_forces = in_out.get_compute_forces()
        in_out.register_data(compute_energy, compute_forces)

    elif self.calculator_type == 'ANNCalculator':
      pass
    else:
      raise Exception('Not supported calculator')


#
#  def set_nprocs(self, nprocs):
#    """ Set the number of processors to be used."""
#    self.nprocs = nprocs
#
#  def set_residual_fn_and_data(self, fn, data):
#    """ Set residual function and data. """
#    self.residual_fn = fn
#    self.residual_fn_data = data


  def minimize(self, method, **kwargs):
    """ Minimize the loss.

    method: str
      minimization methods as specified at:
        https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.optimize.minimize.html
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
        https://www.tensorflow.org/api_guides/python/train#Optimizers
        https://www.tensorflow.org/api_docs/python/tf/contrib/opt/ScipyOptimizerInterface

      Note the usage of tf scipy optimizer interface if the same as on scipy website,
      not the one described on tf website, specifically, this is for the `options`
      argument.

    kwargs: extra keyword arguments that can be used by the scipy optimizer
    """

    if self.calculator_type == 'KIMCalculator':

      if method in self.scipy_least_squares_methods:
        # change unbounded value to np.inf that least_squares needs
        try:
          bounds = kwargs['bounds']
          for i in range(len(bounds)):
            bounds[i][0] = -np.inf
            bounds[i][1] = np.inf
        except KeyError:
          pass
        result = self._scipy_optimize_least_squares(method, **kwargs)
      elif method in self.scipy_minimize_methods:
        result = self._scipy_optimize_minimize(method, **kwargs)
      else:
        raise Exception('minimization method "{}" not supported.'.format(method))

      # update final optimized paramters to ModelParameters object
      self._update_params(result.x)
      return result

    elif self.calculator_type == 'ANNCalculator':

      try:
        import tensorflow as tf
      except ImportError as e:
        raise ImportError(str(e) + '.\nPlease install "tensorflow" first.')

      loss = self.calculator.get_loss()


      with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        if method in self.tf_minimize_methods:
          optimizer = tf.train.AdamOptimizer(**kwargs)
          train = optimizer.minimize(loss)
          while True:
            try:
              sess.run(train)
            except tf.errors.OutOfRangeError:
              break

        elif method in self.tf_scipy_minimize_methods:
          # enable giving options as a dictionary as in scipy website
          scipy_interface_options = kwargs.copy()
          options = scipy_interface_options.pop('options', None)
          if options is not None:
            for key,val in options.items():
              kwargs[key] = val
          optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, method=method,
              options = scipy_interface_options)
          optimizer.minimize(sess)

        else:
          raise Exception('minimization method "{}" not supported.'.format(method))


        self.calculator.model.write_kim_ann(sess, fname='ann_kim.params')


    else:
      raise Exception('Not recognized calculator')



  def get_residual(self, x0):
    """ Compute the residual for the cost.
    This is a callable for optimizer (e.g. scipy.optimize.least_squares), which is
    passed as the first positional argument.

    Parameters
    ----------

    x0: 1D array
      optimizing parameter values
    """

    # publish params x0 to predictor
    self._update_params(x0)

    # parallel computing of residual
    kim_in_out_data = self.calculator.get_kim_input_and_output()

    # compute residual
    if self.nprocs > 1:
      residuals = parallel.parmap2(self._get_residual_single_config, kim_in_out_data,
          self.nprocs, self.calculator, self.residual_fn, self.residual_data)
      residual = np.concatenate(residuals)
    else:
      residual = []
      for in_out in kim_in_out_data:
        current_residual = self._get_residual_single_config(in_out, self.calculator,
            self.residual_fn, self.residual_data)
        residual = np.concatenate((residual, current_residual))

    return residual


  def get_loss(self, x0):
    """ Compute the loss.
    This is a callable for optimizer (e.g. scipy.optimize.minimize), which is
    passed as the first positional argument.

    Parameters
    ----------

    x0: 1D array
      optimizing parameter values
    """
    residual = self.get_residual(x0)
    loss = 0.5*np.linalg.norm(residual)**2
    return loss


  def error_report(self, normalize_by_num_atoms=True, fname='ERROR_REPORT'):
    """Write error of each configuration to fname.

    Parameters
    ----------
    fname: str
      path to the file to write the error report.
    """

    loss = self.get_loss(self.model_params.get_x0())

    with open (fname, 'w') as fout:
      fout.write('\n'+'='*80+'\n')
      fout.write('Final loss: {:18.10e}\n\n'.format(loss))
      fout.write('='*80+'\n')
      if normalize_by_num_atoms:
        fout.write('(Loss, energy RMSE, and forces RMSE are normalized by '
            'number of atoms: Natoms.)\n\n')
      fout.write('      Loss       energy RMSE     forces RMSE  Natoms  '
          'config. identifier\n\n')

      kim_in_out_data = self.calculator.get_kim_input_and_output()
      for in_out in kim_in_out_data:

        # prediction data
        self.calculator.compute(in_out)
        pred = self.calculator.get_prediction(in_out)

        # reference data
        ref = self.calculator.get_reference(in_out)

        conf = in_out.conf
        identifier = conf.get_identifier()
        natoms = conf.get_number_of_atoms()

        compute_energy = in_out.get_compute_energy()
        compute_forces = in_out.get_compute_forces()
        if compute_energy:
          pred_energy = self.calculator.get_energy(in_out)
          ref_energy = in_out.conf.get_energy()
          energy_rmse = pred_energy - ref_energy
        else:
          energy_rmse = None
        if compute_forces:
          pred_forces = self.calculator.get_forces(in_out)
          ref_forces = in_out.conf.get_forces()
          forces_rmse = np.linalg.norm(pred_forces - ref_forces)
        else:
          forces_rmse = None

        residual = self.residual_fn(identifier, natoms, pred, ref, self.residual_data)
        loss = 0.5*np.linalg.norm(residual)**2

        if normalize_by_num_atoms:
          nz = natoms
        else:
          nz = 1

        if energy_rmse is None:
          if forces_rmse is None:
            fout.write('{:14.6e} {} {}   {}   {}\n'.format(
                loss/nz, energy_rmse/nz, forces_rmse/nz, natoms, identifier))
          else:
            fout.write('{:14.6e} {} {:14.6}   {}   {}\n'.format(
                loss/nz, energy_rmse/nz, forces_rmse/nz, natoms, identifier))
        else:
          if forces_rmse is None:
            fout.write('{:14.6e} {:14.6e} {}   {}   {}\n'.format(
                loss/nz, energy_rmse/nz, forces_rmse/nz, natoms, identifier))
          else:
            fout.write('{:14.6e} {:14.6e} {:14.6e}   {}   {}\n'.format(
                loss/nz, energy_rmse/nz, forces_rmse/nz, natoms, identifier))


  def _scipy_optimize_least_squares(self, method, **kwargs):
    residual = self.get_residual
    x0 = self.model_params.get_x0()
    return scipy.optimize.least_squares(residual, x0, method=method, **kwargs)


  def _scipy_optimize_minimize(self, method, **kwargs):
    loss = self.get_loss
    x0 = self.model_params.get_x0()
    return scipy.optimize.minimize(loss, x0, method=method, **kwargs)


  def _update_params(self, x0):
    """Update parameter values from minimizer to KIM calculator.

    Parameters
    ----------

    x0: list
      optimizing parameter values
    """
    # update from minimizer to ModelParameters object
    self.model_params.update_params(x0)

    # update from ModelParameters to KIM calculator
    self.calculator.update_params(self.model_params)


  def _get_residual_single_config(self, in_out, calculator, residual_fn, residual_data):

    # prediction data
    calculator.compute(in_out)
    pred = calculator.get_prediction(in_out)

    # reference data
    ref = calculator.get_reference(in_out)

    conf = in_out.conf
    identifier = conf.get_identifier()
    natoms = conf.get_number_of_atoms()

    residual = residual_fn(identifier, natoms, pred, ref, residual_data)

    return residual



  def __enter__(self):
    return self

  def __exit__(self, exec_type, exec_value, trackback):

    # if there is expections, raise it (not for KeyboardInterrupt)
    if exec_type is not None and exec_type is not KeyboardInterrupt:
      return False # return False will cause Python to re-raise the expection

    # write error report
    self.error_report()

    #write fitted params to `FINAL_FITTED_PARAMS' and stdout at end.
    self.model_params.echo_params(fname='FINAL_FITTED_PARAMS')
    self.model_params.echo_params()

