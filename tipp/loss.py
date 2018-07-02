#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import scipy.optimize
import multiprocessing as mp
import collections
from tipp import parallel


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

  def __init__(self, model_params, calculator, nprocs=1,
      residual_fn=energy_forces_residual, residual_data=None):

    self.model_params = model_params
    self.calculator = calculator
    self.nprocs = nprocs

    self.residual_fn = residual_fn
    self.residual_data = residual_data
    if self.residual_data is None:
      self.residual_data = dict()

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


  def set_nprocs(self, nprocs):
    """ Set the number of processors to be used."""
    self.nprocs = nprocs

  def set_residual_fn_and_data(self, fn, data):
    """ Set residual function and data. """
    self.residual_fn = fn
    self.residual_fn_data = data


  def minimize(self, method, *args, **kwargs):
    """ Minimize the loss.

    method: str
      minimization methods as specified at:
        https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.optimize.minimize.html
      and
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

    args: extra arguments that can be used by the scipy optimizer
    kwargs: extra keyword arguments that can be used by the scipy optimizer
    """

    if method in self.scipy_least_squares_methods:
      result = self._scipy_optimize_least_squares(method, *args, **kwargs)
    elif method in self.scipy_minimize_methods:
      result = self._scipy_optimize_minimize(method, *args, **kwargs)

    # update final optimized paramters to ModelParameters object
    self._update_params(result.x)
    return result


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

    # TODO implement parallel
    # parallel computing of residual
    #results = parallel.parmap(self.calculator.compute, kim_in_out_data, 1)

    # compute residual
    residual = []
    kim_in_out_data = self.calculator.get_kim_input_and_output()
    for in_out in kim_in_out_data:
      current_residual = self._get_residual_single_config(in_out, self.calculator,
          self.residual_fn, self.residual_data)
      residual = np.concatenate((residual, current_residual))

    return residual





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
        fout.write('(Loss, energy RMSE, and forces RMSE are normalized by number of atoms: Natoms.)\n\n')
      fout.write('      Loss       energy RMSE     forces RMSE  Natoms  config. identifier\n\n')

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




  def _scipy_optimize_least_squares(self, method, *args, **kwargs):
    residual = self.get_residual
    x0 = self.model_params.get_x0()
    return scipy.optimize.least_squares(residual, x0, *args, method=method, **kwargs)


  def _scipy_optimize_minimize(self, method, *args, **kwargs):
    loss = self.get_loss
    x0 = self.model_params.get_x0()
    return scipy.optimize.minimize(loss, x0, *args, method=method, **kwargs)



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
























class Cost(object):
  """Objective function that will be minimized.

  Parameters
  ----------

  model_params: ModelParameters object

  nprocs: int
    Number of processors to parallel to run the predictors.

  verbose: int
    Integer code to denote to whether echo running cost to screen.
    0, do not echo anything
    1, echo running cost
  """
  def __init__(self, model_params, calculator, nprocs=1, verbose=0):

    self.params = params
    self.nprocs = nprocs
    self.verbose = verbose
    self.normalize = normalize
    self.pred_obj = []
    self.ref = []
    self.weight = []
    self.func = []
    self.pred_obj_group = None
    self.ref_group = None
    self.weight_group = None
    self.func_group = None

  def __enter__(self):
    return self

  def add(self, pred_obj, reference, weight=1, func=None):
    """Add a contribution part to the cost.

    Parameters
    ----------

    pred_obj: predictor object.
      It must have the 'get_prediction()' method, which returns a float
      list that has the same length as "reference". Optionally, it can have
      the 'update_params()' method that updates the parameters from the
      ModelParams class. However, for OpenKIM test, this is impossible, since
      we have no access to its KIM object. The way we handle it is to crack
      the KIM api to read parameters at Model initialization stage.

    reference: float list
      reference data

    weight: float or list
      weight for the prediction and reference data. If a float number
      is entered, all the prediction and reference data entries will
      use the same value. Its length should match reference data if
      "func" is not set, otherwise, its length should match the length
      of the return value of "func".

    func(pred_obj.get_prediction(), reference):
      a function to generate residual using a method rather than the
      default one. It takes two R^m list arguments, and returns a R^n
      list.
    """

    # check correct length
    pred = pred_obj.get_prediction()
    len_pred = len(pred)
    if len_pred != len(reference):
      raise InputError('Cost.add(): lengths of prediction and reference data '
               'do not match.')
    # construct weight list according to input type
    if func is None:
      len_resid = len_pred
    else:
      residual = func(pred,reference)
      if not isinstance(residual, (collections.Sequence, np.ndarray)):
        raise InputError('Cost.add(): return value of "{}" should be a list.'.format(func.__name__))
      len_resid = len(residual)

    if isinstance(weight, (collections.Sequence, np.ndarray)):
      if len(weight) != len_resid:
        raise InputError('Cost.add(): length of return value of "{}" '
                 'and that of weight do not match.'.format(func.__name__))
    else:
      weight = [weight for i in range(len_resid)]

    # record data
    self.pred_obj.append(pred_obj)
    self.ref.append(reference)
    self.weight.append(weight)
    self.func.append(func)
    # group objects added so far
    self._group_preds()


  def _group_preds(self):
    """ Group the predictors (and the associated references, weights and wrapper
    functions) into 'nprocs' groups, so that each group can be processed by a
    processor. The predictors are randomly assigned to each group to account for
    load balance in each group.
    """
#NOTE we may not want the shuffling while developing the code, such that it gives
# deterministic results
    # shuffle before grouping, in case some processors get quite heavy jobs but
    # others get light ones.
#    combined = zip(self.pred_obj, self.ref, self.weight, self.func)
#    np.random.shuffle(combined)
#    self.pred_obj[:], self.ref[:], self.weight[:], self.func[:] = zip(*combined)

    # grouping
    self.pred_obj_group= np.array_split(self.pred_obj, self.nprocs)
    self.ref_group= np.array_split(self.ref, self.nprocs)
    self.weight_group= np.array_split(self.weight, self.nprocs)
    self.func_group= np.array_split(self.func, self.nprocs)


  def get_residual(self, x0):
    """ Compute the residual for the cost.
    This is a callable for optimizer that usually passed as the first positional
    argument.

    Parameters
    ----------

    x0: list
      optimizing parameter values
    """
    # update params from minimizer to ModelParams
    self._update_params(x0)


    # create jobs and start
    jobs = []
    pipe_list = []
    for g in range(self.nprocs):
      # create pipes
      recv_end, send_end = mp.Pipe(False)
      # get data for this group
      pred = self.pred_obj_group[g]
      ref = self.ref_group[g]
      weight = self.weight_group[g]
      func = self.func_group[g]
      p = mp.Process(target=self._get_residual_group, args=(pred,ref,weight,func,send_end,))
      p.start()
      jobs.append(p)
      pipe_list.append(recv_end)

    # we should place recv() before join()
    residuals = [x.recv() for x in pipe_list]
    # wait for the all the jobs to complete
    for p in jobs:
      p.join()

    residuals = np.concatenate(residuals)

    if self.verbose == 1:
      cost = 0.5*np.linalg.norm(residuals)**2
      print ('cost: {:18.10e}'.format(cost))

    return residuals




## do not use multiprocessing
#  def get_residual(self, x0):
#   # publish params x0 to predictor
#   self._update_params(x0)
#   # compute properties using new params x0
#   #self._compute_predictions()
#
#   residual = []
#   for i in range(len(self.pred_obj)):
#     pred = self.pred_obj[i].get_prediction()
#     ref = self.ref[i]
#     weight = np.sqrt(self.weight[i])
#     fun = self.func[i]
#     if fun == None:
#       difference = np.subtract(pred, ref)
#     else:
#       difference = fun(pred, ref)
#     tmp_resid = np.multiply(weight, difference)
#     residual = np.concatenate((residual, tmp_resid))
#   return residual



  def get_cost(self, x0):
    """ Compute the cost.
    This can be used as the callable for optimizer (e.g. scipy.optimize.minimize
    method='CG')

    Parameters
    ----------

    x0: list
      optimizing parameter values
    """
    residual = self.get_residual(x0)
    cost = 0.5*np.linalg.norm(residual)**2
    return cost


  def set_cost_fn(self, fn):
    self.cost_fn = fn

  def set_minimizer(self, minimizer,*args, **kwargs):
    self.minimizer = minimizer
    self.min_args = args
    self.min_kwargs = kwargs


  def minimize(self):
    pass



  def _update_params(self, x0):
    """Update parameter values from minimizer to ModelParams object.

    Parameters
    ----------

    x0: list
      optimizing parameter values
    """
    self.params.update_params(x0)

# NOTE (we may support print to screen every N steps)
# print to file while minimizion
    self.params.echo_params('params.txt')


  def _get_residual_group(self, pred_g, ref_g, weight_g, func_g, send_end):
    """Helper function to do the computation of residuals.
    """
    residual = []
    for i in range(len(pred_g)):
      pred = pred_g[i].get_prediction()
      ref = ref_g[i]
      weight = np.sqrt(weight_g[i])
      func = func_g[i]
      if func is None:
        difference = np.subtract(pred, ref)
      else:
        difference = func(pred, ref)

      #NOTE this is not good, only for temporary use, see doc in _init__.
      # here we have assumed that difference[0] is energy, difference[1],
      # difference[2], and difference[3] are the x,y,z component forces of
      # atom 1, and so on.
      # we normalize forces by magnitude of reference forces. Do not normalize
      # energy.
      if self.normalize:
        normalizer = [1.] # do not normalize energy
        for i in range(1, len(difference), 3):
          forces = ref[i:i+3]
          fmag = np.linalg.norm(forces)
          normalizer.extend([fmag, fmag, fmag])
        difference = np.divide(difference, normalizer)

      tmp_resid = np.multiply(weight, difference)
      residual = np.concatenate((residual, tmp_resid))
    send_end.send(residual)


  def error_report(self, fname='ERROR_REPORT'):
    """Write error of each configuration to fname.
    """
    residuals = self.get_residual(self.params.get_x0())
    cost_all = 0.5*residuals**2

    # the order of residual is the same as grouped prediction objects
    pred_objs = np.concatenate(self.pred_obj_group)
    refs = np.concatenate(self.ref_group)
    weights = np.concatenate(self.weight_group)

    start = 0
    with open (fname, 'w') as fout:
      fout.write('Total error: {:18.10e}\n\n'.format(sum(cost_all)))
      fout.write('Below is error by each configuration.\n')
      for p_obj, r, w in zip(pred_objs, refs, weights):
        identifier = p_obj.conf.id
        p = p_obj.get_prediction()
        end = start + len(p)
        cost_config = sum(cost_all[start:end])
        start = end
        fout.write('\nconfig: {},    config error:{:18.10e}\n'.format(identifier, cost_config))
        fout.write('     Prediction    Reference   Difference   Diff./Ref.     Weight     Error\n')
        for i,j,k in zip(p,r,w):
          diff = i-j
          try:
            ratio = float(diff)/float(j) # change np.float to float to catch error
          except ZeroDivisionError:
            ratio = np.inf
          error = 0.5*k*diff**2
          fout.write('  {:14.6e} {:14.6e} {:14.6e} {:14.6e} {:14.6e} {:14.6e}\n'.format(i,j,diff,ratio,k,error))


  def __exit__(self, exec_type, exec_value, trackback):

    # if there is expections, raise it (not for KeyboardInterrupt)
    if exec_type is not None and exec_type is not KeyboardInterrupt:
      return False # return False will cause Python to re-raise the expection

    # write error report
    self.error_report()

    #write fitted params to `FINAL_FITTED_PARAMS' and stdout at end.
    self.params.echo_params(fname='FINAL_FITTED_PARAMS')
    self.params.echo_params()

