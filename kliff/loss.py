import numpy as np
import scipy.optimize
import multiprocessing as mp
import kliff
from kliff import parallel

logger = kliff.logger.get_logger(__name__)


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
        raise ValueError(
            "len(prediction) != 3N+1, where N is the number of atoms.")

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


# import tensorflow as tf
# @tf.custom_gradient
def test_residual(conf_id, natoms, prediction, reference, data):

    print('@@ natoms', natoms)

    def grad(dy):
        return None, None, None, None, None

    return prediction - reference
    # return prediction - reference, grad


class Loss(object):
    """Objective function that will be minimized. """

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

    scipy_minimize_methods_not_supported_arguments = [
        'bounds'
    ]

    scipy_least_squares_methods = [
        'trf',
        'dogbox',
        'lm',
    ]

    scipy_least_squares_methods_not_supported_arguments = [
        'ounds'
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

    def __init__(self, calculator, nprocs=1, residual_fn=energy_forces_residual,
                 residual_data=None):
        """
        Parameters
        ----------

        calculator: Calculator object

        nprocs: int
          Number of processors to parallel to run the predictors.

        residual_fn: function
          function to compute residual, see `energy_forces_residual` for an example

        residual_data: dict
          data passed to residual function

        """

        self.calculator = calculator
        self.nprocs = nprocs

        self.residual_fn = residual_fn
        self.residual_data = residual_data
        if self.residual_data is None:
            self.residual_data = dict()

        self.calculator_type = calculator.__class__.__name__

        if self.calculator_type != 'NeuralNetwork':

            self.compute_arguments = self.calculator.get_compute_arguments()
            self.calculator.update_model_params()
            infl_dist = self.calculator.get_influence_distance()
            # TODO can be parallelized
            for ca in self.compute_arguments:
                ca.refresh(infl_dist)

        logger.info('"{}" instantiated.'.format(self.__class__.__name__))
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

        kwargs: extra keyword arguments that can be used by the scipy optimizer.
        """

        if self.calculator_type != 'NeuralNetwork':

            if method in self.scipy_least_squares_methods:
                #    # change unbounded value to np.inf that least_squares needs
                #    try:
                #        bounds = kwargs['bounds']
                #        for i in range(len(bounds)):
                #            bounds[i][0] = -np.inf
                #            bounds[i][1] = np.inf
                #    except KeyError:
                #        pass
                for i in self.scipy_least_squares_methods_not_supported_arguments:
                    if i in kwargs:
                        raise LossError('Argument "{}" should not be set through the '
                                        '"minimize" method.'.format(i))
                bounds = self.calculator.get_opt_params_bounds()
                kwargs['bounds'] = bounds
                result = self._scipy_optimize_least_squares(method, **kwargs)
            elif method in self.scipy_minimize_methods:
                for i in self.scipy_minimize_methods_not_supported_arguments:
                    if i in kwargs:
                        raise LossError('Argument "{}" should not be set through the '
                                        '"minimize" method.'.format(i))
                bounds = self.calculator.get_opt_params_bounds()
                for i in range(len(bounds)):
                    lb = bounds[i][0]
                    ub = bounds[i][1]
                    if lb is None:
                        bounds[i][0] = -np.inf
                    if ub is None:
                        bounds[i][1] = np.inf
                kwargs['bounds'] = bounds
                result = self._scipy_optimize_minimize(method, **kwargs)
            else:
                raise Exception('minimization method "{}" not supported.'.format(method))

            # update final optimized paramters to ModelParameters object
            self.calculator.update_params(result.x)
            return result

        else:

            try:
                import tensorflow as tf
            except ImportError as e:
                raise ImportError(
                    str(e) + '.\nPlease install "tensorflow" first.')

            loss = self.calculator.get_loss()

            with tf.Session() as sess:

                if method in self.tf_minimize_methods:
                    optimizer = getattr(tf.train, method)(**kwargs)
                    train = optimizer.minimize(loss)
                    init_op = tf.global_variables_initializer()
                    sess.run(init_op)
                    while True:
                        try:
                            sess.run(train)
                        except tf.errors.OutOfRangeError:
                            break

                elif method in self.tf_scipy_minimize_methods:
                    # enable giving options as a dictionary as at scipy website
                    scipy_interface_options = kwargs.copy()
                    options = scipy_interface_options.pop('options', None)
                    if options is not None:
                        for key, val in options.items():
                            scipy_interface_options[key] = val
                    optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, method=method,
                                                                       options=scipy_interface_options)
                    init_op = tf.global_variables_initializer()
                    sess.run(init_op)
                    optimizer.minimize(sess)

                else:
                    raise Exception(
                        'minimization method "{}" not supported.'.format(method))

                self.calculator.model.write_kim_ann(
                    sess, fname='ann_kim.params')

    def get_residual(self, x):
        """ Compute the residual for the cost.
        This is a callable for optimizer (e.g. scipy.optimize.least_squares), which is
        passed as the first positional argument.

        Parameters
        ----------

        x: 1D array
          optimizing parameter values
        """

        # publish params x to predictor
        self.calculator.update_params(x)

        # parallel computing of residual
        cas = self.calculator.get_compute_arguments()

        # compute residual
        if self.nprocs > 1:
            residuals = parallel.parmap2(
                self._get_residual_single_config,
                cas,
                self.nprocs,
                self.calculator,
                self.residual_fn,
                self.residual_data)
            residual = np.concatenate(residuals)
        else:
            residual = []
            for ca in cas:
                current_residual = self._get_residual_single_config(
                    ca,
                    self.calculator,
                    self.residual_fn,
                    self.residual_data)
                residual = np.concatenate((residual, current_residual))

        return residual

    def get_loss(self, x):
        """ Compute the loss.
        This is a callable for optimizer (e.g. scipy.optimize.minimize), which is
        passed as the first positional argument.

        Parameters
        ----------

        x: 1D array
          optimizing parameter values
        """
        residual = self.get_residual(x)
        loss = 0.5*np.linalg.norm(residual)**2
        return loss

    def error_report(self, normalize_by_num_atoms=True, fname='ERROR_REPORT'):
        """Write error of each configuration to fname.

        Parameters
        ----------
        fname: str
          path to the file to write the error report.
        """

        loss = self.get_loss(self.calculator.get_opt_params())

        with open(fname, 'w') as fout:
            fout.write('\n'+'='*80+'\n')
            fout.write('Final loss: {:18.10e}\n\n'.format(loss))
            fout.write('='*80+'\n')
            if normalize_by_num_atoms:
                fout.write('(Loss, energy RMSE, and forces RMSE are normalized by '
                           'number of atoms: Natoms.)\n\n')
            fout.write('      Loss       energy RMSE     forces RMSE  Natoms  '
                       'config. identifier\n\n')

            cas = self.calculator.get_compute_arguments()
            for ca in cas:

                # prediction data
                self.calculator.compute(ca)
                pred = self.calculator.get_prediction(ca)

                # reference data
                ref = self.calculator.get_reference(ca)

                conf = ca.conf
                identifier = conf.get_identifier()
                natoms = conf.get_number_of_atoms()

                compute_energy = ca.get_compute_flag('energy')
                compute_forces = ca.get_compute_flag('forces')
                if compute_energy:
                    pred_energy = self.calculator.get_energy(ca)
                    ref_energy = ca.conf.get_energy()
                    energy_rmse = pred_energy - ref_energy
                else:
                    energy_rmse = None
                if compute_forces:
                    pred_forces = self.calculator.get_forces(ca)
                    ref_forces = ca.conf.get_forces()
                    forces_rmse = np.linalg.norm(pred_forces - ref_forces)
                else:
                    forces_rmse = None

                residual = self.residual_fn(
                    identifier, natoms, pred, ref, self.residual_data)
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
                        fout.write('{:14.6e} {} {:14.6e}   {}   {}\n'.format(
                            loss/nz, energy_rmse/nz, forces_rmse/nz, natoms, identifier))
                else:
                    if forces_rmse is None:
                        fout.write('{:14.6e} {:14.6e} {}   {}   {}\n'.format(
                            loss/nz, energy_rmse/nz, forces_rmse/nz, natoms, identifier))
                    else:
                        fout.write('{:14.6e} {:14.6e} {:14.6e}   {}   {}\n'.format(
                            loss/nz, energy_rmse/nz, forces_rmse/nz, natoms, identifier))

    def _scipy_optimize_least_squares(self, method, **kwargs):
        logger.info('scipy least squares method "{}" used.'.format(method))
        residual = self.get_residual
        x0 = self.calculator.get_opt_params()
        return scipy.optimize.least_squares(residual, x0, method=method, **kwargs)

    def _scipy_optimize_minimize(self, method, **kwargs):
        logger.info('scipy optimization method "{}" used.'.format(method))
        loss = self.get_loss
        x0 = self.calculator.get_opt_params()
        return scipy.optimize.minimize(loss, x0, method=method, **kwargs)

    def _get_residual_single_config(self, ca, calculator, residual_fn, residual_data):

        # prediction data
        calculator.compute(ca)
        pred = calculator.get_prediction(ca)

        # reference data
        ref = calculator.get_reference(ca)

        conf = ca.conf
        identifier = conf.get_identifier()
        natoms = conf.get_number_of_atoms()

        residual = residual_fn(identifier, natoms, pred, ref, residual_data)

        return residual

    def __enter__(self):
        return self

    def __exit__(self, exec_type, exec_value, trackback):

        # if there is expections, raise it (not for KeyboardInterrupt)
        if exec_type is not None and exec_type is not KeyboardInterrupt:
            return False  # return False will cause Python to re-raise the expection

        # write error report
        self.error_report()

        # write fitted params to `FINAL_FITTED_PARAMS' and stdout at end.
        self.calculator.echo_fitting_params(fname='FINAL_FITTED_PARAMS')


class LossError(Exception):
    def __init__(self, msg):
        super(LossError, self).__init__(msg)
        self.msg = msg

    def __str__(self):
        return self.msg
