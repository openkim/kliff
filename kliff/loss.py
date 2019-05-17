import os
import sys
import numpy as np
import warnings
import scipy.optimize
import multiprocessing as mp
import kliff
from kliff import parallel
from kliff.error import InputError

try:
    import torch
    from kliff.neuralnetwork import FingerprintsDataset
    from kliff.neuralnetwork import FingerprintsDataLoader

    torch_available = True
except ImportError:
    torch_available = False

try:
    from mpi4py import MPI

    mpi4py_available = True
except ImportError:
    mpi4py_available = False


logger = kliff.logger.get_logger(__name__)


def energy_forces_residual(identifier, natoms, prediction, reference, data):
    """A residual function using both energy and forces.

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
        User provided callback data, and aviailable keys:
            energy_weight: float (default: 1)
            forces_weight: float (default: 1)

    normalize: bool (default: False)
        Whether to normalize the residual by the number of atoms

    Return
    ------
    residual: 1D array

    Note
    ----
    The length of `prediction` and `reference` (call it `S`) are the same, and it
    depends on `use_energy` and `use_forces` in KIMCalculator. Assume the
    configuration contains of `N` atoms.

    1) If `use_energy == True` and `use_forces == False`, then `S = 1`.
    `prediction[0]` is the potential energy computed by the calculator, and
    `reference[0]` is the reference energy.

    2) If `use_energy == False` and `use_forces == True`, then `S = 3N`.
    `prediction[3*i+0]`, `prediction[3*i+1]`, and `prediction[3*i+2]` are the
    x, y, and z component of the forces on atom i in the configuration, respecrively.
    Correspondingly, `reference` is the 3N concatenated reference forces.


    3) If `use_energy == True` and `use_forces == True`, then `S = 3N + 1`.
    `prediction[0]` is the potential energy computed by the calculator, and
    `reference[0]` is the reference energy.
    `prediction[3*i+1]`, `prediction[3*i+2]`, and `prediction[3*i+3]` are the
    x, y, and z component of the forces on atom i in the configuration, respecrively.
    Correspondingly, `reference` is the 3N concatenated reference forces.
    """

    if len(prediction) != 3 * natoms + 1:
        raise ValueError("len(prediction) != 3N+1, where N is the number of atoms.")

    # prepare weight based on user provided data
    energy_weight = data['energy_weight']
    forces_weight = data['forces_weight']
    normalize = data['normalize_by_natoms']
    if energy_weight is None:
        energy_weight = 1.0
    if forces_weight is None:
        forces_weight = 1.0
    if normalize:
        energy_weight /= natoms
        forces_weight /= natoms

    # obtain residual and properly normalize it
    residual = prediction - reference
    residual[0] *= energy_weight
    residual[1:] *= forces_weight

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
    """Objective function class to conduct the optimization."""

    def __new__(
        self, calculator, nprocs=1, residual_fn=energy_forces_residual, residual_data=None
    ):
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
        data = self.check_residual_data(residual_data)

        calc_type = calculator.__class__.__name__

        if calc_type == 'PytorchANNCalculator':
            return LossNeuralNetworkModel(calculator, nprocs, residual_fn, data)
        else:
            return LossPhysicsMotivatedModel(calculator, nprocs, residual_fn, data)

    @staticmethod
    def check_residual_data(data):
        default = {
            'energy_weight': 1.0,
            'forces_weight': 1.0,
            'stress_weight': 1.0,
            'normalize_by_natoms': True,
        }
        if data is not None:
            for key, value in data.items():
                if key not in default:
                    raise InputError('"{}" not supported by "residual_data".'.format(key))
                else:
                    default[key] = value
        return default


class LossPhysicsMotivatedModel(object):
    """Objective function class to conduct the optimization for PM models."""

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
    scipy_minimize_methods_not_supported_args = ['bounds']
    scipy_least_squares_methods = ['trf', 'dogbox', 'lm']
    scipy_least_squares_methods_not_supported_args = ['bounds']

    def __init__(
        self, calculator, nprocs=1, residual_fn=energy_forces_residual, residual_data=None
    ):

        self.calculator = calculator
        self.nprocs = nprocs

        self.residual_fn = residual_fn
        self.residual_data = residual_data if residual_data is not None else dict()
        self.calculator_type = calculator.__class__.__name__

        if self.calculator_type == 'WrapperCalculator':
            calculators = self.calculator.calculators
        else:
            calculators = [self.calculator]
        for calc in calculators:
            infl_dist = calc.model.get_influence_distance()
            cas = calc.get_compute_arguments()
            # TODO can be parallelized
            for ca in cas:
                ca.refresh(infl_dist)

        logger.info('"{}" instantiated.'.format(self.__class__.__name__))

    def minimize(self, method, **kwargs):
        """ Minimize the loss.

        Parameters
        ----------
        method: str
            minimization methods as specified at:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

        kwargs: dict
            extra keyword arguments that can be used by the scipy optimizer
        """
        kwargs = self.adjust_kwargs(method, **kwargs)

        logger.info('Start minimization using method: {}.'.format(method))
        result = self.scipy_optimize(method, **kwargs)
        logger.info('Finish minimization using method: {}.'.format(method))

        # update final optimized parameters
        self.calculator.update_opt_params(result.x)
        return result

    def adjust_kwargs(self, method, **kwargs):

        if method in self.scipy_least_squares_methods:
            # check support status
            for i in self.scipy_least_squares_methods_not_supported_args:
                if i in kwargs:
                    raise LossError(
                        'Argument "{}" should not be set via the "minimize" method. '
                        'It it set internally.'.format(i)
                    )
            # adjust bounds
            if self.calculator.has_opt_params_bounds():
                if method in ['trf', 'dogbox']:
                    bounds = self.calculator.get_opt_params_bounds()
                    lb = [b[0] if b[0] is not None else -np.inf for b in bounds]
                    ub = [b[1] if b[1] is not None else np.inf for b in bounds]
                    bounds = (lb, ub)
                    kwargs['bounds'] = bounds
                else:
                    raise LossError('Method "{}" cannot handle bounds.'.format(method))
        elif method in self.scipy_minimize_methods:
            # check support status
            for i in self.scipy_minimize_methods_not_supported_args:
                if i in kwargs:
                    raise LossError(
                        'Argument "{}" should not be set via the "minimize" method. '
                        'It it set internally.'.format(i)
                    )
            # adjust bounds
            if self.calculator.has_opt_params_bounds():
                if method in ['L-BFGS-B', 'TNC', 'SLSQP']:
                    bounds = self.calculator.get_opt_params_bounds()
                    kwargs['bounds'] = bounds
                else:
                    raise LossError('Method "{}" cannot handle bounds.'.format(method))
        else:
            raise LossError('minimization method "{}" not supported.'.format(method))

        return kwargs

    def scipy_optimize(self, method, **kwargs):

        if method in self.scipy_least_squares_methods:
            minimize_fn = scipy.optimize.least_squares
        elif method in self.scipy_minimize_methods:
            minimize_fn = scipy.optimize.minimize
        else:
            raise LossError('minimization method "{}" not supported.'.format(method))

        size = parallel.get_MPI_world_size()

        if size > 1:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

            msg = 'Running in MPI mode with {} processes.'.format(size)
            print(msg + '\n')
            logger.info(msg)
            if self.nprocs > 1:
                msg = (
                    'Argument "nprocs = {}" provided at initialization is ignored. When '
                    'running in MPI mode, the number of processes provided along with '
                    'the "mpiexec" (or "mpirun") command is used.'.format(self.nprocs)
                )
                logger.warning(msg)
                warnings.warn(msg, category=Warning)

            x = self.calculator.get_opt_params()
            if method in self.scipy_least_squares_methods:
                func = self.get_residual_MPI
            elif method in self.scipy_minimize_methods:
                func = self.get_loss_MPI
            else:
                raise LossError('minimization method "{}" not supported.'.format(method))

            if rank == 0:
                result = minimize_fn(func, x, method=method, **kwargs)
                # notify other process to break func
                break_flag = True
                for i in range(1, size):
                    comm.send(break_flag, dest=i, tag=i)
            else:
                func(x)
                result = None
            result = comm.bcast(result, root=0)
            return result

        else:
            # 1. running MPI with 1 process
            # 2. running without MPI at all
            # both cases are regarded as running without MPI

            if self.nprocs == 1:
                msg = 'Running in serial mode.'
                print(msg + '\n')
                logger.info(msg)
            else:
                msg = 'Running in multiprocessing mode with {} processes.'.format(
                    self.nprocs
                )
                print(msg + '\n')
                logger.info(msg)
                # Maybe one thinks he is using MPI because nprocs is used
                if mpi4py_available:
                    msg = (
                        '"mpi4y" detected. If you try to run in MPI mode, you should '
                        'execute your code via "mpiexec" (or "mpirun"). If not, ignore '
                        'this message.'
                    )
                    logger.warning(msg)
                    warnings.warn(msg, category=Warning)

            x = self.calculator.get_opt_params()
            if method in self.scipy_least_squares_methods:
                func = self.get_residual
            elif method in self.scipy_minimize_methods:
                func = self.get_loss
            else:
                raise LossError('minimization method "{}" not supported.'.format(method))

            result = minimize_fn(func, x, method=method, **kwargs)
            return result

    def get_residual(self, x):
        """ Compute the residual.

        This is a callable for optimizing method in scipy.optimize.least_squares,
        which is passed as the first positional argument.

        Parameters
        ----------

        x: 1D array
          optimizing parameter values
        """

        # publish params x to predictor
        self.calculator.update_opt_params(x)

        cas = self.calculator.get_compute_arguments()

        if self.calculator_type == 'WrapperCalculator':
            calc_list = self.calculator.get_calculator_list()
            X = zip(cas, calc_list)
            if self.nprocs > 1:
                residuals = parallel.parmap2(
                    self.get_residual_single_config,
                    X,
                    self.residual_fn,
                    self.residual_data,
                    nprocs=self.nprocs,
                    tuple_X=True,
                )
                residual = np.concatenate(residuals)
            else:
                residual = []
                for ca, calc in X:
                    current_residual = self.get_residual_single_config(
                        ca, calc, self.residual_fn, self.residual_data
                    )
                    residual = np.concatenate((residual, current_residual))

        else:
            if self.nprocs > 1:
                residuals = parallel.parmap2(
                    self.get_residual_single_config,
                    cas,
                    self.calculator,
                    self.residual_fn,
                    self.residual_data,
                    nprocs=self.nprocs,
                    tuple_X=False,
                )
                residual = np.concatenate(residuals)
            else:
                residual = []
                for ca in cas:
                    current_residual = self.get_residual_single_config(
                        ca, self.calculator, self.residual_fn, self.residual_data
                    )
                    residual = np.concatenate((residual, current_residual))

        return residual

    def get_loss(self, x):
        """Compute the loss.

        This is a callable for optimizing method in scipy.optimize.minimize,
        which is passed as the first positional argument.

        Parameters
        ----------

        x: 1D array
          optimizing parameter values
        """
        residual = self.get_residual(x)
        loss = 0.5 * np.linalg.norm(residual) ** 2
        return loss

    def get_residual_MPI(self, x):
        def residual_my_chunk(x):
            # broadcast parameters
            x = comm.bcast(x, root=0)
            # publish params x to predictor
            self.calculator.update_opt_params(x)

            residual = []
            for ca in cas:
                current_residual = self.get_residual_single_config(
                    ca, self.calculator, self.residual_fn, self.residual_data
                )
                residual.extend(current_residual)
            return residual

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # get my chunk of data
        cas = self.split_data()

        while True:

            if rank == 0:
                break_flag = False
                for i in range(1, size):
                    comm.send(break_flag, dest=i, tag=i)
                residual = residual_my_chunk(x)
                all_residuals = comm.gather(residual, root=0)
                return np.concatenate(all_residuals)
            else:
                break_flag = comm.recv(source=0, tag=rank)
                if break_flag:
                    break
                else:
                    residual = residual_my_chunk(x)
                    all_residuals = comm.gather(residual, root=0)

    def get_loss_MPI(self, x):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        residual = self.get_residual_MPI(x)
        if rank == 0:
            loss = 0.5 * np.linalg.norm(residual) ** 2
        else:
            loss = None

        return loss

    def split_data(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # get a portion of data based on rank
        cas = self.calculator.get_compute_arguments()
        # random.shuffle(cas)

        rank_size = len(cas) // size
        # last rank deal with the case where len(cas) cannot evenly divide size
        if rank == size - 1:
            cas = cas[rank_size * rank :]
        else:
            cas = cas[rank_size * rank : rank_size * (rank + 1)]

        return cas

    def get_residual_single_config(self, ca, calculator, residual_fn, residual_data):

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


class LossNeuralNetworkModel(object):
    """Objective function class to conduct the optimization for ML models."""

    torch_minimize_methods = [
        'Adadelta',
        'Adagrad',
        'Adam',
        'SparseAdam',
        'Adamax',
        'ASGD',
        'LBFGS',
        'RMSprop',
        'Rprop',
        'SGD',
    ]

    def __init__(
        self, calculator, nprocs=1, residual_fn=energy_forces_residual, residual_data=None
    ):
        """
        Parameters
        ----------

        calculator: Calculator object

        nprocs: int
          Number of processors to parallel to run the predictors.

        residual_fn: function
          Function to compute residual, see `energy_forces_residual` for an example.

        residual_data: dict
          Data passed to residual function.

        """

        if not torch_available:
            raise ImportError('Please install "PyTorch" first. See: https://pytorch.org')

        self.calculator = calculator
        self.nprocs = nprocs

        self.residual_fn = residual_fn
        self.residual_data = residual_data if residual_data is not None else dict()

        self.calculator_type = calculator.__class__.__name__

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

    def minimize(self, method, batch_size=100, num_epochs=1000, **kwargs):
        """ Minimize the loss.

        Parameters
        ----------
        method: str
            PyTorch optimization methods, and aviliable ones are:
            [`Adadelta`, `Adagrad`, `Adam`, `SparseAdam`, `Adamax`, `ASGD`,
            `LBFGS`, `RMSprop`, `Rprop`, `SGD`]
            See also: https://pytorch.org/docs/stable/optim.html

        kwargs: extra keyword arguments that can be used by the PyTorch optimizer.
        """
        if method not in self.torch_minimize_methods:
            raise LossError('Minimization method "{}" not supported.'.format(method))
        self.method = method
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # data loader
        fname = self.calculator.get_train_fingerprints_path()
        fp = FingerprintsDataset(fname)
        self.data_loader = FingerprintsDataLoader(dataset=fp, num_epochs=self.num_epochs)

        # optimizing
        try:
            optimizer = getattr(torch.optim, method)(
                self.calculator.model.parameters(), **kwargs
            )
        except TypeError as e:
            print(str(e))
            idx = str(e).index("argument '") + 10
            err_arg = str(e)[idx:].strip("'")
            raise InputError(
                'Argument "{}" not supported by optimizer "{}".'.format(err_arg, method)
            )

        # model save metadata
        save_prefix = self.calculator.model.save_prefix
        save_start = self.calculator.model.save_start
        save_frequency = self.calculator.model.save_frequency
        if save_prefix is None or save_start is None or save_frequency is None:
            logger.info(
                'Model saving meta data not set by user. Now set it '
                'to "prefix=./kliff_saved_model", "start=num_epochs/2",'
                ' and "frequency=10".'
            )
            save_prefix = os.path.join(os.getcwd(), 'kliff_saved_model')
            save_start = self.num_epochs // 2
            save_frequency = 10
            self.calculator.model.set_save_metadata(
                save_prefix, save_start, save_frequency
            )

        # other metadata
        n = 0
        epoch = 0
        DATASET_SIZE = len(self.calculator.configs)

        msg = 'Start minimization using optimization method: {}.'.format(self.method)
        logger.info(msg)
        print(msg)

        while True:
            try:
                if self.method in ['LBFGS']:

                    def closure():
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        # forward + backward + optimize
                        loss = self.get_loss()
                        loss.backward()
                        return loss

                    optimizer.step(closure)
                else:
                    optimizer.zero_grad()
                    # forward + backward + optimize
                    loss = self.get_loss()
                    loss.backward()
                    optimizer.step()
                epoch_new = n * self.batch_size // DATASET_SIZE

                if epoch_new > epoch:
                    epoch = epoch_new
                    print('Epoch = {}, loss = {}'.format(epoch, loss))
                n += 1

                if (
                    epoch_new >= save_start
                    and (epoch_new - save_start) % save_frequency == 0
                ):
                    fname = 'model_epoch{}.pkl'.format(epoch)
                    path = os.path.join(save_prefix, fname)
                    self.calculator.model.save(path)

            except StopIteration:
                break

    def get_loss(self):
        loss = 0
        for _ in range(self.batch_size):
            # raise StopIteration error if out of bounds; This will ignore the last
            # chunk of data whose size is smaller than `batch_size`
            inp = self.data_loader.next_element()
            residual = self.get_residual_single_config(
                inp, self.calculator, self.residual_fn, self.residual_data
            )
            c = torch.sum(torch.pow(residual, 2))
            loss += c
        # TODO  maybe divide batch_size elsewhere
        loss /= self.batch_size
        return loss

    def get_residual_single_config(self, inp, calculator, residual_fn, residual_data):

        # prediction data
        results = calculator.compute(inp)
        pred_energy = results['energy']
        pred_forces = results['forces']

        if calculator.use_energy:
            pred = torch.tensor([pred_energy])
            ref = torch.tensor([inp['energy'][0]])
        if calculator.use_forces:
            ref_forces = inp['forces'][0]
            if calculator.use_energy:
                pred = torch.cat((pred, pred_forces.reshape((-1,))))
                ref = torch.cat((ref, ref_forces.reshape((-1,))))
            else:
                pred = pred_forces.reshape((-1,))
                ref = ref_forces.reshape((-1,))

        identifier = inp['name'][0]
        species = inp['species'][0]
        natoms = len(species)

        residual = residual_fn(identifier, natoms, pred, ref, residual_data)

        return residual


class LossError(Exception):
    def __init__(self, msg):
        super(LossError, self).__init__(msg)
        self.msg = msg

    def __expr__(self):
        return self.msg
