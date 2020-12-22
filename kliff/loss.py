import logging
import os

import numpy as np
import scipy.optimize

from . import parallel
from .error import InputError, report_import_error
from .log import log_entry

try:
    import torch

    torch_avail = True
except ImportError:
    torch_avail = False

try:
    from mpi4py import MPI

    mpi4py_avail = True
except ImportError:
    mpi4py_avail = False


logger = logging.getLogger(__name__)


def energy_forces_residual(identifier, natoms, weight, prediction, reference, data):
    r"""A residual function using both energy and forces.

    Parameters
    ----------
    identifier: str
        identifier of the configuration, i.e. path to the file

    natoms: int
        number of atoms in the configuration

    weight: float
        weight for the configuration

    prediction: 1D array
        prediction computed by calculator

    reference: 1D array
        references data for the prediction

    data: dict
        Additional data to be used to calculate the residual. Supported key value pairs
        are:

        - energy_weight: float (default: 1)
        - forces_weight: float (default: 1)
        - normalize_by_atoms: bool (default: True)

    Return
    ------
    residual: 1D array

    Note
    ----
    The length of `prediction` and `reference` (call it `S`) are the same, and it
    depends on `use_energy` and `use_forces` in Calculator. Assume the
    configuration contains of `N` atoms.

    1) If `use_energy == True` and `use_forces == False`, then `S = 1`.
    `prediction[0]` is the potential energy computed by the calculator, and
    `reference[0]` is the reference energy.

    2) If `use_energy == False` and `use_forces == True`, then `S = 3N`.
    `prediction[3*i+0]`, `prediction[3*i+1]`, and `prediction[3*i+2]` are the
    x, y, and z component of the forces on atom i in the configuration, respectively.
    Correspondingly, `reference` is the 3N concatenated reference forces.


    3) If `use_energy == True` and `use_forces == True`, then `S = 3N + 1`.
    `prediction[0]` is the potential energy computed by the calculator, and
    `reference[0]` is the reference energy.
    `prediction[3*i+1]`, `prediction[3*i+2]`, and `prediction[3*i+3]` are the
    x, y, and z component of the forces on atom i in the configuration, respectively.
    Correspondingly, `reference` is the 3N concatenated reference forces.
    """

    # prepare weight based on user provided data
    energy_weight = data["energy_weight"]
    forces_weight = data["forces_weight"]
    normalize = data["normalize_by_natoms"]
    if normalize:
        energy_weight /= natoms
        forces_weight /= natoms

    # obtain residual and properly normalize it
    residual = weight * (prediction - reference)
    residual[0] *= energy_weight
    residual[1:] *= forces_weight

    return residual


def forces_residual(conf_id, natoms, prediction, reference, data):
    data["energy_weight"] = 0
    return energy_forces_residual(conf_id, natoms, prediction, reference, data)


def energy_residual(conf_id, natoms, prediction, reference, data):
    data["forces_weight"] = 0
    return energy_forces_residual(conf_id, natoms, prediction, reference, data)


class Loss(object):
    """Objective function class to conduct the optimization."""

    def __new__(
        self,
        calculator,
        nprocs=1,
        residual_fn=energy_forces_residual,
        residual_data=None,
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
        self.check_computation_flag(calculator, data)

        calc_type = calculator.__class__.__name__

        if "Torch" in calc_type:
            return LossNeuralNetworkModel(calculator, nprocs, residual_fn, data)
        else:
            return LossPhysicsMotivatedModel(calculator, nprocs, residual_fn, data)

    @staticmethod
    def check_residual_data(data):
        default = {
            "energy_weight": 1.0,
            "forces_weight": 1.0,
            "stress_weight": 1.0,
            "normalize_by_natoms": True,
        }
        if data is not None:
            for key, value in data.items():
                if key not in default:
                    msg = '"{}" not supported by "residual_data".'.format(key)
                    log_entry(logger, msg, level="error")
                    raise InputError(msg)
                else:
                    default[key] = value
        return default

    @staticmethod
    def check_computation_flag(calculator, data):
        ew = data["energy_weight"]
        fw = data["forces_weight"]
        sw = data["stress_weight"]
        msg = (
            '"{0}_weight" set to "{1}". Seems you do not want to use {0} in the fitting. '
            'You can set "use_{0}" of "calculator.create()" to "False" to speed up the '
            "fitting."
        )

        if calculator.use_energy and ew < 1e-12:
            msg = msg.format("energy", ew)
            log_entry(logger, msg, level="warning")
        if calculator.use_forces and fw < 1e-12:
            msg = msg.format("forces", fw)
            log_entry(logger, msg, level="warning")
        if calculator.use_stress and sw < 1e-12:
            msg = msg.format("stress", sw)
            log_entry(logger, msg, level="warning")


class LossPhysicsMotivatedModel(object):
    r"""Objective function class to conduct the optimization for PM models."""

    scipy_minimize_methods = [
        "Nelder-Mead",
        "Powell",
        "CG",
        "BFGS",
        "Newton-CG",
        "L-BFGS-B",
        "TNC",
        "COBYLA",
        "SLSQP",
        "trust-constr",
        "dogleg",
        "trust-ncg",
        "trust-exact",
        "trust-krylov",
    ]
    scipy_minimize_methods_not_supported_args = ["bounds"]
    scipy_least_squares_methods = ["trf", "dogbox", "lm", "geodesiclm"]
    scipy_least_squares_methods_not_supported_args = ["bounds"]

    def __init__(
        self,
        calculator,
        nprocs=1,
        residual_fn=energy_forces_residual,
        residual_data=None,
    ):

        self.calculator = calculator
        self.nprocs = nprocs

        self.residual_fn = residual_fn
        self.residual_data = residual_data if residual_data is not None else dict()
        self.calculator_type = calculator.__class__.__name__

        if self.calculator_type == "WrapperCalculator":
            calculators = self.calculator.calculators
        else:
            calculators = [self.calculator]
        for calc in calculators:
            infl_dist = calc.model.get_influence_distance()
            cas = calc.get_compute_arguments()
            for ca in cas:
                ca.refresh(infl_dist)

        logger.info('"{}" instantiated.'.format(self.__class__.__name__))

    def minimize(self, method, **kwargs):
        r"""Minimize the loss.

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

        msg = "Start minimization using method: {}.".format(method)
        log_entry(logger, msg, level="info")

        result = self.scipy_optimize(method, **kwargs)

        msg = "Finish minimization using method: {}.".format(method)
        log_entry(logger, msg, level="info")

        # update final optimized parameters
        self.calculator.update_opt_params(result.x)

        return result

    def adjust_kwargs(self, method, **kwargs):

        if method in self.scipy_least_squares_methods:

            # check support status
            for i in self.scipy_least_squares_methods_not_supported_args:
                if i in kwargs:
                    msg = (
                        'Argument "{}" should not be set via the "minimize" method. '
                        "It it set internally.".format(i)
                    )
                    log_entry(logger, msg, level="error")
                    raise LossError(msg)

            # adjust bounds
            if self.calculator.has_opt_params_bounds():
                if method in ["trf", "dogbox"]:
                    bounds = self.calculator.get_opt_params_bounds()
                    lb = [b[0] if b[0] is not None else -np.inf for b in bounds]
                    ub = [b[1] if b[1] is not None else np.inf for b in bounds]
                    bounds = (lb, ub)
                    kwargs["bounds"] = bounds
                else:
                    msg = 'Method "{}" cannot handle bounds.'.format(method)
                    log_entry(logger, msg, level="error")
                    raise LossError(msg)

        elif method in self.scipy_minimize_methods:

            # check support status
            for i in self.scipy_minimize_methods_not_supported_args:
                if i in kwargs:
                    msg = (
                        'Argument "{}" should not be set via the "minimize" method. '
                        "It it set internally.".format(i)
                    )
                    log_entry(logger, msg, level="error")
                    raise LossError(msg)

            # adjust bounds
            if self.calculator.has_opt_params_bounds():
                if method in ["L-BFGS-B", "TNC", "SLSQP"]:
                    bounds = self.calculator.get_opt_params_bounds()
                    kwargs["bounds"] = bounds
                else:
                    msg = 'Method "{}" cannot handle bounds.'.format(method)
                    log_entry(logger, msg, level="error")
                    raise LossError(msg)
        else:
            msg = 'minimization method "{}" not supported.'.format(method)
            log_entry(logger, msg, level="error")
            raise LossError(msg)

        return kwargs

    def scipy_optimize(self, method, **kwargs):

        size = parallel.get_MPI_world_size()

        if size > 1:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

            msg = "Running in MPI mode with {} processes.".format(size)
            log_entry(logger, msg, level="info", print_end="\n\n")

            if self.nprocs > 1:
                msg = (
                    'Argument "nprocs = {}" provided at initialization is ignored. When '
                    "running in MPI mode, the number of processes provided along with "
                    'the "mpiexec" (or "mpirun") command is used.'.format(self.nprocs)
                )
                log_entry(logger, msg, level="warning")

            x = self.calculator.get_opt_params()
            if method in self.scipy_least_squares_methods:
                # geodesic LM
                if method == "geodesiclm":
                    from geodesicLM import geodesiclm

                    minimize_fn = geodesiclm
                else:
                    minimize_fn = scipy.optimize.least_squares
                func = self.get_residual_MPI
            elif method in self.scipy_minimize_methods:
                minimize_fn = scipy.optimize.minimize
                func = self.get_loss_MPI

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
                msg = "Running in serial mode."
                log_entry(logger, msg, level="info", print_end="\n\n")
            else:
                msg = "Running in multiprocessing mode with {} processes.".format(
                    self.nprocs
                )
                log_entry(logger, msg, level="info", print_end="\n\n")

                # Maybe one thinks he is using MPI because nprocs is used
                if mpi4py_avail:
                    msg = (
                        '"mpi4y" detected. If you try to run in MPI mode, you should '
                        'execute your code via "mpiexec" (or "mpirun"). If not, ignore '
                        "this message."
                    )
                    log_entry(logger, msg, level="warning")

            x = self.calculator.get_opt_params()
            if method in self.scipy_least_squares_methods:
                if method == "geodesiclm":
                    from geodesicLM import geodesiclm

                    minimize_fn = geodesiclm
                else:
                    minimize_fn = scipy.optimize.least_squares
                func = self.get_residual
            elif method in self.scipy_minimize_methods:
                minimize_fn = scipy.optimize.minimize
                func = self.get_loss

            result = minimize_fn(func, x, method=method, **kwargs)
            return result

    def get_residual(self, x):
        r"""Compute the residual.

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

        if self.calculator_type == "WrapperCalculator":
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
        r"""Compute the loss.

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

    # NOTE this function can be called only once, not need to call each time
    # get_residual_MPI is called
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
        weight = conf.get_weight()

        residual = residual_fn(identifier, natoms, weight, pred, ref, residual_data)

        return residual


class LossNeuralNetworkModel(object):
    r"""Objective function class to conduct the optimization for ML models.

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

    torch_minimize_methods = [
        "Adadelta",
        "Adagrad",
        "Adam",
        "SparseAdam",
        "Adamax",
        "ASGD",
        "LBFGS",
        "RMSprop",
        "Rprop",
        "SGD",
    ]

    def __init__(
        self,
        calculator,
        nprocs=1,
        residual_fn=energy_forces_residual,
        residual_data=None,
    ):

        if not torch_avail:
            report_import_error("pytorch")

        self.calculator = calculator
        self.nprocs = nprocs

        self.residual_fn = residual_fn
        self.residual_data = residual_data if residual_data is not None else dict()

        self.calculator_type = calculator.__class__.__name__

        self.optimizer = None
        self.optimizer_stat_path = None

        logger.info('"{}" instantiated.'.format(self.__class__.__name__))

    def minimize(
        self, method, batch_size=100, num_epochs=1000, start_epoch=0, **kwargs
    ):
        r"""Minimize the loss.

        Parameters
        ----------
        method: str
            PyTorch optimization methods, and available ones are:
            [`Adadelta`, `Adagrad`, `Adam`, `SparseAdam`, `Adamax`, `ASGD`, `LBFGS`,
            `RMSprop`, `Rprop`, `SGD`]
            See also: https://pytorch.org/docs/stable/optim.html

        batch_size: int
            Number of configurations used in in each minimization step.

        num_epochs: int
            Number of epochs to carry out the minimization.

        start_epoch: int
            The starting epoch number. This is typically 0, but if continuing a training,
            it is useful to set this to the last epoch number of the previous training.

        kwargs: dict
            Extra keyword arguments that can be used by the PyTorch optimizer.
        """
        if method not in self.torch_minimize_methods:
            msg = 'Minimization method "{}" not supported.'.format(method)
            log_entry(logger, msg, level="error")
            raise LossError(msg)

        self.method = method
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.start_epoch = start_epoch

        # data loader
        loader = self.calculator.get_compute_arguments(batch_size)

        # model save metadata
        save_prefix = self.calculator.model.save_prefix
        save_start = self.calculator.model.save_start
        save_frequency = self.calculator.model.save_frequency
        if save_prefix is None or save_start is None or save_frequency is None:
            logger.info(
                "Model saving meta data not set by user. Now set it to "
                '"prefix=./kliff_saved_model", "start=1", and "frequency=10".'
            )
            save_prefix = os.path.join(os.getcwd(), "kliff_saved_model")
            save_start = 1
            save_frequency = 10
            self.calculator.model.set_save_metadata(
                save_prefix, save_start, save_frequency
            )

        msg = "Start minimization using optimization method: {}.".format(self.method)
        log_entry(logger, msg, level="info")

        # optimizing
        try:
            self.optimizer = getattr(torch.optim, method)(
                self.calculator.model.parameters(), **kwargs
            )
            if self.optimizer_stat_path is not None:
                self._load_optimizer_stat(self.optimizer_stat_path)

        except TypeError as e:
            print(str(e))
            idx = str(e).index("argument '") + 10
            err_arg = str(e)[idx:].strip("'")
            msg = 'Argument "{}" not supported by optimizer "{}".'.format(
                err_arg, method
            )
            log_entry(logger, msg, level="error")
            raise InputError(msg)

        epoch = 0
        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):

            # get the loss without any optimization if continue a training
            if self.start_epoch != 0 and epoch == self.start_epoch:
                epoch_loss = self._get_loss_epoch(loader)
                print("Epoch = {:<6d}  loss = {:.10e}".format(epoch, epoch_loss))

            else:
                epoch_loss = 0
                for ib, batch in enumerate(loader):

                    def closure():
                        self.optimizer.zero_grad()
                        loss = self.get_loss_batch(batch)
                        loss.backward()
                        return loss

                    loss = self.optimizer.step(closure)
                    # float() such that do not accumulate history, more memory friendly
                    epoch_loss += float(loss)

                print("Epoch = {:<6d}  loss = {:.10e}".format(epoch, epoch_loss))
                if epoch >= save_start and (epoch - save_start) % save_frequency == 0:
                    path = os.path.join(save_prefix, "model_epoch{}.pkl".format(epoch))
                    self.calculator.model.save(path)

        # print loss from final parameter and save last epoch
        epoch += 1
        epoch_loss = self._get_loss_epoch(loader)
        print("Epoch = {:<6d}  loss = {:.10e}".format(epoch, epoch_loss))
        path = os.path.join(save_prefix, "model_epoch{}.pkl".format(epoch))
        self.calculator.model.save(path)

        msg = "Finish minimization using optimization method: {}.".format(self.method)
        log_entry(logger, msg, level="info")

    def _get_loss_epoch(self, loader):
        epoch_loss = 0
        for ib, batch in enumerate(loader):
            loss = self.get_loss_batch(batch)
            epoch_loss += float(loss)
        return epoch_loss

    def get_loss_batch(self, batch, normalize=True):
        r"""Compute the loss of a batch of samples.

        Parameters
        ----------
        batch: list
            A list of samples.

        normalize: bool
            If `True`, normalize the loss of the batch by the size of the batch.
            Note, how to normalize the loss of a single configuration is determined by the
            `normalize` flag of the `residual_data` argument of :mod:`kliff.Loss`.
        """
        results = self.calculator.compute(batch)
        energy_batch = results["energy"]
        forces_batch = results["forces"]
        stress_batch = results["stress"]

        if forces_batch is None:
            forces_batch = [None] * len(batch)
        if stress_batch is None:
            stress_batch = [None] * len(batch)

        # Instead of loss_batch = 0 and loss_batch += loss in the loop, the below one may
        # be faster, considering chain rule it needs to take derivatives.
        # Anyway, it is minimal. Don't worry about it.
        losses = []
        for sample, energy, forces, stress in zip(
            batch, energy_batch, forces_batch, stress_batch
        ):
            loss = self.get_loss_single_config(sample, energy, forces, stress)
            losses.append(loss)
        loss_batch = torch.stack(losses).sum()
        if normalize:
            loss_batch /= len(batch)

        return loss_batch

    def get_loss_single_config(self, sample, pred_energy, pred_forces, pred_stress):

        if self.calculator.use_energy:
            pred = pred_energy.reshape(-1)  # reshape scalar as 1D tensor
            ref = sample["energy"].reshape(-1)

        if self.calculator.use_forces:
            ref_forces = sample["forces"]
            if self.calculator.use_energy:
                pred = torch.cat((pred, pred_forces.reshape(-1)))
                ref = torch.cat((ref, ref_forces.reshape(-1)))
            else:
                pred = pred_forces.reshape(-1)
                ref = ref_forces.reshape(-1)

        if self.calculator.use_stress:
            ref_stress = sample["stress"]
            if self.calculator.use_energy or self.calculator.use_stress:
                pred = torch.cat((pred, pred_stress.reshape(-1)))
                ref = torch.cat((ref, ref_stress.reshape(-1)))
            else:
                pred = pred_stress.reshape(-1)
                ref = ref_stress.reshape(-1)

        conf = sample["configuration"]
        identifier = conf.get_identifier()
        natoms = conf.get_number_of_atoms()
        weight = conf.get_weight()

        residual = self.residual_fn(
            identifier, natoms, weight, pred, ref, self.residual_data
        )
        loss = torch.sum(torch.pow(residual, 2))

        return loss

    def save_optimizer_stat(self, path="optimizer_stat.pkl"):
        torch.save(self.optimizer.state_dict(), path)

    def load_optimizer_stat(self, path="optimizer_stat.pkl"):
        self.optimizer_stat_path = path

    def _load_optimizer_stat(self, path):
        self.optimizer.load_state_dict(torch.load(path))


class LossError(Exception):
    def __init__(self, msg):
        super(LossError, self).__init__(msg)
        self.msg = msg

    def __expr__(self):
        return self.msg
