from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from scipy import optimize as opt
from loguru import logger

from kliff.models import KIMModel
from kliff.models.parameter import OptimizingParameters
from kliff.dataset import Dataset

try:
    import torch
    from torch.nn import Parameter, Module
except ImportError:
    torch = None
    Parameter = None
    Module = None
    logger.warning("Torch is not installed. OptimizerTorch will not work correctly.")

# TODO: MPI and geodesic
# try:
#     from mpi4py import MPI
#
#     mpi4py_avail = True
# except ImportError:
#     mpi4py_avail = False
#
try:
    from geodesicLM import geodesiclm

    geodesicLM_avail = True
except ImportError:
    geodesicLM_avail = False


class OptimizerScipy:
    """
    Container class for wrapping the :class:`scipy.optimize.minimize` function. This class is used to contain
    the model, dataset, and parameters to be optimized. The :meth:`minimize` method will call the
    :class:`scipy.optimize.minimize` function and return the optimized parameters. This optimizer is meant
    to be used with physics-based KIM models. For loss functions currently only mean squared error is supported
    as the loss function (:meth:`loss_fn`), but it can be edited by monkey-patching.
    TODO: better API for loss function
    Args:
        model_fn (KIMModel): KIM model function.
        parameters (list): List of parameters to be optimized.
        dataset (Dataset): :class:`kliff.dataset.Dataset` object.
        weights (dict): Weights for the loss function. Default is ``{"energy": 1.0, "forces": 1.0, "stress": 1.0}``.
        optimizer (str): Optimizer to use. Default is ``"L-BFGS-B"``.
        optimizer_kwargs (dict): Keyword arguments for the optimizer. Default is ``None``.
        maxiter (int): Maximum number of iterations. Default is ``1000``.
        tol (float): Tolerance for termination. Default is ``1e-8``.
        target_property (list): List of target properties to be optimized. Default is ``["energy"]``.
        verbose (bool): If ``True``, print out the loss function value at each iteration. Default is ``False``.
    """

    def __init__(
            self,
            model_fn: KIMModel,
            parameters: Union[List[Parameter], List[OptimizingParameters]],
            dataset: Dataset,
            weights: Dict = {"energy": 1.0, "forces": 1.0, "stress": 1.0},
            optimizer=None,
            optimizer_kwargs: Optional[Dict] = None,
            maxiter: Optional[int] = 1000,
            tol: Optional[float] = 1e-8,
            target_property: List = ["energy"],
            verbose: bool = False
    ):
        # TODO: parallelization of the optimizer based on torch and mpi
        self.model_fn = model_fn
        self.parameters = parameters
        self.maxiter = maxiter
        self.tol = tol
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_str = self._get_optimizer(optimizer)
        self.dataset = dataset
        self.weights = weights
        self.loss_agg_func = lambda x, y: np.mean(np.sum((x - y) ** 2))
        self.target_property = target_property
        self.verbose = verbose
        self.options = {"maxiter": self.maxiter, "disp": self.verbose}

    def _get_optimizer(self, optimizer_str):
        """
        Get the optimizer string for :class:`scipy.optimize.minimize`. Based on the optimizer string,
        the corresponding optimizer will be returned.
        :param optimizer_str:
        :return:
        """
        scipy_minimize_methods = [ "Nelder-Mead", "Powell", "CG", "BFGS", "Newton-CG", "L-BFGS-B", "TNC", "COBYLA",
                                   "SLSQP", "trust-constr", "dogleg", "trust-ncg", "trust-exact", "trust-krylov"]
        # scipy_minimize_methods_not_supported_args = ["bounds"]
        # scipy_least_squares_methods = ["trf", "dogbox", "lm", "geodesiclm"]
        # scipy_least_squares_methods_not_supported_args = ["bounds"]

        if optimizer_str in scipy_minimize_methods:
            return optimizer_str
        elif optimizer_str == "geodesiclm":
            if geodesicLM_avail:
                return optimizer_str
            else:
                logger.error("GeodesicLM not loaded")
        else:
            logger.warning(f"Optimization method {optimizer_str} not found in scipy, switching to default L-BFGS")
            return "L-BFGS-B"

        raise ValueError("No optimizer provided")

    def update_parameters(self, new_parameters: List[Union[Parameter, OptimizingParameters]]):
        """
        Copy the new parameters to the model parameters.
        :param new_parameters:
        :return:
        """
        for new_parameter, parameter in zip(new_parameters, self.parameters):
            self.model_fn.copy_parameters(parameter, new_parameter)

    def loss_fn(self, models, dataset, weights, properties):
        """
        Loss function for the optimizer. Currently only mean squared error is supported.
        :param models: :class:`kliff.models.KIMModel` object.
        :param dataset: :class:`kliff.dataset.Dataset` object.
        :param weights: Broadcast-able weights for the loss function.
        :param properties: Properties to use while computing the loss function
        :return:
        """
        loss = 0.0
        for configuration in dataset:
            model_eval = self.model_fn(configuration)
            for property_ in properties:
                loss += weights[property_] * self.loss_agg_func(model_eval[property_],
                                                                configuration.__getattribute__(property_))
        return loss

    def _scipy_loss_wrapper(self, new_parameters):
        """
        Wrapper for the loss function to be used with :class:`scipy.optimize.minimize`.
        :param new_parameters: Parameter input for the loss function.
        :return:
        """
        self.update_parameters(new_parameters)
        loss = self.loss_fn(self.model_fn, self.dataset, self.weights, self.target_property)
        return loss

    def minimize(self, kwargs=None):
        """
        Call minimize function from :class:`scipy.optimize.minimize`.
        :param kwargs: Optional keyword arguments for the optimizer.
        :return: Scipy result object.
        """
        if kwargs:
            kwargs = self.optimizer_kwargs
        else:
            kwargs = {}
        x0 = list(map(lambda x: x[1], self.parameters))
        logger.info(f"Starting with method {self.optimizer_str}")

        bounds = []
        for param in self.parameters:
            bounds.append([param[-2], param[-1]])

        result = opt.minimize(self._scipy_loss_wrapper, np.array(x0), bounds=bounds, method=self.optimizer_str,
                              tol=self.tol, options=self.options, **kwargs)

        # Update model one last time if successful:
        if result.success:
            for in_param, opt_param in zip(self.parameters, result.x):
                self.model_fn.opt_params.set_one(in_param[0], [[opt_param]], suppress_warnings=True)

        return result


class OptimizerTorch:
    """ Optimizer for torch models. This class provides an optimized for the torch models. It is based on the torch
    optimizers and the torch autograd. It can be used for optimizing general pytorch functions with autograd as well.
    The parameters to be are either inferred from the model or can be provided as a list of torch.nn.Parameter objects.
    Dataset has to be an instance of the torch.utils.data.Dataset class or kliff_torch.dataset.Dataset class.
    Weight is expected to be a dictionary with the keys "energy", "forces", and "stress" and the values should be a
    valid torch broadcastable array. The target property is a list of the properties to be optimized. The default is
    energy. The loss_agg_func is a function that takes the model output and the target and returns a scalar value.
    epochs is the number of iterations to be performed.
    Models should either return a named tuple with  three values (energy, forces, stress) or energy if forces are to
    be computed using autograd.
        params:
            model_fn: model function to be optimized. Has to be a torch.nn.Module
            parameters: list of parameters to optimize.
            dataset: dataset to optimize on
            weights: weights for the different properties
            optimizer: optimizer to use
            optimizer_kwargs: kwargs for the optimizer
            epochs: maximum number of iterations
            target_property: list of properties to optimize on

    """

    def __init__(
            self,
            model_fn: Union[Callable, Module],
            dataset: Dataset,
            weights: Dict = None,
            optimizer: Optional[Callable] = "Adam",
            optimizer_kwargs: Optional[Dict] = None,
            epochs: Optional[int] = 100,
            target_property: List = None,
            parameters: Union[List[Parameter], List[OptimizingParameters]] = None
    ):
        # TODO: parallelization of the optimizer based on torch and mpi
        self.model_fn = model_fn
        if parameters:
            self.parameters = parameters
        elif isinstance(model_fn, Module):
            try:
                self.parameters = list(model_fn.parameters())  # If model_fn torch module
            except TypeError:
                self.parameters = list(model_fn.get_parameters())  # if model_fn is a TrainingWheel
        else:
            raise ValueError("No parameters provided")

        if not weights:
            weights = {"energy": 1.0, "forces": 1.0, "stress": 1.0}
        self.weights = weights

        self.epochs = epochs
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = self._get_optimizer(optimizer)
        self.dataset = dataset
        self.weights = weights
        self.loss_agg_func = lambda x, y: torch.mean(torch.sum((x - y) ** 2))
        if not target_property:
            target_property = ["energy"]
        self.target_property = target_property
        self.print_loss = False

    def _get_optimizer(self, optimizer_str):
        """
        Get optimizer from string or torch optimizer object. Same as in kliff.optimizer.Optimizer
        :param optimizer_str: String or torch optimizer object.
        :return:
        """
        torch_minimize_methods = ["Adadelta", "Adagrad", "Adam", "SparseAdam", "Adamax", "ASGD", "LBFGS",
                                  "RMSprop", "Rprop", "SGD"]
        if isinstance(optimizer_str, str):
            if optimizer_str in torch_minimize_methods:
                return getattr(torch.optim, optimizer_str)(self.parameters)
            else:
                logger.warning(f"Optimization method {optimizer_str} not found currently supported list, "
                               f"switching to default Adam")
                return torch.optim.Adam(self.parameters)
        elif "torch.optim" in str(type(optimizer_str)):
            return optimizer_str
        else:
            raise ValueError("No optimizer provided")

    def update_parameters(self, new_parameters: List[Parameter]):
        """
        Copy new parameters to the model parameters.
        :param new_parameters:
        :return:
        """
        for new_parameter, parameter in zip(new_parameters, self.parameters):
            with torch.no_grad():
                parameter.copy_(new_parameter)

    def loss_fn(self, model, dataset, weights, properties):
        """
        Loss function for the optimizer. This function is called by the optimizer to compute the loss.
        Currently only the mean squared error is supported.
        :param model: :class:`torch.nn.Module` to be optimized
        :param dataset: :class:`torch.utils.data.Dataset` to optimize on
        :param weights: Broadcastable weights for the different properties
        :param properties: Properties to optimize on (energy, forces, stress)
        :return: loss (torch.tensor)
        """
        loss = torch.tensor(0.0)
        for configuration in dataset:
            model_eval = model(configuration)
            for property_ in properties:
                loss += weights[property_] * self.loss_agg_func(model_eval[property_],
                                                                torch.as_tensor(
                                                                    configuration.__getattribute__(property_)))
        return loss

    def minimize(self, kwargs=None):
        """
        Minimize the model function. It contains a simple PyTorch loop to compute loss, and update the parameters.
        :param kwargs:
        :return:
        """
        if kwargs:
            kwargs = self.optimizer_kwargs
        else:
            kwargs = {}

        logger.info(f"Starting with method {self.optimizer}")
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model_fn, self.dataset, self.weights, self.target_property)
            loss.backward()
            self.optimizer.step()
            if self.print_loss:
                print(f"Epoch {epoch} loss {loss}")
        return self.parameters


def _check_residual_data(data: Dict[str, Any], default: Dict[str, Any]):
    """
    Check whether user provided residual data is valid, and add default values if not
    provided.
    """
    if data is not None:
        for key, value in data.items():
            if key not in default:
                raise LossError(
                    f"Expect the keys of `residual_data` to be one or combinations of "
                    f"{', '.join(default.keys())}; got {key}. "
                )
            else:
                default[key] = value

    return default


def _check_compute_flag(calculator, residual_data):
    """
    Check whether compute flag correctly set when the corresponding weight in residual
    data is 0.
    """
    ew = residual_data["energy_weight"]
    fw = residual_data["forces_weight"]
    sw = residual_data["stress_weight"]
    msg = (
        '"{0}_weight" set to "{1}". Seems you do not want to use {0} in the fitting. '
        'You can set "use_{0}" in "calculator.create()" to "False" to speed up the '
        "fitting."
    )

    if calculator.use_energy and ew < 1e-12:
        logger.warning(msg.format("energy", ew))
    if calculator.use_forces and fw < 1e-12:
        logger.warning(msg.format("forces", fw))
    if calculator.use_stress and sw < 1e-12:
        logger.warning(msg.format("stress", sw))


class LossError(Exception):
    def __init__(self, msg):
        super(LossError, self).__init__(msg)
        self.msg = msg
