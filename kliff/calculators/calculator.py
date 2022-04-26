from collections.abc import Sequence
from typing import Any, Dict, List, Union

import numpy as np
from loguru import logger

from kliff.dataset.dataset import Configuration
from kliff.models.model import Model
from kliff.utils import length_equal


class Calculator:
    """
    Calculator to compute properties (e.g. energy and forces) using a potential model for
    each configuration in the dataset.

    The properties, together with the corresponding reference data stored in the
    configurations, are provided to :class:`~kliff.loss.Loss` to construct a loss
    function for the optimizer.

    In the reverse direction, a calculator grabs the new parameters from the optimizer
    and update the model with the new parameters.

    Args:
        model: An instance of :class:`~kliff.models.Model`.
    """

    def __init__(self, model: Model):
        self.model = model

        self.compute_arguments = None

    def create(
        self,
        configs: List[Configuration],
        use_energy: Union[List[bool], bool] = True,
        use_forces: Union[List[bool], bool] = True,
        use_stress: Union[List[bool], bool] = False,
    ):
        """
        Create compute arguments for a collection of configurations.

        By compute arguments, we mean the information needed by a model to carry on a
        calculation, such as the coordinates, species, cutoff distance, neighbor list,
        etc. Each configuration has its own compute arguments, and this function creates
        the compute arguments for all the configurations in ``configs``.

        Args:
            configs: atomic configurations.
                instances.
            use_energy: Whether to require the calculator to compute energy.
                If a list of bool is provided, each component is for one configuration
                in `configs`. If a bool is provided, it is applied to all configurations.
            use_forces: Whether to require the calculator to compute forces.
                If a list of bool is provided, each component is for one configuration
                in `configs`. If a bool is provided, it is applied to all configurations.
            use_stress: Whether to require the calculator to compute stress.
                If a list of bool is provided, each component is for one configuration
                in `configs`. If a bool is provided, it is applied to all configurations.
        """

        self.use_energy = use_energy
        self.use_forces = use_forces
        self.use_stress = use_stress

        if isinstance(configs, Configuration):
            configs = [configs]

        if use_energy and not length_equal(configs, use_energy):
            raise CalculatorError(
                "Expect length of `configs` and `use_energy` have the same size; got "
                f"{len(configs)} and {len(use_energy)}."
            )
        if use_forces and not length_equal(configs, use_forces):
            raise CalculatorError(
                "Expect length of `configs` and `use_forces` have the same size; got "
                f"{len(configs)} and {len(use_forces)}."
            )
        if use_stress and not length_equal(configs, use_stress):
            raise CalculatorError(
                "Expect length of `configs` and `use_stresses` have the same size; got "
                f"{len(configs)} and {len(use_stress)}."
            )

        N = len(configs)
        if not isinstance(use_energy, Sequence):
            use_energy = [use_energy] * N
        if not isinstance(use_forces, Sequence):
            use_forces = [use_forces] * N
        if not isinstance(use_stress, Sequence):
            use_stress = [use_stress] * N

        supported_species = self.model.get_supported_species()
        infl_dist = self.model.get_influence_distance()

        ca_class = self.model.get_compute_argument_class()

        self.compute_arguments = []
        for conf, e, f, s in zip(configs, use_energy, use_forces, use_stress):
            if self._is_kim_model():
                kim_ca = self.model.create_a_kim_compute_argument()
                ca = ca_class(kim_ca, conf, supported_species, infl_dist, e, f, s)
            else:
                ca = ca_class(conf, supported_species, infl_dist, e, f, s)
            self.compute_arguments.append(ca)

        logger.info(f"Create calculator for {len(configs)} configurations.")
        return self.compute_arguments

    def get_compute_arguments(self) -> List[Any]:
        """
        Return a list of compute arguments, each associated with a configuration.
        """
        return self.compute_arguments

    # TODO, maybe change the compute_argument.compute api of kim models, such
    #   that it accept params as the argument, instead of kim_model
    def compute(self, compute_arguments) -> Dict[str, Any]:
        """
        Compute the properties given the compute arguments associated with a
        configuration.

        Args:
            compute_arguments: A compute arguments instance for a configuration.

        Returns:
            A dictionary of properties, with keys of `energy`, `forces` and `stress`,
                values of float or np.array.
        """
        if self._is_kim_model():
            compute_arguments.compute(self.model.kim_model)
        else:
            compute_arguments.compute(self.model.get_model_params())
        return compute_arguments.results

    def get_energy(self, compute_arguments) -> float:
        """
        Get the energy of a configuration.

        Args:
            compute_arguments: A compute arguments instance for a configuration.

        Returns:
            The energy of the configuration associated with the compute arguments.
        """
        return compute_arguments.get_energy()

    def get_forces(self, compute_arguments) -> np.array:
        """
        Get the forces of a configuration.

        Args:
            compute_arguments: A compute arguments instance for a configuration.

        Returns:
            Forces on atoms. 2D array of shape (N, 3), where N is the number of atoms
                in the configuration.
        """
        return compute_arguments.get_forces()

    def get_stress(self, compute_arguments) -> np.array:
        r"""
        Get the stress of a configuration.

        Args:
            compute_arguments: A compute arguments instance for a configuration.

        Returns:
            Virial stress of the configuration associated with the compute arguments,
            1D array with 6 component.
            The returned stress is in Voigt notation, i.e. this function returns:
            [:math:`\sigma_{xx}, \sigma_{yy}, \sigma_{zz}, \sigma_{yz}, \sigma_{xy},
            \sigma_{xz}`]
        """
        return compute_arguments.get_stress()

    # NOTE, we keep the interface of get_prediction, get_opt_params, has_opt_params
    # etc... to make it easier to implement WrapperCalculators, where multiple
    # calculators are used.
    def get_prediction(self, compute_arguments) -> np.array:
        """
        Get the prediction of all properties that are requested to compute.

        The `energy`, `forces`, and `stress` are each flattened to a 1D array, and then
        concatenated (in the order of `energy`, `forces`, and `stress`) to form the
        prediction.  Depending on the values of ``use_energy``, ``use_forces``, and
        ``use_stress`` that are provided in :meth:`create`, one or more of `energy`,
        `forces` and `stress` may not be included in the prediction.

        Args:
            compute_arguments: A compute arguments instance for a configuration.

        Returns:
            1D array of predictions. For a configuration of N atoms:
            - If ``use_energy``, ``use_forces``, and ``use_stress`` are all ``True``, the
              size of prediction is `1+3N+6`, with the 1st component the `energy`, the 2nd
              to 1+3N components the flattened `forces` on N atoms, and the remaining 6
              components the Voigt `stress`.
            - If one or more of ``use_energy``, ``use_forces``, and ``use_stress`` is
              ``False``, its corresponding value is removed from prediction, and the size
              of prediction shrinks accordingly. For example, if both ``use_energy`` and
              ``use_stress`` are ``True`` but ``use_forces`` is ``False``, then the size
              of prediction is `1+6`, with the 1st component the `energy`, and the 2nd to
              7th components the Voigt stress.
        """
        return compute_arguments.get_prediction()

    def get_reference(self, compute_arguments) -> np.array:
        """
        Get the reference data of all properties that are requested to compute.

        Same as :meth:`get_prediction`, the `energy`, `forces`, and `stress` are each
        flattened to a 1D array, and then concatenated (in the order of `energy`,
        `forces`, and `stress`) to form the reference. Depending on the values of
        ``use_energy``, ``use_forces``, and ``use_stress`` that are provided in
        :meth:`create`, one or more of `energy`, `forces` and `stress` may not be included
        in the reference.

        Args:
            compute_arguments: A compute arguments instance for a configuration.

        Returns:
            1D array of reference values. For a configuration of N atoms:
            - If ``use_energy``, ``use_forces``, and ``use_stress`` are all ``True``, the
              size of reference is `1+3N+6`, with the 1st component the `energy`, the 2nd
              to 1+3N components the flattened `forces` on N atoms, and the remaining 6
              components the Voigt `stress`.
            - If one or more of ``use_energy``, ``use_forces``, and ``use_stress`` is
              ``False``, its corresponding value is removed from reference, and the size
              of reference shrinks accordingly. For example, if both ``use_energy`` and
              ``use_stress`` are ``True`` but ``use_forces`` is ``False``, then the size
              of reference is `1+6`, with the 1st component the `energy`, and the 2nd to
              7th components the Voigt stress.
        """
        return compute_arguments.get_reference()

    def get_opt_params(self) -> np.array:
        """
        Return a list of optimizing parameters.

        The optimizing parameters is a list consisting of the values of the model
        parameters that is set to fit via :meth:`kliff.models.Model.set_opt_params` or
        :meth:`kliff.models.Model.set_one_opt_param`.  The returned value is typically
        passed to an optimizer as the initial values to carry out the optimization.
        """
        return self.model.get_opt_params()

    def get_num_opt_params(self) -> int:
        """
        Return the number of optimizing parameters.
        """
        return self.model.get_num_opt_params()

    def has_opt_params_bounds(self):
        """
        Return a bool to indicate whether there are parameters whose bounds are
        provided.
        """
        return self.model.has_opt_params_bounds()

    def get_opt_params_bounds(self):
        """
        Return the lower and upper bounds for the optimizing parameters.

        The returned value is a list of (lower, upper) tuples. Each tuple contains the
        lower and upper bounds for the corresponding parameter obtained from
        :meth:`get_opt_params`. ``None`` for ``lower`` or ``upper`` means that no bound
        should be applied.
        """
        return self.model.get_opt_params_bounds()

    def update_model_params(self, opt_params: List[float]):
        """
        Update model parameters from a list of opt_params.

        This is the reverse of :meth:`get_opt_params`.

        Args:
            opt_params: Optimizing parameters.
        """
        self.model.update_model_params(opt_params)

    def _is_kim_model(self):
        return self.model.__class__.__name__ == "KIMModel"


class _WrapperCalculator(object):
    """
    Wrapper to deal with the fitting of multiple models.

    Args:
        calculators: calculators to wrap. Each calculator for a different model,
        but should be for the same dataset.
    """

    # TODO methods like get_prediction needs to be implemented. The prediction of
    #  energy (force) should be the sum of each calculator.
    def __init__(self, calculators=List[Calculator]):
        """
        Parameters
        ----------

        calculators: instance of Calculator
        """
        self.calculators = calculators
        self._start_end = self._set_start_end()

    def _set_start_end(self):
        """
        Compute the start and end indices of the `opt_params` of each calculator in the
        `opt_params` of the wrapper calculator.
        """
        start_end = []
        i = 0
        for calc in self.calculators:
            n = calc.get_num_opt_params()
            start = i
            end = i + n
            start_end.append((start, end))
            i += n
        return start_end

    def get_compute_arguments(self):
        all_cas = []
        for calc in self.calculators:
            cas = calc.get_compute_arguments()
            all_cas.extend(cas)
        return all_cas

    def get_num_opt_params(self):
        return sum([calc.get_num_opt_params() for calc in self.calculators])

    def get_opt_params(self):
        return np.concatenate([calc.get_opt_params() for calc in self.calculators])

    def get_opt_params_bounds(self):
        bounds = []
        for calc in self.calculators:
            b = calc.get_opt_params_bounds()
            bounds.extend(b)
        return bounds

    def update_model_params(self, opt_params):
        for i, calc in enumerate(self.calculators):
            start, end = self._start_end[i]
            p = opt_params[start:end]
            calc.update_model_params(p)

    def get_calculator_list(self):
        """
        Create a list of calculators.

        Each calculator has `number of configurations` copies in the list.

        In such, we can easily associate the compute arguments with the calculator.
        """
        calc_list = []
        for calc in self.calculators:
            N = len(calc.get_compute_arguments())
            calc_list.extend([calc] * N)
        return calc_list


class CalculatorError(Exception):
    def __init__(self, msg):
        super(CalculatorError, self).__init__(msg)
        self.msg = msg
