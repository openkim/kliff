from collections.abc import Iterable
import kliff
from kliff.dataset import Configuration
from kliff.utils import length_equal

logger = kliff.logger.get_logger(__name__)


class Calculator:
    """ Calcualtor class to exchange information between model and an optimizer.

    It computes the `energy`, `forces`, etc. using a potential model, and provides
    these properties, together with the corresponding reference data, to
    :class:`~kliff.loss.Loss` to construct a cost function for the optimizer.
    In the reverse direction, it grab the new parameters from the optimizer and
    update the model with the new parameters.

    Parameters
    ----------
    model: obj
        An instance of the :class:`~kliff.models.Model` class.
    """

    def __init__(self, model):
        self.model = model

    def create(self, configs, use_energy=True, use_forces=True, use_stress=False):
        """Create compute arguments for a collection of configurations.

        By compute argugments, we mean the information needed by a model to carry on
        a calculation, such as the coordiantes, speices, cutoff distance, neighbor
        list, etc. Each configuration has its own compute arguments, and this
        function creates the compute arguments for all the configurations in
        ``configs``.

        Parameters
        ----------
        configs: list
            A list of :class:`~kliff.dataset.Configuration` instances.

        use_energy: list of bools (optional)
            Whether to require the calculator to compute energy. Each component is
            for one configuration in ``configs``. If a bool instead of a list is
            provided, it is applied to all configurations.

        use_forces: list of bools (optional)
            Whether to require the calculator to compute forces. Each component is
            for one configuration in ``configs``. If a bool instead of a list is
            provided, it is applied to all configurations.

        use_stress: list of bools (optional)
            Whether to require the calculator to compute stress. Each component is
            for one configuration in ``configs``. If a bool instead of a list is
            provided, it is applied to all configurations.
        """

        self.use_energy = use_energy
        self.use_forces = use_forces
        self.use_stress = use_stress

        if isinstance(configs, Configuration):
            configs = [configs]

        if not length_equal(configs, use_energy):
            raise InputError('Lenghs of arguments "configs" and "use_energy" not equal.')
        if not length_equal(configs, use_forces):
            raise InputError('Lenghs of arguments "configs" and "use_forces" not equal.')
        if not length_equal(configs, use_stress):
            raise InputError('Lenghs of arguments "configs" and "use_stress" not equal.')

        N = len(configs)
        if not isinstance(use_energy, Iterable):
            use_energy = [use_energy for _ in range(N)]
        if not isinstance(use_forces, Iterable):
            use_forces = [use_forces for _ in range(N)]
        if not isinstance(use_stress, Iterable):
            use_stress = [use_stress for _ in range(N)]

        # update parameters from fitting parameters to model parameters
        # influence distance may be updated if the user set the related parameters
        self.model.update_model_params()

        supported_species = self.model.get_supported_species()
        infl_dist = self.model.get_influence_distance()

        self.compute_arguments = []
        for conf, e, f, s in zip(configs, use_energy, use_forces, use_stress):
            if self._is_kim_model():
                kim_ca = self.model.create_a_kim_compute_argument()
                ca = self.model.compute_arguments_class(
                    kim_ca, conf, supported_species, infl_dist, e, f, s
                )
            else:
                ca = self.model.compute_arguments_class(
                    conf, supported_species, infl_dist, e, f, s
                )
            self.compute_arguments.append(ca)

        logger.info('calculator for %d configurations created.', len(configs))
        return self.compute_arguments

    def _is_kim_model(self):
        return self.model.__class__.__name__ == 'KIM'

    def get_compute_arguments(self):
        """Return a list of compute arguments, each associated with a configuration.
        """
        return self.compute_arguments

    # TODO, maybe change the compute_argument.compute api of kim models, such
    # that it accept params as the argument, insteand of kim_model
    def compute(self, compute_arguments):
        """Compute the properties given the compute arguments assciated with a
        configuration.

        Parameters
        ----------
        compute_arguments: obj
            A compute arguments instance for a configuration.

        Return
        ------
        dict
            A dictionary of properties, with keys of `energy`, `forces` and `stress`.
        """
        if self._is_kim_model():
            compute_arguments.compute(self.model.kim_model)
        else:
            compute_arguments.compute(self.model.params)
        return compute_arguments.results

    # TODO, possibly, and an argument `reference` to get reference values
    def get_energy(self, compute_arguments):
        """Get the energy of a configuration.

        Parameters
        ----------
        compute_arguments: obj
            A compute arguments instance for a configuration.

        Return
        ------
        float
            The energy of the configuration associated with the compute arguments.
        """
        return compute_arguments.get_energy()

    def get_forces(self, compute_arguments):
        """Get the forces of a configuration.

        Parameters
        ----------
        compute_arguments: obj
            A compute arguments instance for a configuration.

        Return
        ------
        2D array
            The forces of the configuration associated with the compute arguments.
        """

        return compute_arguments.get_forces()

    def get_stress(self, compute_arguments):
        """Get the stress of a configuration.

        Parameters
        ----------
        compute_arguments: obj
            A compute arguments instance for a configuration.

        Return
        ------
        1D array
            The stress of the configuration associated with the compute arguments.
            The returned stress is in Voigt notation, i.e. this function returns:
            [:math:`\sigma_{xx}, \sigma_{yy}, \sigma_{zz}, \sigma_{yz}, \sigma_{xy},
            \sigma_{xz}`]
        """

        return compute_arguments.get_stress()

    # TODO maybe move this and `get_referene` to loss.
    def get_prediction(self, compute_arguments):
        """Get the prediction of all properties that are requested to compute.

        The `energy`, `forces`, and `stress` are each flattened to a 1D array, and
        then concatenated (in the order of `energy`, `forces`, and `stress`) to form
        the prediction.
        Depending on the values of ``use_energy``, ``use_forces``, and ``use_stress``
        that are provided in :meth:`create`, one or more of `energy`, `forces` and
        `stress` may not be included in the prediction.

        Parameters
        ----------
        compute_arguments: obj
            A compute arguments instance for a configuration.

        Return
        ------
        1D array
            For a configuration of N atoms:

            - If ``use_energy``, ``use_forces``, and ``use_stress`` are all ``True``,
              the size of prediction is `1+3N+6`, with the 1st component the
              `energy`, the 2nd to 1+3N components the flattened `forces` on N atoms,
              and the remaining 6 components the Voigt `stress`.
            - If one or more of ``use_energy``, ``use_forces``, and ``use_stress`` is
              ``False``, its corresponding value is removed from prediction, and
              the size of prediction shrinks accordingly. For example, if both
              ``use_energy`` and ``use_stress`` are ``True`` but ``use_forces`` is
              ``False``, then the size of prediction is `1+6`, with the 1st
              component the `energy`, and the 2nd to 7th components the Voigt stress.
        """
        return compute_arguments.get_prediction()

    def get_reference(self, compute_arguments):
        """Get the reference data of all properties that are requested to compute.

        Same as :meth:`get_prediction`, the `energy`, `forces`, and `stress` are each
        flattened to a 1D array, and then concatenated (in the order of `energy`,
        `forces`, and `stress`) to form the reference.
        Depending on the values of ``use_energy``, ``use_forces``, and ``use_stress``
        that are provided in :meth:`create`, one or more of `energy`, `forces` and
        `stress` may not be included in the reference.

        Parameters
        ----------
        compute_arguments: obj
            A compute arguments instance for a configuration.

        Return
        ------
        1D array
            For a configuration of N atoms:

            - If ``use_energy``, ``use_forces``, and ``use_stress`` are all ``True``,
              the size of reference is `1+3N+6`, with the 1st component the
              `energy`, the 2nd to 1+3N components the flattened `forces` on N atoms,
              and the remaining 6 components the Voigt `stress`.
            - If one or more of ``use_energy``, ``use_forces``, and ``use_stress`` is
              ``False``, its corresponding value is removed from reference, and
              the size of reference shrinks accordingly. For example, if both
              ``use_energy`` and ``use_stress`` are ``True`` but ``use_forces`` is
              ``False``, then the size of reference is `1+6`, with the 1st
              component the `energy`, and the 2nd to 7th components the Voigt stress.
        """
        return compute_arguments.get_reference()

    def get_opt_params(self):
        """Return a list of optimizing parameters.

        The optimizing parameters is a list consisting of the values of the model
        parameters that is set to fit via :meth:`kliff.models.Model.set_fitting_params`
        or :meth:`kliff.models.Model.set_one_fitting_param`.
        The returned value is typically passed to an optimizer as the initial values
        to carry out the optimization.
        """
        return self.model.get_opt_params()

    def get_opt_params_bounds(self):
        """Return the lower and upper bounds for the optimizing parameters.

        The returnd value is a list of (lower, upper) tuples. Each tuple contains the
        lower and upper bounds for the corresponding parameter obtained from
        :meth:`get_opt_params`. ``None`` for ``lower`` or ``upper`` means that no
        bound should be applied.
        """
        return self.model.get_opt_params_bounds()

    def update_opt_params(self, opt_params):
        """Update the optimizing parameters from optimizer to model.

        This function is the reverse of :meth:`get_opt_params`.

        Parameters
        ----------
        opt_params: list
            Optimizing parameters.
        """
        self.model.update_fitting_params(opt_params)
        self.model.apply_params_relation()
        self.model.update_model_params()


class _WrapperCalculator(object):
    """Wrapper to deal with the fitting of multiple models."""

    def __init__(self, *calculators):
        """
        Parameters
        ----------

        calculators: instance of Calculator
        """
        self.calculators = calculators
        self._start_end = self._set_start_end()

    def _set_start_end(self):
        """Compute the start and end indices of the `opt_params` of each calculator
        in the `opt_params` of the wrapper calculator."""
        start_end = []
        i = 0
        for calc in self.calculators:
            N = calc.get_number_of_opt_params()
            start = i
            end = i + N
            start_end.append((start, end))
            i += N
        return start_end

    def get_compute_arguments(self):
        all_cas = []
        for calc in self.calculators:
            cas = calc.get_compute_arguments()
            all_cas.extend(cas)
        return all_cas

    def get_number_of_opt_params(self):
        N = 0
        for calc in self.calculators:
            N += calc.get_number_of_opt_params()
        return N

    def get_opt_params(self):
        opt_params = []
        for calc in self.calculators:
            p = calc.get_opt_params()
            opt_params.extend(p)
        return opt_params

    def get_opt_params_bounds(self):
        bounds = []
        for calc in self.calculators:
            b = calc.get_opt_params_bounds()
            bounds.extend(b)
        return bounds

    def update_model_params(self):
        for calc in self.calculators:
            calc.update_model_params()

    def update_params(self, opt_params):
        for i, calc in enumerate(self.calculators):
            start, end = self._start_end[i]
            p = opt_params[start:end]
            calc.update_params(p)

    def get_calculator_list(self):
        """Create a list of calculators.

        Each calculator has `number of configurations` copies in the list.
        """
        calc_list = []
        for calc in self.calculators:
            N = len(calc.get_compute_arguments())
            calc_list.extend([calc] * N)
        return calc_list
