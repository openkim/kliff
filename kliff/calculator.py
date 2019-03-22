from collections.abc import Iterable
import kliff
from kliff.dataset import Configuration
from kliff.utils import length_equal

logger = kliff.logger.get_logger(__name__)


class Calculator(object):
    """ Base calculator deal with parameters if model.

    """

    def __init__(self, model):
        """
        model: str
            Name of the model.
        """

        self.model = model

    def create(self, configs, use_energy=True, use_forces=True, use_stress=False):
        """Create compute arguments for a set of configurations.

        Parameters
        ----------

        configs: list of Configuration object

        use_energy: bool or list of bools (optional)
            Whether to require the calculator to compute energy.

        use_forces: bool or list of bools (optional)
            Whether to require the calculator to compute forces.

        use_stress: bool or list of bools (optional)
            Whether to require the calculator to compute stress.
        """

        # TODO  need not be registered as self
        self.use_energy = use_energy
        self.use_forces = use_forces
        self.use_stress = use_stress

        if isinstance(configs, Configuration):
            configs = [configs]

        if not length_equal(configs, use_energy):
            raise InputError(
                'Lenghs of arguments "configs" and "use_energy" not equal.')
        if not length_equal(configs, use_forces):
            raise InputError(
                'Lenghs of arguments "configs" and "use_forces" not equal.')
        if not length_equal(configs, use_stress):
            raise InputError(
                'Lenghs of arguments "configs" and "use_stress" not equal.')

        N = len(configs)
        if not isinstance(use_energy, Iterable):
            use_energy = [use_energy for _ in range(N)]
        if not isinstance(use_forces, Iterable):
            use_forces = [use_forces for _ in range(N)]
        if not isinstance(use_stress, Iterable):
            use_stress = [use_stress for _ in range(N)]

        # update parameters from fitting parameters to model parameters
        # possibly, influence will be updated if the user set related parameters
        self.model.update_model_params()

        supported_species = self.model.get_supported_species()
        infl_dist = self.model.get_influence_distance()

        self.compute_arguments = []
        for conf, e, f, s in zip(configs, use_energy, use_forces, use_stress):
            if self.is_kim_model():
                kim_ca = self.model.create_a_kim_compute_argument()
                ca = self.model.compute_arguments_class(
                    kim_ca, conf, supported_species, infl_dist, e, f, s)
            else:
                ca = self.model.compute_arguments_class(
                    conf, supported_species, infl_dist, e, f, s)
            self.compute_arguments.append(ca)

        logger.info('calculator for %d configurations created.', len(configs))
        return self.compute_arguments

    def is_kim_model(self):
        return self.model.__class__.__name__ == 'KIM'

    def get_compute_arguments(self):
        return self.compute_arguments

    # TODO, maybe change the compute_argument.compute api of kim models, such
    # that it accept params as the argument, insteand of kim_model
    def compute(self, compute_arguments):
        if self.is_kim_model():
            compute_arguments.compute(self.model.kim_model)
        else:
            compute_arguments.compute(self.model.params)
        return compute_arguments.results

    def get_energy(self, compute_arguments):
        return compute_arguments.get_energy()

    def get_forces(self, compute_arguments):
        return compute_arguments.get_forces()

    def get_stress(self, compute_arguments):
        return compute_arguments.get_stress()

    def get_prediction(self, compute_arguments):
        return compute_arguments.get_prediction()

    def get_reference(self, compute_arguments):
        return compute_arguments.get_reference()

    def get_opt_params(self):
        return self.model.get_opt_params()

    def get_opt_params_bounds(self):
        return self.model.get_opt_params_bounds()

    def update_opt_params(self, opt_params):
        """ Update from optimizer params to model params. """
        self.model.update_fitting_params(opt_params)
        self.model.apply_params_relation()
        self.model.update_model_params()


class WrapperCalculator(object):
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
            end = i+N
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
            calc_list.extend([calc]*N)
        return calc_list
