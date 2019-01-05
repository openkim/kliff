import sys
import copy
from collections import OrderedDict
from collections import Iterable
import numpy as np
import yaml
import kliff
from kliff.dataset import Configuration
from .parameter import FittingParameter

logger = kliff.logger.get_logger(__name__)


class ComputeArguments(object):
    """ Implementation of code to compute energy, forces, and stress.

    """
    implemented_property = []

    def __init__(self, conf, influence_distance, compute_energy=True,
                 compute_forces=True, compute_stress=False):
        self.conf = conf
        self.influence_distance = influence_distance
        self.compute_energy = compute_energy
        self.compute_forces = compute_forces
        self.compute_stress = compute_stress
        self.compute_property = self.check_compute_property()
        self.results = dict([(i, None) for i in self.implemented_property])

    def refresh(self, influence_distance=None, params=None):
        """ Refresh settings.

        Such as recreating the neighbor list due to the change of cutoff.
        """
        if influence_distance is not None:
            infl_dist = influence_distance
        else:
            try:
                infl_dist = params['influence_distance'].get_value()[0]
            except KeyError:
                raise ParameterError('"influence_distance" not provided by calculator."')
        self.influence_distance = infl_dist

        # NOTE to be filled
        # create neighbor list based on `infl_dist`

    # TODO check also that the conf provide these properties
    def check_compute_property(self):
        def add_to_compute_property(compute_property, name):
            if name not in self.implemented_property:
                raise NotImplementedError(
                    '"{}" not implemented in calculator.'.format(name))
            compute_property.append(name)
        compute_property = []
        if self.compute_energy:
            add_to_compute_property(compute_property, 'energy')
        if self.compute_forces:
            add_to_compute_property(compute_property, 'forces')
        if self.compute_stress:
            add_to_compute_property(compute_property, 'stress')
        return compute_property

    def compute(self, params):
        """ Compute the properties required by the compute flags, and store them
        in self.results.

        Parameters
        ----------
        params: dict of 1D array
            Parameters of the model.

        Example
        -------

        energy = a_func_to_compute_energy()
        forces = a_func_to_compute_forces()
        stress = a_func_to_compute_stress()
        self.results['energy'] = energy
        self.results['forces'] = forces
        self.results['stress'] = stress
        """
        # NOTE to be filled
        pass

    def get_compute_flag(self, name):
        if name in self.compute_property:
            return True
        else:
            return False

    def get_property(self, name):
        if name not in self.compute_property:
            raise CalculatorError(
                'Calculator not initialized to comptue "{}".'.format(name))
        result = self.results[name]
        if isinstance(result, np.ndarray):
            result = result.copy()
        return result

    def get_energy(self):
        return self.get_property('energy')

    def get_forces(self):
        return self.get_property('forces')

    def get_stress(self):
        return self.get_property('stress')

    def get_prediction(self):
        if self.compute_energy:
            energy = self.results['energy']
            pred = np.asarray([energy])
        else:
            pred = np.asarray([])

        if self.compute_forces:
            forces = self.results['forces']
            pred = np.concatenate((pred, forces.ravel()))

        if self.compute_stress:
            stress = self.results['stress']
            pred = np.concatenate((pred, stress))

        return pred

    def get_reference(self):
        if self.compute_energy:
            energy = self.conf.energy
            ref = np.asarray([energy])
        else:
            ref = np.asarray([])

        if self.compute_forces:
            forces = self.conf.forces
            ref = np.concatenate((ref, forces.ravel()))

        if self.compute_stress:
            stress = self.conf.stress
            ref = np.concatenate((ref, stress))

        return ref


class Calculator(object):
    """ Base calculator deal with parameters if model.

    """

    def __init__(self, model_name=None, params_relation_callback=None):
        """
        model_name: str (optional)
            Name of the model.

        param_relations_callback: callback function (optional)
            A callback function to set the relations between parameters, which are
            called each minimization step after the optimizer updates the parameters.

            The function with be given the ModelParameters instance as argument, and
            it can use get_value() and set_value() to manipulate the relation
            between parameters.

            Example
            -------

            In the following example, we set the value of B[0] to 2 * A[0].

            def params_relation(params):
                A = params.get_value('A')
                B = params.get_value('B')
                B[0] = 2 * A[0]
                params.set_value('B', B)
        """

        self.model_name = model_name
        self.params_relation_callback = params_relation_callback

        # NOTE to be filled
        self.params = OrderedDict()
        # set up parameters of the calculator
        # e.g.
        # self.params['sigma'] = Parameter(0.5)
        # self.params['epsilon'] = Parameter(0.4)

        # NOTE to be filled
        self.influence_distance = None

        # NOTE to be filled
        self.compute_arguments_class = ComputeArguments

        # TODO maybe use metaclass to call this automatically after initialization
        # NOTE do not forget to call this in the subclass
        self.fitting_params = self.init_fitting_params(self.params)

    def set_params_relation_callback(self, params_relation_callback):
        """Register a function to set the relation between parameters."""
        self.params_relation_callback = params_relation_callback

    def create(self, configs, use_energy=True, use_forces=True, use_stress=False):
        """Create compute arguments for a set of configurations.

        Parameters
        ----------

        configs: list of Configuration object

        use_energy: bool (optional)
            Whether to require the calculator to compute energy.

        use_forces: bool (optional)
            Whether to require the calculator to compute forces.

        use_stress: bool (optional)
            Whether to require the calculator to compute stress.
        """

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

        self.compute_arguments = []
        infl_dist = self.get_influence_distance()
        for conf, e, f, s in zip(configs, use_energy, use_forces, use_stress):
            ca = self.compute_arguments_class(conf, infl_dist, e, f, s)
            self.compute_arguments.append(ca)

        logger.info('calculator for %d configurations created.', len(configs))
        return self.compute_arguments

    def get_influence_distance(self):
        return self.influence_distance

    def get_compute_arguments(self):
        return self.compute_arguments

    def compute(self, compute_arguments):
        compute_arguments.compute(self.params)

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

    def get_model_params(self, name):
        """ Return a copy of the values of parameter.

        Parameters
        ----------
        name: string
            Name of the parameter.
        """
        if name in self.params:
            return self.params[name].get_value()
        else:
            raise CalculatorError('"{}" is not a parameter of calculator.'.format(name))

    def set_model_params(self, name, value):
        """ Update the parameter values.

        Parameters
        ----------
        name: string
            Name of the parameter.
        value: list of floats
            Value of hte parameter.
        """
        if name in self.params:
            self.params[name].set_value_with_shape_check(value)
        else:
            raise CalculatorError('"{}" is not a parameter of calculator.'.format(name))

    def set_model_params_no_shape_check(self, name, value):
        """ Update the parameter values.

        Parameters
        ----------
        name: string
            Name of the parameter.
        value: list of floats
            Value of hte parameter.
        """
        if name in self.params:
            self.params[name].set_value(value)
        else:
            raise CalculatorError('"{}" is not a parameter of calculator.'.format(name))

    def save_model_params(self, fname=None):
        params = dict()
        for i, j in self.params.items():
            v = j.value
            if isinstance(v, np.ndarray):
                v = v.tolist()
            params[i] = v
        if fname is not None:
            with open(fname, 'w') as fout:
                yaml.dump(params, fout, default_flow_style=False)
        else:
            fout = sys.stdout
            yaml.dump(params, fout, default_flow_style=False)

    def restore_model_params(self, fname):
        with open(fname, 'r') as fin:
            params = yaml.safe_load(fin)
        for key, value in params.items():
            self.set_model_params_no_shape_check(key, value)

    def echo_model_params(self, fname=None):
        """Echo the optimizable parameters. """

        if fname is not None:
            fout = open(fname, 'w')
        else:
            fout = sys.stdout

        print(file=fout)
        print('#'+'='*80, file=fout)
        print('# Available parameters to optimize.')
        print(file=fout)
        name = self.__class__.__name__ if self.model_name is None else self.model_name
        print('# Model:', name, file=fout)
        print('#'+'='*80, file=fout)
        # print('Include the names and the initial guesses (optionally, lower and upper bounds)')
        # print('of the parameters that you want to optimize in the input file.')
        print(file=fout)
        for name, p in self.params.items():
            print('name:', name, file=fout)
            print('value:', p.value, file=fout)
            print('size:', p.size, file=fout)
            print('dtype:', p.dtype, file=fout)
            print('description:', p.description, file=fout)
            print(file=fout)

        if fname is not None:
            fout.close()

    def init_fitting_params(self, params):
        return FittingParameter(params)

    def read_fitting_params(self, fname):
        self.fitting_params.read(fname)

    def set_fitting_params(self, **kwargs):
        self.fitting_params.set(**kwargs)

    def set_one_fitting_params(self, name, settings):
        self.fitting_params.set_one(name, settings)

    def save_fitting_params(self):
        self.fitting_params.save()

    def restore_fitting_params(self):
        self.fitting_params.restore()

    def echo_fitting_params(self, fname=None):
        self.fitting_params.echo_params(fname)

    def get_number_of_opt_params(self):
        return self.fitting_params.get_number_of_opt_params()

    def get_opt_params(self):
        return self.fitting_params.get_opt_params()

    def get_opt_param_value_and_indices(self, k):
        return self.fitting_params.get_opt_param_value_and_indices(k)

    def get_opt_params_bounds(self):
        return self.fitting_params.get_opt_params_bounds()

    def update_model_params(self):
        """ Update from fitting params to model params. """
        for name, attr in self.fitting_params.params.items():
            self.set_model_params_no_shape_check(name, attr['value'])

    def update_params(self, opt_params):
        """ Update from optimizer params to model params. """
        # update from optimzier to fitting params
        self.fitting_params.update_params(opt_params)

        # user-specified relation between parameters
        if self.params_relation_callback is not None:
            self.params_relation_callback(self.fitting_params)

        # update from fitting params to model params
        self.update_model_params()


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


def length_equal(a, b):
    if isinstance(a, Iterable) and isinstance(b, Iterable):
        if len(a) == len(b):
            return True
        else:
            return False
    else:
        return True  # if one is Iterable and the other is not, we treat them
        # as equal since it can be groadcasted


class CalculatorError(Exception):
    def __init__(self, msg):
        super(CalculatorError, self).__init__(msg)
        self.msg = msg

    def __str__(self):
        return self.msg


class InputError(Exception):
    def __init__(self, msg):
        super(InputError, self).__init__(msg)
        self.msg = msg

    def __str__(self):
        return self.msg
