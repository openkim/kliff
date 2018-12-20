import sys
import copy
import warnings
from collections import OrderedDict
from collections import Iterable
import numpy as np
import yaml
import kliff
from kliff.dataset import Configuration

logger = kliff.logger.get_logger(__name__)


class ComputeArguments(object):
    """ Implementation of code to compute energy, forces, and stress.

    """
    implemented_property = []

    def __init__(self, conf, compute_energy=True, compute_forces=True,
                 compute_stress=False):
        self.conf = conf
        self.compute_energy = compute_energy
        self.compute_forces = compute_forces
        self.compute_stress = compute_stress
        self.compute_property = self.check_compute_property()
        self.results = dict([(i, None) for i in self.implemented_property])

    # TODO maybe pass params as an argument, also for refresh
    def refresh(self):
        """ Refresh settings.

        Such as recreating the neighbor list due to the change of cutoff.
        """
        # NOTE to be filled
        pass

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
        for conf, e, f, s in zip(configs, use_energy, use_forces, use_stress):
            ca = self.compute_arguments_class(conf, e, f, s)
            self.compute_arguments.append(ca)

        logger.info('calculator for %d configurations created.', len(configs))
        return self.compute_arguments

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

    def get_model_params(self):
        """ Return a copy of the parameters of the calculator.
        """
        return copy.deepcopy(self.params)

    def set_model_params(self, key, value):
        """ Update the parameters in the calculator.
        """
        if key in self.params:
            self.params[key].set_value(value)
        else:
            raise CalculatorError('"{}" is not a parameter of calculator.' .format(key))

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
            self.set_model_params(key, value)

    def echo_model_params(self):
        """Echo the optimizable parameters to stdout.
        """
        name = self.__class__.__name__ if self.model_name is None else self.model_name
        print()
        print('#'+'='*80)
        print('# Available parameters to optimize.')
        print()
        print('# Model:', name)
        print('#'+'='*80)
        # print('Include the names and the initial guesses (optionally, lower and upper bounds)')
        # print('of the parameters that you want to optimize in the input file.')
        print()
        for name, p in self.params.items():
            print('name:', name)
            print('value:', p.value)
            print('size:', p.size)
            print('dtype:', p.dtype)
            print('description:', p.description)
            print()

    def init_fitting_params(self, params):
        return FittingParameter(params)

    def read_fitting_params(self, fname):
        self.fitting_params.read(fname)

    def set_fitting_params(self, **kwargs):
        self.fitting_params.set(**kwargs)

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
        for key, attr in self.fitting_params.params.items():
            self.set_model_params(key, attr['value'])

    def update_params(self, opt_params):
        """ Update from optimizer params to model params. """
        # update from optimzier to fitting params
        self.fitting_params.update_params(opt_params)

        # update from fitting params to model params
        self.update_model_params()

        # TODO should be moved above update_model_params
        # user-specified relations between parameters
        if self.params_relation_callback is not None:
            self.params_relation_callback(self)


# TODO take a look at proporty decorator
class Parameter(object):

    def __init__(self, value, dtype='double', description=None):
        self.set_value_check_shape(value)
        self.dtype = dtype
        self.description = description

    def get_value(self):
        return self.value.copy()

    def get_size(self):
        return len(self.value)

    def get_dtype(self):
        return self.dtype

    def get_description(self):
        return self.description

    def set_value(self, value):
        self.value = np.asarray(value)
        self.size = len(value)

    def set_value_check_shape(self, value):
        if isinstance(value, Iterable):
            if any(isinstance(i, Iterable) for i in value):
                raise ParameterError('Parameter should be a scalar or 1D array.')
            self.value = np.asarray(value)
            self.size = len(value)
        else:
            raise ParameterError('Parameter should be a scalar or 1D array.')

    def to_string(self):
        s = 'value: {}\n'.format(np.array_str(self.value))
        s += 'size: {}\n'.format(self.size)
        s += 'dtype: {}\n'.format(self.dtype)
        s += 'description: {}\n'.format(self.description)
        return s


class FittingParameter(object):
    """Class of model parameters that will be optimzied.

    It interacts with optimizer to provide initial guesses of parameters and
    receive updated paramters from the optimizer.
    """

    def __init__(self, model_params):
        """
        Parameters
        ----------

        model_params: OrderDict
            All the paramters of a model (calculator).
        """
        self.model_params = model_params

        # key: parameter name
        # values: {'size', 'value', 'use_default', 'fix', 'lower_bound', 'upper_bound'}
        self.params = OrderedDict()

        # components of params that are optimized
        self._index = []

    def read(self, fname):
        """Read the parameters that will be optimized. (Interface to user)

        Each parameter is a 1D array, and each component of the parameter array should
        be listed in a new line. Each line can contains 1, 2, or 3 elements.

        1st element: float or `DEFAULT`
            Initial guess of the parameter component. If `DEFAULT` (case insensitive),
            the value from the calculator is used as the intial guess.

        The 2nd and 3rd elements are optional. If 2 elements are provided:

        2nd element: `FIX` (case insensitive)
            If `FIX`, the corresponding component will not be optimized.

        If 3 elements are provided:

        2nd element: float or `INF` (case insensitive)
            Lower bound of the parameter. `INF` indicates that the lower bound is
            negative infinite, i.e. no lower bound is applied.

        3rd element: float or `INF` (case insensitive)
            Upper bound of the parameter. `INF` indicates that the upper bound is
            positive infinite, i.e. no upper bound is applied.

        An alternative is self.set().

        Parameters
        ----------

        fname: str
          name of file that includes the fitting parameters

        Example
        -------

        A
        DEFAULT
        1.1

        B
        DEFAULT FIX
        1.1     FIX

        C
        DEFAULT  0.1  INF
        1.0      INF  2.1
        2.0      FIX
        """

        with open(fname, 'r') as fin:
            lines = fin.readlines()
            lines = remove_comments(lines)
        num_line = 0
        while num_line < len(lines):
            line = lines[num_line].strip()
            num_line += 1
            if line in self.params:
                msg = 'file "{}", line {}. Parameter "{}" already set.' .format(
                    fname, num_line, line)
                warnings.warn(msg, category=Warning)
            if line not in self.model_params:
                raise InputError('file "{}", line {}. Parameter "{}" not supported '
                                 'by calculator.'.format(fname, num_line, line))
            name = line
            size = self.model_params[name].get_size()
            settings = []
            for j in range(size):
                settings.append(lines[num_line].split())
                num_line += 1
            self.set_one(name, settings)

    def set(self, **kwargs):
        """Set the parameters that will be optimized. (Interface to user)

        One or more parameters can be set. Each argument is for one parameter, where
        the argument name is the parameter name, the value of the argument is the
        settings (including intial value, fix flag, lower bound, and upper bound).

        The value of the argument should be a list of list, where each inner list is for
        one component of the parameter, which can contain 1, 2, or 3 elements.
        See self.read() for the options of the elements.

        Example
        -------

        instance.set(A = [['DEFAULT'],
                          [2.0, 1.0, 3.0]],
                     B = [[1.0, 'FIX'],
                          [2.0, 'INF', 3.0]])
        """
        for name, settings in kwargs.items():
            if name in self.params:
                msg = 'Parameter "{}" already set.'.format(name)
                warnings.warn(msg, category=Warning)
            if name not in self.model_params:
                raise InputError('Parameter "{}" not supported.'.format(name))
            self.set_one(name, settings)

    def set_one(self, name, settings):
        """Set one parameter that will be optimized.

        The name of the parameter should be given as the first entry of a list
        (or tuple), and then each data line should be given in in a list.

        Parameters
        ----------

        name, string
            name of a fitting parameter

        settings, list of list
            initial value, flag to fix a parameter, lower and upper bounds of a parameter

        Example
        -------
            name = 'param_A'
            settings = [['default', 0, 20],
                       [2.0, 'fix'],
                       [2.2, 'inf', 3.3]]
          instance.set_one(name, settings)
        """

        size = self.model_params[name].get_size()
        if len(settings) != size:
            raise InputError(
                'Incorrect number of initial values for paramter "{}".'.format(name))

        tmp_dict = {'size': size,
                    'value': [None for _ in range(size)],
                    'use_default': [False for _ in range(size)],
                    'fix': [False for _ in range(size)],
                    'lower_bound': [None for _ in range(size)],
                    'upper_bound': [None for _ in range(size)]}
        self.params[name] = tmp_dict

        for j, line in enumerate(settings):
            num_items = len(line)
            if num_items == 1:
                self._read_1_item(name, j, line)
            elif num_items == 2:
                self._read_2_item(name, j, line)
            elif num_items == 3:
                self._read_3_item(name, j, line)
            else:
                raise InputError('More than 3 elements listed at data line '
                                 '{} for parameter "{}".'.format(j+1, name))
            self._check_bounds(name)

        self._set_index(name)

    def echo_params(self, fname=None, print_size=True):
        """Print the optimizing parameters to stdout or file.

        Parameters
        ----------

        fname: str
          Name of the file to print the optimizing parameters. If None, printing
          to stdout.

        print_size: bool
          Flag to indicate whether print the size of parameter. Recall that a
          parameter may have one or more values.
        """

        if fname:
            fout = open(fname, 'w')
        else:
            fout = sys.stdout

        print('#'+'='*80, file=fout)
        print('# Model parameters that are optimzied.', file=fout)
        print('#'+'='*80, file=fout)
        print(file=fout)

        for name, attr in self.params.items():
            if print_size:
                print(name, attr['size'], file=fout)
            else:
                print(name, file=fout)

            for i in range(attr['size']):
                print('{:24.16e}'.format(attr['value'][i]), end=' ', file=fout)

                if attr['fix'][i]:
                    print('fix', end=' ', file=fout)

                lb = attr['lower_bound'][i]
                ub = attr['upper_bound'][i]
                has_lb = lb is not None
                has_ub = ub is not None
                has_bounds = has_lb or has_ub
                if has_bounds:
                    if has_lb:
                        print('{:24.16e}'.format(lb), end=' ', file=fout)
                    else:
                        print('None', end=' ', file=fout)
                    if has_ub:
                        print('{:24.16e}'.format(ub), end=' ', file=fout)
                    else:
                        print('None', end=' ', file=fout)
                print(file=fout)

            print(file=fout)

        if fname:
            fout.close()

    def get_names(self):
        return self.params.keys()

    def get_size(self, name):
        return self.params[name]['size']

    def get_value(self, name):
        return self.params[name]['value'].copy()

#    def get_index(self, name):
#        return self.params[name]['index']

    def get_lower_bound(self, name):
        return self.params[name]['lower_bound'].copy()

    def get_upper_bound(self, name):
        return self.params[name]['upper_bound'].copy()

    def get_fix(self, name):
        return self.params[name]['fix'].copy()

    def set_value(self, name, value):
        self.params[name]['value'] = value

    def get_number_of_opt_params(self):
        return len(self._index)

    def update_params(self, opt_x):
        """ Update parameter values from optimzier. (Interface to optimizer)

        This is the opposite operation of get_opt_params().

        Parameters
        ----------

        opt_x, list of floats
          parameter values from the optimizer.

        """
        for k, val in enumerate(opt_x):
            name = self._index[k].name
            c_idx = self._index[k].c_idx
            self.params[name]['value'][c_idx] = val

    def get_opt_params(self):
        """Nest all parameter values (except the fix ones) to a list.

        This is the opposite operation of update_params(). This can be fed to the
        optimizer as the starting parameters.

        Return
        ------
          A list of nested optimizing parameter values.
        """
        opt_x0 = []
        for idx in self._index:
            name = idx.name
            c_idx = idx.c_idx
            opt_x0.append(self.params[name]['value'][c_idx])
        if len(opt_x0) == 0:
            raise ParameterError('No parameters specified to optimize.')
        return np.asarray(opt_x0)

    def get_opt_param_value_and_indices(self, k):
        """Get the `value`, `parameter_index`, and `component_index` of an optimizing
        parameter given the slot `k`."""
        name = self._index[k].name
        p_idx = self._index[k].p_idx
        c_idx = self._index[k].c_idx
        value = self.params[name]['value'][c_idx]
        return value, p_idx, c_idx

    def get_opt_params_bounds(self):
        """ Get the lower and upper parameter bounds. """
        bounds = []
        for idx in self._index:
            name = idx.name
            c_idx = idx.c_idx
            lower = self.params[name]['lower_bound'][c_idx]
            upper = self.params[name]['upper_bound'][c_idx]
            bounds.append([lower, upper])
        return bounds

    def _read_1_item(self, name, j, line):
        self._read_1st_item(name, j, line[0])

    def _read_2_item(self, name, j, line):
        self._read_1st_item(name, j, line[0])
        if line[1].lower() == 'fix':
            self.params[name]['fix'][j] = True
        else:
            raise InputError(
                'Data at line {} of {} corrupted.'.format(j+1, name))

    def _read_3_item(self, name, j, line):
        self._read_1st_item(name, j, line[0])

        if (line[1] is not None) and (
                not (isinstance(line[1], str) and line[1].lower() == 'none')):
            try:
                self.params[name]['lower_bound'][j] = float(line[1])
            except ValueError as e:
                raise InputError(
                    '{}.\nData at line {} of {} corrupted.'.format(e, j+1, name))

        if (line[2] is not None) and (
                not (isinstance(line[2], str) and line[2].lower() == 'none')):
            try:
                self.params[name]['upper_bound'][j] = float(line[2])
            except ValueError as e:
                raise InputError(
                    '{}.\nData at line {} of {} corrupted.'.format(e, j+1, name))

    def _read_1st_item(self, name, j, first):
        if isinstance(first, str) and first.lower() == 'default':
            self.params[name]['use_default'][j] = True
            model_value = self.model_params[name].get_value()
            self.params[name]['value'][j] = model_value[j]
        else:
            try:
                self.params[name]['value'][j] = float(first)
            except ValueError as e:
                raise InputError(
                    '{}.\nData at line {} of {} corrupted.'.format(e, j+1, name))

    def _check_bounds(self, name):
        """Check whether the initial guess of a paramter is within its lower and
        upper bounds.
        """
        attr = self.params[name]
        for i in range(attr['size']):
            lower_bound = attr['lower_bound'][i]
            upper_bound = attr['upper_bound'][i]
            value = attr['value'][i]
            if lower_bound is not None:
                if value < lower_bound:
                    raise InputError('Initial guess at line {} of parameter "{}" '
                                     'out of bounds.'.format(i+1, name))
            if upper_bound is not None:
                if value > upper_bound:
                    raise InputError('Initial guess at line {} of parameter "{}" '
                                     'out of bounds.'.format(i+1, name))

    def _set_index(self, name):
        """Check whether a parameter component will be optimized or not (by
        checking its 'fix' attribute). If yes, include it in the index list.

        Given a parameter and its values such as:

        PARAM_FREE_B
        1.1
        2.2  fix
        4.4  3.3  5.5

        the first slot (1.1) and the third slot (4.4) will be included in self._index,
        and later be optimized.
        """

        # TODO check if there is alternative so as not to use OrderedDict
        p_idx = list(self.model_params.keys()).index(name)
        size = self.params[name]['size']
        fix = self.params[name]['fix']
        for c_idx in range(size):
            if not fix[c_idx]:
                idx = Index(name, p_idx, c_idx)
                # check whether already in self._index
                already_in = False
                for k, i in enumerate(self._index):
                    if idx == i:
                        already_in = k
                        break
                if already_in is not False:
                    self._index[already_in] = idx
                else:
                    self._index.append(idx)


class Index(object):

    def __init__(self, name, parameter_index=None, component_index=None):
        self.name = name
        self.parameter_index = self.p_idx = parameter_index
        self.component_index = self.c_idx = component_index

    def set_parameter_index(self, index):
        self.parameter_index = self.p_idx = index

    def set_component_index(self, index):
        self.component_index = self.c_idx = index

    def __expr__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ != other.__dict__
        else:
            return True


def length_equal(a, b):
    if isinstance(a, Iterable) and isinstance(b, Iterable):
        if len(a) == len(b):
            return True
        else:
            return False
    else:
        return True  # if one is Iterable and the other is not, we treat them
        # as equal since it can be groadcasted


def remove_comments(lines):
    """Remove lines in a string list that start with # and content after #."""
    processed_lines = []
    for line in lines:
        line = line.strip()
        if not line or line[0] == '#':
            continue
        if '#' in line:
            line = line[0:line.index('#')]
        processed_lines.append(line)
    return processed_lines


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


class ParameterError(Exception):
    def __init__(self, msg):
        super(ParameterError, self).__init__(msg)
        self.msg = msg

    def __str__(self):
        return self.msg
