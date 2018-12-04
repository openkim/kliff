import sys
import copy
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
            raise NotComputeError(
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

    def register_param_relations_callback(self, param_relations_callback):
        """ Register a function to set the relations between parameters. """
        self.param_relations_callback = param_relations_callback

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
            self.params[key].value = value
        else:
            raise LJCalculatorError(
                '"{}" is not a parameter of calculator.' .format(key))

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

    def set_fitting_params(self, lines):
        self.fitting_params.set(lines)

    def save_fitting_params(self):
        self.fitting_params.save()

    def restore_fitting_params(self):
        self.fitting_params.restore()

    def echo_fitting_params(self, fname=None):
        self.fitting_params.echo_params(fname)

    def get_opt_params(self):
        return self.fitting_params.get_opt_params()

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

        # user-specified relations between parameters
        if self.params_relation_callback is not None:
            self.params_relation_callback(self)


# TODO take a look at proporty decorator
class Parameter(object):

    def __init__(self, value, dtype='double', description=None):

        if isinstance(value, Iterable):
            if any(isinstance(i, Iterable) for i in value):
                raise ParameterError(
                    'Parameter should be a scalar or 1D array.')
            value = np.asarray(value)
            size = len(value)
        else:
            raise ParameterError('Parameter should be a scalar or 1D array.')

        self.value = value
        self.size = size
        self.dtype = dtype
        self.description = description

    def get_value(self):
        if isinstance(self.value, Iterable):
            return self.value.copy()

    def set_value(self, value):
        self.value = value

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
        # values: {'value', 'use_default', 'fix', 'lower_bound', 'upper_bound'}
        self.params = OrderedDict()

        # index of optimizing sub-parameter (recall that a parameter is a 1D array,
        # and a sub-parameter is a component in the array.)
        # a list of dictionary:
        # key: parameter name
        # values: 'p_idx', and 'c_idx'
        # p_idx: index of parameter
        # c_idx: index of component of the parameter
        self._index = []

    def read(self, fname):
        """Read the initial values of parameters. (Interface to user)

        An alternative is set_param().
        For a given model parameter, one or multiple initial values may be required,
        and each must be given in a new line. For each line, the initial guess value
        is mandatory, where `KIM` (case insensitive) can be given to use the value
        from the KIM model. Optionally, `fix` can be followed not to optimize this
        parameters, or lower and upper bounds can be given to limit the parameter
        value in the range. Note that lower or upper bounds may not be effective if
        the optimizer does not support it. The following are valid input examples.

        Parameters
        ----------

        fname: str
          name of the input file where the optimizing parameters are listed

        Examples
        --------

        A
        KIM
        1.1

        B
        KIM  fix
        1.1  fix

        C
        KIM  0.1  2.1
        1.0  0.1  2.1
        2.0  fix
        """

        with open(fname, 'r') as fin:
            lines = fin.readlines()
            lines = remove_comments(lines)
        num_line = 0
        while num_line < len(lines):
            line = lines[num_line].strip()
            num_line += 1
            if line in self.params:
                raise InputError('line: {} file: {}. Parameter {} already '
                                 'set.'.format(num_line, fname, line))
            if line not in self.model_params:
                raise InputError('line: {} file: {}. Parameter {} not supported by '
                                 'the potential model.'.format(num_line, fname, line))
            name = line
            size = self.model_params[name].size
            param_lines = [name]
            for j in range(size):
                param_lines.append(lines[num_line].split())
                num_line += 1
            self.set(param_lines)

    def set(self, lines):
        """Set the parameters that will be optimized. (Interface to user)

        An alternative is Read().
        The name of the parameter should be given as the first entry of a list
        (or tuple), and then each data line should be given in in a list.

        Parameters
        ----------

        lines, str
          optimizing parameter initial values, settings

        Example
        -------

          param_A = ['PARAM_FREE_A',
                     ['kim', 0, 20],
                     [2.0, 'fix'],
                     [2.2, 1.1, 3.3]
                    ]
          instance_of_this_class.set(param_A)
        """

        name = lines[0].strip()
# NOTE we want to use set to perturbe params so as to compute Fisher information
# matrix, where the following two lines are annoying
# Maybe issue an warning
#    if name in self.params:
#      raise InputError('Parameter {} already set.'.format(name))

        if name not in self.model_params:
            raise InputError(
                'Parameter "{}" not supported by the potential model.'.format(name))
        size = self.model_params[name].size
        if len(lines)-1 != size:
            raise InputError(
                'Incorrect number of data lines for paramter "{}".'.format(name))

        # index of parameter in model_params (which is the same as in KIM object)
        index = list(self.model_params.keys()).index(name)

        tmp_dict = {
            'size': size,
            'index': index,
            'value': np.array([None for i in range(size)]),
            'use_default': np.array([False for i in range(size)]),
            'fix': np.array([False for i in range(size)]),
            'lower_bound': np.array([None for i in range(size)]),
            'upper_bound': np.array([None for i in range(size)])
        }
        self.params[name] = tmp_dict

        for j in range(size):
            line = lines[j+1]
            num_items = len(line)
            if num_items == 1:
                self._read_1_item(name, j, line)
            elif num_items == 2:
                self._read_2_item(name, j, line)
            elif num_items == 3:
                self._read_3_item(name, j, line)
            else:
                raise InputError('More than 3 iterms listed at data line '
                                 '{} for parameter {}.'.format(j+1, name))
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

        # print(file=fout)
        print('#'+'='*80, file=fout)
        print('# Model parameters that are optimzied.', file=fout)
        print('#'+'='*80, file=fout)
        print(file=fout)

        for name, attr in self.params.items():
            if print_size:
                print(name, attr['size'], file=fout)
            else:
                print(name, file=fout)

            # print ('index:', attr['index'], file=fout)

            for i in range(attr['size']):
                print('{:24.16e}'.format(attr['value'][i]), end=' ', file=fout)
                if not attr['fix'][i] and attr['lower_bound'][i] == None:
                    print(file=fout)   # print new line if only given value
                if attr['fix'][i]:
                    print('fix', file=fout)
                if attr['lower_bound'][i] != None:
                    print('{:24.16e}'.format(
                        attr['lower_bound'][i]), end=' ', file=fout)
                if attr['upper_bound'][i]:
                    print('{:24.16e}'.format(
                        attr['upper_bound'][i]), file=fout)

            print(file=fout)

        if fname:
            fout.close()

    def update_params(self, opt_x):
        """ Update parameter values from optimzier. (Interface to optimizer)

        This is the opposite operation of get_opt_params().

        Parameters
        ----------

        opt_x, list of floats
          parameter values from the optimizer.

        """
        for k, val in enumerate(opt_x):
            name = self._index[k]['name']
            c_idx = self._index[k]['c_idx']
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
            name = idx['name']
            c_idx = idx['c_idx']
            opt_x0.append(self.params[name]['value'][c_idx])
        if len(opt_x0) == 0:
            raise ParameterError('No parameters specified to optimize.')
        return np.asarray(opt_x0)

    def get_bounds(self):
        """ Get the lower and upper parameter bounds. """
        bounds = []
        for idx in self._index:
            name = idx['name']
            c_idx = idx['c_idx']
            lower = self.params[name]['lower_bound'][c_idx]
            upper = self.params[name]['upper_bound'][c_idx]
            bounds.append([lower, upper])
        return bounds

    def get_names(self):
        return self.params.keys()

    def get_size(self, name):
        return self.params[name]['size']

    def get_value(self, name):
        return self.params[name]['value'].copy()

    def get_index(self, name):
        return self.params[name]['index']

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

    def get_opt_param_value_and_i_j_indices(self, k):
        name = self._index[k]['name']
        p_idx = self._index[k]['p_idx']
        c_idx = self._index[k]['c_idx']
        value = self.params[name]['value'][c_idx]
        return value, p_idx, c_idx

    def _read_1_item(self, name, j, line):
        self._read_1st_item(name, j, line[0])

    def _read_2_item(self, name, j, line):
        self._read_1st_item(name, j, line[0])
        if line[1].lower() == 'fix':
            self.params[name]['fix'][j] = True
        else:
            raise InputError(
                'Data at line {} of {} corrupted.\n'.format(j+1, name))

    def _read_3_item(self, name, j, line):
        self._read_1st_item(name, j, line[0])
        try:
            self.params[name]['lower_bound'][j] = float(line[1])
            self.params[name]['upper_bound'][j] = float(line[2])
        except ValueError as err:
            raise InputError(
                '{}.\nData at line {} of {} corrupted.\n'.format(err, j+1, name))

    def _read_1st_item(self, name, j, first):
        if type(first) == str and first.lower() == 'kim':
            self.params[name]['use_default'][j] = True
            model_value = self.model_params[name].value
            self.params[name]['value'][j] = model_value[j]
        else:
            try:
                self.params[name]['value'][j] = float(first)
            except ValueError as err:
                raise InputError(
                    '{}.\nData at line {} of {} corrupted.\n'.format(err, j+1, name))

    def _check_bounds(self, name):
        """Check whether the initial guess of a paramter is within its lower and
        upper bounds.
        """
        attr = self.params[name]
        for i in range(attr['size']):
            lower_bound = attr['lower_bound'][i]
            upper_bound = attr['upper_bound'][i]
            if lower_bound != None:
                value = attr['value'][i]
                if value < lower_bound or value > upper_bound:
                    raise InputError('Initial guess at line {} of parameter {} is '
                                     'out of bounds.\n'.format(i+1, name))

    def _set_index(self, name):
        """Check whether a specific data value of a parameter will be optimized or
        not (by checking its 'fix' attribute). If yes, include it in the index
        list.

        Given a parameter and its values such as:

        PARAM_FREE_B
        1.1
        2.2  fix
        4.4  3.3  5.5

        the first slot (1.1) and the third slot (4.4) will be included in self._index,
        and later be optimized.
        """

        p_idx = list(self.model_params.keys()).index(name)
        size = self.params[name]['size']
        fix = self.params[name]['fix']
        for j in range(size):
            if not fix[j]:
                idx = {'name': name, 'p_idx': p_idx, 'c_idx': j}
                self._index.append(idx)


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


class NotComputeError(Exception):
    def __init__(self, msg):
        super(NotComputeError, self).__init__(msg)
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
