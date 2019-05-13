import os
import sys
import warnings
import pickle
from collections import OrderedDict
from collections.abc import Iterable
import numpy as np
import kliff

logger = kliff.logger.get_logger(__name__)


# TODO take a look at proporty decorator
class Parameter:
    """Parameter class.
    """

    def __init__(self, value, dtype='double', description=None):
        self.set_value(value)
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

    def set_value(self, value, check_shape=True):
        if check_shape:
            if isinstance(value, Iterable):
                if any(isinstance(i, Iterable) for i in value):
                    raise ParameterError('Parameter should be a 1D array.')
                self._set_val(value)
            else:
                raise ParameterError('Parameter should be a 1D array.')
        else:
            self._set_val(value)

    def _set_val(self, value):
        self.value = np.asarray(value)
        self.size = len(value)

    def to_string(self):
        s = 'value: {}\n'.format(np.array_str(self.value))
        s += 'size: {}\n'.format(self.size)
        s += 'dtype: {}\n'.format(self.dtype)
        s += 'description: {}\n'.format(self.description)
        return s


class FittingParameter:
    """Class of model parameters that will be optimzied.

    It interacts with optimizer to provide initial guesses of parameters and
    receive updated paramters from the optimizer.
    """

    def __init__(self, model_params):
        """
        Parameters
        ----------

        model_params: OrderDict
            All the paramters of a model(calculator).
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
                msg = 'file "{}", line {}. Parameter "{}" already set.'.format(
                    fname, num_line, line
                )
                warnings.warn(msg, category=Warning)
            if line not in self.model_params:
                raise InputError(
                    'file "{}", line {}. Parameter "{}" not supported '
                    'by calculator.'.format(fname, num_line, line)
                )
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
        settings(including intial value, fix flag, lower bound, and upper bound).

        The value of the argument should be a list of list, where each inner list is
        for one component of the parameter, which can contain 1, 2, or 3 elements.
        See self.read() for the options of the elements.

        Example
        -------
        instance.set(A=[['DEFAULT'], [2.0, 1.0, 3.0]], B=[[1.0, 'FIX'], [2.0, 'INF',
        3.0]])
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
        name: string
            name of a fitting parameter

        settings: list of list
            initial value, flag to fix a parameter, lower and upper bounds of a
            parameter

        Example
        -------
        name = 'param_A'
        settings = [['default', 0, 20], [2.0, 'fix'], [2.2, 'inf', 3.3]]
        instance.set_one(name, settings)
        """
        # TODO update Example in docs, see test_parameter.py for example
        size = self.model_params[name].get_size()
        if len(settings) != size:
            raise InputError(
                'Incorrect number of initial values for paramter "{}".'.format(name)
            )

        tmp_dict = {
            'size': size,
            'value': [None for _ in range(size)],
            'use_default': [False for _ in range(size)],
            'fix': [False for _ in range(size)],
            'lower_bound': [None for _ in range(size)],
            'upper_bound': [None for _ in range(size)],
        }
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
                raise InputError(
                    'More than 3 elements listed at data line '
                    '{} for parameter "{}".'.format(j + 1, name)
                )
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

        if fname is not None:
            fout = open(fname, 'w')
        else:
            fout = sys.stdout

        print('#' + '=' * 80, file=fout)
        print('# Model parameters that are optimized.', file=fout)
        print('#' + '=' * 80, file=fout)
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

        if fname is not None:
            fout.close()

    def save(self, path):
        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(path, 'wb') as f:
            pickle.dump(self.params, f)

    def load(self, path):
        # restore parameters
        with open(path, 'rb') as f:
            self.params = pickle.load(f)
        # resotre index
        self._index = []
        for name in self.params.keys():
            self._index.append(self._set_index(name))

    def get_names(self):
        return self.params.keys()

    def get_size(self, name):
        return self.params[name]['size']

    def get_value(self, name):
        return self.params[name]['value'].copy()

    def set_value(self, name, value):
        self.params[name]['value'] = value

    def get_lower_bound(self, name):
        return self.params[name]['lower_bound'].copy()

    def get_upper_bound(self, name):
        return self.params[name]['upper_bound'].copy()

    def get_fix(self, name):
        return self.params[name]['fix'].copy()

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
            raise InputError('Data at line {} of {} corrupted.'.format(j + 1, name))

    def _read_3_item(self, name, j, line):
        self._read_1st_item(name, j, line[0])

        if (line[1] is not None) and (
            not (isinstance(line[1], str) and line[1].lower() == 'none')
        ):
            try:
                self.params[name]['lower_bound'][j] = float(line[1])
            except ValueError as e:
                raise InputError(
                    '{}.\nData at line {} of {} corrupted.'.format(e, j + 1, name)
                )

        if (line[2] is not None) and (
            not (isinstance(line[2], str) and line[2].lower() == 'none')
        ):
            try:
                self.params[name]['upper_bound'][j] = float(line[2])
            except ValueError as e:
                raise InputError(
                    '{}.\nData at line {} of {} corrupted.'.format(e, j + 1, name)
                )

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
                    '{}.\nData at line {} of parameter "{}" corrupted.'.format(
                        e, j + 1, name
                    )
                )

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
                    raise InputError(
                        'Initial guess at line {} of parameter "{}" '
                        'out of bounds.'.format(i + 1, name)
                    )
            if upper_bound is not None:
                if value > upper_bound:
                    raise InputError(
                        'Initial guess at line {} of parameter "{}" '
                        'out of bounds.'.format(i + 1, name)
                    )

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


def remove_comments(lines):
    """Remove lines in a string list that start with # and content after #."""
    processed_lines = []
    for line in lines:
        line = line.strip()
        if not line or line[0] == '#':
            continue
        if '#' in line:
            line = line[0 : line.index('#')]
        processed_lines.append(line)
    return processed_lines


class ParameterError(Exception):
    def __init__(self, msg):
        super(ParameterError, self).__init__(msg)
        self.msg = msg

    def __str__(self):
        return self.msg
