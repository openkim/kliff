from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import copy
import sys


class Fisher:
    """Fisher information matrix.

    Compute the Fisher information according to $I_{ij} = \sum_m df_m/dp_i * df_m/dp_j$,
    where f_m are the forces on atoms in configuration m, p_i is the ith model parameter.
    Derivatives are computed numerically using Ridders' algorithm.

    Parameters
    ----------

    KIMobjs: list of KIMcalculator objects

    params: ModelParameters object
    """

    def __init__(self, params, calculator):
        self.params = params
        self.calculator = calculator
        self.F = None
        self.F_std = None
        self.delta_params = None

    def compute(self):
        """Comptue the Fisher information matrix and the standard deviation.

        Returns
        -------

        F: 2D array, shape(N, N), where N is the number of parameters
          Fisher informaiton matrix (FIM)

        F_std: 2D array, shape(N, N), where N is the number of parameters
          standard deviation of FIM
        """
        F_all = []
        kim_in_out_data = self.calculator.get_kim_input_and_output()
        for in_out in kim_in_out_data:
            dfdp = self._get_derivative_one_conf(in_out)
            F_all.append(np.dot(dfdp, dfdp.T))
        self.F = np.mean(F_all, axis=0)
        self.F_std = np.std(F_all, axis=0)
        return self.F, self.F_std

    def _get_derivative_one_conf(self, in_out):
        """Compute the derivative dfm/dpi for one atom configuration.

        Parameters
        ----------

        in_out: Configuration object
        """
        try:
            import numdifftools as nd
        except ImportError as e:
            raise ImportError(
                str(e) + '.\nFisher information computation needs '
                '"numdifftools". Please install first.'
            )

        derivs = []
        ori_param_vals = self.params.get_x0()
        for i, p in enumerate(ori_param_vals):
            values = copy.deepcopy(ori_param_vals)
            Jfunc = nd.Jacobian(self._get_prediction)
            df = Jfunc(p, i, values, in_out)
            derivs.append(df.reshape((-1,)))
            # restore param values back
            self.params.update_params(ori_param_vals)

        return np.array(derivs)

    def _get_prediction(self, x, idx, values, in_out):
        """ Compute predictions using specific parameter.

        Parameters
        ----------

        values: list of float
          the parameter values

        idx: int
          the index of 'x' in the value list

        x: float
          the specific parameter value at slot 'idx'

        Return
        ------
        forces: list of floats
          the forces on atoms in this configuration
        """
        values[idx] = x
        self.params.update_params(values)
        self.calculator.update_params(self.params)
        self.calculator.compute(in_out)
        forces = self.calculator.get_forces(in_out)
        forces = np.reshape(forces, (-1,))
        return forces
