from __future__ import print_function
import numpy as np
import copy
import sys

class Fisher():
    """Fisher information matrix.
    Computed according to I_ij = \sum_m dfm/dpi*dfm/dpj, where fm are the forces
    on atoms in configuration m, pi is the ith model parameter. Derivatives are computed
    numerically using Ridders' algorithm.

    Parameters
    ----------

    KIMobjs: list of KIMcalculator objects

    params: ModelParams object
    """

    def __init__(self, KIMobjs, params):
        self.KIMobjs = KIMobjs
        self.params = params
        self.F = None
        self.F_std = None
        self.delta_params = None

    def compute(self):
        """Comptue the Fisher information matrix and the standard deviation.

        Returns
        -------

        self.F: N by N matrix, where N is the number of parameters
            Fisher informaiton matrix (FIM)

        self.F_std: N by N matrix
            standard deviation of FIM
        """
        F_all = []
        for m,kimobj in enumerate(self.KIMobjs):
            print('Conducting computation for configuration:', m)
            dfdp = self._get_derivative_one_conf(kimobj)
            F_all.append(np.dot(dfdp, dfdp.T))
        self.F = np.mean(F_all, axis=0)
        self.F_std = np.std(F_all, axis=0)
        return self.F, self.F_std


    def _get_derivative_one_conf(self, kimobj):
        """Compute the derivative dfm/dpi for one atom configuration.

        Parameters
        ----------

        kimobj: object of KIMcalculator
        """
        derivs = []
        ori_param_vals = self.params.get_x0()
        for i,p in enumerate(ori_param_vals):
            values = copy.deepcopy(ori_param_vals)
            h = 0.01
            df,dummy = dfdpi(self._get_prediction, p, h, i, values, kimobj)
            derivs.append(df)
            self.params.update_params(ori_param_vals) # restore param values back
        return np.array(derivs)


    def _get_prediction(self, x, idx, values, kimobj):
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
        self.params.update_params(values) # pass params to ModelParams
        kimobj._update_params() # pass params to KIMcalculator
        kimobj.compute()
        forces = kimobj.get_forces()
        return forces


def dfdpi(func, x, h, *args):
    """Ridders' method to estimate first derivative using polynomial extrapolation.

    In Ridders' method, 'func' is a scalar function that takes a scalar argument
    and return s a scalar. Here, we extend it such that 'func' takes a scalar
    argument but can return a vector. But the derivative of each component is computed
    separatly.

    Parameters
    ----------
    func: function
        The function whose first derivate(s) we want to estimate. 'func' should
        take a scalar argument 'x' and additional arguments 'args', and can return
        either scalar or vector.

    x: float
        The point at which we want to estimate the derivative(s)

    h: float
        initial stepsize

    Returns
    -------
    ans: float or list of floats
        the derivative(s)

    err: float or list of floats
        error(s) of the finite difference estimation

    Reference
    ---------
    C. Ridders, Adv. Eng. Software 4, 75 (1982).
    W. H. Press, et al, Numerical Recipes, 3rd ed. Cambridge University Press, 2007.
    """

    ntab = 10
    con = 1.4; con2 = con**2
    big = np.inf
    safe = 2.0
    size = len(func(x, *args))
    a = np.zeros((ntab, ntab, size))
    if h == 0.:
        raise ValueError('h must be nonzero in fridr.')
    hh = h
    a[0][0] = (func(x+hh, *args) - func(x-hh, *args))/(2.*hh)
    err = [big for i in range(size)]
    ans = [None for i in range(size)]
    stop_update = [False for i in range(size)]
    # Successive columns in the Neville tableau will go to smaller stepsizes and higher
    # orders of extrapolation.
    for i in range(1, ntab):
        hh /= con  # try new, smaller step size
        a[0][i] = (func(x+hh, *args) - func(x-hh, *args))/(2.*hh)
        fac = con2
        # Compute extrapolations of various orders, requiring no new function evaluation.
        for j in range(1,i+1):
            a[j][i] = (a[j-1][i]*fac - a[j-1][i-1])/(fac-1.)
            fac *= con2
            # The error strategy is to compare each new extrapolation to one order lower,
            # both at the present stepsize and the previous one.
            # note, np.maximum is element-wise
            errt = np.maximum(np.absolute(a[j][i]-a[j-1][i]), np.absolute(a[j][i]-a[j-1][i-1]))
            for k in range(size):
                if not stop_update[k] and errt[k] < err[k]:
                    err[k] = errt[k]
                    ans[k] = a[j][i][k]
        #If higher order is worse by a significant factor SAFE, then stop update.
        for k in range(size):
            if not stop_update[k] and abs(a[i][i][k] - a[i-1][i-1][k]) >= safe*err[k]:
                stop_update[k] = True
        if all(stop_update):
            break
    return ans,err


