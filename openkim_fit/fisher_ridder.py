import numpy as np
import copy

class Fisher():
    '''Fisher information matrix.

    Parameters
    ----------

    KIMobjs: list of KIMcalculator objects

    params: ModelParams object
    '''
    def __init__(self, KIMobjs, params):
        self.KIMobjs = KIMobjs
        self.params = params
        self.F = None
        self.delta_params = None


    def compute_fisher_matrix(self):
        '''
        Comptue the derivative of forces with respect to parameters using finite
        difference: df/dpi = (df(pi+dpi) - df(pi-dpi)/2dpi. And then compute the
        Fisher information matrix accprding to Fij = df/dpi*df/dpj
        '''
        F_all = []
        for i,kimobj in enumerate(self.KIMobjs):
            print 'doing work for configuration', i
            dfdp = self._get_derivative_one_conf(kimobj)
            F_all.append(np.dot(dfdp, dfdp.T))
        self.F = np.mean(F_all, axis=0)
        return self.F


    def _get_derivative_one_conf(self, kimobj):
        '''
        Compute the derivative.
        '''
        derivs = []
        ori_values = self.params.get_x0()
        for i,x in enumerate(ori_values):
            values = copy.deepcopy(ori_values)
            h = 0.01
            df,dummy = dfridr(self._get_prediction, x, h, i, values, kimobj)
            derivs.append(df)
            # restore values back
            self.params.update_params(ori_values)
        return np.array(derivs)


    def _get_prediction(self, x, idx, values, kimobj):
        """
        Compute predictions using specifit parameter.

        Parameters
        ----------

        values, list of float
            the parameter values

        idx, int
            The index of 'x' in the value list

        x, float
            The specific parameter value at slot 'idx'

        """
        values[idx] = x
        # pass params to ModelParams
        self.params.update_params(values)
        # pass params to KIMcalculator
        kimobj.update_params()
        kimobj.compute()
        return kimobj.get_forces()



def dfridr(func, x, h, *args):
    """
    Compute numerical derivative using Ridders' method.

    In Ridders method, 'func' is a scalar function that takes a scalar argument
    and return s a scalar. Here, we entend it such that 'func' takes a scalar
    argument but can return a vector.


    Parameters
    ----------
    func: function
        The function whose first derivate we want to estimate. 'func' should
        should take a scalar argument, and can return either scalar or vector.

    x: float
        The point at which we want to estimate the derivative.

    h: float
        initial stepsize

    Returns
    -------
    ans: float or list of floats
        the derivative(s)

    err: float
        error of the finite difference estimation

    Reference
    ---------

    Based on code from Numerical Recipes, Press et al., Second Ed., Cambridge,
    1992.

    Ref: Ridders, C. J. F., 'Two algorithms for the calculation of dF(x)=D',
         Advances in Engineering Software, Vol. 4, no. 2, pp. 75-76, 1982.
    """

    ntab = 10
    con = 1.4
    con2 = con**2
    big = np.inf
    safe = 2.0
    size = len(func(x, *args))
    a = np.zeros((ntab, ntab, size))

    if h <= 0.:
        raise ValueError('h must be larger than 0.0 in dfridr.')
    hh = h
    a[0][0] = (func(x+hh, *args) - func(x-hh, *args))/(2.*hh)
    err = big
    for i in range(1, ntab):
        hh /= con  #try new, smaller step size
        a[0][i] = (func(x+hh, *args) - func(x-hh, *args))/(2.*hh)
        fac = con2
        for j in range(1,i+1):
            a[j][i] = (a[j-1][i]*fac - a[j-1][i-1])/(fac-1.)
            fac *= con2
            # The error strategy is to compare each new extrapolation to one order lower,
            # both at the present stepsize and the previous one.
            # this is different from the original Ridder algorithm, where 'a' is scalar
            # but here we assume 'a' is a 1D array, so we use norm
#            errt = max(np.linalg.norm(a[j][i] - a[j-1][i]), np.linalg.norm(a[j][i] - a[j-1][i-1]))
            errt = max(max(abs((a[j][i] - a[j-1][i]))), max(abs((a[j][i] - a[j-1][i-1]))))
            if errt < err:
                err = errt
                ans = a[j][i]
        #If higher order is worse by a significant factor SAFE, then quit early.
#        if np.linalg.norm(a[i][i] - a[i-1][i-1]) > safe*err:
        if max(abs(a[i][i] - a[i-1][i-1])) > safe*err:
            break
    return ans,err



if __name__ == '__main__':

#
#    # test dfridr
#    def func(x):
#        return np.array((np.sin(x), np.cos(x), x**2))
#    def func_grad(x):
#        return np.array((np.cos(x), -np.sin(x), 2*x))
#    def simple_finite_diff(func, x, stepsize):
#        return (func(x+stepsize)-func(x-stepsize))/(2.*stepsize)
#    x = np.pi/4 #    deri,err = dfridr(func, x, 0.1)
#    analytic=func_grad(x)
#    simple_fi = simple_finite_diff(func, x, 1e-8)
#
#    print 'finite diff:'
#    for i,j,k in zip(deri,analytic, simple_fi):
#        print '{:24.16e}   {:24.16e} {:24.16e}'.format(i,j,k)
#
#    sys.exit(1)
#



    from training import TrainingSet
    from modelparams import ModelParams
    from kimcalculator import KIMcalculator

    # KIM model parameters
    modelname = 'Three_Body_Stillinger_Weber_MoS__MO_000000111111_000'
    params = ModelParams(modelname)
    params.echo_avail_params()
    fname = '../tests/mos2_fitted.txt'
    params.read(fname)
    params.echo_params()

    # read config and reference data
    tset = TrainingSet()
    tset.read('../tests/training_set-T300')
    configs = tset.get_configs()

    # prediction
    KIMobjs=[]
    for i in range(len(configs)):
        obj = KIMcalculator(modelname, params, configs[i])
        obj.initialize()
        KIMobjs.append(obj)

    print 'hello there, started computing Fisher information matrix. It may take a'
    print 'while, so take a cup of coffee.'
    fisher = Fisher(KIMobjs, params)
    fisher_matrix = fisher.compute_fisher_matrix()


    # Temperature
    T = 150
    kB = 8.61733034e-5
    gamma = 1
    fisher_matrix = fisher_matrix/(2.*kB*T*gamma)


    with open('Fij', 'w') as fout:
        for line in fisher_matrix:
            for i in line:
                fout.write('{:24.16e} '.format(i))
            fout.write('\n')

    Fij_diag = np.diag(fisher_matrix)
    with open('Fij_diag', 'w') as fout:
        for line in Fij_diag:
            fout.write('{:13.5e}\n'.format(line))

    # inverse
    Fij_inv = np.linalg.inv(fisher_matrix)
    with open('Fij_inv', 'w') as fout:
        for line in Fij_inv:
            for i in line:
                fout.write(str(i)+ ' ')
            fout.write('\n')

    # inverse_diag
    Fij_inv_diag = np.diag(Fij_inv)
    with open('Fij_inv_diag', 'w') as fout:
        for line in Fij_inv_diag:
            fout.write('{:13.5e}\n'.format(line))

    # eiven analysis
    w,v = np.linalg.eig(fisher_matrix)
    with open('eigenVec','w') as fout:
      for row in v:
        for item in row:
          fout.write('{:13.5e}'.format(float(item)))
        fout.write('\n')

    with open('eigenVal','w') as fout:
      for item in w:
        fout.write('{:13.5e}\n'.format(float(item)))


