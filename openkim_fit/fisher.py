import numpy as np
import copy
import sys

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
        self.F_std = None
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
        self.F_std = np.std(F_all, axis=0)
        return F_all, self.F_std
        #return self.F, self.F_std


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
    Compute numerical derivative using Ridders' method of polynomial extrapolation.

    In Ridders method, 'func' is a scalar function that takes a scalar argument
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
        The point at which we want to estimate the derivative(s).

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

    Based on code from Numerical Recipes, Press et al., Second Ed., Cambridge,
    1992.
    """

    ntab = 10
    con = 1.4; con2 = con**2
    big = np.inf
    safe = 2.0
    size = len(func(x, *args))
    a = np.zeros((ntab, ntab, size))
    if h == 0.:
        raise ValueError('h must be nonzero in dfridr.')
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



if __name__ == '__main__':

    from training import TrainingSet
    from modelparams import ModelParams
    from kimcalculator import KIMcalculator

    # Temperature
    T = 750


    # KIM model parameters
    modelname = 'Three_Body_Stillinger_Weber_MoS__MO_000000111111_000'
    params = ModelParams(modelname)
    params.echo_avail_params()
    #fname = '../tests/mos2_fitted-T'+str(T)
    #fname = '../tests/mos2_fitted-T'+str(T)+'_interval4'
    fname = '/home/wenz/Applications/use_kimfit/mos2_T'+str(T)+'/mos2_fitted-T'+str(T)+'_interval4'
    params.read(fname)
    params.echo_params()

    # read config and reference data
    tset = TrainingSet()
    #fname = '../tests/training_set-T'+str(T)
    #fname = '../tests/training_set-T'+str(T)+'_1'
    #fname = '../tests/training_set-T'+str(T)+'_2'
    fname = '/home/wenz/Applications/use_kimfit/mos2_T'+str(T)+'/training_set_T'+str(T)+'_interval4_1'
    #fname = '/home/wenz/Applications/use_kimfit/mos2_T'+str(T)+'/training_set_T'+str(T)+'_interval4_2'
    tset.read(fname)
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
    Fij, Fij_std = fisher.compute_fisher_matrix()

    # pickle
    fname = 'fisher_mat_T'+str(T)
    np.save(fname, Fij)

    sys.exit(1)
#    # load
#    fname = 'fisher_mat.npy'
#    Fij = np.load(fname)


    kB = 8.61733034e-5
    gamma = 1
    Fij = Fij/(2.*kB*T*gamma)

    # relative variance of Fij
    param_values = params.get_x0()

    for t in range(1):

        # relative variance
        Fij = np.dot(np.dot(np.diag(param_values), Fij), np.diag(param_values))
        Fij_std = np.dot(np.dot(np.diag(param_values), Fij_std), np.diag(param_values))


        with open('Fij'+str(t), 'w') as fout:
            for line in Fij:
                for i in line:
                    fout.write('{:24.16e} '.format(i))
                    #fout.write('{:10.2e} '.format(i))
                fout.write('\n')

        with open('Fij_std'+str(t), 'w') as fout:
            for line in Fij_std:
                for i in line:
                    fout.write('{:24.16e} '.format(i))
                    #fout.write('{:10.2e} '.format(i))
                fout.write('\n')

        Fij_diag = np.diag(Fij)
        with open('Fij_diag'+str(t), 'w') as fout:
            for line in Fij_diag:
                #fout.write('{:13.5e}\n'.format(line))
                fout.write('{:13.5e}\n'.format(line**0.5))

        # inverse
        Fij_inv = np.linalg.inv(Fij)
        with open('Fij_inv'+str(t), 'w') as fout:
            for line in Fij_inv:
                for i in line:
                    fout.write('{:24.16e} '.format(i))
                    #fout.write('{:10.2e} '.format(i))
                fout.write('\n')

        # inverse_diag
        Fij_inv_diag = np.diag(Fij_inv)
        with open('Fij_inv_diag'+str(t), 'w') as fout:
            for line in Fij_inv_diag:
                #fout.write('{:13.5e}\n'.format(line))
                fout.write('{:13.5e}\n'.format(line**0.5))

        # eiven analysis
        w,v = np.linalg.eig(Fij)
        with open('eigenVec'+str(t),'w') as fout:
          for row in v:
            for item in row:
              fout.write('{:13.5e}'.format(float(item)))
            fout.write('\n')

        with open('eigenVal'+str(t),'w') as fout:
          for item in w:
            fout.write('{:13.5e}\n'.format(float(item)))


        # update for the next loop
#        param_values = np.dot(v.T, param_values)
#        Fij = np.dot( np.dot(np.linalg.inv(v), Fij), np.linalg.inv(v.T))


#        # remove the smallest eivenval
#        v = v[:,:-1]
#        # update for the next loop
#        param_values = np.dot(v.T, param_values)
#        Fij = np.dot( np.dot(np.linalg.pinv(v), Fij), np.linalg.pinv(v.T))
#
#
#
