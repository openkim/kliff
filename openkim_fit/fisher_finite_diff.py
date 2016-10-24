import numpy as np

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

    def compute_fisher_matrix(self, delta=0.001):
        '''
        Comptue the derivative of forces with respect to parameters using finite
        difference: df/dpi = (df(pi+dpi) - df(pi-dpi)/2dpi. And then compute the
        Fisher information matrix accprding to Fij = df/dpi*df/dpj
        '''
        delta_params = []
        param_names = self.params.get_names()
        for name in param_names:
            values = self.params.get_value(name)
            for val in values:
                delta_params.append(delta*val)
        self.delta_params = delta_params
        F_all = []
        for i,kimobj in enumerate(self.KIMobjs):
            forces_p1 = self._get_forces_one_conf(kimobj, 'plus', 1, delta)
            forces_p2 = self._get_forces_one_conf(kimobj, 'plus', 2, delta)
            forces_p3 = self._get_forces_one_conf(kimobj, 'plus', 3, delta)
            forces_m1 = self._get_forces_one_conf(kimobj, 'minus', 1, delta)
            forces_m2 = self._get_forces_one_conf(kimobj, 'minus', 2, delta)
            forces_m3 = self._get_forces_one_conf(kimobj, 'minus', 3, delta)
            dfdp = -1/60.*forces_m3+3/20.*forces_m2-3/4.*forces_m1 + 3/4.*forces_p1-3/20.*forces_p2+1/60.*forces_p3
            repeat_delta_params = np.repeat(np.atleast_2d(delta_params).T, len(dfdp[0]), axis=1)
            dfdp = np.divide(dfdp, repeat_delta_params)
            F_all.append(np.dot(dfdp, dfdp.T))
        self.F = np.mean(F_all, axis=0)
        return self.F


    def _get_forces_one_conf(self, kimobj, sign, order, delta):
        '''
        Compute the forces by perturbing the parameters one by one.
        '''
        forces = []
        param_names = self.params.get_names()
        for name in param_names:
            values = self.params.get_value(name)
            for i in range(len(values)):
                val_ori = values[i]
                if sign == 'plus':
                    val_new = val_ori*(1 + order*delta)
                elif sign == 'minus':
                    val_new = val_ori*(1 - order*delta)
                # perturbe one param value
                lines = [name]
                for j in range(len(values)):
                    if j == i:
                        lines.append([val_new])
                    else:
                        lines.append([values[j]])
                self.params.set_param(lines)
                kimobj.update_params()
                # compute
                kimobj.compute()
                forces.append(kimobj.get_forces())
                # restore param value back
                lines = [name]
                for j in range(len(values)):
                    if j == i:
                        lines.append([val_ori])
                    else:
                        lines.append([values[j]])
                self.params.set_param(lines)
                kimobj.update_params()
        return np.array(forces)

    def get_fisher_matrix(self):
        return self.F


if __name__ == '__main__':

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
    fisher.compute_fisher_matrix()
    fisher_matrix = fisher.get_fisher_matrix()


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


