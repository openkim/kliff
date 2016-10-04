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
            forces_plus = self._get_forces_one_conf(kimobj, 'plus', delta)
            forces_minus = self._get_forces_one_conf(kimobj, 'minus', delta)
            dfdp = np.subtract(forces_plus, forces_minus)
            repeat_delta_params = np.repeat(np.atleast_2d(delta_params).T, len(dfdp[0]), axis=1)
            dfdp = np.divide(dfdp, 2.*repeat_delta_params)
            F_all.append(np.dot(dfdp, dfdp.T))
        self.F = np.mean(F_all, axis=0) 
        return self.F


    def _get_forces_one_conf(self, kimobj, sign, delta):
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
                    val_new = val_ori*(1 + delta)
                elif sign == 'minus':
                    val_new = val_ori*(1 - delta)
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
        return forces

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
    tset.read('../tests/training_set/')
    configs = tset.get_configs()

    # prediction 
    KIMobjs=[]
    for i in range(len(configs)):
        obj = KIMcalculator(modelname, params, configs[i])
        obj.initialize()
        KIMobjs.append(obj)

    print 'hello there, started computing Fisher information matrix. It may take a' 
    print 'while, take a cup of coffee.'
    fisher = Fisher(KIMobjs, params)
    fisher.compute_fisher_matrix()
    fisher_matrix = fisher.get_fisher_matrix()

    with open('Fij', 'w') as fout:
        for line in fisher_matrix:
            for i in line:
                fout.write(str(i)+ ' ')
            fout.write('\n')
