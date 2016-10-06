import numpy as np
from modelparams import ModelParams

class Cost:
    '''Cost class.'''

    def __init__(self, params):
        '''

        Parameters
        ----------
        params: ModelParams object
            It's method update_params will be used to update parameters from
            the optimizer.
        '''
        self.params = params
        self.pred_obj = []
        self.ref = []
        self.weight = []
        self.fun = []


    def add(self, pred_obj, reference, weight=1, fun=None):
        '''

        Parameters
        ----------

        pred_obj: predictor object.
            It must have the "get_prediction()" method, which returns a float
            list that has the same length as "reference".

        reference: float list
            reference data

        weight: float or list
            weight for the prediction and reference data. If a float number
            is entered, all the prediction and reference data entries will
            use the same value. Its length should match reference data if
            "fun" is not set, otherwise, its length should match the length
            of the return value of "fun".

        fun(pred_obj.get_prediction(), reference):
            a function to generate residual using a method rather than the
            default one. It takes two R^m list arguments, and returns a R^n
            list.
        '''


        # check correct length
        pred = pred_obj.get_prediction()
        len_pred = len(pred)
        if len_pred != len(reference):
            raise InputError('Cost.add(): lenghs of prediction and reference data '
                             'do not match.')
        if fun == None:
            if type(weight) == list or type(weight) == tuple:
                if len(weight) != len_pred:
                    raise InputError('Cost.add(): lenghs of prediction data and '
                                     'weight do not match.')
            else:
                weight = [weight for i in range(len_pred)]
        else:
            residual = func(pred,reference)
            len_resid = len(residual)
            if type(weight) == list or type(weight) == tuple:
                if len(weight) != len_resid:
                    raise InputError('Cost.add(): lenghs of return data of "fun" '
                                     'data and weight do not match.')
                else:
                    weight = [weight for i in range(len_pred)]
        # record data
        self.pred_obj.append(pred_obj)
        self.ref.append(reference)
        self.weight.append(weight)
        self.fun.append(fun)


    def _update_params(self, x0):
        # update params x0 to ModelParams
        self.params.update_params(x0)
        # set enviroment variable such that standard KIM test can update params
        self.params.set_env_var()
# NOTE this may change, since for standard KIM test, we need to crack the
# KIM API to do it
        # update params to predictor
        for p in self.pred_obj:
            p.update_params()

    def _compute_predictions(self):
        for p in self.pred_obj:
            p.get_prediction()

    def get_residual(self, x0):
        # publish params x0 to predictor
        self._update_params(x0)
        # compute properties using new params x0
        self._compute_predictions()

        residual = []
        for i in range(len(self.pred_obj)):
            pred = self.pred_obj[i].get_prediction()
            ref = self.ref[i]
            weight = np.sqrt(self.weight[i])
            fun = self.fun[i]
            if fun == None:
                difference = np.subtract(pred, ref)
            else:
                difference = fun(pred, ref)
            tmp_resid = np.multiply(weight, difference)
            residual = np.concatenate((residual, tmp_resid))
        return residual


    def get_cost(self, x0):
        '''
        least squares cost function
        '''
        residual = self.get_residual(x0)
        cost = 0.5*np.linalg.norm(residual)**2
        return cost

#
#def conf_residual(kimobj, conf):
#    '''
#    Compute the residual of a configruation according to the following cost
#    funtion:
#
#    .. math:
#        C = \frac{1}{2} w \sum r_i^2, i.e., ..math:
#    i.e.,
#    .. math:
#        C = \frac{1}{2} \sum (\sqrt{w} r_i)^2
#    '''
#
#    energy_weight = 1
#    force_weight = 1
#
#    kimobj.compute()
#    kim_energy = kimobj.get_energy()
#    kim_forces = kimobj.get_forces()
#    ref_energy = conf.get_energy()
#    ref_forces = conf.get_forces()
#
#    resid_energy = np.sqrt(energy_weight)*(kim_energy - ref_energy)
#    resid_forces = np.sqrt(force_weight)*np.subtract(kim_forces, ref_forces)
#    resid = np.concatenate(([resid_energy], resid_forces))
#
#    return resid
#
#
#def get_residual(x, kimobjs, confs):
#    '''
#    Compute the residual of the whole training set, which may incldue multiple
#    configurations.
#    '''
#    update_params(x, kimobjs)
#
#    residual = np.array([])
#
#    for obj,conf in zip(kimobjs, confs):
#        tmp_resid = conf_residual(obj, conf)
#        residual = np.concatenate((residual, tmp_resid))
#
#    return residual
#
#
#def update_params(x, kimobjs):
#    '''
#    Wrapper function to update parameters to KIM potential model.
#    '''
#    for obj in kimobjs:
#        obj.publish_params(x)
#
#

