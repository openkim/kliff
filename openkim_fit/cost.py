import numpy as np
import collections
import multiprocessing as mp
from modelparams import ModelParams
import time
import copy


class Cost:
    """Objective cost function that will be minimized.

    Parameters
    ----------

    params: ModelParams object
        It's method 'update_params' will be used to update parameters from
        the optimizer.

    nprocs: int
        Number of processors to parallel to run the predictors.
    """
    def __init__(self, params, nprocs=1):

        self.params = params
        self.nprocs = nprocs
        self.pred_obj = []
        self.ref = []
        self.weight = []
        self.fun = []
        self.pred_obj_group = None
        self.ref_group = None
        self.weight_group = None
        self.fun_group = None

# NOTE Ause np.array_split
    def _group_preds(self):
        """
        Group the predictors (and the associated references, weights and wrapper
        functions) into 'nprocs' groups, so that each group can be processed by a
        processor. The predictors are randomly assigned to each group to account for
        load balance in each group.
        """
        tot_size = len(self.pred_obj)
        avg_size = int(tot_size/self.nprocs)
        remainder = tot_size % self.nprocs
        # random order
        np.random.seed(1)               # we may not want this
        idx = np.arange(tot_size)
        #np.random.shuffle(idx)
        # size of each group
        group_size = []
        for i in range(self.nprocs):
            if i < remainder:
                group_size.append(avg_size+1)
            else:
                group_size.append(avg_size)
        # split data into groups
        self.pred_obj_group = []
        self.ref_group = []
        self.weight_group = []
        self.fun_group = []
        k = 0
        for i in range(self.nprocs):
            tmp_pred_obj = []
            tmp_ref = []
            tmp_weight = []
            tmp_fun = []
            for j in range(group_size[i]):
                tmp_pred_obj.append(self.pred_obj[idx[k]])
                tmp_ref.append(self.ref[idx[k]])
                tmp_weight.append(self.weight[idx[k]])
                tmp_fun.append(self.fun[idx[k]])
                k += 1
            self.pred_obj_group.append(tmp_pred_obj)
            self.ref_group.append(tmp_ref)
            self.weight_group.append(tmp_weight)
            self.fun_group.append(tmp_fun)


    def add(self, pred_obj, reference, weight=1, fun=None):
        """

        Parameters
        ----------

        pred_obj: predictor object.
            It must have the 'get_prediction()' method, which returns a float
            list that has the same length as "reference". Optionally, it can have
            the 'update_params()' method that updates the parameters from the
            ModelParams class. However, for OpenKIM test, this is impossible, since
            we have no access to its KIM object. The way we handle it is to crack
            the KIM api to read parameters at Model initialization stage.

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
        """


        # check correct length
        pred = pred_obj.get_prediction()
        len_pred = len(pred)
        if len_pred != len(reference):
            raise InputError('Cost.add(): lengths of prediction and reference data '
                             'do not match.')
        if fun == None:
# NOTE this is unsafe, what if we pass a numpy array
            if type(weight) == list or type(weight) == tuple:
                if len(weight) != len_pred:
                    raise InputError('Cost.add(): lenghs of prediction data and '
                                     'weight do not match.')
            else:
                weight = [weight for i in range(len_pred)]
        else:
            residual = fun(pred,reference)
            len_resid = len(residual)
# NOTE this is unsafe, what if we pass a numpy array
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
        # group objects added so far
        self._group_preds()



    def _update_params(self, x0):
        # update params x0 from minimizer to ModelParams
        self.params.update_params(x0)
        # set enviroment variable such that standard KIM test can update params
        self.params.set_env_var()


    def get_residual(self, x0):
        # update params from minimizer to ModelParams
        self._update_params(x0)
        # create jobs and start
        jobs = []
        pipe_list = []
        #residuals = [None for i in range(self.nprocs)]
        for g in range(self.nprocs):
            # create pipes
            recv_end, send_end = mp.Pipe(False)
            # get data for this group
            pred = self.pred_obj_group[g]
            ref = self.ref_group[g]
            weight = self.weight_group[g]
            fun = self.fun_group[g]
            p = mp.Process(target=self._get_residual_group, args=(pred, ref, weight, fun, send_end,))
            p.start()
            jobs.append(p)
            pipe_list.append(recv_end)

        # we should place recv() before join()
        residuals = [x.recv() for x in pipe_list]
        #for g in range(self.nprocs):
        #    residuals[g] = pipe_list[g].recv()

        # wait for the all the jobs to complete
        for p in jobs:
            p.join()
        return np.concatenate(residuals)


    def _get_residual_group(self, pred_g, ref_g, weight_g, fun_g, send_end):
        residual = []
        for i in range(len(pred_g)):
            pred = pred_g[i].get_prediction()
            ref = ref_g[i]
            weight = np.sqrt(weight_g[i])
            fun = fun_g[i]
            if fun == None:
                difference = np.subtract(pred, ref)
            else:
                difference = fun(pred, ref)
            tmp_resid = np.multiply(weight, difference)
            residual = np.concatenate((residual, tmp_resid))
        send_end.send(residual)




    def get_cost(self, x0):
        '''
        least squares cost function
        '''
        residual = self.get_residual(x0)
        cost = 0.5*np.linalg.norm(residual)**2
        return cost



