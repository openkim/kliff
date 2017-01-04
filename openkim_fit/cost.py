import numpy as np
import multiprocessing as mp
import collections
from modelparams import ModelParams


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
        self.func = []
        self.pred_obj_group = None
        self.ref_group = None
        self.weight_group = None
        self.func_group = None


    def add(self, pred_obj, reference, weight=1, func=None):
        """Add a contribution part to the cost.

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
            "func" is not set, otherwise, its length should match the length
            of the return value of "func".

        func(pred_obj.get_prediction(), reference):
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
        # construct weight list according to input type
        if func is None:
            len_resid = len_pred
        else:
            residual = func(pred,reference)
            len_resid = len(residual)
        if isinstance(weight, (collections.Sequence, np.ndarray)):
            if len(weight) != len_resid:
                raise InputError('Cost.add(): lenghs of return data of "func" '
                                 'data and weight do not match.')
        else:
            weight = [weight for i in range(len_resid)]

        # record data
        self.pred_obj.append(pred_obj)
        self.ref.append(reference)
        self.weight.append(weight)
        self.func.append(func)
        # group objects added so far
        self._group_preds()


    def _group_preds(self):
        """ Group the predictors (and the associated references, weights and wrapper
        functions) into 'nprocs' groups, so that each group can be processed by a
        processor. The predictors are randomly assigned to each group to account for
        load balance in each group.
        """
#NOTE we may not want the shuffling while developing the code, such that it gives
# deterministic results
        # shuffle before grouping, in case some processors get quite heavy jobs but
        # others get light ones.
#        combined = zip(self.pred_obj, self.ref, self.weight, self.func)
#        np.random.shuffle(combined)
#        self.pred_obj[:], self.ref[:], self.weight[:], self.func[:] = zip(*combined)

        # grouping
        self.pred_obj_group= np.array_split(self.pred_obj, self.nprocs)
        self.ref_group= np.array_split(self.ref, self.nprocs)
        self.weight_group= np.array_split(self.weight, self.nprocs)
        self.func_group= np.array_split(self.func, self.nprocs)


    def get_residual(self, x0):
        """ Compute the residual for the cost.
        This is a callable for optimizer that usually passed as the first positional
        argument.

        Parameters
        ----------

        x0: list
            optimizing parameter values
        """
        # update params from minimizer to ModelParams
        self._update_params(x0)
        # create jobs and start
        jobs = []
        pipe_list = []
        for g in range(self.nprocs):
            # create pipes
            recv_end, send_end = mp.Pipe(False)
            # get data for this group
            pred = self.pred_obj_group[g]
            ref = self.ref_group[g]
            weight = self.weight_group[g]
            func = self.func_group[g]
            p = mp.Process(target=self._get_residual_group, args=(pred, ref, weight, func, send_end,))
            p.start()
            jobs.append(p)
            pipe_list.append(recv_end)

        # we should place recv() before join()
        residuals = [x.recv() for x in pipe_list]
        # wait for the all the jobs to complete
        for p in jobs:
            p.join()
        return np.concatenate(residuals)


    def get_cost(self, x0):
        """ Compute the cost.
        This can be used as the callable for optimizer (e.g. scipy.optimize.minimize
        method='CG')

        Parameters
        ----------

        x0: list
            optimizing parameter values
        """
        residual = self.get_residual(x0)
        cost = 0.5*np.linalg.norm(residual)**2
        return cost


    def _update_params(self, x0):
        """Update parameter values from minimizer to ModelParams object.

        Parameters
        ----------

        x0: list
            optimizing parameter values
        """

        self.params.update_params(x0)
        # set enviroment variable such that standard KIM test can update params
        self.params.set_env_var()


    def _get_residual_group(self, pred_g, ref_g, weight_g, func_g, send_end):
        """Helper function to do the computation of residuals.
        """
        residual = []
        for i in range(len(pred_g)):
            pred = pred_g[i].get_prediction()
            ref = ref_g[i]
            weight = np.sqrt(weight_g[i])
            func = func_g[i]
            if func is None:
                difference = np.subtract(pred, ref)
            else:
                difference = func(pred, ref)
            tmp_resid = np.multiply(weight, difference)
            residual = np.concatenate((residual, tmp_resid))
        send_end.send(residual)


