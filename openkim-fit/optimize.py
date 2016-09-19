from scipy.optimize import least_squares
import numpy as np
import sys
sys.path.append('../lib/geodesicLMv1.1/pythonInterface')
from geodesiclm import geodesiclm


def fit(x0, kim_objects, configs, method='geodesiclm'):
    ''' Choose the correct method and do the fitting.'''

    if method == 'geodesiclm': 
        xf, info = geodesiclm(func, x0, args=(kim_objects, configs),
                              full_output=1, print_level=5, iaccel=1, maxiters=10000,
                              artol=-1.0, xtol=-1, ftol=-1, avmax = 2.0)
        print info
        print xf

    elif method == 'scipy-lm':
        # test scipy.optimize minimization method
        x0 = np.array([ -1.0, 1.0])
        res_1 = least_squares(func, x0, args=(kim_objects,configs), method='lm')
        print res_1


def func(x, kim_objects, configs):
    '''
    Function call to update potential model parameters and compute residuals.
    '''
    update_params(x, kim_objects)
    return compute_residual(kim_objects, configs)


def conf_residual(kimobj, conf):
    '''
    Compute the residual of a configruation according to the following cost 
    funtion:

    .. math:
        C = \frac{1}{2} w \sum r_i^2, i.e., ..math:
    i.e.,
    .. math:
        C = \frac{1}{2} \sum (\sqrt{w} r_i)^2
    '''

    energy_weight = 1
    force_weight = 1

    kimobj.compute() 
    kim_energy = kimobj.get_energy()
    kim_forces = kimobj.get_forces()
    ref_energy = conf.get_energy()
    ref_forces = conf.get_forces()

    resid_energy = np.sqrt(energy_weight)*(kim_energy - ref_energy) 
    resid_forces = np.sqrt(force_weight)*np.subtract(kim_forces, ref_forces) 
    resid = np.concatenate(([resid_energy], resid_forces))

    return resid


def compute_residual(kimobjs, confs):
    '''
    Compute the residual of the whole training set, which may incldue multiple
    configurations.
    '''
    
    residual = np.array([])
   
    for obj,conf in zip(kimobjs, confs):
        tmp_resid = conf_residual(obj, conf)
        residual = np.concatenate((residual, tmp_resid))
 
    return residual


def update_params(x, kimobjs):
    '''
    Wrapper function to update parameters to KIM potential model.
    '''
    for obj in kimobjs:
        obj.publish_params(x)




#
#def fun_rosenbrock(x, A):
#    '''Rosenbrock function'''
#    return np.array( [ (1 - x[0]), A*(x[1] - x[0]**2)] )
#
#    
#    if method == 'geodesiclm': 
#        x0 = np.array([ -1.0, 1.0])
#        xf, info = geodesiclm(fun_rosenbrock, x0, args = (1000,), full_output=1, print_level = 5, 
#                              iaccel = 1, maxiters = 10000, artol = -1.0, xtol = -1, ftol = -1, avmax = 2.0)
#        print info
#        print xf
#
#    elif method == 'scipy-lm':
#        # test scipy.optimize minimization method
#        x0 = np.array([ -1.0, 1.0])
#        res_1 = least_squares(fun_rosenbrock, x0, args = (1000,), method='lm')
#        print res_1
#


#if __name__ == '__main__':
#    fit(method='scipy-lm')

