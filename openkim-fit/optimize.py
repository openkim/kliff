from scipy.optimize import least_squares
import numpy as np
import sys
sys.path.append('./lib/geodesicLMv1.1/pythonInterface')
from geodesiclm import geodesiclm



def fun_rosenbrock(x, A):
    '''Rosenbrock function'''
    return np.array( [ (1 - x[0]), A*(x[1] - x[0]**2)] )



def fit(method='geodesiclm'):
    ''' Choose the correct method and do the fitting.'''
    
    if method == 'geodesiclm': 
        x0 = np.array([ -1.0, 1.0])
        xf, info = geodesiclm(fun_rosenbrock, x0, args = (1000,), full_output=1, print_level = 5, 
                              iaccel = 1, maxiters = 10000, artol = -1.0, xtol = -1, ftol = -1, avmax = 2.0)
        print info
        print xf

    elif method == 'scipy-lm':
        # test scipy.optimize minimization method
        x0 = np.array([ -1.0, 1.0])
        res_1 = least_squares(fun_rosenbrock, x0, args = (1000,), method='lm')
        print res_1



if __name__ == '__main__':
    fit(method='scipy-lm')

