from __future__ import print_function
import functools
import itertools
import shutil
from modelparams import ModelParams


def alternate_fix_params(modelname, paramfiles, Ntimes=10):
    """
    Decorator function to start a new fitting using the previous fitted parameters.

    This is mainly for the case where we want to fit different params alternatively.
    In this implementation, all the work is centered on param files: parsing and writing
    new param files.


    Parameters
    ----------

    paramfiles, list of string
      Parameter file names where the fixed params are defined. Note that
      1) the name of the parameter file used in the function that is to be
         decorated and that in the first slot of `paramfiles' should be the same.
      2) the inital values will be read from the param file in first slot of `paramfiles'
         and other values will be ignored, except the fixes.
      3) Note that the first param file will be rewritten, so make sure to make a copy
         for later reference before invoking this decorator.

    Ntimes, int
      The number of times to execute alternatively the fitting with different fixed params.
    """

    Nparamfiles = len(paramfiles)
    firstname = paramfiles[0]

    # read the parameter files (Actually, we just need the fixes defined there, but we
    # still use the full )
    init_params = []
    for f in paramfiles:
        params = ModelParams(modelname)
        params.read(f)
        init_params.append(params)

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):

                for i, j in itertools.product(range(Ntimes), range(Nparamfiles)):
                    print()
                    print('='*80)
                    print("Number of iterations of running function `{}': {}.\n"
                          .format(func.__name__, i*Nparamfiles+j+1))

                    # run the fitting
                    try:
                        func(*args, **kwargs)
                    except Exception as e:
                        raise Exception("Function `{}' passed to decorator `alter_fix_params' "
                                        "cannot be executed.\n{}".format(func.__name__, e))

                    # generete param file for the next run
                    # index of the param file in init_params for the next run
                    idx = (j+1) % Nparamfiles
                    next_run_params = init_params[idx]
                    next_run_params_names = next_run_params.get_names()

                    # get fitted params values
                    fitted_params = ModelParams(modelname)
                    fitted_params.read('FINAL_FITTED_PARAMS')
                    fitted_param_names = fitted_params.get_names()
                    for name in fitted_param_names:
                        value = fitted_params.get_value(name)
                        if name in next_run_params_names:
                            next_run_params.set_value(name, value)

                    # write params info to file
                    next_run_params.echo_params(firstname)

            return wrapper

    return decorator
