Train a Lennard-Jones potential
===============================

In this tutorial, we train a Lennard-Jones potential that is build in
KLIFF (i.e. not models archived on `OpenKIM <https://openkim.org>`_). From a user’s perspective,
a KLIFF built-in model is not different from a KIM model.

Compare this with the tutorial on `Stillinger-Weber
potential <kim_SW_Si>`__.

.. code-block:: python

    from kliff.legacy.calculators import Calculator
    from kliff.dataset import Dataset
    from kliff.legacy.loss import Loss
    from kliff.models import LennardJones
    from kliff.utils import download_dataset
    
    # training set
    dataset_path = download_dataset(dataset_name="Si_training_set_4_configs")
    tset = Dataset.from_path(dataset_path)
    configs = tset.get_configs()
    
    # calculator
    model = LennardJones()
    model.echo_model_params()
    
    # fitting parameters
    model.set_opt_params(sigma=[["default"]], epsilon=[["default"]])
    model.echo_opt_params()
    
    calc = Calculator(model)
    calc.create(configs)
    
    # loss
    loss = Loss(calc, nprocs=1)
    result = loss.minimize(method="L-BFGS-B", options={"disp": True, "maxiter": 10})
    
    
    # print optimized parameters
    model.echo_opt_params()
    model.save("kliff_model.yaml")


.. parsed-literal::

    2025-05-16 21:18:35.032 \| INFO     \| kliff.dataset.dataset:add_weights:1128 - No explicit weights provided.
    2025-05-16 21:18:35.036 \| INFO     \| kliff.legacy.calculators.calculator:create:107 - Create calculator for 4 configurations.
    2025-05-16 21:18:35.036 \| INFO     \| kliff.legacy.loss:minimize:327 - Start minimization using method: L-BFGS-B.
    2025-05-16 21:18:35.037 \| INFO     \| kliff.legacy.loss:_scipy_optimize:444 - Running in serial mode.
     This problem is unconstrained.


.. parsed-literal::

    #================================================================================
    # Available parameters to optimize (In MODEL SPACE).
    # Model: LJ6-12
    #================================================================================
    
    name: epsilon
    value: [1.]
    size: 1
    
    name: sigma
    value: [2.]
    size: 1
    
    name: cutoff
    value: [5.]
    size: 1
    
    #================================================================================
    # Following parameters have transformation objects attached, 
    # Parameter value in PARAM SPACE: 
    #================================================================================
    
    Parameter:epsilon : [1.]
    Parameter:sigma : [2.]
    RUNNING THE L-BFGS-B CODE
    
               * * *
    
    Machine precision = 2.220D-16
     N =            2     M =           10
    
    At X0         0 variables are exactly at the bounds
    
    At iterate    0    f=  6.40974D+00    \|proj g\|=  2.92791D+01
    At iterate    1    f=  2.98676D+00    \|proj g\|=  3.18782D+01
    At iterate    2    f=  1.56102D+00    \|proj g\|=  1.02614D+01
    At iterate    3    f=  9.61568D-01    \|proj g\|=  8.00167D+00
    At iterate    4    f=  3.20489D-02    \|proj g\|=  7.63381D-01
    At iterate    5    f=  2.42400D-02    \|proj g\|=  5.96986D-01
    At iterate    6    f=  1.49911D-02    \|proj g\|=  6.87761D-01
    At iterate    7    f=  9.48598D-03    \|proj g\|=  1.59359D-01
    At iterate    8    f=  6.69584D-03    \|proj g\|=  1.14377D-01
    At iterate    9    f=  4.11014D-03    \|proj g\|=  3.20704D-01
    At iterate   10    f=  2.97204D-03    \|proj g\|=  7.03415D-02

    2025-05-16 21:18:37.138 \| INFO     \| kliff.legacy.loss:minimize:329 - Finish minimization using method: L-BFGS-B.
               * * *
    
    Tit   = total number of iterations
    Tnf   = total number of function evaluations
    Tnint = total number of segments explored during Cauchy searches
    Skip  = number of BFGS updates skipped
    Nact  = number of active bounds at final generalized Cauchy point
    Projg = norm of the final projected gradient
    F     = final function value
    
               * * *
    
       N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
        2     10     13      1     0     0   7.034D-02   2.972D-03
      F =   2.9720423776281963E-003
    
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
    Parameter:epsilon : [1.5614863]
    Parameter:sigma : [2.06290476]

