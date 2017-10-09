from dataset import DataSet
from keywords import InputKeywords
from modelparams import ModelParams
#from kimobject import init_KIMobjects
from kimcalculator import KIMcalculator
import sys
sys.path.append('../libs/geodesicLMv1.1/pythonInterface')
from geodesiclm import geodesiclm
from traincost import Cost
from scipy.optimize import least_squares, minimize
import numpy as np
import random


def split_kfold(preds, refs, kfold):
  '''split the data into "kfold" of equal size as much as possible.'''
  combined = zip(preds, refs)
  random.shuffle(combined)
  preds,refs = zip(*combined)

  avg_size = int(len(preds)/kfold)
  remainder = len(preds) % kfold
  fold_size = []

#NOTE we can use np.arrray_split to achieve the nearly even split
  for i in range(kfold):
    if i < remainder:
      fold_size.append(avg_size+1)
    else:
      fold_size.append(avg_size)
  preds_fold = []
  refs_fold = []
  for i in range(kfold):
    end = sum(fold_size[:i+1])
    start = end - fold_size[i]
    preds_fold.append(preds[start:end])
    refs_fold.append(refs[start:end])
  return preds_fold, refs_fold



if __name__ == '__main__':

  # KIM model parameters
  modelname = 'Pair_Lennard_Jones_Truncated_Nguyen_Ar__MO_398194508715_000'
  modelname = 'EDIP_BOP_Bazant_Kaxiras_Si__MO_958932894036_001'
  modelname = 'Three_Body_Stillinger_Weber_MoS__MO_000000111111_001'
  params = ModelParams(modelname)
  params.echo_avail_params()
  #fname = '../tests/test_params.txt'
  #params.read(fname)
  #param_A = ('PARAM_FREE_A',
  #     ('kim', 0, 20),
  #     ('kim', 'fix'),
  #     ('kim', 0.1, 9.9))
  #params.set_param(param_A)
  #params.echo_params()
  #params.set_cutoff(6.0)

  fname = '../tests/mos2_init_guess.txt'
  params.read(fname)
  params.echo_params()
  # initial guess of params
  x0 = params.get_x0()

  # read config and reference data
  tset = DataSet()
  #tset.read('../tests/config.txt_20x20')
  #tset.read('../tests/T150_training_1000.xyz')
  #tset.read('../tests/training_set-T300/')
  tset.read('../tests/training_set/')
  configs = tset.get_configs()


  # prediction
  KIMobjs=[]
  for i in range(len(configs)):
    obj = KIMcalculator(modelname, params, configs[i])
    obj.initialize()
    #obj.update_params()
    #obj.compute()
    KIMobjs.append(obj)

  # reference
  refs = []
  for i in range(len(configs)):
    f = configs[i].get_forces()
    refs.append(f)


  kfold = 10
  KIMobjs_kfold,refs_kfold = split_kfold(KIMobjs, refs, kfold)
  fitted_params = []
  train_cost = []
  test_cost = []


  for i in range(kfold):
    KIMobjs_test = KIMobjs_kfold[i]
    refs_test = refs_kfold[i]
    KIMobjs_train = np.concatenate(KIMobjs_kfold[:i] + KIMobjs_kfold[i+1:])
    refs_train = np.concatenate(refs_kfold[:i] + refs_kfold[i+1:])
    # cost function
    cst_train = Cost(params)
    cst_test = Cost(params)
    for j in range(len(KIMobjs_train)):
      cst_train.add(KIMobjs_train[j], refs_train[j], weight=1)
    for j in range(len(KIMobjs_test)):
      cst_test.add(KIMobjs_test[j], refs_test[j], weight=1)

    # optimize
    func = cst_train.get_residual
    #method = 'CG'
    #method = 'geodesiclm'
    method = 'scipy-lm'
    if method == 'geodesiclm':
      print 'started geodesic lm'
      xf = geodesiclm(func, x0, h1=1e-5, h2=0.1, factoraccept=5, factorreject=2,
                imethod=2, initialfactor=1, damp_model=0, print_level=2,
                maxiters=10000, Cgoal=0.5e-7, artol=1.E-8, gtol=1.5e-8, xtol=1e-6,
                xrtol=1.5e-6, ftol=-1, frtol=1.5e-6)

    elif method == 'scipy-lm':
      print 'started leqst_squares'
      res_1 = least_squares(func, x0, method='trf', verbose=2)
      print 'end leqst_squares'
    elif method == 'CG':
      print 'started CG'
      func = cst_train.get_cost
      minimize(func, x0, method='CG')


    x = params.get_x0()   # actually x equal xf
    print 'fitted params', x
    params.echo_params(fname='fitted_params'+str(i))
    fitted_params.append(x)

    cost = cst_train.get_cost(x)
    train_cost.append(cost)

        # optimize
    cost = cst_test.get_cost(x)
    test_cost.append(cost)


  print 'mean params', np.mean(fitted_params, axis=0)
  print 'train cost', train_cost
  print 'mean of train cost', np.mean(train_cost)
  print 'test cost', test_cost
  print 'mean of test cost', np.mean(test_cost)


