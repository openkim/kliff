#
# Workaround to mimic multiprocessing.Pool.map, which does not support class method
# as the function for Pool, and requires that the arguments for map are picklable.
#
# Adopted from answer by klaus at StackOverflow:
# https://stackoverflow.com/questions/3288595/multiprocessing-how-to-use-pool-map-on-a-function-defined-in-a-class
#

import multiprocessing as mp
from random import shuffle
import numpy as np

#
# Accept class method as function, but X needs to be picklable.
#
def parmap(f, X, nprocs, *args):
  q_in = mp.Queue(1)
  q_out = mp.Queue()

  proc = [mp.Process(target=func, args=(f, q_in, q_out)) for _ in range(nprocs)]
  for p in proc:
    p.daemon = True
    p.start()

  sent = [q_in.put((i, x, args)) for i, x in enumerate(X)]
  [q_in.put((None, None, None)) for _ in range(nprocs)]
  res = [q_out.get() for _ in range(len(sent))]

  [p.join() for p in proc]

  return [x for i, x in sorted(res)]


def func(f, q_in, q_out):
  while True:
    i, x, args = q_in.get()
    if i is None:
      break
    q_out.put((i, f(x, *args)))


#
# Accept class method as function, and X needs not to be picklable.
#
# TODO `pairs` and `sorted` may be removed since order here is not important
#
def parmap2(f, X, nprocs, *args):

  pairs = [(i, x) for i,x in enumerate(X)]

  # shuffle and divide into `nprocs` equally-numbered parts
  shuffle(pairs)
  groups = np.array_split(pairs, nprocs)

  processes = []
  managers = []
  for g in range(nprocs):
    manager_end, worker_end = mp.Pipe(duplex=False)
    p = mp.Process(target=func2, args=(f, groups[g], args, worker_end,))
    p.daemon = True
    p.start()
    processes.append(p)
    managers.append(manager_end)

  results = []
  for m in managers:
    results.extend(m.recv())
  [p.join() for p in processes]

  results = [r for i, r in sorted(results)]
  return results


def func2(f, X, args, worker_end):
  results = []
  for i,x in X:
    results.append((i, f(x, *args)))
  worker_end.send(results)


if __name__ == '__main__':

  def sq_cub(i, plus):
    return i**2+plus, i**3+plus


  rslt = parmap(sq_cub, [1, 2, 3, 4, 6, 7, 8], 2, 4)
  print (rslt)
