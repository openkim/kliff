from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, xavier_initializer, variance_scaling_initializer
from descriptor import get_descriptor
import openkim_fit.ann as ann
import random

# set a global random seed
tf.set_random_seed(1)
DTYPE = tf.float32
EVAL_SIZE = 5

# create all descriptors
desc = get_descriptor()
num_desc = desc.get_num_descriptors()

#######################################
# read data
#######################################
dataset = ann.read_from_tfrecords('./dataset_tfrecords/validation.tfrecords')
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
num_atoms,atomic_coords,gen_coords,dgen_datomic_coords,energy_label,forces_label = next_element


#######################################
# create graph
#######################################

# decorator to output statistics of variables
initializer = ann.weight_decorator(xavier_initializer())
layer = ann.layer_decorator(fully_connected)

size = 20
keep_ratio = 0.9
all_energy = []

for i in range(EVAL_SIZE):

  # input layer
  in_layer = ann.input_layer_given_data(atomic_coords, gen_coords,
      dgen_datomic_coords, num_descriptor=num_desc)
  # dropout
  in_layer_drop = tf.nn.dropout(in_layer, keep_ratio, seed=i)

  if i == 0:  # create weights and biases
    hidden1 = layer(in_layer_drop, size, activation_fn=tf.nn.tanh,
        weights_initializer=initializer, scope='hidden1')
    hidden1_drop = tf.nn.dropout(hidden1, keep_ratio, seed=i)
  else: # reuse weights and biases
    hidden1 = layer(in_layer_drop, size, activation_fn=tf.nn.tanh,
        weights_initializer=initializer, reuse=True, scope='hidden1')
    hidden1_drop = tf.nn.dropout(hidden1, keep_ratio, seed=i)

  if i == 0:  # create weights and biases
    hidden2 = layer(hidden1_drop, size, activation_fn=tf.nn.tanh,
        weights_initializer=initializer, scope='hidden2')
    hidden2_drop = tf.nn.dropout(hidden2, keep_ratio, seed=i)
  else: # reuse weights and biases
    hidden2 = layer(hidden1_drop, size, activation_fn=tf.nn.tanh,
        weights_initializer=initializer, reuse=True, scope='hidden2')
    hidden2_drop = tf.nn.dropout(hidden2, keep_ratio, seed=i)

  if i == 0:  # create weights and biases
    output = layer(hidden2_drop, 1, activation_fn=None,
        weights_initializer=initializer, scope='output')
  else: # reuse weights and biases
    output = layer(hidden2_drop, 1, activation_fn=None,
        weights_initializer=initializer, reuse=True, scope='output')

  # keep record of output
  energy = tf.reduce_sum(output)
  all_energy.append(energy)

# checkpoint saver
saver = tf.train.Saver()

#######################################
# execting graph
#######################################
with tf.Session() as sess:

  # restore parameters
  ckpt_step = 20
  saver.restore(sess, "/tmp/tensorflow/ckpt/model.ckpt-{}".format(ckpt_step))

  i = 0
  while True:
    try:
      # restore weight and biases parameters
      E = sess.run(all_energy)
      m = np.mean(E)
      v = np.std(E)
      print('config # =', i, 'mean energy = ', m, 'std =', np.sqrt(v))
      i += 1
    except tf.errors.OutOfRangeError:
      break



