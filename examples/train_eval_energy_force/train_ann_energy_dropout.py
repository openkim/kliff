from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, xavier_initializer, variance_scaling_initializer
from descriptor import get_descriptor
from openkim_fit.dataset import DataSet
import openkim_fit.ann as ann
import random
import time

start = time.time()

# set a global random seed
tf.set_random_seed(1)
DTYPE = tf.float32
NUM_EPOCHS = 60
BATCH_SIZE = 4


# create all descriptors
desc = get_descriptor()
num_desc = desc.get_num_descriptors()


#######################################
# read data
#######################################
# read config and reference data
tset = DataSet()
#tset.read('./training_set/T300_xyz_tiny')
tset.read('./training_set/T300_xyz_small_not_same_size')
#tset.read('./training_set/T300_xyz_medium')
#tset.read('./training_set/T300_xyz_100')
#tset.read('./training_set/T300_xyz')
configs = tset.get_configs()


# preprocess data to generate tfrecords
train_name, validation_name = ann.convert_raw_to_tfrecords(configs, desc,
    size_validation = 2, directory='./dataset_tfrecords', do_generate=False,
    do_shuffle=True)
# read data from tfrecords into tensors
dataset = ann.read_from_tfrecords(train_name)
# number of epoches
dataset = dataset.repeat(NUM_EPOCHS)
iterator = dataset.make_one_shot_iterator()


#######################################
# create graph
#######################################

# decorator to output statistics of variables
initializer = ann.weight_decorator(xavier_initializer())
layer = ann.layer_decorator(fully_connected)

size = 20
keep_ratio = 0.9
subloss = []

for i in range(BATCH_SIZE):

  conf_name, num_atoms,atomic_coords,gen_coords,dgen_datomic_coords,energy_label,forces_label = iterator.get_next()

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
  subloss.append(tf.square(energy - energy_label))

# loss
loss = tf.reduce_mean(subloss)
# add regularization
#lam = 1e2
#for w, b in zip(weights, biases):
#  loss += lam * (tf.reduce_sum(tf.square(w)) + tf.reduce_sum(tf.square(b)) )

# log loss
tf.summary.scalar('loss_value', loss)

# optimization
with tf.name_scope('train_node'):
  optimizer = tf.train.AdamOptimizer(0.01)
  train = optimizer.minimize(loss)

# checkpoint saver
saver = tf.train.Saver(max_to_keep=100)


# get weights and variables
weights, biases = ann.get_weights_and_biases(['hidden1', 'hidden2', 'output'])



#######################################
# execting graph
#######################################
with tf.Session() as sess:

  # Merge all the summaries and write them out to /tmp/tensorflow/potential
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter('/tmp/tensorflow/train', sess.graph)

  # init global vars
  init_op = tf.global_variables_initializer()
  sess.run(init_op)

  try:

    i = 0
    while True:
      try:
        sess.run(train)

        if i%10 == 0:
          out, summary = sess.run([loss, merged])
          save_path = saver.save(sess, "/tmp/tensorflow/ckpt/model.ckpt", global_step=i)
          train_writer.add_summary(summary, i)

          # output results to a KIM model
          w,b = sess.run([weights, biases])
          ann.write_kim_ann(desc, w, b, tf.nn.tanh, fname='ann_kim.params-{}'.format(i))

          print ('i =',i, 'loss =', out)

        i += 1
      except tf.errors.OutOfRangeError:
        break
    #NOTE, we may need to run the last batch of data here

  finally:
    # output results to a KIM model
    w,b = sess.run([weights, biases])
    ann.write_kim_ann(desc, w, b, tf.nn.tanh)
    print('total running time:', time.time() - start)

