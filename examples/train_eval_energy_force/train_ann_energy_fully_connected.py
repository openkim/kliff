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
NUM_EPOCHS = 10
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
tset.read('./training_set/T300_xyz_small')
#tset.read('./training_set/T300_xyz_medium')
#tset.read('./training_set/T300_xyz_100')
#tset.read('./training_set/T300_xyz')
configs = tset.get_configs()


# preprocess data to generate tfrecords
train_name, validation_name = ann.convert_raw_to_tfrecords(configs, desc,
    size_validation = 2, directory='./dataset_tfrecords', do_generate=True,
    do_shuffle=True)
# read data from tfrecords into tensors
dataset = ann.read_from_tfrecords(train_name)
# number of epoches
dataset = dataset.repeat(NUM_EPOCHS)
# batch size
dataset = dataset.batch(BATCH_SIZE)
iterator = dataset.make_one_shot_iterator()
next_batch = iterator.get_next()


#######################################
# create graph
#######################################
atomic_coords = next_batch[0]
gen_coords = next_batch[1]
dgen_datomic_coords = next_batch[2]
energy_label = next_batch[3]
forces_label = next_batch[4]


# decorator to output statistics of variables
initializer = ann.weight_decorator(xavier_initializer())
layer = ann.layer_decorator(fully_connected)

size = 20
subloss = []
for i in range(BATCH_SIZE):

  # input layer
  in_layer = ann.input_layer_given_data(atomic_coords[i], gen_coords[i], dgen_datomic_coords[i], num_descriptor=num_desc)

  if i == 0:  # create weights and biases
    hidden1 = layer(in_layer, size, activation_fn=tf.nn.tanh, weights_initializer=initializer, scope='hidden1')
  else: # reuse weights and biases
    hidden1 = layer(in_layer, size, activation_fn=tf.nn.tanh, weights_initializer=initializer, reuse=True, scope='hidden1')

  if i == 0:  # create weights and biases
    hidden2 = layer(hidden1, size, activation_fn=tf.nn.tanh, weights_initializer=initializer, scope='hidden2')
  else: # reuse weights and biases
    hidden2 = layer(hidden1, size, activation_fn=tf.nn.tanh, weights_initializer=initializer, reuse=True, scope='hidden2')

  if i == 0:  # create weights and biases
    output = layer(hidden2, 1, activation_fn=None, weights_initializer=initializer, scope='output')
  else: # reuse weights and biases
    output = layer(hidden2, 1, activation_fn=None, weights_initializer=initializer, reuse=True, scope='output')

  energy = tf.reduce_sum(output)
  subloss.append(tf.square(energy - energy_label[i]))

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

