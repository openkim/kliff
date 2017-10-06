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
NUM_EPOCHS = 100
BATCH_SIZE = 4

# create all descriptors
desc = get_descriptor()
num_desc = desc.get_num_descriptors()


#######################################
# read data
#######################################
dataset = ann.read_from_tfrecords('./dataset_tfrecords/validation.tfrecords')
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
keep_ratio = 1
subloss = []

for i in range(BATCH_SIZE):

  num_atoms,atomic_coords,gen_coords,dgen_datomic_coords,energy_label,forces_label = iterator.get_next()

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

# checkpoint saver
saver = tf.train.Saver()

# Merge all the summaries and write them out to /tmp/tensorflow/potential
merged = tf.summary.merge_all()


ckpt_step = 10
ckpt_start = 0
ckpt_end = 100

#######################################
# execting graph
#######################################
with tf.Session() as sess:

  train_writer = tf.summary.FileWriter('/tmp/tensorflow/validation', sess.graph)

  for i in range(ckpt_start, ckpt_end, ckpt_step):
    saver.restore(sess, "/tmp/tensorflow/ckpt/model.ckpt-{}".format(i))
    out, summary = sess.run([loss, merged])
    train_writer.add_summary(summary, i)
    print('eval error, i =', i, 'loss =', out)

