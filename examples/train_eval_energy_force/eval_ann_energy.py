from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from descriptor import get_descriptor
import openkim_fit.ann as ann
import random

# set a global random seed
tf.set_random_seed(1)
DTYPE = tf.float32
NUM_EPOCHS = 100
BATCH_SIZE = 2

# create all descriptors
desc = get_descriptor()

# read data
dataset = ann.read_from_tfrecords('./dataset_tfrecords/validation.tfrecords')
# number of epoches
dataset = dataset.repeat(NUM_EPOCHS)
# batch size
dataset = dataset.batch(BATCH_SIZE)
iterator = dataset.make_one_shot_iterator()
next_batch = iterator.get_next()

# create shared params (we need to share params among different config, so create first)
num_desc = desc.get_num_descriptors()
weights, biases = ann.parameters(num_desc, [20, 20, 1], dtype=DTYPE)


#######################################
# create graph
#######################################
conf_name = next_batch[0]
num_atoms = next_batch[1]
atomic_coords = next_batch[2]
gen_coords = next_batch[3]
dgen_datomic_coords = next_batch[4]
energy_label = next_batch[5]
forces_label = next_batch[6]


subloss = []
for i in range(BATCH_SIZE):
  in_layer = ann.input_layer_given_data(atomic_coords[i], gen_coords[i],
      dgen_datomic_coords[i])
  dense1 = ann.nn_layer(in_layer, weights[0], biases[0], 'hidden1',act=tf.nn.tanh)
  dense2 = ann.nn_layer(dense1, weights[1], biases[1], 'hidden2', act=tf.nn.tanh)
  output = ann.output_layer(dense2, weights[2], biases[2], 'outlayer')

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

# checkpoint saver
saver = tf.train.Saver()

#######################################
# execting graph
#######################################
with tf.Session() as sess:

  # Merge all the summaries and write them out to /tmp/tensorflow/potential
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter('/tmp/tensorflow/validation', sess.graph)

  ckpt_start = 0
  ckpt_step = 10
  ckpt_end = 30
  for ckpt in range(ckpt_start,ckpt_end,ckpt_step):
    # restore check points data
    saver.restore(sess, "/tmp/tensorflow/ckpt/model.ckpt-{}".format(ckpt))
    out, summary = sess.run([loss, merged])
    train_writer.add_summary(summary, i)
    print('eval error, ckpt =', ckpt, 'loss =', out)

