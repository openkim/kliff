from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops


@ops.RegisterGradient("IntPot")
def _int_pot_grad(op, *grad_outputs):
  """The gradients for `zero_out`.
    Args:
  op: The `int_pot` `Operation` that we are differentiating, which we can use
    to find the inputs and outputs of the original op.
  grads: Gradients with respect to the outputs of the `int_pot` op.

  Returns:
    Gradients with respect to the input of `int_pot`.
  """

  dgen_datomic_coords = op.outputs[1]
  grad1, dummy = grad_outputs
  grad_input = tf.tensordot(grad1, dgen_datomic_coords, axes=([0,1], [0,1]))

  return [grad_input, None, None] # List of one Tensor, since we have 7 input
