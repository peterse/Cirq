import numpy as np
import tensorflow as tf
from typing import List
import cirq

from cirq.contrib.tf_backend.tf_apply_unitary import (
    tf_apply_unitary,
    ApplyTFUnitaryArgs,
)
from cirq.contrib.tf_backend.utils import tensorboard_session

# unparametrized, no buffer necessary
# Xgate = tf.convert_to_tensor([[0,1],[1,0]])
# x = tf.convert_to_tensor([0,1])
# x = tf.einsum("ij,i", Xgate, x)


# parametrized, no buffer necessary
# init_t = np.pi /2
# theta = tf.Variable(init_t)
# theta = tf.cast(theta, tf.float64)
# RXgate = tf.convert_to_tensor([
#     [tf.cos(theta), -1.0 * tf.sin(theta)],
#     [-1.0 * tf.sin(theta), tf.cos(theta)],
# ])
# tf.summary.scalar('theta', theta)
# x = tf.convert_to_tensor([0,1], dtype=tf.float64)
# x = tf.einsum("ij,i", RXgate, x)

# unparametrized X gate, use buffer
Xgate = tf.convert_to_tensor([[0,1],[1,0]], dtype=tf.float64)
wf = tf.convert_to_tensor([0,1], dtype=tf.float64)
buf = tf.Variable(tf.zeros_like(wf, dtype=tf.float64))
theta = tf.Variable(0)
tf.summary.scalar('theta', theta)

# stand-in for apply_unitary
zero = cirq.linalg.slice_for_qubits_equal_to([0], 0)
one = cirq.linalg.slice_for_qubits_equal_to([0], 1)
inds = [0]
ref0 = wf[one]
ref1 = wf[zero]
refs = [ref0, ref1]
x = buf
with tf.control_dependencies([x[inds[i]].assign(refs[i]) for i in range(1)]):
    x = tf.identity(x)


tensorboard_session(x, {theta:.37})
