import numpy as np
import tensorflow as tf
from typing import List
import cirq

# most basic test possible...
# Xgate = tf.convert_to_tensor([[0,1],[1,0]])
# x = tf.convert_to_tensor([0,1])
# x = tf.einsum("ij,i", Xgate, x)


# parametrized
init_t = np.pi /2
theta = tf.Variable(init_t)
theta = tf.cast(theta, tf.complex64)
RXgate = tf.convert_to_tensor([
    [tf.cos(theta), -1.0j * tf.sin(theta)],
    [-1.0j * tf.sin(theta), tf.cos(theta)],
])
x = tf.convert_to_tensor([0,1], dtype=tf.complex64)
x = tf.einsum("ij,i", RXgate, x)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(x)
    print(result)
    print([n.name for n in tf.get_default_graph().as_graph_def().node])
