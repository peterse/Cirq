
import cirq
from cirq.contrib.tpu import (
    circuit_to_tensorflow_runnable
)
import tensorflow as tf
import pysnooper

import numpy as np
q0 = cirq.LineQubit(0)
init_t = np.pi /2

t = tf.placeholder(dtype=tf.float64)

c = cirq.Circuit.from_ops(cirq.YPowGate(exponent=t)(q0))
simulator = cirq.DensityMatrixSimulator()

# convert to tf

r = circuit_to_tensorflow_runnable(c)
with tf.Session() as session:
    output = session.run(r.compute(), feed_dict=r.feed_dict)
print(output)
