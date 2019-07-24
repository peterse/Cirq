import tensorflow as tf
import cirq
from cirq.contrib.tf_backend.tf_simulator import (
    TFWaveFunctionSimulator
)
from cirq.contrib.tf_backend.utils import tensorboard_session

import numpy as np

def q(i):
    return cirq.LineQubit(i)

def compile_tensorboard_session():
    theta = tf.Variable(np.pi)
    tf.summary.scalar('theta', theta)
    circuit = cirq.Circuit.from_ops(
        cirq.Ry(theta)(q(0)),
        cirq.CNOT(q(0), q(1))
    )

    initial_state = np.asarray([1, 0, 0, 0])
    circuit_op = TFWaveFunctionSimulator(dtype=tf.complex64).simulate(circuit, initial_state=initial_state)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        wf = sess.run(circuit_op)

    tensorboard_session(wf, {theta:np.pi/2})

if __name__ == "__main__":
    compile_tensorboard_session()
