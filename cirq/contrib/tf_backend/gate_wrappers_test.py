import cirq
import tensorflow as tf
import numpy as np

from cirq.contrib.tf_backend.gate_wrappers import (
    tf_gate_wrapper,
    ALL_WRAPPERS,
)

"""
LOCAL TODO:
    - parametrize over all remaining gates
"""

def q(i):
    return cirq.LineQubit(i)


def test_tf_gate_wrapper_gate_inheritance():
    """Wrapping a gate instance with a wrapper base class."""
    np.set_printoptions(precision=2)
    for g in [cirq.X, cirq.Y, cirq.Z, cirq.H]:
        inst = g(q(0))
        wrapped = tf_gate_wrapper(inst, tf.complex64)
        with tf.Session() as sess:
            tf_inst = sess.run(wrapped._tensor)
        np.testing.assert_array_almost_equal(cirq.unitary(inst), tf_inst)
    for g in [cirq.Rx, cirq.Ry, cirq.Rz]:
        inst = g(0)(q(0))
        wrapped = tf_gate_wrapper(inst, tf.complex64)
        with tf.Session() as sess:
            tf_inst = sess.run(wrapped._tensor)
        np.testing.assert_array_almost_equal(cirq.unitary(inst), tf_inst)

    for g in [cirq.CNOT, cirq.SWAP]:
        inst = g(q(0), q(1))
        wrapped = tf_gate_wrapper(inst, tf.complex64)
        with tf.Session() as sess:
            tf_inst = sess.run(wrapped._tensor).reshape((4,4))
        np.testing.assert_array_almost_equal(cirq.unitary(inst), tf_inst)


def test_tf_gate_wrapper_eigengate():
    for g in [cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate,  cirq.HPowGate]:
        inst = g(exponent=1)(q(0))
        wrapped = tf_gate_wrapper(inst, tf.complex64)
        with tf.Session() as sess:
            tf_inst = sess.run(wrapped._tensor)
        np.testing.assert_array_almost_equal(cirq.unitary(inst), tf_inst)
    for g in [cirq.CNotPowGate, cirq.SwapPowGate,]:
        inst = g(exponent=3.84)(q(0), q(1))
        wrapped = tf_gate_wrapper(inst, tf.complex64)
        with tf.Session() as sess:
            tf_inst = sess.run(wrapped._tensor)
        np.testing.assert_array_almost_equal(cirq.unitary(inst), tf_inst.reshape((4,4)))

def test_tf_gate_wrapper_parity_gate():
    for g in [cirq.ZZPowGate]:
        inst = g(exponent=3.84)(q(0), q(1))
        a = tf_gate_wrapper(inst)
        print(a)

def test_tf_gate_wrapper_tensor_inputs():
    # TODO
    init_t = np.pi /2
    t = tf.Variable(init_t)
    inst = cirq.YPowGate(exponent=t)(q(0))
    wrapped = tf_gate_wrapper(inst)
    print(wrapped._tensor)



test_tf_gate_wrapper_parity_gate()
