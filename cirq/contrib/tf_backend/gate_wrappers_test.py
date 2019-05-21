import cirq
import tensorflow as tf
import numpy as np

from cirq.contrib.tf_backend.gate_wrappers import (
    tf_gate_wrapper
)

"""
LOCAL TODO:
    - parametrize over all remaining gates
"""
Q0 = cirq.LineQubit(0)


def test_tf_gate_wrapper_inst():
    inst = cirq.YPowGate(exponent=1.0)(Q0)
    tf_gate_wrapper(inst)

def test_tf_gate_wrapper_variable():
    init_t = np.pi /2
    t = tf.Variable(init_t)
    inst = cirq.YPowGate(exponent=t)(Q0)
    wrapped = tf_gate_wrapper(inst)
    print(wrapped._tensor)



test_tf_gate_wrapper_variable()
