"""
    Master TODO list:
        - setup state OBJECT that contains state and qubits

"""



import numpy as np
import tensorflow as tf
import cirq
from cirq.contrib.tf_backend.tf_simulator import (
    TFWaveFunctionSimulator
)

INITIAL_STATE = np.asarray([1, 0])
TEST_VAR = tf.Variable(1.0)
TEST_GATES = [
    cirq.YPowGate(exponent=TEST_VAR)(cirq.LineQubit(0)),
    cirq.YPowGate(exponent=TEST_VAR)(cirq.LineQubit(0))
]
TEST_CIRCUIT = cirq.Circuit.from_ops(TEST_GATES)


def test_tf_wavefunction_simulator_instantiate():
    _ = TFWaveFunctionSimulator()


def test_tf_wavefunction_simulator_dense_circuit_conversion():

    # single qubit
    tf_sim = TFWaveFunctionSimulator().simulate(TEST_CIRCUIT, INITIAL_STATE)


test_tf_wavefunction_simulator_dense_circuit_conversion()
