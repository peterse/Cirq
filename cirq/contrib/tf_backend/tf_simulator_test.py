"""
    Master TODO list:
        - setup state OBJECT that contains state and qubits

"""


from unittest import mock
import pytest
import sympy

import numpy as np
import tensorflow as tf
import cirq
from cirq.contrib.tf_backend.tf_simulator import (
    TFWaveFunctionSimulator
)

def test_tf_wavefunction_simulator_instantiate():
    _ = TFWaveFunctionSimulator()


def test_tf_wavefunction_simulator_dense_circuit_conversion():

    # FIXME: impose assertions
    circuit = cirq.Circuit.from_ops([
        cirq.YPowGate(exponent=tf.Variable(1, tf.float64))(cirq.LineQubit(0)),
        cirq.YPowGate(exponent=tf.Variable(1, tf.float64))(cirq.LineQubit(1)),
        cirq.XPowGate(exponent=tf.Variable(1, tf.float64))(cirq.LineQubit(0))])
    initial_state = np.asarray([1, 0, 0, 0])
    tf_sim = TFWaveFunctionSimulator().simulate(circuit, initial_state=initial_state)



@pytest.mark.parametrize('dtype', [tf.complex64, tf.complex128])
def test_run_bit_flips(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = TFWaveFunctionSimulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit.from_ops((cirq.X**b0)(q0), (cirq.X**b1)(q1))
            circuit_op = simulator.simulate(circuit)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            wf = sess.run(circuit_op)
        measurements = cirq.sample_state_vector(wf, [0, 1])
        np.testing.assert_array_almost_equal(measurements[0], [b0, b1])
        expected_state = np.zeros(shape=(2, 2))
        expected_state[b0][b1] = 1.0
        cirq.testing.assert_allclose_up_to_global_phase(wf.reshape(-1), np.reshape(expected_state, 4), atol=1e-6)


@pytest.mark.parametrize('dtype', [tf.complex64, tf.complex128])
def test_run_correlations(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = TFWaveFunctionSimulator(dtype=dtype)
    circuit = cirq.Circuit.from_ops(cirq.H(q0), cirq.CNOT(q0, q1))
    for _ in range(10):
        circuit_op = simulator.simulate(circuit)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            wf = sess.run(circuit_op)
        measurements = cirq.sample_state_vector(wf.reshape(-1), [0, 1])




def get_parametrized_two_qubit_gates():
    return [
        cirq.SwapPowGate,
        cirq.CNotPowGate,
        cirq.ISwapPowGate,
        cirq.ZZPowGate,
        cirq.CZ,
    ]

def get_two_qubit_gates():
    return [

    ]

def get_parametrized_single_qubit_gates():
    return [

    ]


@pytest.mark.parametrize(
    'gate', [cirq.X, cirq.Y, cirq.Z, cirq.H, cirq.S, cirq.T])
def test_tf_wavefunction_simulator_vs_cirq_single_qubit_gates():
    return


@pytest.mark.parametrize(
    'gate', [cirq.Rx, cirq.Ry, cirq.Rz])
def test_tf_wavefunction_simulator_vs_cirq_parametrized_single_qubit_gates():
    return


@pytest.mark.parametrize(
    'gate', [cirq.CNOT, cirq.SWAP])
def test_tf_wavefunction_simulator_vs_cirq_two_qubit_gates():
    return


@pytest.mark.parametrize('gate', [cirq.SwapPowGate,
                                  cirq.CNotPowGate,
                                  cirq.ISwapPowGate,
                                  cirq.ZZPowGate,
                                  cirq.CZ])
def test_tf_wavefunction_simulator_vs_cirq_parametrized_two_qubit_gates():
    return


# @pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
# def test_simulate_initial_state(dtype):
#     q0, q1 = cirq.LineQubit.range(2)
#     simulator = cirq.Simulator(dtype=dtype)
#     for b0 in [0, 1]:
#         for b1 in [0, 1]:
#             circuit = cirq.Circuit.from_ops((cirq.X**b0)(q0), (cirq.X**b1)(q1))
#             result = simulator.simulate(circuit, initial_state=1)
#             expected_state = np.zeros(shape=(2, 2))
#             expected_state[b0][1 - b1] = 1.0
#             np.testing.assert_equal(result.final_state,
#                                     np.reshape(expected_state, 4))




if __name__ == "__main__":
    test_run_bit_flips(tf.complex64)
#
# @pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
# def test_run_repetitions_measure_at_end(dtype):
#     q0, q1 = cirq.LineQubit.range(2)
#     simulator = cirq.Simulator(dtype=dtype)
#     with mock.patch.object(simulator, '_base_iterator',
#                            wraps=simulator._base_iterator) as mock_sim:
#         for b0 in [0, 1]:
#             for b1 in [0, 1]:
#                 circuit = cirq.Circuit.from_ops((cirq.X**b0)(q0),
#                                                 (cirq.X**b1)(q1),
#                                                 cirq.measure(q0),
#                                                 cirq.measure(q1))
#                 result = simulator.run(circuit, repetitions=3)
#                 np.testing.assert_equal(result.measurements,
#                                         {'0': [[b0]] * 3, '1': [[b1]] * 3})
#                 assert result.repetitions == 3
#         assert mock_sim.call_count == 4
#
#
# @pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
# def test_run_repetitions_measurement_not_terminal(dtype):
#     q0, q1 = cirq.LineQubit.range(2)
#     simulator = cirq.Simulator(dtype=dtype)
#     with mock.patch.object(simulator, '_base_iterator',
#                            wraps=simulator._base_iterator) as mock_sim:
#         for b0 in [0, 1]:
#             for b1 in [0, 1]:
#                 circuit = cirq.Circuit.from_ops((cirq.X**b0)(q0),
#                                                 (cirq.X**b1)(q1),
#                                                 cirq.measure(q0),
#                                                 cirq.measure(q1),
#                                                 cirq.H(q0),
#                                                 cirq.H(q1))
#                 result = simulator.run(circuit, repetitions=3)
#                 np.testing.assert_equal(result.measurements,
#                                         {'0': [[b0]] * 3, '1': [[b1]] * 3})
#                 assert result.repetitions == 3
#         assert  mock_sim.call_count == 12
#
#
# @pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
# def test_run_param_resolver(dtype):
#     q0, q1 = cirq.LineQubit.range(2)
#     simulator = cirq.Simulator(dtype=dtype)
#     for b0 in [0, 1]:
#         for b1 in [0, 1]:
#             circuit = cirq.Circuit.from_ops((cirq.X**sympy.Symbol('b0'))(q0),
#                                             (cirq.X**sympy.Symbol('b1'))(q1),
#                                             cirq.measure(q0),
#                                             cirq.measure(q1))
#             param_resolver = cirq.ParamResolver({'b0': b0, 'b1': b1})
#             result = simulator.run(circuit, param_resolver=param_resolver)
#             np.testing.assert_equal(result.measurements,
#                                     {'0': [[b0]], '1': [[b1]] })
#             np.testing.assert_equal(result.params, param_resolver)
#
#
# @pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
# def test_run_mixture(dtype):
#     q0 = cirq.LineQubit(0)
#     simulator = cirq.Simulator(dtype=dtype)
#     circuit = cirq.Circuit.from_ops(cirq.bit_flip(0.5)(q0), cirq.measure(q0))
#     result = simulator.run(circuit, repetitions=100)
#     assert sum(result.measurements['0'])[0] < 80
#     assert sum(result.measurements['0'])[0] > 20
#
#
# @pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
# def test_run_mixture_with_gates(dtype):
#     q0 = cirq.LineQubit(0)
#     simulator = cirq.Simulator(dtype=dtype)
#     circuit = cirq.Circuit.from_ops(cirq.H(q0), cirq.phase_flip(0.5)(q0),
#                                     cirq.H(q0), cirq.measure(q0))
#     result = simulator.run(circuit, repetitions=100)
#     assert sum(result.measurements['0'])[0] < 80
#     assert sum(result.measurements['0'])[0] > 20
#
#
#
# @pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
# def test_run_measure_multiple_qubits(dtype):
#     q0, q1 = cirq.LineQubit.range(2)
#     simulator = cirq.Simulator(dtype=dtype)
#     for b0 in [0, 1]:
#         for b1 in [0, 1]:
#             circuit = cirq.Circuit.from_ops((cirq.X**b0)(q0),
#                                             (cirq.X**b1)(q1),
#                                             cirq.measure(q0, q1))
#             result = simulator.run(circuit, repetitions=3)
#             np.testing.assert_equal(result.measurements,
#                                     {'0,1': [[b0, b1]] * 3})
#
#
# @pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
# def test_run_sweeps_param_resolvers(dtype):
#     q0, q1 = cirq.LineQubit.range(2)
#     simulator = cirq.Simulator(dtype=dtype)
#     for b0 in [0, 1]:
#         for b1 in [0, 1]:
#             circuit = cirq.Circuit.from_ops((cirq.X**sympy.Symbol('b0'))(q0),
#                                             (cirq.X**sympy.Symbol('b1'))(q1),
#                                             cirq.measure(q0),
#                                             cirq.measure(q1))
#             params = [cirq.ParamResolver({'b0': b0, 'b1': b1}),
#                       cirq.ParamResolver({'b0': b1, 'b1': b0})]
#             results = simulator.run_sweep(circuit, params=params)
#
#             assert len(results) == 2
#             np.testing.assert_equal(results[0].measurements,
#                                     {'0': [[b0]], '1': [[b1]] })
#             np.testing.assert_equal(results[1].measurements,
#                                     {'0': [[b1]], '1': [[b0]] })
#             assert results[0].params == params[0]
#             assert results[1].params == params[1]
#
#
# @pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
# def test_simulate_random_unitary(dtype):
#     q0, q1 = cirq.LineQubit.range(2)
#     simulator = cirq.Simulator(dtype=dtype)
#     for _ in range(10):
#         random_circuit = cirq.testing.random_circuit(qubits=[q0, q1],
#                                                      n_moments=8,
#                                                      op_density=0.99)
#         circuit_unitary = []
#         for x in range(4):
#             result = simulator.simulate(random_circuit, qubit_order=[q0, q1],
#                                         initial_state=x)
#             circuit_unitary.append(result.final_state)
#         np.testing.assert_almost_equal(
#             np.transpose(circuit_unitary),
#             random_circuit.to_unitary_matrix(qubit_order=[q0, q1]))
#
#
# @pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
# def test_simulate_no_circuit(dtype,):
#     q0, q1 = cirq.LineQubit.range(2)
#     simulator = cirq.Simulator(dtype=dtype)
#     circuit = cirq.Circuit()
#     result = simulator.simulate(circuit, qubit_order=[q0, q1])
#     np.testing.assert_almost_equal(result.final_state,
#                                    np.array([1, 0, 0, 0]))
#     assert len(result.measurements) == 0
#
#
# @pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
# def test_simulate(dtype,):
#     q0, q1 = cirq.LineQubit.range(2)
#     simulator = cirq.Simulator(dtype=dtype)
#     circuit = cirq.Circuit.from_ops(cirq.H(q0), cirq.H(q1))
#     result = simulator.simulate(circuit, qubit_order=[q0, q1])
#     np.testing.assert_almost_equal(result.final_state,
#                                    np.array([0.5, 0.5, 0.5, 0.5]))
#     assert len(result.measurements) == 0
#
#
# @pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
# def test_simulate_mixtures(dtype,):
#     q0 = cirq.LineQubit(0)
#     simulator = cirq.Simulator(dtype=dtype)
#     circuit = cirq.Circuit.from_ops(cirq.bit_flip(0.5)(q0), cirq.measure(q0))
#     count = 0
#     for _ in range(100):
#         result = simulator.simulate(circuit, qubit_order=[q0])
#         if result.measurements['0']:
#             np.testing.assert_almost_equal(result.final_state,
#                                             np.array([0, 1]))
#             count += 1
#         else:
#             np.testing.assert_almost_equal(result.final_state,
#                                            np.array([1, 0]))
#     assert count < 80 and count > 20
#
#
