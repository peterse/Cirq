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

from unittest import mock
import numpy as np
import pytest
import sympy

import cirq


def test_invalid_dtype():
    with pytest.raises(ValueError, match='complex'):
        cirq.Simulator(dtype=np.int32)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_no_measurements(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)

    circuit = cirq.Circuit.from_ops(cirq.X(q0), cirq.X(q1))
    with pytest.raises(ValueError, match="no measurements"):
        simulator.run(circuit)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_no_results(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)

    circuit = cirq.Circuit.from_ops(cirq.X(q0), cirq.X(q1))
    with pytest.raises(ValueError, match="no measurements"):
        simulator.run(circuit)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_empty_circuit(dtype):
    simulator = cirq.Simulator(dtype=dtype)
    with pytest.raises(ValueError, match="no measurements"):
        simulator.run(cirq.Circuit())


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_bit_flips(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit.from_ops((cirq.X**b0)(q0),
                                            (cirq.X**b1)(q1),
                                            cirq.measure(q0),
                                            cirq.measure(q1))
            result = simulator.run(circuit)
            np.testing.assert_equal(result.measurements,
                                    {'0': [[b0]], '1': [[b1]]})


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_repetitions_measure_at_end(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)
    with mock.patch.object(simulator, '_base_iterator',
                           wraps=simulator._base_iterator) as mock_sim:
        for b0 in [0, 1]:
            for b1 in [0, 1]:
                circuit = cirq.Circuit.from_ops((cirq.X**b0)(q0),
                                                (cirq.X**b1)(q1),
                                                cirq.measure(q0),
                                                cirq.measure(q1))
                result = simulator.run(circuit, repetitions=3)
                np.testing.assert_equal(result.measurements,
                                        {'0': [[b0]] * 3, '1': [[b1]] * 3})
                assert result.repetitions == 3
        assert mock_sim.call_count == 4


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_repetitions_measurement_not_terminal(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)
    with mock.patch.object(simulator, '_base_iterator',
                           wraps=simulator._base_iterator) as mock_sim:
        for b0 in [0, 1]:
            for b1 in [0, 1]:
                circuit = cirq.Circuit.from_ops((cirq.X**b0)(q0),
                                                (cirq.X**b1)(q1),
                                                cirq.measure(q0),
                                                cirq.measure(q1),
                                                cirq.H(q0),
                                                cirq.H(q1))
                result = simulator.run(circuit, repetitions=3)
                np.testing.assert_equal(result.measurements,
                                        {'0': [[b0]] * 3, '1': [[b1]] * 3})
                assert result.repetitions == 3
        assert  mock_sim.call_count == 12


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_param_resolver(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit.from_ops((cirq.X**sympy.Symbol('b0'))(q0),
                                            (cirq.X**sympy.Symbol('b1'))(q1),
                                            cirq.measure(q0),
                                            cirq.measure(q1))
            param_resolver = cirq.ParamResolver({'b0': b0, 'b1': b1})
            result = simulator.run(circuit, param_resolver=param_resolver)
            np.testing.assert_equal(result.measurements,
                                    {'0': [[b0]], '1': [[b1]] })
            np.testing.assert_equal(result.params, param_resolver)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_mixture(dtype):
    q0 = cirq.LineQubit(0)
    simulator = cirq.Simulator(dtype=dtype)
    circuit = cirq.Circuit.from_ops(cirq.bit_flip(0.5)(q0), cirq.measure(q0))
    result = simulator.run(circuit, repetitions=100)
    assert sum(result.measurements['0'])[0] < 80
    assert sum(result.measurements['0'])[0] > 20


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_mixture_with_gates(dtype):
    q0 = cirq.LineQubit(0)
    simulator = cirq.Simulator(dtype=dtype)
    circuit = cirq.Circuit.from_ops(cirq.H(q0), cirq.phase_flip(0.5)(q0),
                                    cirq.H(q0), cirq.measure(q0))
    result = simulator.run(circuit, repetitions=100)
    assert sum(result.measurements['0'])[0] < 80
    assert sum(result.measurements['0'])[0] > 20


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_correlations(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)
    circuit = cirq.Circuit.from_ops(cirq.H(q0), cirq.CNOT(q0, q1),
                                    cirq.measure(q0, q1))
    for _ in range(10):
        result = simulator.run(circuit)
        bits = result.measurements['0,1'][0]
        assert bits[0] == bits[1]


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_measure_multiple_qubits(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit.from_ops((cirq.X**b0)(q0),
                                            (cirq.X**b1)(q1),
                                            cirq.measure(q0, q1))
            result = simulator.run(circuit, repetitions=3)
            np.testing.assert_equal(result.measurements,
                                    {'0,1': [[b0, b1]] * 3})


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_sweeps_param_resolvers(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit.from_ops((cirq.X**sympy.Symbol('b0'))(q0),
                                            (cirq.X**sympy.Symbol('b1'))(q1),
                                            cirq.measure(q0),
                                            cirq.measure(q1))
            params = [cirq.ParamResolver({'b0': b0, 'b1': b1}),
                      cirq.ParamResolver({'b0': b1, 'b1': b0})]
            results = simulator.run_sweep(circuit, params=params)

            assert len(results) == 2
            np.testing.assert_equal(results[0].measurements,
                                    {'0': [[b0]], '1': [[b1]] })
            np.testing.assert_equal(results[1].measurements,
                                    {'0': [[b1]], '1': [[b0]] })
            assert results[0].params == params[0]
            assert results[1].params == params[1]


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_random_unitary(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)
    for _ in range(10):
        random_circuit = cirq.testing.random_circuit(qubits=[q0, q1],
                                                     n_moments=8,
                                                     op_density=0.99)
        circuit_unitary = []
        for x in range(4):
            result = simulator.simulate(random_circuit, qubit_order=[q0, q1],
                                        initial_state=x)
            circuit_unitary.append(result.final_state)
        np.testing.assert_almost_equal(
            np.transpose(circuit_unitary),
            random_circuit.to_unitary_matrix(qubit_order=[q0, q1]))


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_no_circuit(dtype,):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)
    circuit = cirq.Circuit()
    result = simulator.simulate(circuit, qubit_order=[q0, q1])
    np.testing.assert_almost_equal(result.final_state,
                                   np.array([1, 0, 0, 0]))
    assert len(result.measurements) == 0


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate(dtype,):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)
    circuit = cirq.Circuit.from_ops(cirq.H(q0), cirq.H(q1))
    result = simulator.simulate(circuit, qubit_order=[q0, q1])
    np.testing.assert_almost_equal(result.final_state,
                                   np.array([0.5, 0.5, 0.5, 0.5]))
    assert len(result.measurements) == 0


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_mixtures(dtype,):
    q0 = cirq.LineQubit(0)
    simulator = cirq.Simulator(dtype=dtype)
    circuit = cirq.Circuit.from_ops(cirq.bit_flip(0.5)(q0), cirq.measure(q0))
    count = 0
    for _ in range(100):
        result = simulator.simulate(circuit, qubit_order=[q0])
        if result.measurements['0']:
            np.testing.assert_almost_equal(result.final_state,
                                            np.array([0, 1]))
            count += 1
        else:
            np.testing.assert_almost_equal(result.final_state,
                                           np.array([1, 0]))
    assert count < 80 and count > 20


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_bit_flips(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit.from_ops((cirq.X**b0)(q0),
                                            (cirq.X**b1)(q1),
                                            cirq.measure(q0),
                                            cirq.measure(q1))
            result = simulator.simulate(circuit)
            np.testing.assert_equal(result.measurements, {'0': [b0], '1': [b1]})
            expected_state = np.zeros(shape=(2, 2))
            expected_state[b0][b1] = 1.0
            np.testing.assert_equal(result.final_state,
                                    np.reshape(expected_state, 4))


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_initial_state(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit.from_ops((cirq.X**b0)(q0), (cirq.X**b1)(q1))
            result = simulator.simulate(circuit, initial_state=1)
            expected_state = np.zeros(shape=(2, 2))
            expected_state[b0][1 - b1] = 1.0
            np.testing.assert_equal(result.final_state,
                                    np.reshape(expected_state, 4))
