"""
    Master TODO list:
        - setup state OBJECT that contains state and qubits

"""



import numpy as np
import tensorflow as tf
import cirq
from cirq.contrib.tf_backend.tf_apply_unitary import (
    tf_apply_unitary,
    ApplyTFUnitaryArgs,
)

INITIAL_STATE = np.asarray([1, 0])
TEST_VAR = tf.Variable(1.0)
TEST_GATES = [
    cirq.YPowGate(exponent=TEST_VAR)(cirq.LineQubit(0)),
    cirq.YPowGate(exponent=TEST_VAR)(cirq.LineQubit(0))
]
TEST_CIRCUIT = cirq.Circuit.from_ops(TEST_GATES)


def test_apply_unitary_presence_absence():
    m = np.diag([1, -1])

    class HasUnitary:
        def _unitary_(self) -> np.ndarray:
            return m

    class HasApplyReturnsNotImplementedButHasUnitary:
        def _apply_unitary_(self, args: ApplyTFUnitaryArgs):
            return NotImplemented

        def _unitary_(self) -> np.ndarray:
            return m

    class HasApplyOutputInBuffer:
        def _apply_unitary_(self, args: ApplyTFUnitaryArgs) -> np.ndarray:
            zero = args.subspace_index(0)
            one = args.subspace_index(1)
            args.available_buffer[zero] = args.target_tensor[zero]
            args.available_buffer[one] = -args.target_tensor[one]
            return args.available_buffer

    class HasApplyMutateInline:
        def _apply_unitary_(self, args: ApplyTFUnitaryArgs) -> np.ndarray:
            one = args.subspace_index(1)
            args.target_tensor[one] *= -1
            return args.target_tensor

    passes = [
        HasUnitary(),
        HasApplyReturnsNotImplementedButHasUnitary(),
        HasApplyOutputInBuffer(),
        HasApplyMutateInline(),
    ]

    def make_input():
        return np.ones((2, 2))

    def assert_works(val):
        expected_outputs = [
            np.array([1, 1, -1, -1]).reshape((2, 2)),
            np.array([1, -1, 1, -1]).reshape((2, 2)),
        ]
        for axis in range(2):
            result = tf_apply_unitary(
                val, ApplyTFUnitaryArgs(make_input(), buf, [axis]))
            np.testing.assert_allclose(result, expected_outputs[axis])

    buf = np.empty(shape=(2, 2), dtype=np.complex128)

    for s in passes:
        assert_works(s)
        assert tf_apply_unitary(
            s,
            ApplyTFUnitaryArgs(make_input(), buf, [0]),
            default=None) is not None
