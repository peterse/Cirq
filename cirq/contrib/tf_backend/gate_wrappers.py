from typing import List
import cirq
import tensorflow as tf
import numpy as np
from cirq.contrib.tf_backend.tf_apply_unitary import (
    ApplyTFUnitaryArgs,
)
"""
LOCAL TODO:
    - wrapper constructs for every cirq gate
    - Check that these gate matrix definitions are self consistent; cirq
        has a weird tendency to introduce/cut global phases out...
    - unit tests for each individual instantiation
"""


class BaseTFGate(cirq.SupportsUnitary, cirq.SupportsApplyUnitary):
    """
    Main wrapper for a cirq gate. The purpose of this object is to
        1) wrap an initialized cirq gate (that may include placeholders)
        2) divert further processing away from the cirq native pipeline,
                towards tensorflow native evaluation

    Meanwhile, this should be generally compatible with cirq protocols and
    operations
    """


    def __init__(self, ):

        pass

    def _apply_unitary_(self, state):
        """Apply the action of this gate upon a state"""
        return NotImplemented


    @property
    def _has_unitary_(self):
        return True

    def _unitary_(self):
        return self._tensor


class WrapXPowGate(BaseTFGate):

    def __init__(self, qubit: int,
        theta: tf.Tensor = None,
        global_shift: float = None,
        dtype = tf.complex64
    ):
        """Wrap a XPoweGate instance.
        learnability is handled at exponent instantiation.

        """

        # FIXME: IS THETA A SCALAR OR A TENSOR IN GENERAL???
        self._exponent = theta # save for later use
        self._global_shift = global_shift # FIXME my inheritance is all fucked up...
        theta = tf.cast(theta, dtype)
        # FIXME: linequbits only...
        self._qubits = [qubit.x]
        self._tensor = tf.convert_to_tensor([
            [tf.cos(theta), -1.0j * tf.sin(theta)],
            [-1.0j * tf.sin(theta), tf.cos(theta)],
        ])
        # TODO: different classing structure that will let me track qubits
        # super().__init__(tensor, [qubit], [theta])


class WrapYPowGate(BaseTFGate):

    def __init__(self, qubit: int,
        theta: tf.Tensor = None,
        global_shift: float = None,
        dtype = tf.complex64
    ):
        """Wrap a YPowGate instance.
        """

        theta = tf.scalar_mul(np.pi/2, theta)
        theta = tf.cast(theta, dtype)
        self._exponent = theta
        self._global_shift = global_shift
        self._qubits = [qubit.x]
        self._tensor = tf.convert_to_tensor([
            [tf.cos(theta), -1.0*tf.sin(theta)],
            [tf.sin(theta), tf.cos(theta)]
        ])


class WrapZPowGate(BaseTFGate):

    def __init__(self, qubit: int,
        theta: tf.Tensor = None,
        global_shift: float = None,
        dtype = tf.complex64
    ):
        """Wrap a ZPowGate instance.
        """

        theta = tf.scalar_mul(np.pi/2, theta)
        theta = tf.cast(theta, dtype)
        self._exponent = theta
        self._global_shift = global_shift
        self._qubits = [qubit.x]
        self._tensor = tf.convert_to_tensor([
            [tf.cos(theta) - 1j * tf.sin(theta), 0],
            [0, tf.cos(theta) + 1j * tf.sin(theta)]
        ])


class WrapHPowGate(BaseTFGate):

    def __init__(self, qubit: int,
        theta: tf.Tensor = None,
        global_shift: float = None,
        dtype = tf.complex64
    ):
        """Wrap a HPowGate instance.
        """

        theta = tf.scalar_mul(np.pi/2, theta)
        theta = tf.cast(theta, dtype)
        self._exponent = theta
        self._global_shift = global_shift
        self._qubits = [qubit.x]
        self._tensor = tf.convert_to_tensor([
            [tf.cos(theta) - 1j * tf.sin(theta), -1j * tf.sin(theta)],
            [-1j * tf.sin(theta), tf.cos(theta) + 1j * tf.sin(theta)]
        ])

        # FIXME: can't do this... need variable to be carried thru op
        self._tensor = tf.scalar_mul(np.exp(1j*theta)/np.sqrt(2), self._tensor)


class WrapCNotPowGate(BaseTFGate):

    def __init__(self, *qubits: List[int],
                 theta: tf.Tensor = None,
                 global_shift: float = None,
                 dtype = tf.complex64
    ):
        """Wrap a CNotPowGate instance.
        """

        theta = tf.scalar_mul(np.pi/2, theta)
        theta = tf.cast(theta, dtype)
        self._exponent = theta
        self._global_shift = global_shift
        self._qubits = [qubits[0].x, qubits[1].x]
        self._tensor = tf.convert_to_tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, tf.exp(1j*theta) * tf.cos(theta), -1j * tf.exp(1j*theta) * tf.sin(theta)],
            [0, 0, -1j * tf.exp(1j*theta) * tf.sin(theta), tf.exp(1j*theta) * tf.cos(theta)]
        ])


class WrapSwapPowGate(BaseTFGate):

    def __init__(self, *qubits: List[int],
                 theta: tf.Tensor = None,
                 global_shift: float = None,
                 dtype = tf.complex64
    ):
        """Wrap a ISwapPowGate instance.
        """

        theta = tf.scalar_mul(np.pi/2, theta)
        theta = tf.cast(theta, dtype)
        self._exponent = theta
        self._global_shift = global_shift
        self._qubits = [qubits[0].x, qubits[1].x]
        self._tensor = tf.convert_to_tensor([
            [1, 0, 0, 0],
            [0, tf.exp(1j*theta) * tf.cos(theta), -1j * tf.exp(1j*theta) * tf.sin(theta), 0],
            [0, -1j * tf.exp(1j*theta) * tf.sin(theta), tf.exp(1j*theta) * tf.cos(theta), 0],
            [0, 0, 0, 1]
        ])


ALL_WRAPPERS = {
    cirq.ops.pauli_gates._PauliX: WrapXPowGate,
    cirq.ops.pauli_gates._PauliY: WrapYPowGate,
    cirq.ops.pauli_gates._PauliZ: WrapZPowGate,
    cirq.XPowGate: WrapXPowGate,
    cirq.YPowGate: WrapYPowGate,
    cirq.ZPowGate: WrapZPowGate,
    cirq.HPowGate: WrapHPowGate,
    cirq.CNotPowGate: WrapCNotPowGate,
    cirq.SwapPowGate: WrapSwapPowGate,
}

def tf_gate_wrapper(
    inst: cirq.EigenGate,
    dtype) -> BaseTFGate:

    # todo: notimplemented case checking
    # cirq has spaghetti inheritance. Why were these getters so difficult?
    theta = getattr(inst._gate, 'exponent', 1)
    global_shift = getattr(inst._gate, '_global_shift', 0)
    if theta is not None:
        print(ALL_WRAPPERS.get(type(inst._gate)))
        return ALL_WRAPPERS.get(
            type(inst._gate))(*inst.qubits, theta=theta, global_shift=global_shift, dtype=dtype
        )


### DO NOT DELETE
# Below is working prototype code for WrapXPowGate._apply_unitary
#     def _apply_unitary_(self, args: ApplyTFUnitaryArgs
#                         ) -> tf.Tensor:
#
#         if self._exponent != 1:
#             return None
#         zero = args.subspace_index(0)
#         one = args.subspace_index(1)
#         inds = [zero, one]
#         ref0 = args.target_tensor[one]
#         ref1 = args.target_tensor[zero]
#         refs = [ref0, ref1]
#         x = args.available_buffer
#         with tf.control_dependencies([x[inds[i]].assign(refs[i]) for i in range(2)]):
#             x = tf.identity(x)
#
#         p = 1j**(2 * self._exponent * self._global_shift)
#         if p != 1:
#             x = tf.scalar_mul(p, x)
#         return x
