
import cirq
import tensorflow as tf
import numpy as np
from cirq.contrib.tf_backend.tf_apply_unitary import (
    ApplyTFUnitaryArgs,
)
"""
LOCAL TODO:
    - wrapper constructs for every cirq gate.
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
        # TODO: tf implementation of eigenvalue shortcut.
        # indices = state.qubits
        #tensor = tf.matmul(self._tensor, state._tensor, indices)
        return tensor

    @property
    def _has_unitary_(self):
        return True

    def _unitary_(self):
        return self._tensor


class WrapYPowGate(BaseTFGate):

    def __init__(self, qubit: int,
        theta: tf.Tensor = None,
        dtype = tf.complex64
    ):
        """Wrap a YPoweGate instance.
        learnability is handled at exponent instantiation.
        """

        theta = tf.multiply(theta, [[np.pi/2]])
        theta = tf.cast(theta, dtype)

        # FIXME: linequbits only...
        self._qubits = [qubit.x]
        self._tensor = tf.convert_to_tensor([
            [tf.cos(theta), -1.0*tf.sin(theta)],
            [tf.sin(theta), tf.cos(theta)]
        ])

        # TODO: different classing structure that will let me track qubits
        # super().__init__(tensor, [qubit], [theta])


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

    def _apply_unitary_(self, args: ApplyTFUnitaryArgs
                        ) -> tf.Tensor:

        if self._exponent != 1:
            return None
        zero = args.subspace_index(0)
        one = args.subspace_index(1)
        inds = [zero, one]
        ref0 = args.target_tensor[one]
        ref1 = args.target_tensor[zero]
        refs = [ref0, ref1]
        x = args.available_buffer
        with tf.control_dependencies([x[inds[i]].assign(refs[i]) for i in range(2)]):
            x = tf.identity(x)

        p = 1j**(2 * self._exponent * self._global_shift)
        if p != 1:
            x = tf.scalar_mul(p, x)
        return x


ALL_WRAPPERS = {
    cirq.YPowGate: WrapYPowGate,
    cirq.XPowGate: WrapXPowGate,
    cirq.ops.pauli_gates._PauliX: WrapXPowGate,
}

def tf_gate_wrapper(
    inst: cirq.EigenGate,
    dtype) -> BaseTFGate:

    # todo: notimplemented case checking
    # cirq = spaghetti inheritance. Why were these getters so difficult?
    theta = getattr(inst._gate, 'exponent')
    global_shift = getattr(inst._gate, '_global_shift')
    if theta is not None:
        return ALL_WRAPPERS.get(
            type(inst._gate))(*inst.qubits, theta=theta, global_shift=global_shift, dtype=dtype
        )
