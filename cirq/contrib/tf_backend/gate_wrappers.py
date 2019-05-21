
import cirq
from cirq.contrib.tf_backend.state import (
    State
)
import tensorflow as tf

"""
LOCAL TODO:
    - wrapper constructs for every cirq gate.
"""


class BaseTFGate:
    """
    Main wrapper for a cirq gate. The purpose of this object is to
        1) wrap an initialized cirq gate (that may include placeholders)
        2) divert further processing away from the cirq native pipeline,
                towards tensorflow native evaluation
    """


    def __init__(self, ):

        pass

    def _apply_unitary_(self, state: State):
        """Apply the action of this gate upon a state"""
        indices = state.qubits
        tensor = tf.matmul(self._tensor, state._tensor, indices)
        return State(tensor, state.qubits)


class WrapYPowGate(BaseTFGate):

    def __init__(self, qubit: int, theta: tf.Tensor = None):
        """Wrap a YPoweGate instance.
        learnability is handled at exponent instantiation.
        """

        theta = tf.cast(theta, tf.complex64)
        # FIXME: linequbits only...
        self._qubits = [qubit.x]
        self._tensor = tf.convert_to_tensor([
            [tf.cos(theta / 2.0), -1.0*tf.sin(theta / 2.0)],
            [tf.sin(theta / 2.0), tf.cos(theta / 2.0)]
        ])

        # TODO: different classing structure that will let me track qubits
        # super().__init__(tensor, [qubit], [theta])



ALL_WRAPPERS = {cirq.YPowGate: WrapYPowGate}


def tf_gate_wrapper(inst: cirq.EigenGate) -> BaseTFGate:

    # todo: notimplemented case checking
    # cirq = spaghetti inheritance. Why were these getters so difficult?
    theta = inst._gate.exponent
    return ALL_WRAPPERS.get(
        type(inst._gate))(*inst.qubits, theta=theta
    )
