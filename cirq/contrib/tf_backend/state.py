import tensorflow as tf

class State:
    """
    Container class for a quantum state and its target qubits.
    """

    def __init__(self, state, qubits):
        """Create a new State from a wavefunction or density matrix representation.
        Args:
            tensor: A vector or tensor of state amplitudes
            qubits: A sequence of qubit indices
                FIXME: interface with location..? do I care?
        """

        # TODO: input validation
        self._tensor = tf.convert_to_tensor(
            value=state, dtype=tf.complex64
        )
        self._qubits = qubits
