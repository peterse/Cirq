import abc
import tensorflow as tf
import numpy as np
import math

from typing import Any
import cirq
from cirq.contrib.tf_backend.state import (
    State
)
from cirq.contrib.tf_backend.gate_wrappers import (
    tf_gate_wrapper
)


def wf_tensor_prod(tensor0, inds0, tensor1, inds1):
    """
    TODO:
        this is not set up to handle anything more than 1-qubit gates/states.
    tensor0 (tf.Tensor): gate (matrix, rank 2)
    tensor1 (tf.Tensor): wavefunction (column matrix, rank1)
    """
    all_inds = list(set(inds0) & set(inds1))
    N = len(tensor0.shape) # gate dim
    K = len(tensor1.shape) # wavefunction dim
    assert K == len(all_inds)

    gate = tf.reshape(tensor0, (2**K, 2**K))
    tensor = tensor1

    tensor = tf.reshape(tensor, (2**K, 1))
    tensor = tf.matmul(gate, tensor)
    tensor = tf.reshape(tensor, (2,)*K)

    return tensor


class TFSimulator:

    def __init__(self):
        pass


class TFWaveFunctionSimulator(TFSimulator):
    """
    TODO:
        - validate input state
        - default initial State
        - proper typing on intial state
    """
    def __init__(self, ):

        pass
        # 2. do TFWrapping for all cirq gates
        #   a. guarantee preserved ordering..
        #   b. tensor up the shapes; relies on initial_state

    def simulate(self, circuit: cirq.Circuit, initial_state: Any = None):
        """
        Construct the tf graph from pre-padded matrices and input state.
        Accepts placeholders as parameters

        return staged graph for session run
        """
        # 1. checking/promotion of initial state to track qubits
        # TODO: enforce sizing on initial_state; don't want any stray/missing qubits
        if isinstance(initial_state, np.ndarray):
            initial_qubits = range(int(np.log2(initial_state.shape[0])))
            # enforce tensor shape 2,2,2,2,...
            initial_state = initial_state.reshape((2,)*len(initial_qubits))
            state = tf.convert_to_tensor(
                value=initial_state, dtype = tf.complex64
            )
            # fixme: util here:
            state = State(state, initial_qubits)

        # TODO: gather a set of all qubits acted on in this circuit

        # Moment-wise construction of a set of matrices to apply wall
        ops = []
        for moment in circuit:
            # FIXME: empty moment?
            for op in moment.operations:
                ops.append(tf_gate_wrapper(op))

        # sparse tensor-up of each gate's tensor repr
        for op in ops:
            new = wf_tensor_prod(op._tensor, op._qubits, state._tensor, state._qubits)
            # repack state for next processing step
            state = State(new, initial_qubits)
        return state

#
# def astensor(array):
#     """Covert numpy array to tensorflow tensor"""
#     tensor = tf.convert_to_tensor(value=array, dtype=tf.complex128)
#     return tensor
# #
# def size(tensor):
#     return np.prod(np.array(tensor.get_shape().as_list()))
#
# def astensorproduct(array):
#     tensor = astensor(array)
#     N = int(math.log2(size(tensor)))
#     tensor = tf.reshape(tensor, ([2]*N))
#     return tensor
#
