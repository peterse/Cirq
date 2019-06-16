# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility methods for transforming matrices."""

from typing import Tuple, Optional, Sequence, List, Union

import numpy as np

from cirq.protocols.approximate_equality import approx_eq
from cirq.linalg import predicates

def reflection_matrix_pow(reflection_matrix: np.ndarray, exponent: float):
    """Raises a matrix with two opposing eigenvalues to a power.

    Args:
        reflection_matrix: The matrix to raise to a power.
        exponent: The power to raise the matrix to.

    Returns:
        The given matrix raised to the given power.
    """

    # The eigenvalues are x and -x for some complex unit x. Determine x.
    squared_phase = np.dot(reflection_matrix[:, 0],
                           reflection_matrix[0, :])
    phase = complex(np.sqrt(squared_phase))

    # Extract +x and -x eigencomponents of the matrix.
    i = np.eye(reflection_matrix.shape[0]) * phase
    pos_part = (i + reflection_matrix) * 0.5
    neg_part = (i - reflection_matrix) * 0.5

    # Raise the matrix to a power by raising its eigencomponents to that power.
    pos_factor = phase**(exponent - 1)
    neg_factor = pos_factor * complex(-1)**exponent
    pos_part_raised = pos_factor * pos_part
    neg_part_raised = neg_part * neg_factor
    return pos_part_raised + neg_part_raised


def match_global_phase(a: np.ndarray,
                       b: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Phases the given matrices so that they agree on the phase of one entry.

    To maximize precision, the position with the largest entry from one of the
    matrices is used when attempting to compute the phase difference between
    the two matrices.

    Args:
        a: A numpy array.
        b: Another numpy array.

    Returns:
        A tuple (a', b') where a' == b' implies a == b*exp(i t) for some t.
    """

    # Not much point when they have different shapes.
    if a.shape != b.shape:
        return a, b

    # Find the entry with the largest magnitude in one of the matrices.
    k = max(np.ndindex(*a.shape), key=lambda t: abs(b[t]))

    def dephase(v):
        r = np.real(v)
        i = np.imag(v)

        # Avoid introducing floating point error when axis-aligned.
        if i == 0:
            return -1 if r < 0 else 1
        if r == 0:
            return 1j if i < 0 else -1j

        return np.exp(-1j * np.arctan2(i, r))

    # Zero the phase at this entry in both matrices.
    return a * dephase(a[k]), b * dephase(b[k])


def targeted_left_multiply(left_matrix: np.ndarray,
                           right_target: np.ndarray,
                           target_axes: Sequence[int],
                           out: Optional[np.ndarray] = None
                           ) -> np.ndarray:
    """Left-multiplies the given axes of the target tensor by the given matrix.

    Note that the matrix must have a compatible tensor structure.

    For example, if you have an 6-qubit state vector `input_state` with shape
    (2, 2, 2, 2, 2, 2), and a 2-qubit unitary operation `op` with shape
    (2, 2, 2, 2), and you want to apply `op` to the 5'th and 3'rd qubits
    within `input_state`, then the output state vector is computed as follows:

        output_state = cirq.targeted_left_multiply(op, input_state, [5, 3])

    This method also works when the right hand side is a matrix instead of a
    vector. If a unitary circuit's matrix is `old_effect`, and you append
    a CNOT(q1, q4) operation onto the circuit, where the control q1 is the qubit
    at offset 1 and the target q4 is the qubit at offset 4, then the appended
    circuit's unitary matrix is computed as follows:

        new_effect = cirq.targeted_left_multiply(
            left_matrix=cirq.unitary(cirq.CNOT).reshape((2, 2, 2, 2)),
            right_target=old_effect,
            target_axes=[1, 4])

    Args:
        left_matrix: What to left-multiply the target tensor by.
        right_target: A tensor to carefully broadcast a left-multiply over.
        target_axes: Which axes of the target are being operated on.
        out: The buffer to store the results in. If not specified or None, a new
            buffer is used. Must have the same shape as right_target.

    Returns:
        The output tensor.
    """
    k = len(target_axes)
    d = len(right_target.shape)
    work_indices = tuple(range(k))
    data_indices = tuple(range(k, k + d))
    used_data_indices = tuple(data_indices[q] for q in target_axes)
    input_indices = work_indices + used_data_indices
    output_indices = list(data_indices)
    for w, t in zip(work_indices, target_axes):
        output_indices[t] = w

    all_indices = set(input_indices + data_indices + tuple(output_indices))

    return np.einsum(left_matrix, input_indices,
                     right_target, data_indices,
                     output_indices,
                     # We would prefer to omit 'optimize=' (it's faster),
                     # but this is a workaround for a bug in numpy:
                     #     https://github.com/numpy/numpy/issues/10926
                     optimize=len(all_indices) >= 26,
                     # And this is workaround for *another* bug!
                     # Supposed to be able to just say 'old=old'.
                     **({'out': out} if out is not None else {}))


def targeted_conjugate_about(tensor: np.ndarray,
                             target: np.ndarray,
                             indices: Sequence[int],
                             conj_indices: Sequence[int] = None,
                             buffer: Optional[np.ndarray] = None,
                             out: Optional[np.ndarray] = None) -> np.ndarray:
    r"""Conjugates the given tensor about the target tensor.

    This method computes a target tensor conjugated by another tensor.
    Here conjugate is used in the sense of conjugating by a matrix, i.a.
    A conjugated about B is $A B A^\dagger$ where $\dagger$ represents the
    conjugate transpose.

    Abstractly this compute $A \cdot B \cdot A^\dagger$ where A and B are
    multi-dimensional arrays, and instead of matrix multiplication $\cdot$
    is a contraction between the given indices (indices for first $\cdot$,
    conj_indices for second $\cdot$).

    More specifically this computes
        sum tensor_{i_0,...,i_{r-1},j_0,...,j_{r-1}}
        * target_{k_0,...,k_{r-1},l_0,...,l_{r-1}
        * tensor_{m_0,...,m_{r-1},n_0,...,n_{r-1}}^*
    where the sum is over indices where j_s = k_s and s is in `indices`
    and l_s = m_s and s is in `conj_indices`.

    Args:
        tensor: The tensor that will be conjugated about the target tensor.
        target: The tensor that will receive the conjugation.
        indices: The indices which will be contracted between the tensor and
            target.
        conj_indices; The indices which will be contracted between the
            complex conjugate of the tensor and the target. If this is None,
            then these will be the values in indices plus half the number
            of dimensions of the target (`ndim`). This is the most common case
            and corresponds to the case where the target is an operator on
            a n-dimensional tensor product space (here `n` would be `ndim`).
        buffer: A buffer to store partial results in.  If not specified or None,
            a new buffer is used.
        out: The buffer to store the results in. If not specified or None, a new
            buffer is used. Must have the same shape as target.

    Returns:
        The result the conjugation.
    """
    conj_indices = conj_indices or [i + target.ndim // 2 for i in indices]
    first_multiply = targeted_left_multiply(tensor, target, indices, out=buffer)
    return targeted_left_multiply(np.conjugate(tensor),
                                  first_multiply,
                                  conj_indices,
                                  out=out)


_TSliceAtom = Union[int, slice, 'ellipsis']
_TSlice = Union[_TSliceAtom, Sequence[_TSliceAtom]]


def apply_matrix_to_slices(
        target: np.ndarray,
        matrix: np.ndarray,
        slices: List[_TSlice],
        *,
        out: Optional[np.ndarray] = None) -> np.ndarray:
    """Left-multiplies an NxN matrix onto N slices of a numpy array.

    Example:
        The 4x4 matrix of a fractional SWAP gate can be expressed as

           [ 1       ]
           [   X**t  ]
           [       1 ]

        Where X is the 2x2 Pauli X gate and t is the power of the swap with t=1
        being a full swap. X**t is a power of the Pauli X gate's matrix.
        Applying the fractional swap is equivalent to applying a fractional X
        within the inner 2x2 subspace; the rest of the matrix is identity. This
        can be expressed using `apply_matrix_to_slices` as follows:

            def fractional_swap(target):
                assert target.shape == (4,)
                return apply_matrix_to_slices(
                    target=target,
                    matrix=cirq.unitary(cirq.X**t),
                    slices=[1, 2]
                )

    Args:
        target: The input array with slices that need to be left-multiplied.
        matrix: The linear operation to apply to the subspace defined by the
            slices.
        slices: The parts of the tensor that correspond to the "vector entries"
            that the matrix should operate on. May be integers or complicated
            multi-dimensional slices into a tensor. The slices must refer to
            non-overlapping sections of the input all with the same shape.
        out: Where to write the output. If not specified, a new numpy array is
            created, with the same shape and dtype as the target, to store the
            output.

    Returns:
        The transformed array.
    """
    # Validate arguments.
    if out is target:
        raise ValueError("Can't write output over the input.")
    if matrix.shape != (len(slices), len(slices)):
        raise ValueError("matrix.shape != (len(slices), len(slices))")

    # Fill in default values and prepare space.
    if out is None:
        out = np.copy(target)
    else:
        out[...] = target[...]

    # Apply operation.
    for i, s_i in enumerate(slices):
        out[s_i] *= matrix[i, i]
        for j, s_j in enumerate(slices):
            if i != j:
                out[s_i] += target[s_j] * matrix[i, j]

    return out


def partial_trace(tensor: np.ndarray,
                  keep_indices: List[int]) -> np.ndarray:
    """Takes the partial trace of a given tensor.

    The input tensor must have shape `(d_0, ..., d_{k-1}, d_0, ..., d_{k-1})`.
    The trace is done over all indices that are not in keep_indices. The
    resulting tensor has shape `(d_{i_0}, ..., d_{i_r}, d_{i_0}, ..., d_{i_r})`
    where `i_j` is the `j`th element of `keep_indices`.

    Args:
        tensor: The tensor to sum over. This tensor must have a shape
            `(d_0, ..., d_{k-1}, d_0, ..., d_{k-1})`.
        keep_indices: Which indices to not sum over. These are only the indices
            of the first half of the tensors indices (i.e. all elements must
            be between `0` and `tensor.ndims / 2 - 1` inclusive).

    Raises:
        ValueError: if the tensor is not of the correct shape or the indices
            are not from the first half of valid indices for the tensor.
    """
    ndim = tensor.ndim // 2
    if not all(tensor.shape[i] == tensor.shape[i + ndim] for i in range(ndim)):
        raise ValueError('Tensors must have shape (d_0,...,d_{{k-1}},d_0,...,'
                         'd_{{k-1}}) but had shape ({}).'.format(tensor.shape))
    if not all(i < ndim for i in keep_indices):
        raise ValueError('keep_indices were {} but must be in first half, '
                         'i.e. have index less that {}.'.format(keep_indices,
                                                                ndim))
    keep_set = set(keep_indices)
    keep_map = dict(zip(keep_indices, sorted(keep_indices)))
    left_indices = [keep_map[i] if i in keep_set else i for i in range(ndim)]
    right_indices = [ndim + i if i in keep_set else i for i in left_indices]
    return np.einsum(tensor, left_indices + right_indices)


def keep_qubits(state: np.ndarray, keep_indices: List[int]) -> np.ndarray:
    """Return a representation of the given state over a subset of its qubits.

    A given wavefunction can be expressed over a subset of its qubits by
    tracing out undesired qubits from the state's corresponding density matrix.
    If the traced density matrix is rank 1 (pure state), then it can be
    refactored into wavefunction form via eigendecomposition. Otherwise, the
    outcome is a mixed state and cannot be expressed as a wavefunction.

    Global phase of the input state will not necessariy be preserved in the
    output state.

    Args:
        state: A wavefunction-like

    """

    if not np.isclose(np.linalg.norm(state), 1):
        raise ValueError("Input state must be normalized.")
    if not len(set(keep_indices)) == len(keep_indices):
        raise ValueError(
            "keep_indices were {} but must be unique.".format(keep_indices))
    dim = np.log2(state.shape[0])
    if not dim.is_integer():
        raise ValueError("Input wavefunction has bad length of {}.".format(
            state.shape[0]))
    dim = int(dim)
    # REMOVEME: 'attempt wavefunction trace' via slicing
    state = np.array(state).reshape((2,)*dim)
    all_indices = range(dim)
    remove_indices = list(set(range(dim)) - set(keep_indices))
    keep_dim = len(keep_indices)
    remove_dim = dim - keep_dim
    import itertools
    # iterate over all possible length `keep_dim` sets of indices to remove
    # the set(s) that retain the largest magnitude is a factor candidate
    max_mag = 0.0
    # ordered list of all wfs computed
    wfs = []
    largest_seen_ind = []

    for i, try_keep_indices in enumerate(itertools.combinations(all_indices,keep_dim)):
        # try_keep_dim = len(try_keep_indices)
        try_keep = np.asarray([state[predicates.slice_for_qubits_equal_to(try_keep_indices, i)].sum() for i in range(2**keep_dim)])
        # print("keep:", try_keep_indices)
        # print("keep state", try_keep)
        # print("mag:", mag)
        # # the other factor
        # try_remove_indices = list(set(all_indices) - set(try_keep_indices))
        # try_remove_dim = len(try_remove_indices)
        # try_remove = np.asarray([state[predicates.slice_for_qubits_equal_to(try_remove_indices, i)].sum() for i in range(2**try_remove_dim)])

        mag = np.linalg.norm(try_keep)
        if approx_eq(mag, max_mag):
            largest_seen_ind.append(i)
        elif mag > max_mag:
            largest_seen_ind = [i]
            max_mag = mag

        print("unnormalized try", try_keep)
        wfs.append(try_keep / np.linalg.norm(try_keep))
    print("biggest mag", max_mag)

    # do a dot product between this candidate and _all_ other wfs of the same size
    cum = 0.0
    for i in largest_seen_ind:
        cum = 0.0
        # wf_i = np.array(wfs[i]) / np.linalg.norm(wfs[i]) #normalized candidate
        wf_i = wfs[i]

        print("wf_i", wf_i)
        for j, wf_j in enumerate(wfs):
            if i != j:
                print(wf_j) # un-normalized 'other'
                print(wf_i)
                cum += np.linalg.norm(np.dot(np.conj(wf_j).T, wf_i)) ** 2
        print("ind", i, cum)
    # rule out the case where _none_ of the candidates are pure factors


    return
    rho = np.kron(np.conj(np.reshape(state, (state.shape[0], 1)).T),
                  state).reshape((2, 2) * int(dim))
    new_dim = 2**len(keep_indices)
    keep_rho = partial_trace(rho, keep_indices).reshape((new_dim,) * 2)
    if not any(keep_indices):
        return keep_rho

    # FIXME: reasonable tolerance for purity check?
    if approx_eq(np.trace(keep_rho @ keep_rho), 1):
        w, v = np.linalg.eig(keep_rho)
        # FIXME: reasonable tolerance for lone eigenvalue check?
        return v[:, np.isclose(w, 1)].reshape((new_dim,))

    # FIXME: not sure how to deal with warnings
    print("Removing qubits from an entangled state resulted in mixture.")
    return keep_rho
