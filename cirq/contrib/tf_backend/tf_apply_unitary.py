"""A shallow reproduction of cirq.protocols.apply_unitary.

NOT FOR DISTRIBUTION I don't want to get sued.
"""

import string
from typing import (
    Any, Union, TypeVar, Tuple, Iterable, Sequence, Optional, List
)
import numpy as np
import tensorflow as tf
from typing_extensions import Protocol, _TSlice

from cirq import linalg
from cirq.protocols.unitary import unitary
from cirq.type_workarounds import NotImplementedType


# This is a special indicator value used by the apply_unitary method
# to determine whether or not the caller provided a 'default' argument. It must
# be of type np.ndarray to ensure the method has the correct type signature in
# that case. It is checked for using `is`, so it won't have a false positive if
# the user provides a different np.array([]) value.

RaiseTypeErrorIfNotProvided = np.array([])  # type: np.ndarray

TDefault = TypeVar('TDefault')


TF_INDEX_MAPPINGS = string.ascii_lowercase
def tf_index_map(inds):
    return [TF_INDEX_MAPPINGS[i] for i in inds]


def tf_targeted_left_multiply(left_matrix:np.ndarray,
                           right_target:np.ndarray,
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



def tf_apply_matrix_to_slices(
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


class ApplyTFUnitaryArgs:
    """
    Basic overwrite of cirq.ApplyUnitaryArgs with correct type hints.
    """

    def __init__(self,
                 target_tensor: tf.Tensor,
                 available_buffer: np.ndarray,
                 axes: Iterable[int]):
        """
        Args:
            target_tensor: The input tensor that needs to be left-multiplied by
                the unitary effect of the receiving object. The tensor will
                have the shape (2, 2, 2, ..., 2). It usually corresponds to
                a multi-qubit superposition, but it could also be a multi-qubit
                unitary transformation or some other concept.
            available_buffer: Pre-allocated workspace with the same shape and
                dtype as the target tensor.
            axes: Which axes the unitary effect is being applied to (e.g. the
                qubits that the gate is operating on).
        """
        self.target_tensor = target_tensor
        self.available_buffer = available_buffer
        self.axes = tuple(axes)

    def subspace_index(self, little_endian_bits_int: int
                       ) -> Tuple[Union[slice, int, 'ellipsis'], ...]:
        """An index for the subspace where the target axes equal a value.
        Args:
            little_endian_bits_int: The desired value of the qubits at the
                targeted `axes`, packed into an integer. The least significant
                bit of the integer is the desired bit for the first axis, and
                so forth in increasing order.
        Returns:
            A value that can be used to index into `target_tensor` and
            `available_buffer`, and manipulate only the part of Hilbert space
            corresponding to a given bit assignment.
        Example:
            If `target_tensor` is a 4 qubit tensor and `axes` is `[1, 3]` and
            then this method will return the following when given
            `little_endian_bits=0b01`:
                `(slice(None), 0, slice(None), 1, Ellipsis)`
            Therefore the following two lines would be equivalent:
                args.target_tensor[args.subspace_index(0b01)] += 1
                args.target_tensor[:, 0, :, 1] += 1
        """
        return linalg.slice_for_qubits_equal_to(self.axes,
                                                little_endian_bits_int)


def tf_apply_unitary(unitary_value: Any,
                  args: ApplyTFUnitaryArgs,
                  default: TDefault = RaiseTypeErrorIfNotProvided
                  ) -> Union[tf.Tensor, TDefault]:
    """High performance left-multiplication of a unitary effect onto a tensor.
    If `unitary_value` defines an `_apply_unitary_` method, that method will be
    used to apply `unitary_value`'s unitary effect to the target tensor.
    Otherwise, if `unitary_value` defines a `_unitary_` method, its unitary
    matrix will be retrieved and applied using a generic method. Otherwise the
    application fails, and either an exception is raised or the specified
    default value is returned.
    Args:
        unitary_value: The value with a unitary effect to apply to the target.
        args: A mutable `cirq.ApplyTFUnitaryArgs` object describing the target
            tensor, available workspace, and axes to operate on. The attributes
            of this object will be mutated as part of computing the result.
        default: What should be returned if `unitary_value` doesn't have a
            unitary effect. If not specified, a TypeError is raised instead of
            returning a default value.
    Returns:
        If the receiving object is not able to apply its unitary effect,
        the specified default value is returned (or a TypeError is raised). If
        this occurs, then `target_tensor` should not have been mutated.
        If the receiving object was able to work inline, directly
        mutating target_tensor it will return target_tensor. The caller is
        responsible for checking if the result is target_tensor.
        If the receiving object wrote its output over available_buffer, the
        result will be available_buffer. The caller is responsible for
        checking if the result is available_buffer (and e.g. swapping
        the buffer for the target tensor before the next call).
        The receiving object may also write its output over a new buffer
        that it created, in which case that new array is returned.
    Raises:
        TypeError: `unitary_value` doesn't have a unitary effect and `default`
            wasn't specified.
    """

    # Check if the specialized method is present.
    func = getattr(unitary_value, '_apply_unitary_', None)
    if func is not None:
        print("WHOOPS: DONT IMPLEMENT FUNC YET")

        result = func(args)
        if result is not NotImplemented and result is not None:
            return result

    # Fallback to using the object's _unitary_ matrix.
    matrix = unitary(unitary_value, None)
    if matrix is not None:
        print("WHOOPS: DONT IMPLEMENT matrix YET")
        # Special case for single-qubit operations.
        if matrix.shape == (2, 2):
            zero = args.subspace_index(0)
            one = args.subspace_index(1)
            return tf_apply_matrix_to_slices(args.target_tensor,
                                                 matrix,
                                                 [zero, one],
                                                 out=args.available_buffer)

        # Fallback to tf.einsum for the general case.
        return tf_targeted_left_multiply(
            matrix.astype(args.target_tensor.dtype).reshape(
                (2,) * (2 * len(args.axes))),
            args.target_tensor,
            args.axes,
            out=args.available_buffer)

    # Don't know how to apply. Fallback to specified default behavior.
    if default is not RaiseTypeErrorIfNotProvided:
        return default
    raise TypeError(
        "object of type '{}' has no _apply_unitary_ or _unitary_ methods "
        "(or they returned None or NotImplemented).".format(
            type(unitary_value)))
