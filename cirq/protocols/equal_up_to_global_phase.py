# Copyright 2019 The Cirq Developers
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

from collections import Iterable
import numbers

import numpy as np
from typing import Any, Union
from typing_extensions import Protocol

import cirq


class SupportsEqualUpToGlobalPhase(Protocol):
    """Object which can be compared for equality mod global phase."""

    def _equal_up_to_global_phase_(self, other: Any, *,
                                   atol: Union[int, float]) -> bool:
        """Approximate comparator.

        Types implementing this protocol define their own logic for comparison
        with other types.

        Args:
            other: Target object for comparison of equality up to global phase.
            atol: The minimum absolute tolerance. See np.isclose() documentation
                  for details.

        Returns:
            True if objects are equal up to a global phase, False otherwise.
            Returns NotImplemented when checking equality up to a global phase
            is not implemented for given types.
        """


def equal_up_to_global_phase(val: Any,
                             other: Any,
                             *,
                             atol: Union[int, float] = 1e-8) -> bool:
    """Determine whether two objects are equal up to global phase.

    If `val` implements SupportsEqualUpToGlobalPhase protocol then it is
    invoked and takes precedence over all other checks:
     - For complex primitive type the magnitudes of the values are compared.
     - For `val` and `other` both iterable of the same length, consecutive
       elements are compared recursively. Types of `val` and `other` does not
       necessarily needs to match each other. They just need to be iterable and
       have the same structure.
     - For all other types, fall back to _approx_eq_

    Args:
        val: Source object for approximate comparison.
        other: Target object for approximate comparison.
        atol: The minimum absolute tolerance. This places an upper bound on
        the differences in _magnitudes_ of two compared complex numbers.

    Returns:
        True if objects are approximately equal up to phase, False otherwise.
    """

    # attempt _equal_up_to_global_phase_ for val.
    eq_up_to_phase_getter = getattr(val, '_equal_up_to_global_phase_', None)
    if eq_up_to_phase_getter is not None:
        result = eq_up_to_phase_getter(other, atol)
        if result is not NotImplemented:
            return result

    # fall back to _equal_up_to_global_phase_ for other.
    other_eq_up_to_phase_getter = getattr(other, '_equal_up_to_global_phase_',
                                          None)
    if other_eq_up_to_phase_getter is not None:
        result = other_eq_up_to_phase_getter(val, atol)
        if result is not NotImplemented:
            return result

    # fall back to special check for numeric arrays
    # defer to numpy automatic type casting to determine numeric type
    if isinstance(val, Iterable) and isinstance(other, Iterable):
        a = np.array(val)
        b = np.array(other)
        # FIXME: allclose_up_to_global_phase doesn't support mismatched sizes
        # FIXME: allclose_up_to_global_phase doesn't support empty arrays
        if a.shape == b.shape and any(a) and any(b) and \
                a.dtype.kind in set('uifc') and b.dtype.kind in set('uifc'):
            return cirq.linalg.allclose_up_to_global_phase(a, b, atol=atol)

    # fall back to approx_eq for compare the magnitude of two numbers.
    if isinstance(val, numbers.Number):
        if not isinstance(other, numbers.Number):
            return False
        result = cirq.approx_eq(abs(val), abs(other), atol=atol)
        if result is not NotImplemented:
            return result

    # Fallback to cir approx_eq for remaining types
    return cirq.approx_eq(val, other, atol=atol)
