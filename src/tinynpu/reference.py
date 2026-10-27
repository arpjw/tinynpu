"""Exact, schedule-independent oracle for the initial INT8 matrix products."""

import numpy as np
from numpy.typing import NDArray


def matmul_int8(a: NDArray[np.int8], b: NDArray[np.int8]) -> NDArray[np.int32]:
    """Multiply INT8 matrices exactly for the baseline's reduction lengths 1..64.

    Wide arithmetic prevents NumPy INT8 overflow. The supported reduction bound
    proves the result fits INT32: 64 * 128 * 128 = 1,048,576.
    Padding, timing, bias, and quantization are deliberately outside this oracle.
    """
    if a.dtype != np.int8 or b.dtype != np.int8:
        raise TypeError("Both operands must have dtype int8; implicit narrowing is forbidden.")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Both operands must be two-dimensional matrices.")
    if a.shape[1] != b.shape[0]:
        raise ValueError("Reduction dimensions must match.")
    if a.shape[0] == 0 or b.shape[1] == 0 or not 1 <= a.shape[1] <= 64:
        raise ValueError("Require nonempty outputs and reduction length 1..64.")
    result = a.astype(np.int64) @ b.astype(np.int64)
    return result.astype(np.int32)
