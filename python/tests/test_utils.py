import numpy as np
import torch as t

from ktest import dtype


def test_dtype():
    """Testing ktest_dtype class."""
    dtype_float16 = dtype("float16")
    assert dtype_float16.torch() == t.float16
    assert dtype_float16.numpy() == np.float16

    dtype_float32 = dtype("float32")
    assert dtype_float32.torch() == t.float32
    assert dtype_float32.numpy() == np.float32

    dtype_float64 = dtype("float64")
    assert dtype_float64.torch() == t.float64
    assert dtype_float64.numpy() == np.float64
