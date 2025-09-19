import numpy as np
import torch as t


class dtype(object):
    """
    Implement custom class for seamless dtype management during computations.

    Users specify `"float16"`, `"float32"` or `"float64"` as
    character string, which is automatically converted to corresponding
    numpy and torch dtype.

    Attributes:
        t_dtype (torch.dtype): corresponding torch dtype.
        np_dtype (numpy.dtype): corresponding numpy dtype.
    """

    def __init__(self, dtype: str):
        """
        Create ktest_dtype object.

        Input:
            dtype (str): user required dtype `"float16"`, `"float32"` or`"float64"`
                to run computations
        """
        assert dtype in ["float16", "float32", "float64"], \
            "Wrong input, should be 'float16', 'float32' or 'float64'."
        match dtype:
            case "float16":
                self.t_dtype = t.float16
                self.np_dtype = np.float16
            case "float32":
                self.t_dtype = t.float32
                self.np_dtype = np.float32
            case "float64":
                self.t_dtype = t.float64
                self.np_dtype = np.float64

    def torch(self):
        """Return corresponding torch dtype."""
        return self.t_dtype

    def numpy(self):
        """Return corresponding tnumpy dtype."""
        return self.np_dtype
