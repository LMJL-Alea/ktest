from numbers import Number
import warnings
import numpy as np
import torch as to


def verbosity(msg: str, msg_type: str = "print", verbose: int | bool = 0):
    """
    Verbosity output function.

    Input:
        msg (str): message to print.
        msg_type (str): type of message among `"print"`, `"warning"`.
        verbose (int | bool): verbosity level, `0` means no output,
            `1` means print only, `>=2` means print and warning.
            Note: `verbose=False` is equivalent to `verbose=0` and
            `verbose=True` is equivalent to `verbose=1`

    Output: no output
    """

    # check input
    assert isinstance(msg, str), "`msg` input should be a character string"
    assert isinstance(msg_type, str) and msg_type in ["print", "warning"], \
        "`msg_type` input should be either \"print\" or \"warning\""
    assert isinstance(verbose, bool) or \
        isinstance(verbose, Number) and verbose.is_integer(), \
        "`verbosity` input should be a boolean or an integer value"

    # manage verbosity
    if verbose == 1 and msg_type == "print":
        print(msg)
    elif verbose > 1:
        if msg_type == "print":
            print(msg)
        elif msg_type == "warning":
            warnings.warn(msg, UserWarning)


def pred_threshold_fun(
    x: float,
    left_val: float | np.ndarray | to.Tensor,
    right_val: float | np.ndarray | to.Tensor
):
    """
    Define a piecewise linear function that fit the following points:
    [0, -left_val], [1/2, 0] and [1, right_val] to compute the prediction
    bias in kFDA.

    x -> y = 2*left_val*x - left_val if x < 1/2
         y = 2*right_val*x - right_val if x >= 1/2

    Note: if array-like object, `left_val` and `right_val` should have the
    same shape.

    Input:
        x (float): input number between 0 and 1.
        left_val (float | np.ndarray | t.Tensor): positive number or
            array of positive numbers corresponding to the y-axis value
            for x = 0.
        right_val (float | np.ndarray | t.Tensor): positive number or
            array of positive numbers corresponding to the y-axis value
            for x = 1.

    Output: result value or array of result values when the piecewise linear
        function is applied to input x.
    """

    return (x < 1/2) * (2 * left_val * x - left_val) + \
        (x >= 1/2) * (2 * right_val * x - right_val)
