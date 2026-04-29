from collections.abc import Iterable
from numbers import Integral, Number
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


def compute_accuracy(
    prediction: list[np.ndarray], ground_truth: np.ndarray,
    ref: Integral | str | None = None
) -> list[np.ndarray]:
    """
    Compute accuracy indicator between an array of predictions (observations x
    truncations) and the vector of ground truth values.

    Note: prediction and ground_truth should have the same encoding for the
    two classes, i.e. either two value strings or 0-1 integers.

    Input:
        prediction (list of numpy.ndarray): list of 2-D arrays of class
            predictions for observations (in rows) and truncation values
            (in columns), corresponding to different predication bias values.
        ground_truth (numpy.ndarray): vector (1-D array) of class ground truth
            values for each observations.
        ref (numbers.Integral or str or None): class of reference for
            computing true positive rate. The other class will be used for
            computing true negative rate. `ref` type should correspond to
            `prediction` and `ground_truth` array dtype. Default is `None`
            and first class appearing in `ground_truth` vector is used as
            reference.

    Output:
        accuracy (list of numpy.ndarray): list of 2-D arrays of
            prediction accuracy (`1` for correct, `0` for wrong)
            for observations (in rows) and increasing truncation values
            (in columns), corresponding to different predication bias values.
        true_pos (list of numpy.ndarray): list of 2-D arrays of
            prediction true positives (`1` for correct, `0` for wrong)
            for positive only observations (in rows) and increasing truncation
            values (in columns), corresponding to different predication bias
            values.s
        true_neg (list of numpy.ndarray): list of 2-D arrays of
            prediction true negatives (`1` for correct, `0` for wrong)
            for negative only observations (in rows) and increasing truncation
            values (in columns), corresponding to different predication bias
            values.

    """

    # check reference
    if ref is None:
        ref = ground_truth[0]

    # convert input to binary indicator
    ground_truth_ind = (ground_truth == ref).astype(np.uint8)
    prediction_ind = [
        (pred_tab == ref).astype(np.uint8) for pred_tab in prediction
    ]
    ref_ind = 1

    # init output
    accuracy = []
    true_pos = []
    true_neg = []

    # loop over prediction arrays
    for i, pred_tab in enumerate(prediction_ind):
        # compute accuracy indicator
        accuracy_tab = 1 - (
            pred_tab != ground_truth_ind[:, None]
        ).astype(np.uint8)
        # store error
        accuracy.append(accuracy_tab)
        # compute and store true positive
        true_pos.append(accuracy_tab[ground_truth_ind == ref_ind])
        # compute and store true negative
        true_neg.append(accuracy_tab[ground_truth_ind != ref_ind])

    # output
    return accuracy, true_pos, true_neg
