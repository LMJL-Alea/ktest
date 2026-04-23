import functools
import pytest
import numpy as np
import torch as to

from ktest.utils import verbosity, pred_threshold_fun


torch_assert_equal = functools.partial(to.testing.assert_close, rtol=0, atol=0)


def test_verbosity(capsys, recwarn):
    """Test function managing verbosity output."""

    # no output
    msg = "no output"
    verbosity(msg, msg_type="print", verbose=0)
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err == ''
    assert len(recwarn) == 0

    msg = "no output"
    verbosity(msg, msg_type="warning", verbose=0)
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err == ''
    assert len(recwarn) == 0

    # print only output
    msg = "print only"
    verbosity(msg, msg_type="print", verbose=1)
    captured = capsys.readouterr()
    assert captured.out == f"{msg}\n"
    assert captured.err == ''
    assert len(recwarn) == 0

    msg = "print only"
    verbosity(msg, msg_type="warning", verbose=1)
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err == ''
    assert len(recwarn) == 0

    # print or warning output
    msg = "this is a print"
    verbosity(msg, msg_type="print", verbose=2)
    captured = capsys.readouterr()
    assert captured.out == f"{msg}\n"
    assert captured.err == ''
    assert len(recwarn) == 0

    msg = "this is a warning"
    with pytest.warns(UserWarning, match=msg):
        verbosity(msg, msg_type="warning", verbose=2)
        captured = capsys.readouterr()
        assert captured.out == ''
        assert captured.err == ''

    # boolean verbosity level
    msg = "no output"
    verbosity(msg, msg_type="print", verbose=False)
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err == ''
    assert len(recwarn) == 0

    msg = "print only"
    verbosity(msg, msg_type="print", verbose=True)
    captured = capsys.readouterr()
    assert captured.out == f"{msg}\n"
    assert captured.err == ''
    assert len(recwarn) == 0

    # check bad input
    err_msg = "`msg` input should be a character string"
    with pytest.raises(AssertionError) as excinfo:
        verbosity(msg=111, msg_type="print", verbose=0)
    assert str(excinfo.value) == err_msg

    err_msg = "`msg_type` input should be either \"print\" or \"warning\""
    with pytest.raises(AssertionError) as excinfo:
        verbosity(msg="this is a message", msg_type="error", verbose=0)
    assert str(excinfo.value) == err_msg

    err_msg = "`verbosity` input should be a boolean or an integer value"
    with pytest.raises(AssertionError) as excinfo:
        verbosity(msg="this is a message", msg_type="print", verbose="0")
    assert str(excinfo.value) == err_msg


def test_pred_threshold_fun():
    """Testing function that define biased prediction threshold."""

    # testing scalar value input for left/right value parameters
    for left_val, right_val in zip(
        [5, 10, 15, 20, 40, 50], reversed([5, 10, 15, 20, 40, 50])
    ):

        # at x=0, y should be -left_val
        assert pred_threshold_fun(0, left_val, right_val) == -left_val
        # at x=0.5, y should be 0
        assert pred_threshold_fun(0.5, left_val, right_val) == 0
        # at x=1, y should be right_val
        assert pred_threshold_fun(1, left_val, right_val) == right_val

    # testing array-like input for left/right value parameters
    # numpy array input
    left_val = np.array([5, 10, 15, 20, 40, 50])
    right_val = left_val[::-1]

    for x in np.linspace(0, 0.5, 100):
        res = pred_threshold_fun(x, left_val, right_val)
        np.testing.assert_equal(res, 2 * left_val * x - left_val)
    for x in np.linspace(0.5, 1, 100):
        res = pred_threshold_fun(x, left_val, right_val)
        np.testing.assert_equal(res, 2 * right_val * x - right_val)

    # torch array input
    left_val = to.Tensor([5, 10, 15, 20, 40, 50])
    right_val = left_val.flip(dims=(0,))

    for x in to.linspace(0, 0.5, 100):
        res = pred_threshold_fun(x, left_val, right_val)
        torch_assert_equal(res, 2 * left_val * x - left_val)
    for x in to.linspace(0.5, 1, 100):
        res = pred_threshold_fun(x, left_val, right_val)
        torch_assert_equal(res, 2 * right_val * x - right_val)


