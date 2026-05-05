import functools
import pytest
import numpy as np
import torch as to

from ktest.utils import verbosity, pred_threshold_fun, compute_accuracy


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


def test_compute_accuracy():
    """Testing function computing accuracy in kFDA cross-validation."""

    # Pseudo-random number generator
    rng = np.random.default_rng(42)

    # consider 2-level factor
    group_name = np.array(['c1', 'c2'])

    # Case 1a: prediction = ground truth (expect no error), ref is provided
    # ground truth
    ground_truth_index = rng.choice([0, 1], size=(20,)).astype(np.uint)
    ground_truth = group_name[ground_truth_index]
    # create prediction
    prediction_index = [np.repeat(ground_truth_index[:, None], 10, axis=1)]
    prediction = [group_name[tab] for tab in prediction_index]

    # try computing accuracy
    accuracy, true_pos, true_neg = compute_accuracy(
        prediction, ground_truth, ref=group_name[0]
    )

    # check output
    assert isinstance(accuracy, list)
    assert isinstance(true_pos, list)
    assert isinstance(true_neg, list)

    for accuracy_tab, true_pos_tab, true_neg_tab in zip(
        accuracy, true_pos, true_neg
    ):
        # type
        assert isinstance(accuracy_tab, np.ndarray)
        assert isinstance(true_pos_tab, np.ndarray)
        assert isinstance(true_neg_tab, np.ndarray)

        # dimension
        assert accuracy_tab.shape == (20, 10)
        ref = group_name[0]
        assert true_pos_tab.shape == \
            (np.sum(ground_truth == ref), 10)
        assert true_neg_tab.shape == \
            (np.sum(ground_truth != ref), 10)

        # value
        assert np.all(accuracy_tab == 1)
        assert np.all(true_pos_tab == 1)
        assert np.all(true_neg_tab == 1)

    # Case 1b: prediction = ground truth (expect no error), ref is not provided
    # try computing accuracy
    accuracy, true_pos, true_neg = compute_accuracy(
        prediction, ground_truth, ref=group_name[0]
    )

    # check output
    assert isinstance(accuracy, list)
    assert isinstance(true_pos, list)
    assert isinstance(true_neg, list)

    for accuracy_tab, true_pos_tab, true_neg_tab in zip(
        accuracy, true_pos, true_neg
    ):
        # type
        assert isinstance(accuracy_tab, np.ndarray)
        assert isinstance(true_pos_tab, np.ndarray)
        assert isinstance(true_neg_tab, np.ndarray)

        # dimension
        assert accuracy_tab.shape == (20, 10)
        ref = ground_truth[0]
        assert true_pos_tab.shape == \
            (np.sum(ground_truth == ref), 10)
        assert true_neg_tab.shape == \
            (np.sum(ground_truth != ref), 10)

        # value
        assert np.all(accuracy_tab == 1)
        assert np.all(true_pos_tab == 1)
        assert np.all(true_neg_tab == 1)

    # Case 2: prediction = 1 - ground truth (expect only errors)
    # create prediction
    prediction = [group_name[1 - tab] for tab in prediction_index]

    # try computing accuracy
    accuracy, true_pos, true_neg = compute_accuracy(
        prediction, ground_truth, ref=group_name[0]
    )

    # check
    for accuracy_tab, true_pos_tab, true_neg_tab in zip(
        accuracy, true_pos, true_neg
    ):
        assert np.all(accuracy_tab == 0)
        assert np.all(true_pos_tab == 0)
        assert np.all(true_neg_tab == 0)

    # Case 3: multiple random predictions
    # generate data (list of 2-level factor arrays)
    prediction = []
    rng = np.random.default_rng(42)
    for i in range(10):
        prediction.append(
            rng.choice(group_name, size=(20, 10))
        )

    # try computing accuracy
    accuracy, true_pos, true_neg = compute_accuracy(
        prediction, ground_truth, ref=group_name[0]
    )

    # check
    for accuracy_tab, true_pos_tab, true_neg_tab in zip(
        accuracy, true_pos, true_neg
    ):
        assert np.all(np.isin(accuracy_tab, [0, 1]))
        assert np.all(np.isin(true_pos_tab, [0, 1]))
        assert np.all(np.isin(true_neg_tab, [0, 1]))
