import numpy as np
import numpy.testing as npt
import os
import pandas as pd
import pytest
import secrets
import torch as t
import warnings

from ktest.tester import Ktest

from .test_data import (
    data_shape, dummy_data_array, dummy_data, dummy_zidata, dummy_separated_data
)


@pytest.fixture
def dummy_ktest(dummy_data):
    """
    Function to create Ktest object from dummy data for testing using a given
    floating point number type/precision.
    """
    def _ktest(nystrom=False, dtype=t.float64):
        # init object
        kt = Ktest(
            data=dummy_data[0], metadata=dummy_data[1], nystrom=nystrom,
            dtype=dtype
        )
        # run kfda test
        kt.test(verbose=0)
        # output
        return kt
    # fixture output
    yield _ktest


@pytest.mark.parametrize("nystrom", [False, True])
@pytest.mark.parametrize("dtype", [t.float64, t.float32])
def test_ktest(dummy_ktest, dummy_data, nystrom, dtype):
    """Testing Ktest class."""
    # create ktest objects
    kt = dummy_ktest(nystrom, dtype)

    # check object class
    assert isinstance(kt, Ktest)

    ## check content
    # input data
    for group in np.unique(dummy_data[1]):
        np.testing.assert_allclose(
            kt.data.data[group].numpy(),
            dummy_data[0][dummy_data[1] == group].to_numpy(),
            rtol=0, atol=1e-6
        )
    pd.testing.assert_series_equal(kt.metadata, dummy_data[1])

    # output
    assert isinstance(kt.kfda_statistic, pd.Series)
    assert isinstance(kt.kfda_pval_asymp, pd.Series)


def test_ktest_precision(dummy_ktest, assert_equal_ktest):
    """Testing Ktest computing with various precision."""
    # create ktest objects (for a given precision)
    kt_f32 = dummy_ktest(dtype=t.float32)
    kt_f64 = dummy_ktest(dtype=t.float64)

    # check output type
    assert kt_f32.kfda_statistic.dtype == "float32"
    assert kt_f32.kfda_pval_asymp.dtype == "float32"
    assert kt_f64.kfda_statistic.dtype == "float64"
    assert kt_f64.kfda_pval_asymp.dtype == "float64"

    # compare results
    # (tolerance threshold not too tight because of expected result differences
    # due to precision difference)
    max_len = min([len(kt_f32.kfda_statistic), len(kt_f64.kfda_statistic)])
    npt.assert_allclose(
        kt_f32.kstat.sp.numpy()[:max_len],
        kt_f64.kstat.sp.numpy()[:max_len],
        rtol=0, atol=1e-7
    )
    try:
        npt.assert_allclose(
            kt_f32.kfda_statistic.to_numpy()[:max_len],
            kt_f64.kfda_statistic.to_numpy()[:max_len],
            rtol=0, atol=1e-3
        )
    except AssertionError as e:
        # use warning in that case because max absolute difference is
        # often higher than required threshold (but not always)
        warnings.warn(e)
    try:
        npt.assert_allclose(
            kt_f32.kfda_pval_asymp.to_numpy()[:max_len],
            kt_f64.kfda_pval_asymp.to_numpy()[:max_len],
            rtol=0, atol=1e-5
        )
    except AssertionError as e:
        # use warning in that case because max absolute difference is
        # often higher than required threshold (but not always)
        warnings.warn(e)


@pytest.fixture
def output_file():
    """
    Dummy file to test saving/loading.

    Note: file will be removed during test teardown.
    """
    # add a random unique tag to file
    yield os.path.join(
        pytest.output_dir, f"ktest_obj_{secrets.token_hex(4)}.pkl"
    )


def test_save_load(dummy_ktest, output_file, assert_equal_ktest):
    """Test Ktest object saving and loading."""
    # create ktest objects
    kt = dummy_ktest()
    # saving (no compression)
    kt.save(output_file, compress=False)
    # check
    assert os.path.isfile(output_file)
    # saving (compression)
    kt.save(output_file, compress=True)
    # check
    assert os.path.isfile(f"{output_file}.gz")

    # loading (no compression)
    kt_1 = Ktest.load(output_file, compressed=False)
    # loading (compression)
    kt_2 = Ktest.load(f"{output_file}.gz", compressed=True)

    ## checks (compare before and after loading)
    assert_equal_ktest(kt, kt_1, atol=1e-9)
    assert_equal_ktest(kt, kt_2, atol=1e-9)


@pytest.fixture
def exp_data():
    """Data and metadata from experimental transcriptomic dataset."""
    # data frame
    data = pd.read_csv(pytest.data_file, index_col=0)
    # metadata
    meta = pd.Series(data.index).apply(lambda x: x.split(sep='.')[1])
    meta.index = data.index
    # sample names
    sample_names = ['48HREV', '48HDIFF']
    # output
    yield data, meta, sample_names


@pytest.fixture
def kt_data(exp_data):
    """Create Ktest object from dummy data for testing."""
    # init object
    kt = Ktest(
        data=exp_data[0], metadata=exp_data[1], sample_names=exp_data[2]
    )
    # run kfda test
    kt.test(verbose=0)
    # output
    yield kt


def test_num_stability(kt_data, assert_equal_ktest):
    """
    Compare numerical results to previous version of ktest (if available).
    """

    # saving current results
    kt_data.save(pytest.res_file, compress=True)

    # load previous results (if available)
    if pytest.previous_res_file is not None:
        kt_data_prev = Ktest.load(pytest.previous_res_file, compressed=True)
        assert_equal_ktest(
            kt_data, kt_data_prev, trunc=len(kt_data.kfda_statistic), atol=1e-8
        )


@pytest.mark.parametrize("nystrom", [False, True])
def test_constant_var(dummy_data, data_shape, nystrom):
    """Testing the case of data with a constant column."""

    # collect input data
    data, metadata = dummy_data

    # insert one constant column in data
    data[data.columns[0]].values[:] = 0

    # init object
    kt = Ktest(data=data, metadata=metadata, nystrom=nystrom)
    # run kfda test
    kt.test(verbose=0)

    # insert constant columns in data (except final one)
    for col in data.columns[:99]:
        data[col].values[:] = 0

    # init object
    kt = Ktest(data=data, metadata=metadata, nystrom=nystrom)
    # run kfda test
    kt.test(verbose=0)

    # insert constant columns in data (except final one)
    for col in data.columns[:99]:
        data[col].values[:] = 0

    # init object
    kt = Ktest(data=data, metadata=metadata, nystrom=nystrom)
    # run kfda test
    kt.test(verbose=0)

    # insert constant columns in final column (in one group only)
    data[data.columns[99]].values[::2] = 0

    # init object
    kt = Ktest(data=data, metadata=metadata, nystrom=nystrom)
    # run kfda test
    kt.test(verbose=0)

    # insert constant columns in final column (in all groups)
    data[data.columns[99]].values[:] = 0

    err_msg = "All variables have constant values in your data."
    with pytest.raises(RuntimeError) as excinfo:
        # init object
        kt = Ktest(data=data, metadata=metadata, nystrom=nystrom)
        # run kfda test
        kt.test(verbose=0)
    assert str(excinfo.value) == err_msg


@pytest.mark.parametrize("nystrom", [False, True])
def test_zi_data(dummy_zidata, data_shape, nystrom):
    """Testing the case of data with zero-inflation."""

    # init object
    kt = Ktest(data=dummy_zidata[0], metadata=dummy_zidata[1], nystrom=nystrom)
    # run kfda test
    kt.test(verbose=0)


def test_ktest_project(dummy_ktest):
    """Testing projection in Ktest class."""
    # create ktest objects
    kt = dummy_ktest()

    # Case 1 - centering, no new obs
    proj_kfda, proj_kpca = kt.project(
        t=10, center=True, verbose=1, new_obs=None
    )

    # expected results
    stat_val, _ = kt.kstat.compute_kfda()
    exp_proj_kfda, exp_proj_kpca = kt.kstat.compute_projections(
        stat=stat_val, t=10, center=True
    )

    # check
    assert isinstance(proj_kfda, dict)
    assert len(proj_kfda) == len(exp_proj_kfda)
    assert proj_kfda.keys() == exp_proj_kfda.keys()
    assert isinstance(proj_kpca, dict)
    assert len(proj_kpca) == len(exp_proj_kpca)
    assert proj_kpca.keys() == exp_proj_kpca.keys()

    for (
        proj_kfda_tab, exp_proj_kfda_tab,
        proj_kpca_tab, exp_proj_kpca_tab,
        data_tab
    ) in zip(
        proj_kfda.values(), exp_proj_kfda.values(),
        proj_kpca.values(), exp_proj_kpca.values(),
        kt.data.data.values()
    ):
        assert isinstance(proj_kfda_tab, pd.DataFrame)
        assert proj_kfda_tab.shape == (data_tab.shape[0], 10)
        pd.testing.assert_frame_equal(proj_kfda_tab, exp_proj_kfda_tab)
        assert isinstance(proj_kpca_tab, pd.DataFrame)
        assert proj_kpca_tab.shape == (data_tab.shape[0], 10)
        pd.testing.assert_frame_equal(proj_kpca_tab, exp_proj_kpca_tab)

    # Case 2 - not centering, no new obs
    proj_kfda, proj_kpca = kt.project(
        t=10, center=False, verbose=1, new_obs=None
    )

    # expected results
    stat_val, _ = kt.kstat.compute_kfda()
    exp_proj_kfda, exp_proj_kpca = kt.kstat.compute_projections(
        stat=stat_val, t=10, center=False
    )

    # check
    assert isinstance(proj_kfda, dict)
    assert len(proj_kfda) == len(exp_proj_kfda)
    assert proj_kfda.keys() == exp_proj_kfda.keys()
    assert isinstance(proj_kpca, dict)
    assert len(proj_kpca) == len(exp_proj_kpca)
    assert proj_kpca.keys() == exp_proj_kpca.keys()

    for (
        proj_kfda_tab, exp_proj_kfda_tab,
        proj_kpca_tab, exp_proj_kpca_tab,
        data_tab
    ) in zip(
        proj_kfda.values(), exp_proj_kfda.values(),
        proj_kpca.values(), exp_proj_kpca.values(),
        kt.data.data.values()
    ):
        assert isinstance(proj_kfda_tab, pd.DataFrame)
        assert proj_kfda_tab.shape == (data_tab.shape[0], 10)
        pd.testing.assert_frame_equal(proj_kfda_tab, exp_proj_kfda_tab)
        assert isinstance(proj_kpca_tab, pd.DataFrame)
        assert proj_kpca_tab.shape == (data_tab.shape[0], 10)
        pd.testing.assert_frame_equal(proj_kpca_tab, exp_proj_kpca_tab)

    # Case 3 - centering, providing new obs
    new_obs = list(kt.kstat.data.data.values())[0]
    proj_kfda, proj_kpca = kt.project(
        t=10, center=True, verbose=1, new_obs=new_obs
    )

    # expected results
    stat_val, _ = kt.kstat.compute_kfda()
    exp_proj_kfda, exp_proj_kpca = kt.kstat.compute_projections(
        stat=stat_val, t=10, center=True, new_obs=new_obs
    )

    # check
    assert isinstance(proj_kfda, dict)
    assert len(proj_kfda) == len(exp_proj_kfda)
    assert proj_kfda.keys() == exp_proj_kfda.keys()
    assert isinstance(proj_kpca, dict)
    assert len(proj_kpca) == len(exp_proj_kpca)
    assert proj_kpca.keys() == exp_proj_kpca.keys()

    for (
        proj_kfda_tab, exp_proj_kfda_tab,
        proj_kpca_tab, exp_proj_kpca_tab
    ) in zip(
        proj_kfda.values(), exp_proj_kfda.values(),
        proj_kpca.values(), exp_proj_kpca.values()
    ):
        assert isinstance(proj_kfda_tab, pd.DataFrame)
        assert proj_kfda_tab.shape == (new_obs.shape[0], 10)
        pd.testing.assert_frame_equal(proj_kfda_tab, exp_proj_kfda_tab)
        assert isinstance(proj_kpca_tab, pd.DataFrame)
        assert proj_kpca_tab.shape == (new_obs.shape[0], 10)
        pd.testing.assert_frame_equal(proj_kpca_tab, exp_proj_kpca_tab)

    # Case 4 - not centering, providing new obs
    new_obs = list(kt.kstat.data.data.values())[0]
    proj_kfda, proj_kpca = kt.project(
        t=10, center=False, verbose=1, new_obs=new_obs
    )

    # expected results
    stat_val, _ = kt.kstat.compute_kfda()
    exp_proj_kfda, exp_proj_kpca = kt.kstat.compute_projections(
        stat=stat_val, t=10, center=False, new_obs=new_obs
    )

    # check
    assert isinstance(proj_kfda, dict)
    assert len(proj_kfda) == len(exp_proj_kfda)
    assert proj_kfda.keys() == exp_proj_kfda.keys()
    assert isinstance(proj_kpca, dict)
    assert len(proj_kpca) == len(exp_proj_kpca)
    assert proj_kpca.keys() == exp_proj_kpca.keys()

    for (
        proj_kfda_tab, exp_proj_kfda_tab,
        proj_kpca_tab, exp_proj_kpca_tab
    ) in zip(
        proj_kfda.values(), exp_proj_kfda.values(),
        proj_kpca.values(), exp_proj_kpca.values()
    ):
        assert isinstance(proj_kfda_tab, pd.DataFrame)
        assert proj_kfda_tab.shape == (new_obs.shape[0], 10)
        pd.testing.assert_frame_equal(proj_kfda_tab, exp_proj_kfda_tab)
        assert isinstance(proj_kpca_tab, pd.DataFrame)
        assert proj_kpca_tab.shape == (new_obs.shape[0], 10)
        pd.testing.assert_frame_equal(proj_kpca_tab, exp_proj_kpca_tab)


def test_ktest_predict(dummy_ktest, dummy_separated_data, capsys):
    """Testing prediction in Ktest class."""

    # Case 1a - no new obs, a single prediction bias (threshold) value
    # (no bias)
    # data under H0

    # create ktest objects
    kt = dummy_ktest()

    # run CV
    pred, loss = kt.predict(
        t=10, new_obs=None, pred_threshold=0.5, verbose=1
    )

    # check
    assert isinstance(pred, dict)
    assert isinstance(loss, dict)
    assert pred.keys() == kt.data.data.keys()
    assert loss.keys() == kt.data.data.keys()

    for group in kt.data.data.keys():
        n_obs = kt.data.data[group].shape[0]

        assert isinstance(pred[group], list)
        assert isinstance(loss[group], list)

        assert len(pred[group]) == 1
        assert len(loss[group]) == 1

        pred_val = pred[group][0]
        loss_val = loss[group][0]

        assert isinstance(pred_val, np.ndarray)
        assert isinstance(loss_val, np.ndarray)

        assert list(pred_val.shape) == [n_obs, 10]
        assert list(loss_val.shape) == [n_obs, 10]

        assert np.all(np.isin(pred_val, list(kt.data.data.keys())))
        assert np.issubdtype(loss_val.dtype, np.floating)

        # group 1 and 2 are similar so we expect 50%-50% prediction
        count_pred = np.count_nonzero(pred_val == group, axis=0)
        np.testing.assert_allclose(count_pred / n_obs, 1/2, atol=0.1)

    # Case 1b - full data (no Nystrom), no new_obs,
    # with a list of threshold values
    # default: new_obs=None
    threshold_values = np.linspace(0, 1, 11)

    # create ktest objects
    kt = dummy_ktest()

    # run CV
    pred, loss = kt.predict(
        t=10, new_obs=None, pred_threshold=threshold_values, verbose=1
    )

    # check
    assert isinstance(pred, dict)
    assert isinstance(loss, dict)
    assert pred.keys() == kt.data.data.keys()
    assert loss.keys() == kt.data.data.keys()

    for group_ind, group in enumerate(kt.data.data.keys()):
        n_obs = kt.data.data[group].shape[0]

        assert isinstance(pred[group], list)
        assert isinstance(loss[group], list)

        assert len(pred[group]) == len(threshold_values)
        assert len(loss[group]) == len(threshold_values)

        for pred_val, loss_val, pred_threshold in zip(
            pred[group], loss[group], threshold_values
        ):

            assert isinstance(pred_val, np.ndarray)
            assert isinstance(loss_val, np.ndarray)

            assert list(pred_val.shape) == [n_obs, 10]
            assert list(loss_val.shape) == [n_obs, 10]

            assert np.all(np.isin(pred_val, list(kt.data.data.keys())))
            assert np.issubdtype(loss_val.dtype, np.floating)

            # group 1 and 2 are similar so we expect a prediction
            # corresponding to the bias
            # (or 1 - bias depending on the group)
            count_pred = np.count_nonzero(pred_val == group, axis=0)
            np.testing.assert_allclose(
                count_pred / n_obs,
                (1 - group_ind) * pred_threshold +
                group_ind * (1 - pred_threshold),
                atol=0.3
            )

    # Case 2a - full data (no Nystrom), providing new_obs
    # new observations: use one population subsample

    # create ktest objects
    kt = dummy_ktest()

    # new data
    new_obs = list(kt.data.data.values())[0]

    # run CV
    pred, loss = kt.predict(
        t=10, new_obs=new_obs, pred_threshold=0.5, verbose=1
    )

    # check
    assert isinstance(pred, dict)
    assert isinstance(loss, dict)
    assert list(pred.keys()) == ["new_obs"]
    assert list(loss.keys()) == ["new_obs"]

    group = "new_obs"
    n_obs = new_obs.shape[0]

    assert isinstance(pred[group], list)
    assert isinstance(loss[group], list)

    assert len(pred[group]) == 1
    assert len(loss[group]) == 1

    pred_val = pred[group][0]
    loss_val = loss[group][0]

    assert isinstance(pred_val, np.ndarray)
    assert isinstance(loss_val, np.ndarray)

    assert list(pred_val.shape) == [n_obs, 10]
    assert list(loss_val.shape) == [n_obs, 10]

    assert np.all(np.isin(pred_val, list(kt.data.data.keys())))
    assert np.issubdtype(loss_val.dtype, np.floating)

    # group 1 and 2 are similar so we expect 50%-50% prediction
    count_pred = np.count_nonzero(pred_val == "c1", axis=0)
    np.testing.assert_allclose(count_pred / n_obs, 1/2, atol=0.1)

    # Case 2b - full data (no Nystrom), providing new_obs,
    # biasing prediction (expect only group 2 prediction)
    # new observations: use one population subsample

    # create ktest objects
    kt = dummy_ktest()

    # new data
    new_obs = list(kt.data.data.values())[0]

    # run CV
    pred, loss = kt.predict(
        t=10, new_obs=new_obs, pred_threshold=0, verbose=1
    )

    # check
    assert isinstance(pred, dict)
    assert isinstance(loss, dict)
    assert list(pred.keys()) == ["new_obs"]
    assert list(loss.keys()) == ["new_obs"]

    group = "new_obs"
    n_obs = new_obs.shape[0]

    assert isinstance(pred[group], list)
    assert isinstance(loss[group], list)

    assert len(pred[group]) == 1
    assert len(loss[group]) == 1

    pred_val = pred[group][0]
    loss_val = loss[group][0]

    assert isinstance(pred_val, np.ndarray)
    assert isinstance(loss_val, np.ndarray)

    assert list(pred_val.shape) == [n_obs, 10]
    assert list(loss_val.shape) == [n_obs, 10]

    assert np.all(np.isin(pred_val, list(kt.data.data.keys())))
    assert np.issubdtype(loss_val.dtype, np.floating)

    # we expect only "group 2" prediction
    count_pred = np.count_nonzero(pred_val == "c1", axis=0)
    np.testing.assert_allclose(count_pred / n_obs, 0, atol=0)

    # Case 2c - full data (no Nystrom), providing new_obs,
    # biasing prediction (expect only group 1 prediction)
    # new observations: use one population subsample

    # create ktest objects
    kt = dummy_ktest()

    # new data
    new_obs = list(kt.data.data.values())[0]

    # run CV
    pred, loss = kt.predict(
        t=10, new_obs=new_obs, pred_threshold=1, verbose=1
    )

    # check
    assert isinstance(pred, dict)
    assert isinstance(loss, dict)
    assert list(pred.keys()) == ["new_obs"]
    assert list(loss.keys()) == ["new_obs"]

    group = "new_obs"
    n_obs = new_obs.shape[0]

    assert isinstance(pred[group], list)
    assert isinstance(loss[group], list)

    assert len(pred[group]) == 1
    assert len(loss[group]) == 1

    pred_val = pred[group][0]
    loss_val = loss[group][0]

    assert isinstance(pred_val, np.ndarray)
    assert isinstance(loss_val, np.ndarray)

    assert list(pred_val.shape) == [n_obs, 10]
    assert list(loss_val.shape) == [n_obs, 10]

    assert np.all(np.isin(pred_val, list(kt.data.data.keys())))
    assert np.issubdtype(loss_val.dtype, np.floating)

    # we expect only "group 1" prediction
    count_pred = np.count_nonzero(pred_val == "c1", axis=0)
    np.testing.assert_allclose(count_pred / n_obs, 1, atol=0)

    # Case 3 - Nystrom approximation, no new_obs
    # default: new_obs=None

    # create ktest objects
    kt = dummy_ktest(nystrom=True)

    # run CV
    pred, loss = kt.predict(
        t=10, new_obs=None, pred_threshold=0.5, verbose=1
    )

    # check
    assert isinstance(pred, dict)
    assert isinstance(loss, dict)
    assert pred.keys() == kt.data.data.keys()
    assert loss.keys() == kt.data.data.keys()

    for group in kt.data.data.keys():
        n_obs = kt.data.data[group].shape[0]

        assert isinstance(pred[group], list)
        assert isinstance(loss[group], list)

        assert len(pred[group]) == 1
        assert len(loss[group]) == 1

        pred_val = pred[group][0]
        loss_val = loss[group][0]

        assert isinstance(pred_val, np.ndarray)
        assert isinstance(loss_val, np.ndarray)

        assert list(pred_val.shape) == [n_obs, 10]
        assert list(loss_val.shape) == [n_obs, 10]

        assert np.all(np.isin(
            pred_val, list(kt.data.data.keys())
        ))
        assert np.issubdtype(loss_val.dtype, np.floating)

        # group 1 and 2 are similar so we expect 50%-50% prediction
        count_pred = np.count_nonzero(pred_val == group, axis=0)
        np.testing.assert_allclose(count_pred / n_obs, 1/2, atol=0.1)

    # Case 4 - Nystrom approximation, providing new_obs
    # new observations: use one population subsample

    # create ktest objects
    kt = dummy_ktest(nystrom=True)

    # new data
    new_obs = list(kt.data.data.values())[0]

    # run CV
    pred, loss = kt.predict(
        t=10, new_obs=new_obs, pred_threshold=1/2, verbose=1
    )

    # check
    assert isinstance(pred, dict)
    assert isinstance(loss, dict)
    assert list(pred.keys()) == ["new_obs"]
    assert list(loss.keys()) == ["new_obs"]

    group = "new_obs"
    n_obs = new_obs.shape[0]

    assert isinstance(pred[group], list)
    assert isinstance(loss[group], list)

    assert len(pred[group]) == 1
    assert len(loss[group]) == 1

    pred_val = pred[group][0]
    loss_val = loss[group][0]

    assert isinstance(pred_val, np.ndarray)
    assert isinstance(loss_val, np.ndarray)

    assert list(pred_val.shape) == [n_obs, 10]
    assert list(loss_val.shape) == [n_obs, 10]

    assert np.all(np.isin(pred_val, list(kt.data_nystrom.data.keys())))
    assert np.issubdtype(loss_val.dtype, np.floating)

    # group 1 and 2 are similar so we expect 50%-50% prediction
    count_pred = np.count_nonzero(pred_val == "c1", axis=0)
    np.testing.assert_allclose(count_pred / n_obs, 1/2, atol=0.1)

    # Case 5 - separated data (Nystrom approximation, no new_obs)

    # create ktest objects
    kt = Ktest(
        data=dummy_separated_data[0],
        metadata=dummy_separated_data[1],
        nystrom=True
    )

    # run CV
    pred, loss = kt.predict(
        t=50, new_obs=None, pred_threshold=1/2, verbose=1
    )

    # check
    assert isinstance(pred, dict)
    assert isinstance(loss, dict)
    assert pred.keys() == kt.data.data.keys()
    assert loss.keys() == kt.data.data.keys()

    for i, group in enumerate(kt.data.data.keys()):
        n_obs = kt.data.data[group].shape[0]

        assert isinstance(pred[group], list)
        assert isinstance(loss[group], list)

        assert len(pred[group]) == 1
        assert len(loss[group]) == 1

        pred_val = pred[group][0]
        loss_val = loss[group][0]

        assert isinstance(pred_val, np.ndarray)
        assert isinstance(loss_val, np.ndarray)

        assert list(pred_val.shape) == [n_obs, 50]
        assert list(loss_val.shape) == [n_obs, 50]

        assert np.all(np.isin(
            pred_val, list(kt.data.data.keys())
        ))
        assert np.issubdtype(loss_val.dtype, np.floating)

        # group 1 and 2 are very different so we expect 100%
        # prediction on each group (at least for large truncations)
        count_pred = np.count_nonzero(
            pred_val[:, -10:] == group, axis=0
        )
        np.testing.assert_allclose(count_pred / n_obs, 1, atol=0.1)


def test_ktest_cv(dummy_ktest, dummy_separated_data, capsys):
    """Test cross-validation for kFDA prediction."""

    # Case 1a - data under H0, a single prediction bias (threshold) value
    # no bias
    # create ktest objects
    kt = dummy_ktest()
    # run CV
    accuracy, true_pos, true_neg = kt.cv(
        t=50, pred_threshold=1/2, n_fold=5, n_repeat=1,
        random_state=None, verbose=1
    )
    # check stdout (expect output)
    captured = capsys.readouterr()
    assert captured.out == 'Starting cross-validation...' + \
        '\nSplit 0\nSplit 1\nSplit 2\nSplit 3\nSplit 4\n' + \
        '...Aggregating CV fold results\n'

    # check output
    assert isinstance(accuracy, list)
    assert isinstance(true_pos, list)
    assert isinstance(true_neg, list)

    assert len(accuracy) == 1
    assert len(true_pos) == 1
    assert len(true_neg) == 1

    for accuracy_tab, true_pos_tab, true_neg_tab in zip(
        accuracy, true_pos, true_neg
    ):
        # type
        assert isinstance(accuracy_tab, np.ndarray)
        assert isinstance(true_pos_tab, np.ndarray)
        assert isinstance(true_neg_tab, np.ndarray)

        # dimension
        assert accuracy_tab.shape == (50,)
        assert true_pos_tab.shape == (50,)
        assert true_neg_tab.shape == (50,)

        # value (expect 50%-50% accuracy)
        np.testing.assert_allclose(accuracy_tab, 1/2, atol=0.1)
        np.testing.assert_allclose(true_pos_tab, 1/2, atol=0.1)
        np.testing.assert_allclose(true_neg_tab, 1/2, atol=0.1)

    # Case 1b - data under H0, a single prediction bias (threshold) value
    # bias towards ref group
    # create ktest objects
    kt = dummy_ktest()
    # run CV
    accuracy, true_pos, true_neg = kt.cv(
        t=50, pred_threshold=1, n_fold=5, n_repeat=1,
        random_state=None, verbose=0
    )

    # check stdout (expect no output)
    captured = capsys.readouterr()
    assert captured.out == ''

    # check output
    assert isinstance(accuracy, list)
    assert isinstance(true_pos, list)
    assert isinstance(true_neg, list)

    assert len(accuracy) == 1
    assert len(true_pos) == 1
    assert len(true_neg) == 1

    for accuracy_tab, true_pos_tab, true_neg_tab in zip(
        accuracy, true_pos, true_neg
    ):
        # type
        assert isinstance(accuracy_tab, np.ndarray)
        assert isinstance(true_pos_tab, np.ndarray)
        assert isinstance(true_neg_tab, np.ndarray)

        # dimension
        assert accuracy_tab.shape == (50,)
        assert true_pos_tab.shape == (50,)
        assert true_neg_tab.shape == (50,)

        # value (expect 50%-50% accuracy but biased sens/spec)
        np.testing.assert_allclose(accuracy_tab, 1/2, atol=0.1)
        np.testing.assert_allclose(true_pos_tab, 1, atol=0.1)
        np.testing.assert_allclose(true_neg_tab, 0, atol=0.1)

    # Case 1c - data under H0, a single prediction bias (threshold) value
    # bias towards non ref group
    # create ktest objects
    kt = dummy_ktest()
    # run CV
    accuracy, true_pos, true_neg = kt.cv(
        t=50, pred_threshold=0, n_fold=5, n_repeat=1,
        random_state=None, verbose=0
    )

    # check output
    assert isinstance(accuracy, list)
    assert isinstance(true_pos, list)
    assert isinstance(true_neg, list)

    assert len(accuracy) == 1
    assert len(true_pos) == 1
    assert len(true_neg) == 1

    for accuracy_tab, true_pos_tab, true_neg_tab in zip(
        accuracy, true_pos, true_neg
    ):
        # type
        assert isinstance(accuracy_tab, np.ndarray)
        assert isinstance(true_pos_tab, np.ndarray)
        assert isinstance(true_neg_tab, np.ndarray)

        # dimension
        assert accuracy_tab.shape == (50,)
        assert true_pos_tab.shape == (50,)
        assert true_neg_tab.shape == (50,)

        # value (expect 50%-50% accuracy but biased sens/spec)
        np.testing.assert_allclose(accuracy_tab, 1/2, atol=0.1)
        np.testing.assert_allclose(true_pos_tab, 0, atol=0.1)
        np.testing.assert_allclose(true_neg_tab, 1, atol=0.1)

    # Case 2 - data under H0, multiple prediction bias (threshold) values
    # create ktest objects
    kt = dummy_ktest()
    # run CV
    threshold_values = np.linspace(0, 1, 11)
    accuracy, true_pos, true_neg = kt.cv(
        t=50, pred_threshold=threshold_values, n_fold=5, n_repeat=1,
        random_state=None, verbose=0
    )
    # check output
    assert isinstance(accuracy, list)
    assert isinstance(true_pos, list)
    assert isinstance(true_neg, list)

    assert len(accuracy) == len(threshold_values)
    assert len(true_pos) == len(threshold_values)
    assert len(true_neg) == len(threshold_values)

    for accuracy_tab, true_pos_tab, true_neg_tab, pred_threshold in zip(
        accuracy, true_pos, true_neg, threshold_values
    ):
        # type
        assert isinstance(accuracy_tab, np.ndarray)
        assert isinstance(true_pos_tab, np.ndarray)
        assert isinstance(true_neg_tab, np.ndarray)

        # dimension
        assert accuracy_tab.shape == (50,)
        assert true_pos_tab.shape == (50,)
        assert true_neg_tab.shape == (50,)

        # value (expect 50%-50% accuracy)
        np.testing.assert_allclose(accuracy_tab, 1/2, atol=0.1)
        np.testing.assert_allclose(true_pos_tab, pred_threshold, atol=0.3)
        np.testing.assert_allclose(true_neg_tab, 1-pred_threshold, atol=0.3)

    # Case 3 - data under H1
    # create ktest objects
    kt = Ktest(
        data=dummy_separated_data[0],
        metadata=dummy_separated_data[1],
        nystrom=True
    )
    # run CV
    accuracy, true_pos, true_neg = kt.cv(
        t=50, pred_threshold=1/2, n_fold=5, n_repeat=1,
        random_state=None, verbose=1
    )

    # check output
    assert isinstance(accuracy, list)
    assert isinstance(true_pos, list)
    assert isinstance(true_neg, list)

    assert len(accuracy) == 1
    assert len(true_pos) == 1
    assert len(true_neg) == 1

    for accuracy_tab, true_pos_tab, true_neg_tab in zip(
        accuracy, true_pos, true_neg
    ):
        # type
        assert isinstance(accuracy_tab, np.ndarray)
        assert isinstance(true_pos_tab, np.ndarray)
        assert isinstance(true_neg_tab, np.ndarray)

        # dimension
        assert accuracy_tab.shape == (50,)
        assert true_pos_tab.shape == (50,)
        assert true_neg_tab.shape == (50,)

        # value (expect almost 100% accuracy for non-small truncations)
        np.testing.assert_allclose(accuracy_tab[10:], 1, atol=0.1)
        np.testing.assert_allclose(true_pos_tab[10:], 1, atol=0.1)
        np.testing.assert_allclose(true_neg_tab[10:], 1, atol=0.1)
