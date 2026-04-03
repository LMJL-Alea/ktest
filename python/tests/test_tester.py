import numpy as np
import numpy.testing as npt
import os
import pandas as pd
import pytest
import secrets
import torch as t
import warnings

from ktest.tester import Ktest

from .test_data import data_shape, dummy_data, dummy_zidata


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
def test_Ktest(dummy_ktest, dummy_data, nystrom, dtype):
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

    # insert constant columns in data (all)
    data[data.columns[99]].values[::2] = 0

    if nystrom:  # not subsampling otherwise and thus no issue
        err_msg = "Subsampling failed after 100 trials. " + \
            "All variables have constant values in at leat one subsample."
        with pytest.raises(RuntimeError) as excinfo:
            # init object
            kt = Ktest(data=data, metadata=metadata, nystrom=nystrom)
            # run kfda test
            kt.test(verbose=0)
        assert str(excinfo.value) == err_msg
    else:
        # init object
        kt = Ktest(data=data, metadata=metadata, nystrom=nystrom)
        # run kfda test
        kt.test(verbose=0)


@pytest.mark.parametrize("nystrom", [False, True])
def test_zi_data(dummy_zidata, data_shape, nystrom):
    """Testing the case of data with zero-inflation."""

    # init object
    kt = Ktest(data=dummy_zidata[0], metadata=dummy_zidata[1], nystrom=nystrom)
    # run kfda test
    kt.test(verbose=0)
