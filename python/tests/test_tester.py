import numpy as np
import os
import pandas as pd
import pytest
import secrets

from ktest.tester import Ktest


@pytest.fixture(scope="module")
def data_shape():
    """Number of rows and columns in dummy data for tests."""
    yield (1000, 100)


@pytest.fixture(scope="module")
def dummy_data(data_shape):
    """Generate dummy data for testing (two groups, no difference)"""
    # generate random data (under H0, no separation)
    rng = np.random.default_rng()
    data_array = rng.normal(loc=0, scale=1, size=data_shape)

    # create a data frame from random gaussian data
    data = pd.DataFrame(
        data=data_array,
        columns=[f"col{i+1}" for i in range(data_shape[1])]
    )

    # create meta data frame indicating two groups
    meta = pd.Series(
        data = [f"c{i+1}" for i in range(2)] * (data_shape[0] // 2)
    )

    # output
    yield data, meta


@pytest.fixture(scope="module")
def kt(dummy_data):
    """Create Ktest object from dummy data for testing."""
    # init object
    kt = Ktest(data=dummy_data[0], metadata=dummy_data[1])
    # run kfda test
    kt.test()
    # output
    yield kt


def test_Ktest(kt, dummy_data):
    """Testing Ktest class."""
    # check object class
    assert isinstance(kt, Ktest)

    ## check content
    # input data
    for group in np.unique(dummy_data[1]):
        np.testing.assert_array_equal(
            kt.data.data[group].numpy(),
            dummy_data[0][dummy_data[1] == group].to_numpy()
        )
    pd.testing.assert_series_equal(kt.metadata, dummy_data[1])

    # output
    assert isinstance(kt.kfda_statistic, pd.Series)
    assert isinstance(kt.kfda_pval_asymp, pd.Series)


@pytest.fixture
def output_file():
    """
    Dummy file to test saving/loading.

    Note: file will be removed during test teardown.
    """
    # add a random unique tag to file
    yield os.path.join(pytest.output_dir, f"ktest_obj_{secrets.token_hex(4)}.pkl")


def test_save_load(kt, output_file, assert_equal_ktest):
    """Test Ktest object saving and loading."""
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
    assert_equal_ktest(kt, kt_1)
    assert_equal_ktest(kt, kt_2)


@pytest.fixture(scope="module")
def exp_data():
    """Data and metadata from experimental transcriptomic dataset."""
    # data frame
    data = pd.read_csv(pytest.data_file, index_col=0)
    # metadata
    meta = pd.Series(data.index).apply(lambda x: x.split(sep='.')[1])
    meta.index = data.index
    # sample names
    sample_names = ['48HREV','48HDIFF']
    # output
    yield data, meta, sample_names


@pytest.fixture(scope="module")
def kt_data(exp_data):
    """Create Ktest object from dummy data for testing."""
    # init object
    kt = Ktest(
        data=exp_data[0], metadata=exp_data[1], sample_names=exp_data[2]
    )
    # run kfda test
    kt.test()
    # output
    yield kt


def test_num_stability(kt_data, assert_equal_ktest):
    """Compare numerical results to previous version of ktest (if available)."""

    # saving current results
    kt_data.save(pytest.res_file, compress=True)

    # load previous results (if available)
    if pytest.previous_res_file is not None:
        kt_data_prev = Ktest.load(pytest.previous_res_file, compressed=True)
        assert_equal_ktest(kt_data, kt_data_prev)
