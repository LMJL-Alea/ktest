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
        np.testing.assert_equal(
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


def test_save_load(kt, output_file):
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
    # input data
    for group in np.unique(kt.metadata):
        np.testing.assert_equal(
            kt.data.data[group].numpy(),
            kt_1.data.data[group].numpy()
        )
        np.testing.assert_equal(
            kt.data.data[group].numpy(),
            kt_2.data.data[group].numpy()
        )
    pd.testing.assert_series_equal(kt.metadata, kt_1.metadata)
    pd.testing.assert_series_equal(kt.metadata, kt_2.metadata)
    # test statistics
    pd.testing.assert_series_equal(kt.kfda_statistic, kt_1.kfda_statistic)
    pd.testing.assert_series_equal(kt.kfda_statistic, kt_2.kfda_statistic)
    # asymptotic p-values
    pd.testing.assert_series_equal(kt.kfda_pval_asymp, kt_1.kfda_pval_asymp)
    pd.testing.assert_series_equal(kt.kfda_pval_asymp, kt_2.kfda_pval_asymp)

