import numpy as np
import pandas as pd
import pytest
import torch as t

from ktest.data import Data


@pytest.fixture
def data_shape():
    """Number of rows and columns in dummy data for tests."""
    yield (1000, 100)


@pytest.fixture
def dummy_data(data_shape):
    """Generate dummy data for testing (two groups, no difference)"""
    # generate random data (under H0, no separation)
    rng = np.random.default_rng(42)
    data_array = rng.normal(loc=0, scale=1, size=data_shape)

    # create a data frame from random gaussian data
    data = pd.DataFrame(
        data=data_array,
        columns=[f"col{i+1}" for i in range(data_shape[1])]
    )

    # create meta data frame indicating two groups
    meta = pd.Series(
        data=[f"c{i+1}" for i in range(2)] * (data_shape[0] // 2)
    )

    # output
    yield data, meta


@pytest.fixture
def ktest_data(dummy_data):
    """Convert pandas array to ktest `Data` type."""
    data = Data(
        data=dummy_data[0],
        metadata=dummy_data[1],
        sample_names=None,
        dtype=t.float64
    )

    yield data


@pytest.fixture
def ktest_data_nystrom(dummy_data):
    """
    Convert pandas array to ktest `Data` type using Nystrom approximation.
    """
    ny_data = Data(
        data=dummy_data[0],
        metadata=dummy_data[1],
        sample_names=None,
        nystrom=True,
        n_landmarks=None,
        landmark_method='random',
        random_state=None,
        dtype=t.float64
    )

    yield ny_data


class TestData:
    """Implement unit tests for Data class."""

    def test_init(self, ktest_data, dummy_data, data_shape):
        """Testing Data object instantiation."""

        # check subsample names
        assert list(ktest_data.sample_names) == ['c1', 'c2']

        # check total number of observations
        assert ktest_data.ntot == data_shape[0]

        # check number of variables
        assert ktest_data.nvar == data_shape[1]

        # check data type
        assert ktest_data.dtype == t.float64

        # check data attribute
        assert isinstance(ktest_data.data, dict)

        # check variable index
        assert all(ktest_data.variables == dummy_data[0].columns)

        # check details about data, index and nobs attributes
        for subsample in ktest_data.sample_names:
            # expected number of observations in subsample
            exp_nobs = np.sum(dummy_data[1] == subsample)

            # check number of observations in subsample
            assert ktest_data.nobs[subsample] == exp_nobs

            # check subsample index
            assert all(
                ktest_data.index[subsample] ==
                dummy_data[0][dummy_data[1] == subsample].index
            )

            # check subsample data
            assert isinstance(ktest_data.data[subsample], t.Tensor)
            assert list(ktest_data.data[subsample].shape) == \
                [exp_nobs, data_shape[1]]
            np.testing.assert_equal(
                ktest_data.data[subsample].numpy(),
                dummy_data[0][dummy_data[1] == subsample].to_numpy()
            )

    def test_init_nystrom(self, ktest_data_nystrom, dummy_data, data_shape):
        """Testing Data object instantiation when using Nystrom."""

        # assert True

        # check subsample names
        assert list(ktest_data_nystrom.sample_names) == ['c1', 'c2']

        # expected Nystrom sampling dimension
        exp_nobs_ny = data_shape[0] // 5

        # check total number of observations
        assert ktest_data_nystrom.ntot == exp_nobs_ny

        # check number of variables
        assert ktest_data_nystrom.nvar == data_shape[1]

        # check data type
        assert ktest_data_nystrom.dtype == t.float64

        # check data attribute
        assert isinstance(ktest_data_nystrom.data, dict)

        # check variable index
        assert all(ktest_data_nystrom.variables == dummy_data[0].columns)

        # check details about data, index and nobs attributes
        for subsample in ktest_data_nystrom.sample_names:
            # expected number of observations in subsample
            exp_nobs = exp_nobs_ny // 2

            # check number of observations in subsample
            assert ktest_data_nystrom.nobs[subsample] == exp_nobs

            # check subsample index
            assert all(
                ktest_data_nystrom.index[subsample] ==
                dummy_data[0].iloc[ktest_data_nystrom.index[subsample]].index
            )

            # check subsample data
            assert isinstance(ktest_data_nystrom.data[subsample], t.Tensor)
            assert list(ktest_data_nystrom.data[subsample].shape) == \
                [exp_nobs, data_shape[1]]
            np.testing.assert_equal(
                ktest_data_nystrom.data[subsample].numpy(),
                dummy_data[0].iloc[
                    ktest_data_nystrom.index[subsample]
                ].to_numpy()
            )
