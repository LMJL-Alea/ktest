import numpy as np
import pandas as pd
import pytest
import torch as to

from ktest.data import Data


@pytest.fixture
def data_shape():
    """Number of rows and columns in dummy data for tests."""
    yield (1000, 100)
    # yield (50, 20)


@pytest.fixture
def dummy_data_array(data_shape):
    """
    Generate dummy data as numpy array for testing
    (two groups, no difference).
    """
    # generate random data (under H0, no separation)
    rng = np.random.default_rng(42)
    data_array = rng.normal(loc=0, scale=1, size=data_shape)

    # create meta data frame indicating two groups
    meta_array = np.array(
        [f"c{i+1}" for i in range(2)] * (data_shape[0] // 2)
    )

    # output
    yield data_array, meta_array


@pytest.fixture
def dummy_data(dummy_data_array, data_shape):
    """
    Generate dummy data as pandas array for testing
    (two groups, no difference).
    """
    # create a data frame from random gaussian data
    data = pd.DataFrame(
        data=dummy_data_array[0],
        columns=[f"col{i+1}" for i in range(data_shape[1])]
    )

    # create meta data frame indicating two groups
    meta = pd.Series(data=dummy_data_array[1])

    # output
    yield data, meta


@pytest.fixture
def dummy_separated_data(data_shape):
    """Generate dummy data for testing (two groups, significant difference)"""
    # generate random data with different mean in different the 2 groups
    # for 10 features, and only noise in other features
    rng = np.random.default_rng(42)
    loc = np.hstack([
        np.tile(
            np.array([-2, 2])[np.arange(data_shape[0]) % 2][:, np.newaxis],
            (1, 10)
        ) + rng.normal(loc=0, scale=1, size=[data_shape[0], 10]),
        np.zeros([data_shape[0], data_shape[1] - 10], dtype=np.float64)
    ])
    data_array = rng.normal(loc=loc, scale=1, size=data_shape)

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
def dummy_zidata(data_shape):
    """
    Generate dummy zero-inflated data for testing (two groups, no difference).
    """
    # generate random data (under H0, no separation)
    rng = np.random.default_rng(42)
    data_array = rng.normal(loc=0, scale=1, size=data_shape) * \
        rng.binomial(n=1, p=0.001, size=data_shape)

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
        dtype=to.float64
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
        dtype=to.float64
    )

    yield ny_data


@pytest.fixture
def ktest_separated_data(dummy_separated_data):
    """Convert pandas array to ktest `Data` type."""
    data = Data(
        data=dummy_separated_data[0],
        metadata=dummy_separated_data[1],
        sample_names=None,
        nystrom=False,
        dtype=to.float64
    )

    yield data


@pytest.fixture
def ktest_separated_data_nystrom(dummy_separated_data):
    """Convert pandas array to ktest `Data` type."""
    data = Data(
        data=dummy_separated_data[0],
        metadata=dummy_separated_data[1],
        sample_names=None,
        nystrom=True,
        n_landmarks=None,
        landmark_method='random',
        random_state=None,
        dtype=to.float64
    )

    yield data


def _check_ktest_data_object(
    data_obj, data_tab, metadata_tab, data_tab_shape, nystrom=False
):
    """
    Function to check a ktest data object with or without Nystrom
    approximation.
    """

    # check verbose level
    assert isinstance(data_obj.verbose, bool) or \
        isinstance(data_obj.verbose, int) and data_obj.verbose >= 0

    # check subsample names
    assert list(data_obj.sample_names) == ['c1', 'c2']

    # expected number of total number of observation
    exp_tot_nobs = data_tab_shape[0] if not nystrom else data_tab_shape[0] // 5

    # check total number of observations
    assert data_obj.ntot == exp_tot_nobs

    # check number of variables
    assert data_obj.nvar == data_tab_shape[1]

    # check data type
    assert data_obj.dtype == to.float64

    # check data attribute
    assert isinstance(data_obj.data, dict)

    # check variable index
    if isinstance(data_obj, pd.DataFrame):
        assert all(data_obj.variables == data_tab.columns)

    # check details about data, index and nobs attributes
    for i, subsample in enumerate(data_obj.sample_names):
        # expected number of observations in subsample
        exp_nobs = np.sum(metadata_tab == subsample) if not nystrom \
            else exp_tot_nobs // 2

        # check number of observations in subsample
        assert data_obj.nobs[subsample] == exp_nobs

        # check subsample index
        if isinstance(data_obj, pd.DataFrame):
            assert all(
                data_obj.index[subsample] ==
                data_tab.iloc[data_obj.index[subsample]].index
            )

        # check subsample data
        assert isinstance(data_obj.data[subsample], to.Tensor)
        assert list(data_obj.data[subsample].shape) == \
            [exp_nobs, data_tab_shape[1]]

        if isinstance(data_obj, pd.DataFrame):
            np.testing.assert_equal(
                data_obj.data[subsample].numpy(),
                data_tab.iloc[
                    data_obj.index[subsample]
                ].to_numpy()
            )
        elif isinstance(data_obj, np.ndarray):
            np.testing.assert_equal(
                data_obj.data[subsample].numpy(),
                data_tab[metadata_tab == subsample]
            )


class TestData:
    """Implement unit tests for Data class."""

    def test_init(self, ktest_data, dummy_data, data_shape):
        """Testing Data object instantiation."""
        _check_ktest_data_object(
            ktest_data, dummy_data[0], dummy_data[1], data_shape, nystrom=False
        )

    def test_init_nystrom(self, ktest_data_nystrom, dummy_data, data_shape):
        """Testing Data object instantiation when using Nystrom."""
        _check_ktest_data_object(
            ktest_data_nystrom, dummy_data[0], dummy_data[1], data_shape,
            nystrom=True
        )

    def test_init_zidata(self, dummy_zidata, data_shape):
        """
        Testing Data object instantiation in case of zero-inflated data.
        """

        # collect input data
        data, metadata = dummy_zidata

        # init data object without nystrom
        base_data = Data(
            data=data,
            metadata=metadata,
            sample_names=None,
            dtype=to.float64
        )

        _check_ktest_data_object(
            base_data, dummy_zidata[0], dummy_zidata[1], data_shape,
            nystrom=False
        )

        # init data object with nystrom
        ny_data = Data(
            data=data,
            metadata=metadata,
            sample_names=None,
            nystrom=True,
            n_landmarks=None,
            landmark_method='random',
            random_state=None,
            dtype=to.float64
        )

        _check_ktest_data_object(
            ny_data, dummy_zidata[0], dummy_zidata[1], data_shape, nystrom=True
        )

    def test_init_constant_var(self, dummy_data, data_shape):
        """
        Testing Data object instantiation in case of one or
        multiple constant columns in data.
        """

        # collect input data
        data, metadata = dummy_data

        # insert one constant column in data
        data[data.columns[0]].values[:] = 0

        # init data object without nystrom
        base_data = Data(
            data=data,
            metadata=metadata,
            sample_names=None,
            dtype=to.float64
        )

        _check_ktest_data_object(
            base_data, data, metadata, data_shape, nystrom=False
        )

        # init data object with nystrom
        ny_data = Data(
            data=data,
            metadata=metadata,
            sample_names=None,
            nystrom=True,
            n_landmarks=None,
            landmark_method='random',
            random_state=None,
            dtype=to.float64
        )

        _check_ktest_data_object(
            ny_data, data, metadata, data_shape, nystrom=True
        )

        # insert constant columns in data (except final one)
        for col in data.columns[:99]:
            data[col].values[:] = 0

        # init data object without nystrom
        base_data = Data(
            data=data,
            metadata=metadata,
            sample_names=None,
            dtype=to.float64
        )

        _check_ktest_data_object(
            base_data, data, metadata, data_shape, nystrom=False
        )

        # init data object with nystrom
        ny_data = Data(
            data=data,
            metadata=metadata,
            sample_names=None,
            nystrom=True,
            n_landmarks=None,
            landmark_method='random',
            random_state=None,
            dtype=to.float64
        )

        _check_ktest_data_object(
            ny_data, data, metadata, data_shape, nystrom=True
        )

        # insert constant columns in final column (in one group only)
        data[data.columns[99]].values[::2] = 0

        # init data object without nystrom
        base_data = Data(
            data=data,
            metadata=metadata,
            sample_names=None,
            dtype=to.float64
        )

        _check_ktest_data_object(
            base_data, data, metadata, data_shape, nystrom=False
        )

        # init data object with nystrom
        ny_data = Data(
            data=data,
            metadata=metadata,
            sample_names=None,
            nystrom=True,
            n_landmarks=None,
            landmark_method='random',
            random_state=None,
            dtype=to.float64
        )

        _check_ktest_data_object(
            ny_data, data, metadata, data_shape, nystrom=True
        )

        # insert constant columns in final column (in all groups)
        data[data.columns[99]].values[:] = 0

        # init data object without nystrom
        err_msg = "All variables have constant values in your data."
        with pytest.raises(RuntimeError) as excinfo:
            base_data = Data(
                data=data,
                metadata=metadata,
                sample_names=None,
                dtype=to.float64
            )
        assert str(excinfo.value) == err_msg

        # init data object without nystrom (no safe subsampling)
        try:
            base_data = Data(
                data=data,
                metadata=metadata,
                sample_names=None,
                dtype=to.float64,
                safe_subsample=False
            )
        except Exception as e:
            pytest.fail(f"Unexpected error: {e}")

        # insert constant columns in final column but one value (in all groups)
        data[data.columns[99]].values[0] = 1

        # init data object with nystrom
        err_msg = "Subsampling failed after 10 trials. " + \
            "All variables have constant values."
        # IMPORTANT: issue may not be raised if non null value in last
        # column is selected in Nystrom subsample (should be rare)
        with pytest.raises(RuntimeError) as excinfo:
            ny_data = Data(
                data=data,
                metadata=metadata,
                sample_names=None,
                nystrom=True,
                n_landmarks=None,
                landmark_method='random',
                random_state=42,
                dtype=to.float64,
                safe_subsample=True,
                n_subsample_trial=10
            )
        assert str(excinfo.value) == err_msg

        # init data object with nystrom (no safe subsampling)
        try:
            ny_data = Data(
                data=data,
                metadata=metadata,
                sample_names=None,
                nystrom=True,
                n_landmarks=None,
                landmark_method='random',
                random_state=None,
                dtype=to.float64,
                safe_subsample=False
            )
        except Exception as e:
            pytest.fail(f"Unexpected error: {e}")

    def test_init_numpy_array(self, dummy_data_array, data_shape):
        """Testing Data object instantiation with numpy array."""
        # init data object
        base_data = Data(
            data=dummy_data_array[0],
            metadata=dummy_data_array[1],
            sample_names=None,
            dtype=to.float64
        )
        _check_ktest_data_object(
            base_data, dummy_data_array[0], dummy_data_array[1], data_shape, nystrom=False
        )


