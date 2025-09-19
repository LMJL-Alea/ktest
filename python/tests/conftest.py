from glob import glob
import os
import pandas.testing as pdt
import pytest
import shutil
import toml
import torch.testing as tt
import warnings

## current test directory
test_dir = os.path.dirname(os.path.abspath(__file__))


## temporary output directory (cleaned after tests if required)
# output directory for tests
pytest.output_dir = os.path.join(test_dir, "results")
# enable/disable custom teardown and cleaning after tests
pytest.clean = True


## data for tests
# dataset
pytest.dataset = "RTqPCR_reversion_logcentered"
# data directory
pytest.data_dir = os.path.abspath(os.path.join(
    test_dir, os.pardir, os.pardir, "tutorials", "v5_data"
))
# data file
pytest.data_file = os.path.join(pytest.data_dir, f"{pytest.dataset}.csv")
assert os.path.isfile(pytest.data_file), "Missing data file for unit tests."
# test asset directory (for result file that are kept)
pytest.asset_dir = os.path.join(test_dir, "assets")
os.makedirs(pytest.asset_dir, exist_ok=True)


## current package version
pyproject_toml_file = os.path.join(test_dir, os.pardir, "pyproject.toml")
assert os.path.isfile(pyproject_toml_file), "Missing pyproject file."
pyproject_meta = toml.load(pyproject_toml_file)
assert "project" in pyproject_meta and "version" in pyproject_meta["project"], \
    "Missing 'version' field in 'pyproject.toml'."
pytest.pkg_version = pyproject_meta["project"]["version"]


## result file for data analysis
pytest.res_file = os.path.join(
    pytest.asset_dir, f"ktest_{pytest.pkg_version}_{pytest.dataset}.pkl.gz"
)


## result file for data analysis from previous package version (if existing)
# list previous result files
previous_res_files = glob(os.path.join(
    pytest.asset_dir, f"ktest_*_{pytest.dataset}.pkl.gz"
))
# remove result file for current package version (if existing)
try:
    previous_res_files.remove(pytest.res_file)
except ValueError:
    pass
# sort existing files
previous_res_files.sort()
# get file from previous package version to compare results
try:
    pytest.previous_res_file = previous_res_files[-1]
except IndexError:
    msg = "No existing data analysis result file " + \
        "(for comparing with previous package version run)."
    warnings.warn(msg)
    pytest.previous_res_file = None


## define function to compare ktest objects
@pytest.fixture
def assert_equal_ktest():
    """
    Function to compare two Ktest objects and assert that they are
    (almost) equal or not.

    See <https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close>
    for default tolerance value (when `atol=None`).

    Note: partial comparison to check data input and result output (test
    statistic and p-values).

    Parameters
    ----------
        kt_1 : Ktest object.
        kt_2 : Ktest object.
        atol : float,
            absolute tolerance for comparison.
    """
    def _assert_equal_ktest(kt_1, kt_2, atol=1e-5):
        # dataset
        pdt.assert_frame_equal(kt_1.dataset, kt_2.dataset)
        # data
        for (group1, array1), (group2, array2) in zip(
            kt_1.data.data.items(), kt_2.data.data.items()
        ):
            assert group1 == group2, \
                f"Unmatching samples '{group1}' and '{group2}'"
            tt.assert_close(array1, array2, rtol=0, atol=atol)
        # metadata
        pdt.assert_series_equal(kt_1.metadata, kt_2.metadata)
        # data nystrom (if relevant)
        if kt_1.data_nystrom is not None and kt_2.data_nystrom is not None:
            for (group1, array1), (group2, array2) in zip(
                kt_1.data_nystrom.data.items(), kt_2.data_nystroms.data.items()
            ):
                assert group1 == group2, \
                    f"Unmatching samples '{group1}' and '{group2}'"
                tt.assert_close(array1, array2, rtol=0, atol=atol)
        else:
            assert kt_1.data_nystrom == kt_2.data_nystrom
        # test statistics
        pdt.assert_series_equal(kt_1.kfda_statistic, kt_2.kfda_statistic)
        # asymptotic p-values
        pdt.assert_series_equal(kt_1.kfda_pval_asymp, kt_2.kfda_pval_asymp)
        pdt.assert_series_equal(kt_1.kfda_pval_asymp, kt_2.kfda_pval_asymp)

    yield _assert_equal_ktest


## custom test env preparation and teardown
@pytest.fixture(scope="session", autouse=True)
def run_before_and_after_test_session():
    """Custom test environment setup and teardown"""

    # Before Tests
    print(f"Creating test output directory: {pytest.output_dir}")
    os.makedirs(pytest.output_dir, exist_ok=True)
    assert os.path.isdir(pytest.output_dir)

    # Testing
    yield

    # After tests
    if pytest.clean:
        print(f"Deleting test output directory: {pytest.output_dir}")
        shutil.rmtree(pytest.output_dir)
    else:
        print(f"!!! Not deleting test output directory: {pytest.output_dir}")
