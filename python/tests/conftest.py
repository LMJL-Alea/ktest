import os
import pytest
import shutil

## current test dir
test_dir = os.path.dirname(os.path.abspath(__file__))

## global setup
# output directory for tests
pytest.output_dir = os.path.join(test_dir, "results")
# enable/disable custom teardown and cleaning after tests
pytest.clean = True

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
