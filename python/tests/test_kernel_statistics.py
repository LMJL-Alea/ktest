import numpy as np
import pytest
import torch as t

from ktest.kernel_statistics import Statistics
from ktest.data import Data

from .test_data import data_shape, dummy_data, ktest_data, ktest_data_nystrom


@pytest.fixture
def kstat(ktest_data, ktest_data_nystrom):
    """Define a kernel statistics object."""
    # kernel stat object
    kstat = Statistics(
        data=ktest_data,
        kernel_function='gauss',
        bandwidth='median',
        median_coef=1,
        data_nystrom=ktest_data_nystrom,
        n_anchors=None,
        anchor_basis='w',
        eps=None, clip_eigval=True
    )
    # output
    yield kstat


class TestStatistics:
    """Implement test for Statistics class methods."""

    def test_ordered_eigsy(self, dummy_data):
        """Test eigen decomposition"""
        # get data
        data_tensor = t.from_numpy(dummy_data[0].values)

        # compute a symmetric matrix to decompose
        sym_matrix = t.matmul(data_tensor, data_tensor.T)

        # compute eigen decomposition (without clipping)
        sp, ev = Statistics.ordered_eigsy(sym_matrix, eps=None, clip=False)

        # check output
        assert isinstance(sp, t.Tensor)
        assert list(sp.shape) == [sym_matrix.shape[0]]
        assert isinstance(ev, t.Tensor)
        assert list(ev.shape) == [sym_matrix.shape[0], sym_matrix.shape[0]]

        # compute eigen decomposition with numpy to compare
        # /!\ eigen values/vectors are in reverse order
        exp_sp, exp_ev = np.linalg.eigh(sym_matrix.numpy())

        # revert order
        exp_sp = np.flip(exp_sp)
        exp_ev = np.flip(exp_ev, axis=1)

        # check eigen values
        np.testing.assert_allclose(
            sp.numpy(), exp_sp, rtol=0, atol=1e-11
        )

        # check eigen vectors for non null eigen values
        np.testing.assert_allclose(
            ev.numpy()[:, sp >= 1e-12],
            exp_ev[:, exp_sp >= 1e-12],
            rtol=0, atol=1e-12
        )

        # compute eigen decomposition
        # with clipping, no specified threshold
        with pytest.warns(UserWarning):
            sp, ev = Statistics.ordered_eigsy(sym_matrix, eps=None, clip=True)

        # check output
        assert isinstance(sp, t.Tensor)
        assert list(sp.shape) < [sym_matrix.shape[0]]
        assert isinstance(ev, t.Tensor)
        assert list(ev.shape) == [sym_matrix.shape[0], sp.shape[0]]

        # check eigen values
        np.testing.assert_allclose(
            sp.numpy(), exp_sp[:sp.shape[0]], rtol=0, atol=1e-11
        )

        # check eigen vectors for non null eigen values
        np.testing.assert_allclose(
            ev.numpy()[:, sp >= 1e-12],
            exp_ev[:, exp_sp >= 1e-12],
            rtol=0, atol=1e-12
        )

        # compute eigen decomposition
        # with clipping, specified threshold
        with pytest.warns(UserWarning):
            sp, ev = Statistics.ordered_eigsy(sym_matrix, eps=1e-12, clip=True)

        # check output
        assert isinstance(sp, t.Tensor)
        assert list(sp.shape) < [sym_matrix.shape[0]]
        assert isinstance(ev, t.Tensor)
        assert list(ev.shape) == [sym_matrix.shape[0], sp.shape[0]]

        # check eigen values
        np.testing.assert_allclose(
            sp.numpy(), exp_sp[:sp.shape[0]], rtol=0, atol=1e-11
        )

        # check eigen vectors for non null eigen values
        np.testing.assert_allclose(
            ev.numpy(),
            exp_ev[:, :sp.shape[0]],
            rtol=0, atol=1e-12
        )
