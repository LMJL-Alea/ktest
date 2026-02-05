import numpy as np
import pytest
import torch as t
import types
from scipy.linalg import block_diag

from ktest.kernel_statistics import Statistics
from ktest.data import Data

from .test_data import data_shape, dummy_data, ktest_data, ktest_data_nystrom


@pytest.fixture
def kstat(ktest_data):
    """Define a kernel statistics object without Nystrom approximation."""
    # kernel stat object
    kstat = Statistics(
        data=ktest_data,
        kernel_function='gauss',
        bandwidth='median',
        median_coef=1,
        data_nystrom=None,
        eps=None, clip_eigval=True
    )
    # output
    yield kstat


@pytest.fixture
def kstat_nystrom(ktest_data, ktest_data_nystrom):
    """Define a kernel statistics object using Nystrom approximation."""
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
    """Implement unit tests for Statistics class."""

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

    def test_init(self, kstat, kstat_nystrom, data_shape):
        """Testing Statictics object instantiation."""

        # no Nystrom
        assert isinstance(kstat.data, Data)
        assert kstat.data_ny is None
        assert kstat.dtype == t.float64
        assert kstat.eps is None or isinstance(kstat.eps, float)
        assert isinstance(kstat.clip_eigval, bool) and kstat.clip_eigval
        assert kstat.n_anchors is None
        assert isinstance(kstat.anchor_basis, str) and \
            kstat.anchor_basis == 'w'
        assert isinstance(kstat.kernel_function, str) and \
            kstat.kernel_function == 'gauss'
        assert isinstance(kstat.bandwidth, str) and kstat.bandwidth == 'median'
        assert isinstance(kstat.median_coef, int) and kstat.median_coef == 1
        assert isinstance(kstat.kernel, types.FunctionType)
        assert isinstance(kstat.computed_bandwidth, t.Tensor) and \
            len(kstat.computed_bandwidth.shape) == 0 and \
            kstat.computed_bandwidth.dtype == t.float64
        assert kstat.sp is None
        assert kstat.ev is None
        assert kstat.sp_anchors is None
        assert kstat.ev_anchors is None

        # Nystrom
        assert isinstance(kstat_nystrom.data, Data)
        assert isinstance(kstat_nystrom.data_ny, Data)
        assert kstat_nystrom.dtype == t.float64
        assert kstat_nystrom.eps is None or \
            isinstance(kstat_nystrom.eps, float)
        assert isinstance(kstat_nystrom.clip_eigval, bool) and \
            kstat_nystrom.clip_eigval
        assert isinstance(kstat_nystrom.n_anchors, int) and \
            kstat_nystrom.n_anchors == data_shape[0] // 5
        assert isinstance(kstat_nystrom.anchor_basis, str) and \
            kstat_nystrom.anchor_basis == 'w'
        assert isinstance(kstat_nystrom.kernel_function, str) and \
            kstat_nystrom.kernel_function == 'gauss'
        assert isinstance(kstat_nystrom.bandwidth, str) and \
            kstat_nystrom.bandwidth == 'median'
        assert isinstance(kstat_nystrom.median_coef, int) and \
            kstat_nystrom.median_coef == 1
        assert isinstance(kstat_nystrom.kernel, types.FunctionType)
        assert isinstance(kstat_nystrom.computed_bandwidth, t.Tensor) and \
            len(kstat_nystrom.computed_bandwidth.shape) == 0 and \
            kstat_nystrom.computed_bandwidth.dtype == t.float64
        assert kstat_nystrom.sp is None
        assert kstat_nystrom.ev is None
        assert kstat_nystrom.sp_anchors is None
        assert kstat_nystrom.ev_anchors is None

    def test_compute_centering_matrix(self, kstat, kstat_nystrom):
        """Testing centering matrix computation."""

        # compute expected centering matrix
        def _exp_cent_mat(data: Data, stacked: bool = False):
            """
            Compute expected centering matrix for a given data object.
            """

            mat_block_list = []

            for subdata in data.data.values():

                nobs_subsample = subdata.shape[0]

                subsample_block = np.eye(nobs_subsample) - \
                    1/nobs_subsample * \
                        np.ones((nobs_subsample, nobs_subsample))

                mat_block_list.append(subsample_block)

            if not stacked:
                return block_diag(*mat_block_list)
            else:
                return np.vstack(mat_block_list)

        # no Nystrom, standard block diagonal centering matrix
        res = kstat.compute_centering_matrix(landmarks=False, stacked=False)
        assert isinstance(res, t.Tensor)
        assert len(res.shape) == 2

        exp_res = _exp_cent_mat(kstat.data, stacked=False)

        np.testing.assert_allclose(res.numpy(), exp_res, rtol=0, atol=1e-11)

        # no Nystrom, stacked block centering matrix
        res = kstat.compute_centering_matrix(landmarks=False, stacked=True)
        assert isinstance(res, t.Tensor)
        assert len(res.shape) == 2

        exp_res = _exp_cent_mat(kstat.data, stacked=True)

        np.testing.assert_allclose(res.numpy(), exp_res, rtol=0, atol=1e-11)

        # Nystrom (but not available), standard block diagonal centering matrix
        txt = "Cannot use landmarks, Nystrom approximation not provided."
        with pytest.raises(ValueError, match=txt):
            res = kstat.compute_centering_matrix(landmarks=True, stacked=False)

        # Nystrom, standard block diagonal centering matrix
        res = kstat_nystrom.compute_centering_matrix(
            landmarks=True, stacked=False
        )
        assert isinstance(res, t.Tensor)
        assert len(res.shape) == 2

        exp_res = _exp_cent_mat(kstat_nystrom.data_ny, stacked=False)

        np.testing.assert_allclose(res.numpy(), exp_res, rtol=0, atol=1e-11)

        # Nystrom, stacked block centering matrix
        res = kstat_nystrom.compute_centering_matrix(
            landmarks=True, stacked=True
        )
        assert isinstance(res, t.Tensor)
        assert len(res.shape) == 2

        exp_res = _exp_cent_mat(kstat_nystrom.data_ny, stacked=True)

        np.testing.assert_allclose(res.numpy(), exp_res, rtol=0, atol=1e-11)
