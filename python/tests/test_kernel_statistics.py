import numpy as np
import pandas as pd
import pytest
import torch as to
import types
from scipy.linalg import block_diag

from ktest.kernel_statistics import Statistics
from ktest.data import Data

from .test_data import (
    data_shape, dummy_data_array, dummy_data,
    ktest_data, ktest_data_nystrom,
    dummy_separated_data,
    ktest_separated_data, ktest_separated_data_nystrom
)


def assert_eigenvectors(
    a: np.ndarray | to.Tensor, b: np.ndarray | to.Tensor,
    rtol: float = 0, atol: float = 1e-11
):
    """
    Assert that two set of ordered eigen vectors (by increasing or decreasing
    eigen value order) are identical up to a rotation of angle pi, i.e.
    assert that for any column index j, we have `a[:, j] == b[:, j]` or
    `a[:, j] == -b[:, j]`, meaning that scalar product
    <a[:, j], b[:, j]> = 1 or -1.
    """

    # check type
    assert isinstance(a, np.ndarray) and isinstance(b, np.ndarray) or \
        isinstance(a, to.Tensor) and isinstance(b, to.Tensor)

    # check dimension
    assert a.shape == b.shape

    # setup required functions
    einsum_fun = np.einsum if isinstance(a, np.ndarray) else to.einsum

    abs_fun = np.abs if isinstance(a, np.ndarray) else to.abs

    def assert_eigendecomp(x):
        """Check that a vector is equal to a vector of 1 in absolute value."""
        return \
            np.testing.assert_allclose(
                x, np.ones(x.shape, dtype=x.dtype), atol=atol, rtol=rtol
            ) if isinstance(a, np.ndarray) else \
            to.testing.assert_close(
                x, to.ones(x.shape, dtype=x.dtype), atol=atol, rtol=rtol
            )

    # compute column-wise scalar product
    col_scal_prod = einsum_fun('ij,ij->j', a, b)

    # check results (assert that all scalar product result are 1 or -1)
    assert_eigendecomp(abs_fun(col_scal_prod))


def test_assert_eigenvectors(dummy_data):
    """Test function to compare eigen vectors."""

    # get data
    data_tensor = to.from_numpy(dummy_data[0].values)

    # compute a symmetric matrix to decompose
    sym_matrix = to.matmul(data_tensor, data_tensor.T)

    # compute eigen decomposition with torch
    sp1, ev1 = to.linalg.eigh(sym_matrix)

    # compute eigen decomposition with numpy
    sp2, ev2 = np.linalg.eigh(sym_matrix.numpy())

    # check eigen values
    np.testing.assert_allclose(
        sp1.numpy(), sp2, rtol=0, atol=1e-11
    )

    # torch input
    assert_eigenvectors(
        ev1[:, sp1 >= 1e-12],
        to.from_numpy(ev2[:, sp2 >= 1e-12]),
        rtol=0, atol=1e-11
    )

    # numpy input
    assert_eigenvectors(
        ev1[:, sp1 >= 1e-12].numpy(),
        ev2[:, sp2 >= 1e-12],
        rtol=0, atol=1e-11
    )



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


class TestStatistics(object):
    """Implement unit tests for Statistics class."""

    def test_ordered_eigsy(self, dummy_data):
        """Test eigen decomposition"""
        # get data
        data_tensor = to.from_numpy(dummy_data[0].values)

        # compute a symmetric matrix to decompose
        sym_matrix = to.matmul(data_tensor, data_tensor.T)

        # compute eigen decomposition (without clipping)
        sp, ev = Statistics.ordered_eigsy(sym_matrix, eps=None, clip=False)

        # check output
        assert isinstance(sp, to.Tensor)
        assert list(sp.shape) == [sym_matrix.shape[0]]
        assert isinstance(ev, to.Tensor)
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
        assert_eigenvectors(
            ev.numpy()[:, sp >= 1e-12],
            exp_ev[:, exp_sp >= 1e-12],
            rtol=0, atol=1e-12
        )

        # compute eigen decomposition
        # with clipping, no specified threshold
        with pytest.warns(UserWarning):
            sp, ev = Statistics.ordered_eigsy(sym_matrix, eps=None, clip=True)

        # check output
        assert isinstance(sp, to.Tensor)
        assert list(sp.shape) < [sym_matrix.shape[0]]
        assert isinstance(ev, to.Tensor)
        assert list(ev.shape) == [sym_matrix.shape[0], sp.shape[0]]

        # check eigen values
        np.testing.assert_allclose(
            sp.numpy(), exp_sp[:sp.shape[0]], rtol=0, atol=1e-11
        )

        # check eigen vectors for non null eigen values
        assert_eigenvectors(
            ev.numpy()[:, sp >= 1e-12],
            exp_ev[:, exp_sp >= 1e-12],
            rtol=0, atol=1e-12
        )

        # compute eigen decomposition
        # with clipping, specified threshold
        with pytest.warns(UserWarning):
            sp, ev = Statistics.ordered_eigsy(sym_matrix, eps=1e-12, clip=True)

        # check output
        assert isinstance(sp, to.Tensor)
        assert list(sp.shape) < [sym_matrix.shape[0]]
        assert isinstance(ev, to.Tensor)
        assert list(ev.shape) == [sym_matrix.shape[0], sp.shape[0]]

        # check eigen values
        np.testing.assert_allclose(
            sp.numpy(), exp_sp[:sp.shape[0]], rtol=0, atol=1e-11
        )

        # check eigen vectors for non null eigen values
        assert_eigenvectors(
            ev.numpy(),
            exp_ev[:, :sp.shape[0]],
            rtol=0, atol=1e-12
        )

    def test_init(self, kstat, kstat_nystrom, data_shape):
        """Testing Statictics object instantiation."""

        # no Nystrom
        assert isinstance(kstat.data, Data)
        assert kstat.data_ny is None
        assert kstat.dtype == to.float64
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
        assert isinstance(kstat.computed_bandwidth, to.Tensor) and \
            len(kstat.computed_bandwidth.shape) == 0 and \
            kstat.computed_bandwidth.dtype == to.float64
        assert kstat.sp is None
        assert kstat.ev is None
        assert kstat.sp_anchors is None
        assert kstat.ev_anchors is None

        # Nystrom
        assert isinstance(kstat_nystrom.data, Data)
        assert isinstance(kstat_nystrom.data_ny, Data)
        assert kstat_nystrom.dtype == to.float64
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
        assert isinstance(kstat_nystrom.computed_bandwidth, to.Tensor) and \
            len(kstat_nystrom.computed_bandwidth.shape) == 0 and \
            kstat_nystrom.computed_bandwidth.dtype == to.float64
        assert kstat_nystrom.sp is None
        assert kstat_nystrom.ev is None
        assert kstat_nystrom.sp_anchors is None
        assert kstat_nystrom.ev_anchors is None

    def test_init_nystrom_constant_var(self, dummy_data, data_shape):
        """Testing the case of data with a constant column."""

        # collect input data
        data, metadata = dummy_data

        # insert constant column in data
        data[data.columns[0]].values[:] = 0

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

        # init data object without nystrom
        base_data = Data(
            data=data,
            metadata=metadata,
            sample_names=None,
            dtype=to.float64
        )

        # kernel stat object
        kstat = Statistics(
            data=base_data,
            kernel_function='gauss',
            bandwidth='median',
            median_coef=1,
            data_nystrom=ny_data,
            n_anchors=None,
            anchor_basis='w',
            eps=None, clip_eigval=True
        )

        # try to compute kfda
        kstat.compute_kfda()

    def test_compute_centering_matrix(self, kstat, kstat_nystrom):
        """Testing centering matrix computation."""

        # compute expected centering matrix
        def _exp_cent_mat(data: Data):
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

            return block_diag(*mat_block_list)

        # no Nystrom
        res = kstat.compute_centering_matrix(landmarks=False)
        assert isinstance(res, to.Tensor)
        assert len(res.shape) == 2

        exp_dim = kstat.data.ntot
        assert list(res.shape) == [exp_dim, exp_dim]

        exp_res = _exp_cent_mat(kstat.data)

        np.testing.assert_allclose(res.numpy(), exp_res, rtol=0, atol=1e-11)

        # Nystrom (but not available)
        txt = "Cannot use landmarks, Nystrom approximation not provided."
        with pytest.raises(ValueError, match=txt):
            res = kstat.compute_centering_matrix(landmarks=True)

        # Nystrom: anchor_basis == 'w' (default mode)
        res = kstat_nystrom.compute_centering_matrix(landmarks=True)
        assert isinstance(res, to.Tensor)
        assert len(res.shape) == 2

        exp_dim = kstat_nystrom.data_ny.ntot
        assert list(res.shape) == [exp_dim, exp_dim]

        exp_res = _exp_cent_mat(kstat_nystrom.data_ny)

        np.testing.assert_allclose(res.numpy(), exp_res, rtol=0, atol=1e-11)

        # Nystrom: anchor_basis == 'k'
        kstat_nystrom.anchor_basis = 'k'
        res = kstat_nystrom.compute_centering_matrix(landmarks=True)
        assert isinstance(res, to.Tensor)
        assert len(res.shape) == 2

        exp_dim = kstat_nystrom.data_ny.ntot
        assert list(res.shape) == [exp_dim, exp_dim]

        exp_res = np.eye(exp_dim)

        np.testing.assert_allclose(res.numpy(), exp_res, rtol=0, atol=1e-11)

        # Nystrom: anchor_basis == 's'
        kstat_nystrom.anchor_basis = 's'
        res = kstat_nystrom.compute_centering_matrix(landmarks=True)
        assert isinstance(res, to.Tensor)
        assert len(res.shape) == 2

        exp_dim = kstat_nystrom.data_ny.ntot
        assert list(res.shape) == [exp_dim, exp_dim]

        exp_res = np.eye(exp_dim) - 1/exp_dim * np.ones((exp_dim, exp_dim))

        np.testing.assert_allclose(res.numpy(), exp_res, rtol=0, atol=1e-11)

    def test_compute_gram(self, kstat, kstat_nystrom, data_shape):
        """Testing Gram matrix computation."""
        # Case 1 - full data, no landmarks, no new_obs
        # default: landmarks=False, new_obs=None
        K = kstat.compute_gram()

        # Expected Gram matrix using the same gaussian kernel
        D = to.cat(tuple(kstat.data.data.values()), dim=0)
        expected = kstat.kernel(D, D)

        assert isinstance(K, to.Tensor)
        assert K.shape == (data_shape[0], data_shape[0])
        to.testing.assert_close(K, expected)

        # Case 2 - compute K(D, new_obs)
        # new observations: use one population subsample
        new_obs = list(kstat.data.data.values())[0]

        K = kstat.compute_gram(new_obs=new_obs)

        # Expected Gram matrix using the same gaussian kernel
        D = to.cat(tuple(kstat.data.data.values()), dim=0)
        expected = kstat.kernel(D, new_obs)

        assert isinstance(K, to.Tensor)
        assert K.shape == (data_shape[0], new_obs.shape[0])
        to.testing.assert_close(K, expected)

        # Case 3 - landmarks=True and a Nyström dataset is provided
        K = kstat_nystrom.compute_gram(landmarks=True)

        # Expected Gram matrix using the same gaussian kernel
        D = to.cat(tuple(kstat_nystrom.data_ny.data.values()), dim=0)
        expected = kstat_nystrom.kernel(D, D)

        assert isinstance(K, to.Tensor)
        assert K.shape == (data_shape[0]//5, data_shape[0]//5)
        to.testing.assert_close(K, expected)

        # Case 4 - landmarks=True but a Nyström dataset is not provided
        with pytest.raises(ValueError, match="Cannot use landmarks"):
            K = kstat.compute_gram(landmarks=True)

        # Case 5 - compute K(D, new_obs) with landmarks=True
        # new observations: use one population subsample
        new_obs = list(kstat.data.data.values())[0]

        K = kstat_nystrom.compute_gram(landmarks=True, new_obs=new_obs)

        # Expected Gram matrix using the same gaussian kernel
        D = to.cat(tuple(kstat_nystrom.data_ny.data.values()), dim=0)
        expected = kstat_nystrom.kernel(D, new_obs)

        assert isinstance(K, to.Tensor)
        assert K.shape == (data_shape[0]//5, new_obs.shape[0])
        to.testing.assert_close(K, expected)

    def test_compute_kmn(self, kstat, kstat_nystrom, data_shape):
        """Testing Gram matrix computation between landmarks and data."""
        # Case 0 - not using Nystrom
        with pytest.raises(
            AttributeError, match="'NoneType' object has no attribute 'data'"
        ):
            Kmn = kstat.compute_kmn()

        # Case 1 - full data, no new_obs
        # default: new_obs=None
        Kmn = kstat_nystrom.compute_kmn()

        # Expected Gram matrix using the same gaussian kernel
        landmarks = to.cat(tuple(kstat_nystrom.data_ny.data.values()), dim=0)
        D = to.cat(tuple(kstat_nystrom.data.data.values()), dim=0)
        expected = kstat_nystrom.kernel(landmarks, D)

        assert isinstance(Kmn, to.Tensor)
        assert Kmn.shape == (landmarks.shape[0], data_shape[0])
        to.testing.assert_close(Kmn, expected)

        # Case 2 - compute K(landmarks, new_obs)
        # new observations: use one population subsample
        new_obs = list(kstat_nystrom.data.data.values())[0]

        Kmn = kstat_nystrom.compute_kmn(new_obs=new_obs)

        # Expected Gram matrix using the same gaussian kernel
        landmarks = to.cat(tuple(kstat_nystrom.data_ny.data.values()), dim=0)
        D = to.cat(tuple(kstat_nystrom.data.data.values()), dim=0)
        expected = kstat_nystrom.kernel(landmarks, new_obs)

        assert isinstance(Kmn, to.Tensor)
        assert Kmn.shape == (landmarks.shape[0], new_obs.shape[0])
        to.testing.assert_close(Kmn, expected)

    def test_compute_centered_gram(self, kstat, kstat_nystrom, data_shape):
        """
        Testing centered Gram matrix computation,
        with or without the computing trick to avoid storing the full
        n x n centering matrix."""

        # no Nystrom: no effect of 'low_mem_footprint' option
        res1 = kstat.compute_centered_gram(low_mem_footprint=False)
        res2 = kstat.compute_centered_gram(low_mem_footprint=True)

        assert isinstance(res1, to.Tensor)
        assert list(res1.shape) == [data_shape[0], data_shape[0]]

        to.testing.assert_close(res1, res2, rtol=0, atol=1e-12)

        # Nystrom: anchor_basis == 'w' (default mode)
        res1 = kstat_nystrom.compute_centered_gram(low_mem_footprint=False)
        res2 = kstat_nystrom.compute_centered_gram(low_mem_footprint=True)

        exp_dim = kstat_nystrom.sp_anchors.shape[0]

        assert isinstance(res1, to.Tensor)
        assert list(res1.shape) == [exp_dim, exp_dim]
        assert isinstance(res2, to.Tensor)
        assert list(res2.shape) == [exp_dim, exp_dim]

        to.testing.assert_close(res1, res2, rtol=0, atol=1e-12)

        # Nystrom: anchor_basis == 'k'
        kstat_nystrom.anchor_basis = 'k'
        res1 = kstat_nystrom.compute_centered_gram(low_mem_footprint=False)
        res2 = kstat_nystrom.compute_centered_gram(low_mem_footprint=True)

        exp_dim = kstat_nystrom.sp_anchors.shape[0]

        assert isinstance(res1, to.Tensor)
        assert list(res1.shape) == [exp_dim, exp_dim]
        assert isinstance(res2, to.Tensor)
        assert list(res2.shape) == [exp_dim, exp_dim]

        to.testing.assert_close(res1, res2, rtol=0, atol=1e-12)

        # Nystrom: anchor_basis == 's'
        kstat_nystrom.anchor_basis = 's'
        res1 = kstat_nystrom.compute_centered_gram(low_mem_footprint=False)
        res2 = kstat_nystrom.compute_centered_gram(low_mem_footprint=True)

        exp_dim = kstat_nystrom.sp_anchors.shape[0]

        assert isinstance(res1, to.Tensor)
        assert list(res1.shape) == [exp_dim, exp_dim]
        assert isinstance(res2, to.Tensor)
        assert list(res2.shape) == [exp_dim, exp_dim]

        to.testing.assert_close(res1, res2, rtol=0, atol=1e-12)

    def test_diagonalize_centered_gram(self, kstat, kstat_nystrom, data_shape):
        """
        Testing centered Gram matrix diagonalization,
        with or without the computing trick to avoid storing the full
        n x n centering matrix.
        """

        # no Nystrom: no effect of 'low_mem_footprint' option
        sp1, ev1 = kstat.diagonalize_centered_gram(low_mem_footprint=False)
        sp2, ev2 = kstat.diagonalize_centered_gram(low_mem_footprint=True)

        # check output
        assert isinstance(sp1, to.Tensor)
        assert list(sp1.shape) == [data_shape[0] - 2]
        assert isinstance(ev1, to.Tensor)
        assert list(ev1.shape) == [data_shape[0], data_shape[0] - 2]
        # note: last 2 eigen values are clipped

        to.testing.assert_close(sp1, sp2, rtol=0, atol=1e-12)
        to.testing.assert_close(ev1, ev2, rtol=0, atol=1e-12)

        # compute eigen decomposition with numpy to compare
        # /!\ eigen values/vectors are in reverse order
        Kw = kstat.compute_centered_gram(low_mem_footprint=True)
        exp_sp, exp_ev = np.linalg.eigh(Kw.numpy())

        # revert order
        exp_sp = np.flip(exp_sp)
        exp_ev = np.flip(exp_ev, axis=1)

        # check eigen values
        np.testing.assert_allclose(
            sp1.numpy()[sp1 >= 1e-12], exp_sp[exp_sp >= 1e-12],
            rtol=0, atol=1e-11
        )

        # check eigen vectors for non null eigen values
        assert_eigenvectors(
            ev1.numpy()[:, sp1 >= 1e-12],
            exp_ev[:, exp_sp >= 1e-12],
            rtol=0, atol=1e-12
        )

        # Nystrom
        sp1, ev1 = kstat_nystrom.diagonalize_centered_gram(
            low_mem_footprint=False
        )
        sp2, ev2 = kstat_nystrom.diagonalize_centered_gram(
            low_mem_footprint=True
        )

        # check output
        assert isinstance(sp1, to.Tensor)
        assert list(sp1.shape) == [data_shape[0] // 5 - 2]
        assert isinstance(ev1, to.Tensor)
        assert list(ev1.shape) == \
            [data_shape[0] // 5 - 2, data_shape[0] // 5 - 2]
        # note: last 2 eigen values are clipped

        to.testing.assert_close(sp1, sp2, rtol=0, atol=1e-12)
        assert_eigenvectors(ev1, ev2, rtol=0, atol=1e-12)

        # compute eigen decomposition with numpy to compare
        # /!\ eigen values/vectors are in reverse order
        Kw = kstat_nystrom.compute_centered_gram(low_mem_footprint=True)
        exp_sp, exp_ev = np.linalg.eigh(Kw.numpy())

        # revert order
        exp_sp = np.flip(exp_sp)
        exp_ev = np.flip(exp_ev, axis=1)

        # check eigen values
        np.testing.assert_allclose(
            sp1.numpy()[sp1 >= 1e-12], exp_sp[exp_sp >= 1e-12],
            rtol=0, atol=1e-11
        )

        # check eigen vectors for non null eigen values
        assert_eigenvectors(
            ev1.numpy()[:, sp1 >= 1e-12],
            exp_ev[:, exp_sp >= 1e-12],
            rtol=0, atol=1e-12
        )

    def test_compute_upk(self, kstat, kstat_nystrom, data_shape):
        """Testing uPK product computation."""
        # FIXME: only result format is tested, not result values

        # Case 1 - full data (no Nystrom), no new_obs
        # default: new_obs=None
        upk = kstat.compute_upk(t=10)

        assert isinstance(upk, to.Tensor)
        assert upk.shape == (data_shape[0], 10)

        # Case 2 - providing new obs
        # new observations: use one population subsample
        new_obs = list(kstat_nystrom.data.data.values())[0]

        upk = kstat.compute_upk(t=10, new_obs=new_obs)

        assert isinstance(upk, to.Tensor)
        assert upk.shape == (new_obs.shape[0], 10)

        # Case 3 - Nystrom, no new_obs
        # default: new_obs=None
        upk = kstat_nystrom.compute_upk(t=10)

        assert isinstance(upk, to.Tensor)
        assert upk.shape == (data_shape[0], 10)

        # Case 4 - Nystrom, providing new obs
        # new observations: use one population subsample
        new_obs = list(kstat_nystrom.data.data.values())[0]

        upk = kstat_nystrom.compute_upk(t=10, new_obs=new_obs)

        assert isinstance(upk, to.Tensor)
        assert upk.shape == (new_obs.shape[0], 10)

    def test_compute_projections(self, kstat, kstat_nystrom):
        """Testing kFDA axis projection computation."""
        # FIXME: only result format is tested, not result values

        def _check_proj(proj_kfda, proj_kpca, n_obs_val, t_val):
            """Function to check projection results on the fly."""
            assert isinstance(proj_kfda, dict)
            assert len(proj_kfda) == len(n_obs_val)
            assert isinstance(proj_kpca, dict)
            assert len(proj_kpca) == len(n_obs_val)

            for proj_kfda_tab, proj_kpca_tab, n_obs in zip(
                proj_kfda.values(), proj_kpca.values(), n_obs_val
            ):
                assert isinstance(proj_kfda_tab, pd.DataFrame)
                assert proj_kfda_tab.shape == (n_obs, t_val)
                assert isinstance(proj_kpca_tab, pd.DataFrame)
                assert proj_kpca_tab.shape == (n_obs, t_val)

        # Case 1 - full data (no Nystrom), no centering, no new_obs
        # default: new_obs=None
        stat_val, _ = kstat.compute_kfda()
        proj_kfda, proj_kpca = kstat.compute_projections(
            stat=stat_val, t=10, center=False
        )
        n_obs_val = [tab.shape[0] for tab in kstat.data.data.values()]
        _check_proj(proj_kfda, proj_kpca, n_obs_val, t_val=10)

        # Case 2 - full data (no Nystrom), centering, no new_obs
        # default: new_obs=None
        stat_val, _ = kstat.compute_kfda()
        proj_kfda, proj_kpca = kstat.compute_projections(
            stat=stat_val, t=10, center=True
        )
        n_obs_val = [tab.shape[0] for tab in kstat.data.data.values()]
        _check_proj(proj_kfda, proj_kpca, n_obs_val, t_val=10)

        # Case 3 - full data (no Nystrom), no centering, providing new_obs
        # new observations: use one population subsample
        new_obs = list(kstat_nystrom.data.data.values())[0]
        stat_val, _ = kstat.compute_kfda()
        proj_kfda, proj_kpca = kstat.compute_projections(
            stat=stat_val, t=10, center=False, new_obs=new_obs
        )
        n_obs_val = [new_obs.shape[0]]
        _check_proj(proj_kfda, proj_kpca, n_obs_val, t_val=10)

        # Case 4 - full data (no Nystrom), centering, providing new_obs
        # new observations: use one population subsample
        new_obs = list(kstat_nystrom.data.data.values())[0]
        stat_val, _ = kstat.compute_kfda()
        proj_kfda, proj_kpca = kstat.compute_projections(
            stat=stat_val, t=10, center=True, new_obs=new_obs
        )
        n_obs_val = [new_obs.shape[0]]
        _check_proj(proj_kfda, proj_kpca, n_obs_val, t_val=10)

        # Case 5 - full data (no Nystrom), no centering, no new_obs
        # default: new_obs=None
        stat_val, _ = kstat_nystrom.compute_kfda()
        proj_kfda, proj_kpca = kstat_nystrom.compute_projections(
            stat=stat_val, t=10, center=False
        )
        n_obs_val = [tab.shape[0] for tab in kstat_nystrom.data.data.values()]
        _check_proj(proj_kfda, proj_kpca, n_obs_val, t_val=10)

        # Case 6 - full data (no Nystrom), centering, no new_obs
        # default: new_obs=None
        stat_val, _ = kstat_nystrom.compute_kfda()
        proj_kfda, proj_kpca = kstat_nystrom.compute_projections(
            stat=stat_val, t=10, center=True
        )
        n_obs_val = [tab.shape[0] for tab in kstat_nystrom.data.data.values()]
        _check_proj(proj_kfda, proj_kpca, n_obs_val, t_val=10)

        # Case 7 - full data (no Nystrom), no centering, providing new_obs
        # new observations: use one population subsample
        new_obs = list(kstat_nystrom.data.data.values())[0]
        stat_val, _ = kstat_nystrom.compute_kfda()
        proj_kfda, proj_kpca = kstat_nystrom.compute_projections(
            stat=stat_val, t=10, center=False, new_obs=new_obs
        )
        n_obs_val = [new_obs.shape[0]]
        _check_proj(proj_kfda, proj_kpca, n_obs_val, t_val=10)

        # Case 8 - full data (no Nystrom), centering, providing new_obs
        # new observations: use one population subsample
        new_obs = list(kstat_nystrom.data.data.values())[0]
        stat_val, _ = kstat_nystrom.compute_kfda()
        proj_kfda, proj_kpca = kstat_nystrom.compute_projections(
            stat=stat_val, t=10, center=True, new_obs=new_obs
        )
        n_obs_val = [new_obs.shape[0]]
        _check_proj(proj_kfda, proj_kpca, n_obs_val, t_val=10)

    def test_kfda_loss(
        self, kstat, kstat_nystrom, ktest_separated_data,
        ktest_separated_data_nystrom
    ):
        """Testing kFDA prediction computation."""

        # Case 1a - full data (no Nystrom), no new_obs
        # default: new_obs=None
        dist_g1, dist_g2 = kstat.kfda_loss(t=10)

        assert isinstance(dist_g1, dict)
        assert isinstance(dist_g2, dict)
        assert dist_g1.keys() == kstat.data.data.keys()
        assert dist_g2.keys() == kstat.data.data.keys()

        for group in kstat.data.data.keys():
            n_obs = kstat.data.data[group].shape[0]

            assert isinstance(dist_g1[group], to.Tensor)
            assert isinstance(dist_g2[group], to.Tensor)

            assert list(dist_g1[group].shape) == [n_obs, 10]
            assert list(dist_g2[group].shape) == [n_obs, 10]

            assert np.issubdtype(dist_g1[group].numpy().dtype, np.floating)
            assert np.issubdtype(dist_g2[group].numpy().dtype, np.floating)

        # Case 1b - full data (no Nystrom), no new_obs, providing stat value
        # default: new_obs=None
        stat_val, _ = kstat.compute_kfda()
        dist_g1, dist_g2 = kstat.kfda_loss(t=10, stat=stat_val)

        assert isinstance(dist_g1, dict)
        assert isinstance(dist_g2, dict)
        assert dist_g1.keys() == kstat.data.data.keys()
        assert dist_g2.keys() == kstat.data.data.keys()

        for group in kstat.data.data.keys():
            n_obs = kstat.data.data[group].shape[0]

            assert isinstance(dist_g1[group], to.Tensor)
            assert isinstance(dist_g2[group], to.Tensor)

            assert list(dist_g1[group].shape) == [n_obs, 10]
            assert list(dist_g2[group].shape) == [n_obs, 10]

            assert np.issubdtype(dist_g1[group].numpy().dtype, np.floating)
            assert np.issubdtype(dist_g2[group].numpy().dtype, np.floating)

        # Case 2 - full data (no Nystrom), providing new_obs
        # new observations: use one population subsample
        new_obs = list(kstat.data.data.values())[0]
        dist_g1, dist_g2 = kstat.kfda_loss(t=10, new_obs=new_obs)

        assert isinstance(dist_g1, dict)
        assert isinstance(dist_g2, dict)
        assert list(dist_g1.keys()) == ["new_obs"]
        assert list(dist_g2.keys()) == ["new_obs"]

        group = "new_obs"
        n_obs = new_obs.shape[0]

        assert isinstance(dist_g1[group], to.Tensor)
        assert isinstance(dist_g2[group], to.Tensor)

        assert list(dist_g1[group].shape) == [n_obs, 10]
        assert list(dist_g2[group].shape) == [n_obs, 10]

        assert np.issubdtype(dist_g1[group].numpy().dtype, np.floating)
        assert np.issubdtype(dist_g2[group].numpy().dtype, np.floating)

        # Case 3 - Nystrom approximation, no new_obs
        # default: new_obs=None
        dist_g1, dist_g2 = kstat_nystrom.kfda_loss(t=10)

        assert isinstance(dist_g1, dict)
        assert isinstance(dist_g2, dict)
        assert dist_g1.keys() == kstat_nystrom.data.data.keys()
        assert dist_g2.keys() == kstat_nystrom.data.data.keys()

        for group in kstat_nystrom.data.data.keys():
            n_obs = kstat_nystrom.data.data[group].shape[0]

            assert isinstance(dist_g1[group], to.Tensor)
            assert isinstance(dist_g2[group], to.Tensor)

            assert list(dist_g1[group].shape) == [n_obs, 10]
            assert list(dist_g2[group].shape) == [n_obs, 10]

            assert np.issubdtype(dist_g1[group].numpy().dtype, np.floating)
            assert np.issubdtype(dist_g2[group].numpy().dtype, np.floating)

        # Case 4 - Nystrom approximation, providing new_obs
        # new observations: use one population subsample
        new_obs = list(kstat.data.data.values())[0]
        dist_g1, dist_g2 = kstat_nystrom.kfda_loss(t=10, new_obs=new_obs)

        assert isinstance(dist_g1, dict)
        assert isinstance(dist_g2, dict)
        assert list(dist_g1.keys()) == ["new_obs"]
        assert list(dist_g2.keys()) == ["new_obs"]

        group = "new_obs"
        n_obs = new_obs.shape[0]

        assert isinstance(dist_g1[group], to.Tensor)
        assert isinstance(dist_g2[group], to.Tensor)

        assert list(dist_g1[group].shape) == [n_obs, 10]
        assert list(dist_g2[group].shape) == [n_obs, 10]

        assert np.issubdtype(dist_g1[group].numpy().dtype, np.floating)
        assert np.issubdtype(dist_g2[group].numpy().dtype, np.floating)

        # Case 5 - separated data (Nystrom approximation, no new_obs)
        # kernel stat object
        kstat_sep = Statistics(
            data=ktest_separated_data,
            kernel_function='gauss',
            bandwidth='median',
            median_coef=1,
            data_nystrom=ktest_separated_data_nystrom,
            n_anchors=None,
            anchor_basis='w',
            eps=None, clip_eigval=True
        )

        dist_g1, dist_g2 = kstat_sep.kfda_loss(t=10)

        assert isinstance(dist_g1, dict)
        assert isinstance(dist_g2, dict)
        assert dist_g1.keys() == kstat_sep.data.data.keys()
        assert dist_g2.keys() == kstat_sep.data.data.keys()

        for group in kstat_nystrom.data.data.keys():
            n_obs = kstat_sep.data.data[group].shape[0]

            assert isinstance(dist_g1[group], to.Tensor)
            assert isinstance(dist_g2[group], to.Tensor)

            assert list(dist_g1[group].shape) == [n_obs, 10]
            assert list(dist_g2[group].shape) == [n_obs, 10]

            assert np.issubdtype(dist_g1[group].numpy().dtype, np.floating)
            assert np.issubdtype(dist_g2[group].numpy().dtype, np.floating)

    def test_kfda_predict(
        self, kstat, kstat_nystrom, ktest_separated_data,
        ktest_separated_data_nystrom
    ):
        """Testing kFDA prediction computation."""

        # Case 1a - full data (no Nystrom), no new_obs
        # default: new_obs=None
        pred, loss = kstat.kfda_predict(t=10, pred_threshold=1/2)

        assert isinstance(pred, dict)
        assert isinstance(loss, dict)
        assert pred.keys() == kstat.data.data.keys()
        assert loss.keys() == kstat.data.data.keys()

        for group in kstat.data.data.keys():
            n_obs = kstat.data.data[group].shape[0]

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

            assert np.all(np.isin(pred_val, list(kstat.data.data.keys())))
            assert np.issubdtype(loss_val.dtype, np.floating)

            # group 1 and 2 are similar so we expect 50%-50% prediction
            count_pred = np.count_nonzero(pred_val == group, axis=0)
            np.testing.assert_allclose(count_pred / n_obs, 1/2, atol=0.1)

        # Case 1b - full data (no Nystrom), no new_obs,
        # with a list of threshold values
        # default: new_obs=None
        threshold_values = np.linspace(0, 1, 11)
        pred, loss = kstat.kfda_predict(
            t=10, pred_threshold=threshold_values
        )

        assert isinstance(pred, dict)
        assert isinstance(loss, dict)
        assert pred.keys() == kstat.data.data.keys()
        assert loss.keys() == kstat.data.data.keys()

        for group_ind, group in enumerate(kstat.data.data.keys()):
            n_obs = kstat.data.data[group].shape[0]

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

                assert np.all(np.isin(pred_val, list(kstat.data.data.keys())))
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
        new_obs = list(kstat.data.data.values())[0]
        pred, loss = kstat.kfda_predict(
            t=10, new_obs=new_obs, pred_threshold=1/2
        )

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

        assert np.all(np.isin(pred_val, list(kstat.data.data.keys())))
        assert np.issubdtype(loss_val.dtype, np.floating)

        # group 1 and 2 are similar so we expect 50%-50% prediction
        count_pred = np.count_nonzero(pred_val == "c1", axis=0)
        np.testing.assert_allclose(count_pred / n_obs, 1/2, atol=0.1)

        # Case 2b - full data (no Nystrom), providing new_obs,
        # biasing prediction (expect only group 2 prediction)
        # new observations: use one population subsample
        new_obs = list(kstat.data.data.values())[0]
        pred, loss = kstat.kfda_predict(
            t=10, new_obs=new_obs, pred_threshold=0
        )

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

        assert np.all(np.isin(pred_val, list(kstat.data.data.keys())))
        assert np.issubdtype(loss_val.dtype, np.floating)

        # we expect only "group 2" prediction
        count_pred = np.count_nonzero(pred_val == "c1", axis=0)
        np.testing.assert_allclose(count_pred / n_obs, 0, atol=0)

        # Case 2c - full data (no Nystrom), providing new_obs,
        # biasing prediction (expect only group 1 prediction)
        # new observations: use one population subsample
        new_obs = list(kstat.data.data.values())[0]
        pred, loss = kstat.kfda_predict(
            t=10, new_obs=new_obs, pred_threshold=1
        )

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

        assert np.all(np.isin(pred_val, list(kstat.data.data.keys())))
        assert np.issubdtype(loss_val.dtype, np.floating)

        # we expect only "group 1" prediction
        count_pred = np.count_nonzero(pred_val == "c1", axis=0)
        np.testing.assert_allclose(count_pred / n_obs, 1, atol=0)

        # Case 3 - Nystrom approximation, no new_obs
        # default: new_obs=None
        pred, loss = kstat_nystrom.kfda_predict(t=10, pred_threshold=1/2)

        assert isinstance(pred, dict)
        assert isinstance(loss, dict)
        assert pred.keys() == kstat_nystrom.data.data.keys()
        assert loss.keys() == kstat_nystrom.data.data.keys()

        for group in kstat.data.data.keys():
            n_obs = kstat_nystrom.data.data[group].shape[0]

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
                pred_val, list(kstat_nystrom.data.data.keys())
            ))
            assert np.issubdtype(loss_val.dtype, np.floating)

            # group 1 and 2 are similar so we expect 50%-50% prediction
            count_pred = np.count_nonzero(pred_val == group, axis=0)
            np.testing.assert_allclose(count_pred / n_obs, 1/2, atol=0.1)

        # Case 4 - Nystrom approximation, providing new_obs
        # new observations: use one population subsample
        new_obs = list(kstat_nystrom.data.data.values())[0]
        pred, loss = kstat_nystrom.kfda_predict(
            t=10, new_obs=new_obs, pred_threshold=1/2
        )

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

        assert np.all(np.isin(pred_val, list(kstat.data.data.keys())))
        assert np.issubdtype(loss_val.dtype, np.floating)

        # group 1 and 2 are similar so we expect 50%-50% prediction
        count_pred = np.count_nonzero(pred_val == "c1", axis=0)
        np.testing.assert_allclose(count_pred / n_obs, 1/2, atol=0.1)

        # Case 5 - separated data (Nystrom approximation, no new_obs)
        # kernel stat object
        kstat_sep = Statistics(
            data=ktest_separated_data,
            kernel_function='gauss',
            bandwidth='median',
            median_coef=1,
            data_nystrom=ktest_separated_data_nystrom,
            n_anchors=None,
            anchor_basis='w',
            eps=None, clip_eigval=True
        )

        # no bias in prediction, no new observations
        pred, loss = kstat_sep.kfda_predict(t=50, pred_threshold=1/2)

        assert isinstance(pred, dict)
        assert isinstance(loss, dict)
        assert pred.keys() == kstat_sep.data.data.keys()
        assert loss.keys() == kstat_sep.data.data.keys()

        for i, group in enumerate(kstat_sep.data.data.keys()):
            n_obs = kstat_sep.data.data[group].shape[0]

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
                pred_val, list(kstat_nystrom.data.data.keys())
            ))
            assert np.issubdtype(loss_val.dtype, np.floating)

            # group 1 and 2 are very different so we expect 100%
            # prediction on each group (at least for large truncations)
            count_pred = np.count_nonzero(
                pred_val[:, -10:] == group, axis=0
            )
            np.testing.assert_allclose(count_pred / n_obs, 1, atol=0.1)
