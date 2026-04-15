import numpy as np
import pytest
import torch as t
import types
from scipy.linalg import block_diag

from ktest.kernel_statistics import Statistics
from ktest.data import Data

from .test_data import data_shape, dummy_data, ktest_data, ktest_data_nystrom


def assert_eigenvectors(
    a: np.ndarray | t.Tensor, b: np.ndarray | t.Tensor,
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
        isinstance(a, t.Tensor) and isinstance(b, t.Tensor)

    # check dimension
    assert a.shape == b.shape

    # setup required functions
    einsum_fun = np.einsum if isinstance(a, np.ndarray) else t.einsum

    abs_fun = np.abs if isinstance(a, np.ndarray) else t.abs

    def assert_eigendecomp(x):
        """Check that a vector is equal to a vector of 1 in absolute value."""
        return \
            np.testing.assert_allclose(
                x, np.ones(x.shape, dtype=x.dtype), atol=atol, rtol=rtol
            ) if isinstance(a, np.ndarray) else \
            t.testing.assert_close(
                x, t.ones(x.shape, dtype=x.dtype), atol=atol, rtol=rtol
            )

    # compute column-wise scalar product
    col_scal_prod = einsum_fun('ij,ij->j', a, b)

    # check results (assert that all scalar product result are 1 or -1)
    assert_eigendecomp(abs_fun(col_scal_prod))


def test_assert_eigenvectors(dummy_data):
    """Test function to compare eigen vectors."""

    # get data
    data_tensor = t.from_numpy(dummy_data[0].values)

    # compute a symmetric matrix to decompose
    sym_matrix = t.matmul(data_tensor, data_tensor.T)

    # compute eigen decomposition with torch
    sp1, ev1 = t.linalg.eigh(sym_matrix)

    # compute eigen decomposition with numpy
    sp2, ev2 = np.linalg.eigh(sym_matrix.numpy())

    # check eigen values
    np.testing.assert_allclose(
        sp1.numpy(), sp2, rtol=0, atol=1e-11
    )

    # torch input
    assert_eigenvectors(
        ev1[:, sp1 >= 1e-12],
        t.from_numpy(ev2[:, sp2 >= 1e-12]),
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
        assert isinstance(sp, t.Tensor)
        assert list(sp.shape) < [sym_matrix.shape[0]]
        assert isinstance(ev, t.Tensor)
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
        assert isinstance(sp, t.Tensor)
        assert list(sp.shape) < [sym_matrix.shape[0]]
        assert isinstance(ev, t.Tensor)
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
            dtype=t.float64
        )

        # init data object without nystrom
        base_data = Data(
            data=data,
            metadata=metadata,
            sample_names=None,
            dtype=t.float64
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
        assert isinstance(res, t.Tensor)
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
        assert isinstance(res, t.Tensor)
        assert len(res.shape) == 2

        exp_dim = kstat_nystrom.data_ny.ntot
        assert list(res.shape) == [exp_dim, exp_dim]

        exp_res = _exp_cent_mat(kstat_nystrom.data_ny)

        np.testing.assert_allclose(res.numpy(), exp_res, rtol=0, atol=1e-11)

        # Nystrom: anchor_basis == 'k'
        kstat_nystrom.anchor_basis = 'k'
        res = kstat_nystrom.compute_centering_matrix(landmarks=True)
        assert isinstance(res, t.Tensor)
        assert len(res.shape) == 2

        exp_dim = kstat_nystrom.data_ny.ntot
        assert list(res.shape) == [exp_dim, exp_dim]

        exp_res = np.eye(exp_dim)

        np.testing.assert_allclose(res.numpy(), exp_res, rtol=0, atol=1e-11)

        # Nystrom: anchor_basis == 's'
        kstat_nystrom.anchor_basis = 's'
        res = kstat_nystrom.compute_centering_matrix(landmarks=True)
        assert isinstance(res, t.Tensor)
        assert len(res.shape) == 2

        exp_dim = kstat_nystrom.data_ny.ntot
        assert list(res.shape) == [exp_dim, exp_dim]

        exp_res = np.eye(exp_dim) - 1/exp_dim * np.ones((exp_dim, exp_dim))

        np.testing.assert_allclose(res.numpy(), exp_res, rtol=0, atol=1e-11)

    def test_compute_gram(self, kstat, kstat_nystrom, data_shape):
        # Case 1 – full data, no landmarks, no new_obs
        # default: landmarks=False, new_obs=None
        K = kstat.compute_gram()

        # Expected Gram matrix using the same gaussian kernel
        D = t.cat(tuple(kstat.data.data.values()), dim=0)
        expected = kstat.kernel(D, D)

        assert isinstance(K, t.Tensor)
        assert K.shape == (data_shape[0], data_shape[0])
        t.testing.assert_close(K, expected)

        # Case 2 – compute K(D, new_obs)
        # New observations: use one population subsample
        new_obs = list(kstat.data.data.values())[0]

        K = kstat.compute_gram(new_obs=new_obs)

        # Expected Gram matrix using the same gaussian kernel
        D = t.cat(tuple(kstat.data.data.values()), dim=0)
        expected = kstat.kernel(D, new_obs)

        assert isinstance(K, t.Tensor)
        assert K.shape == (data_shape[0], new_obs.shape[0])
        t.testing.assert_close(K, expected)

        # Case 3 – landmarks=True and a Nyström dataset is provided
        K = kstat_nystrom.compute_gram(landmarks=True)

        # Expected Gram matrix using the same gaussian kernel
        D = t.cat(tuple(kstat_nystrom.data_ny.data.values()), dim=0)
        expected = kstat_nystrom.kernel(D, D)

        assert isinstance(K, t.Tensor)
        assert K.shape == (data_shape[0]//5, data_shape[0]//5)
        t.testing.assert_close(K, expected)

        # Case 4 – landmarks=True but a Nyström dataset is not provided
        with pytest.raises(ValueError, match="Cannot use landmarks"):
            K = kstat.compute_gram(landmarks=True)

        # Case 5 – compute K(D, new_obs) with landmarks=True
        # New observations: use one population subsample
        new_obs = list(kstat.data.data.values())[0]

        K = kstat_nystrom.compute_gram(landmarks=True, new_obs=new_obs)

        # Expected Gram matrix using the same gaussian kernel
        D = t.cat(tuple(kstat_nystrom.data_ny.data.values()), dim=0)
        expected = kstat_nystrom.kernel(D, new_obs)

        assert isinstance(K, t.Tensor)
        assert K.shape == (data_shape[0]//5, new_obs.shape[0])
        t.testing.assert_close(K, expected)

    def test_compute_kmn(self, kstat, kstat_nystrom, data_shape):
        # Case 0 – not using Nystrom
        with pytest.raises(
            AttributeError, match="'NoneType' object has no attribute 'data'"
        ):
            Kmn = kstat.compute_kmn()

        # Case 1 – full data, no new_obs
        # default: new_obs=None
        Kmn = kstat_nystrom.compute_kmn()

        # Expected Gram matrix using the same gaussian kernel
        landmarks = t.cat(tuple(kstat_nystrom.data_ny.data.values()), dim=0)
        D = t.cat(tuple(kstat_nystrom.data.data.values()), dim=0)
        expected = kstat_nystrom.kernel(landmarks, D)

        assert isinstance(Kmn, t.Tensor)
        assert Kmn.shape == (landmarks.shape[0], data_shape[0])
        t.testing.assert_close(Kmn, expected)


        # Case 2 – compute K(landmarks, new_obs)
        # New observations: use one population subsample
        new_obs = list(kstat_nystrom.data.data.values())[0]

        Kmn = kstat_nystrom.compute_kmn(new_obs=new_obs)

        # Expected Gram matrix using the same gaussian kernel
        landmarks = t.cat(tuple(kstat_nystrom.data_ny.data.values()), dim=0)
        D = t.cat(tuple(kstat_nystrom.data.data.values()), dim=0)
        expected = kstat_nystrom.kernel(landmarks, new_obs)

        assert isinstance(Kmn, t.Tensor)
        assert Kmn.shape == (landmarks.shape[0], new_obs.shape[0])
        t.testing.assert_close(Kmn, expected)

    def test_compute_centered_gram(self, kstat, kstat_nystrom, data_shape):
        """
        Testing centered Gram matrix computation,
        with or without the computing trick to avoid storing the full
        n x n centering matrix."""

        # no Nystrom: no effect of 'low_mem_footprint' option
        res1 = kstat.compute_centered_gram(low_mem_footprint=False)
        res2 = kstat.compute_centered_gram(low_mem_footprint=True)

        assert isinstance(res1, t.Tensor)
        assert list(res1.shape) == [data_shape[0], data_shape[0]]

        t.testing.assert_close(res1, res2, rtol=0, atol=1e-12)

        # Nystrom: anchor_basis == 'w' (default mode)
        res1 = kstat_nystrom.compute_centered_gram(low_mem_footprint=False)
        res2 = kstat_nystrom.compute_centered_gram(low_mem_footprint=True)

        exp_dim = kstat_nystrom.sp_anchors.shape[0]

        assert isinstance(res1, t.Tensor)
        assert list(res1.shape) == [exp_dim, exp_dim]
        assert isinstance(res2, t.Tensor)
        assert list(res2.shape) == [exp_dim, exp_dim]

        t.testing.assert_close(res1, res2, rtol=0, atol=1e-12)

        # Nystrom: anchor_basis == 'k'
        kstat_nystrom.anchor_basis = 'k'
        res1 = kstat_nystrom.compute_centered_gram(low_mem_footprint=False)
        res2 = kstat_nystrom.compute_centered_gram(low_mem_footprint=True)

        exp_dim = kstat_nystrom.sp_anchors.shape[0]

        assert isinstance(res1, t.Tensor)
        assert list(res1.shape) == [exp_dim, exp_dim]
        assert isinstance(res2, t.Tensor)
        assert list(res2.shape) == [exp_dim, exp_dim]

        t.testing.assert_close(res1, res2, rtol=0, atol=1e-12)

        # Nystrom: anchor_basis == 's'
        kstat_nystrom.anchor_basis = 's'
        res1 = kstat_nystrom.compute_centered_gram(low_mem_footprint=False)
        res2 = kstat_nystrom.compute_centered_gram(low_mem_footprint=True)

        exp_dim = kstat_nystrom.sp_anchors.shape[0]

        assert isinstance(res1, t.Tensor)
        assert list(res1.shape) == [exp_dim, exp_dim]
        assert isinstance(res2, t.Tensor)
        assert list(res2.shape) == [exp_dim, exp_dim]

        t.testing.assert_close(res1, res2, rtol=0, atol=1e-12)

    def test_diagonalize_centered_gram(self, kstat, kstat_nystrom, data_shape):
        """
        Testing centered Gram matrix diagonalization,
        with or without the computing trick to avoid storing the full
        n x n centering matrix."""

        # no Nystrom: no effect of 'low_mem_footprint' option
        sp1, ev1 = kstat.diagonalize_centered_gram(low_mem_footprint=False)
        sp2, ev2 = kstat.diagonalize_centered_gram(low_mem_footprint=True)

        # check output
        assert isinstance(sp1, t.Tensor)
        assert list(sp1.shape) == [data_shape[0] - 2]
        assert isinstance(ev1, t.Tensor)
        assert list(ev1.shape) == [data_shape[0], data_shape[0] - 2]
        # note: last 2 eigen values are clipped

        t.testing.assert_close(sp1, sp2, rtol=0, atol=1e-12)
        t.testing.assert_close(ev1, ev2, rtol=0, atol=1e-12)

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
        assert isinstance(sp1, t.Tensor)
        assert list(sp1.shape) == [data_shape[0] // 5 - 2]
        assert isinstance(ev1, t.Tensor)
        assert list(ev1.shape) == \
            [data_shape[0] // 5 - 2, data_shape[0] // 5 - 2]
        # note: last 2 eigen values are clipped

        t.testing.assert_close(sp1, sp2, rtol=0, atol=1e-12)
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
