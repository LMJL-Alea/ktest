from collections.abc import Iterable
from numbers import Real
import warnings
import numpy as np
import pandas as pd
from torch import cdist, cat, matmul, exp, mv, dot, diag, sqrt
from torch import ones, eye, zeros, finfo
import torch as to
from torch.linalg import multi_dot, eigh

from .utils import pred_threshold_fun


def distances(x, y=None):
    """
    Computes the distances between each pair of the two collections of row
    vectors of x and y, or x and itself if y is not provided.

    """
    if y is None:
        sq_dists = cdist(
            x, x,
            compute_mode='use_mm_for_euclid_dist_if_necessary'
        ).pow(2)
        # [sq_dists]_ij=||X_j - X_i \\^2
    else:
        assert y.ndim == 2
        assert x.shape[1] == y.shape[1]
        sq_dists = cdist(
            x, y,
            compute_mode='use_mm_for_euclid_dist_if_necessary'
        ).pow(2)
        # [sq_dists]_ij=||x_j - y_i \\^2
    return sq_dists


def mediane(x, y=None):
    dxx = distances(x)
    if y is None:
        return dxx.median()
    dyy = distances(y)
    dxy = distances(x, y)
    dyx = dxy.t()
    dtot = cat((cat((dxx, dxy), dim=1), cat((dyx, dyy), dim=1)), dim=0)
    median = dtot.median()
    if median == 0:
        warnings.warn(
            'The median is null. To avoid a kernel with zero bandwidth, ' +
            'we replace the median by the mean'
        )
        mean = dtot.mean()
        if mean == 0:
            warnings.warn('warning: all your dataset is null')
        return mean
    else:
        return median


def linear_kernel(x, y):
    """
    Computes the standard linear kernel k(x,y)= <x,y>

    X - 2d array, samples on left hand side
    Y - 2d array, samples on right hand side, can be None in which case they
    are replaced by X

    returns: kernel matrix
    """
    K = matmul(x, y.T)
    return K


def gauss_kernel(x, y, sigma=1):
    """
    Computes the standard Gaussian kernel
        k(x,y)=exp(- ||x-y||**2 / (2 * sigma**2))

    X - 2d array, samples on left hand side
    Y - 2d array, samples on right hand side, can be None in which case they
    are replaced by X

    returns: kernel matrix
    """
    d = distances(x, y)   # [sq_dists]_ij=||X_j - Y_i \\^2
    K = exp(-d / (2 * sigma**2))  # Gram matrix
    return K


def gauss_kernel_median(
    x, y, bandwidth='median', median_coef=1, return_bandwidth=False
):
    """
    Returns the gaussian kernel with bandwidth set as the median of the
    distances between pairs of observations (`bandwidth`='median').
    """
    if bandwidth == 'median':
        computed_bandwidth = sqrt(mediane(x, y) * median_coef)
    else:
        computed_bandwidth = bandwidth

    def kernel(x, y):
        return gauss_kernel(x, y, computed_bandwidth)

    if return_bandwidth:
        return (kernel, computed_bandwidth)
    else:
        return kernel


class Statistics(object):
    """
    Class containing the technical tools for computing kernel statistics.

    Parameters
    ----------
    data : instance of class Data
        Contains various information on the original dataset, see the
        documentation of the class Data for more details.

    kernel_function : callable or str, optional
        Specifies the kernel function. Acceptable values in the form of a
        string are 'gauss' (default) and 'linear'. Pass a callable for a
        user-defined kernel function.

    bandwidth : 'median' or float, optional
        Value of the bandwidth for kernels using a bandwidth. If 'median'
        (default), the bandwidth will be set as the median or its multiple,
        depending on the value of the parameter `median_coef`. Pass a float
        for a user-defined value of the bandwidth.

    median_coef : float, optional
        Multiple of the median to compute bandwidth if bandwidth=='median'.
        The default is 1.

    data_nystrom : None or instance of class Data
        Contains various information on the Nystrom dataset, see the
        documentation of the class Data for more details. If None, Nystrom is
        not taken into account in the computations.

    n_anchors : int, optional
        Number of anchors used in the Nystrom method, by default equal to
        the number of landamarks.

    anchor_basis : str, optional
        Options for different ways of computing the covariance operator of
        the landmarks in the Nystrom method, of which the anchors are the
        eigenvalues. Possible values are 'w' (default),'s' and 'k'.

    eps : float, optional
        minimum threshold value to clip lower eigen values to zeros.
        If `None` (default), then machine precision (given by
        `torch.finfo()`) for specified dtype is used as threshold.

    clip_eigval : boolean,
        flag to enable/disable eigen value clipping.

    Attributes
    ----------
    kernel: callable
        Kernel function used for calculations.

    computed_bandwidth : float
        The value of the kernel bandwidth.

    sp : 1-dimensional torch tensor
        Eigenvalues ('sp' for spectrum) associated with the diagonalization of
        the centered within covariance operator of the original data using
        the kernel trick.

    ev : 2-dimensional torch tensor
        Eigenvectors associated with the diagonalization of the centered within
        covariance operator of the original data using the kernel trick.

    sp_anchors : 1-dimensional torch tensor
        Eigenvalues ('sp' for spectrum) associated with the diagonalization of
        the centered within covariance operator of the Nystrom landmarks using
        the kernel trick.

    ev_anchors : 2-dimensional torch tensor
        Eigenvectors associated with the diagonalization of the centered within
        covariance operator of the Nystrom landmarks using the kernel trick.

    dtype : torch.dtype, optional
        Floating point number type/precision used for number storage and
        computations given by data.
    """

    def __init__(
        self, data, kernel_function='gauss', bandwidth='median',
        median_coef=1, data_nystrom=None, n_anchors=None, anchor_basis='w',
        eps=None, clip_eigval=True
    ):

        # data
        self.data = data

        # dtype
        self.dtype = data.dtype

        # epsilon for eigen value clipping
        self.eps = eps

        # clipping eigen values?
        self.clip_eigval = clip_eigval

        ### Nystrom:
        self.data_ny = data_nystrom
        if self.data_ny is None or n_anchors is not None:
            self.n_anchors = n_anchors
        else:
            self.n_anchors = self.data_ny.ntot
        self.anchor_basis = anchor_basis
        assert self.anchor_basis in ['w', 's', 'k'], 'invalid anchor basis'

        ### Kernel:
        self.kernel_function = kernel_function
        self.bandwidth = bandwidth
        self.median_coef = median_coef

        if self.kernel_function == 'gauss':
            if self.data_ny is not None:
                x = self.data_ny.data[self.data.sample_names[0]]
                y = self.data_ny.data[self.data.sample_names[1]]
            else:
                x = self.data.data[self.data.sample_names[0]]
                y = self.data.data[self.data.sample_names[1]]

            self.kernel, self.computed_bandwidth = \
                gauss_kernel_median(
                    x,
                    y,
                    bandwidth=bandwidth,
                    median_coef=median_coef,
                    return_bandwidth=True
                )
        elif self.kernel_function == 'linear':
            self.kernel = linear_kernel
        else:
            self.kernel = self.kernel_function

        ### Spectrum and eigenvectors:
        self.sp = None
        self.ev = None
        self.sp_anchors = None
        self.ev_anchors = None

    @staticmethod
    def ordered_eigsy(matrix, eps=None, clip=True):
        """
        Compute eigen values and eigen vectors of input matrix stored by
        decreasing eigen value order.

        Note 1: only positive non-null eigen values and corresponding eigen
        vectors are returned. Eigen values lower than `eps` threshold a
        clipped to zeros.

        Note 2: input matrix is assumed to be symmetrical and positive
        semi-definite, such that its eigen values are positive or null. It
        may happen due to numerical issues that the eigen decomposition finds
        negative eigen values for such matrix anyway. These are also clipped
        to 0.

        Parameters
        ----------
            matrix : 2-D array torch.tensor,
                symmetrical matrix to get eigen decomposition.
            eps : float, optional
                minimum threshold value to clip lower eigen values to zeros.
                If `None` (default), then machine precision (given by
                `torch.finfo()`) for matrix dtype is used as threshold.
            clip : boolean,
                flag to enable/disable eigen value clipping.

        Returns
        -------
            sp : 1-D array torch.tensor,
                vector of decreasing strictly positive eigen values.
            ev : 2-D array torch.tensor,
                corresponding eigen vectors (following the same order).
        """
        # eigen values, eigen vectors
        # following ascending eigen value order (no need for sorting)
        sp, ev = eigh(matrix)
        # revert to get decreasing order for eigen values
        sp = sp.flip(dims=(0,))
        ev = ev.flip(dims=(1,))
        # clipping?
        if clip:
            # eigen value clipping threshold
            if eps is None:
                eps = finfo(matrix.dtype).eps
            # select non null eigen val
            sp_mask = sp >= eps
            # reduc dimension
            reduc_dim = sp_mask.sum()
            # warning
            if reduc_dim < len(sp):
                warnings.warn(
                    f"Clipping last {len(sp) - reduc_dim} eigen values " +
                    f"out of {len(sp)} dimensions, " +
                    f"that are lower than threshold {eps}."
                )

            # output
            return sp[sp_mask], ev[:, sp_mask]
        else:
            # no clipping
            # output
            return sp, ev

    def compute_centering_matrix(self, landmarks=False):
        """
        Computes a projection matrix usefull for the kernel trick.

        Example for the within-group covariance :
            Let I1,I2 the identity matrix of size n1 and n2 (or nxanchors and
             nyanchors if nystrom). J1,J2 the squared matrix full of ones of
            size n1 and n2 (or nxanchors and nyanchors if nystrom).
            012, 021 the matrix full of zeros of size n1 x n2 and n2 x n1
            (or nxanchors x nyanchors and nyanchors x nxanchors if nystrom).

        Pn = [I1 - 1/n1 J1 ,    012     ]
             [     021     ,I2 - 1/n2 J2]

        Parameters
        ----------
            landmarks : bool, optional
                False by default. If True, performs the computations on the
                the Nystrom dataset (landmarks).

        Returns
        -------
            P : torch.tensor,
                The centering matrix.

        """
        if landmarks and self.data_ny is None:
            raise ValueError(
                "Cannot use landmarks, Nystrom approximation not provided."
            )
        data = self.data if not landmarks else self.data_ny
        if not landmarks or self.anchor_basis == 'w':

            In = eye(data.ntot)
            effectifs = list(data.nobs.values())

            cumul_effectifs = np.cumsum([0]+effectifs)

            # Computing a bloc diagonal matrix where the ith diagonal bloc
            # is J_ni, an (ni x ni) matrix full of 1/ni where ni is the
            # size of the ith group
            diag_Jn_by_n = cat([
                cat([
                    zeros(nprec, nell, dtype=self.dtype),
                    1/nell*ones(nell, nell, dtype=self.dtype),
                    zeros(data.ntot - nprec - nell, nell, dtype=self.dtype)
                ], dim=0)
                for nell, nprec in zip(effectifs, cumul_effectifs)
            ], dim=1)
            return In - diag_Jn_by_n
        elif self.anchor_basis == 'k':
            return eye(data.ntot, dtype=self.dtype)
        elif self.anchor_basis == 's':
            In = eye(data.ntot, dtype=self.dtype)
            Jn = ones(data.ntot, data.ntot, dtype=self.dtype)
            return In - 1/data.ntot * Jn

    def compute_omega(self):
        """
        Returns the weights vector used to compute the mean.

        Returns
        -------
            omega : torch.tensor
                a vector of size corresponding to the group of which we
                compute the mean.

        """
        n1, n2 = self.data.nobs.values()
        m_mu1 = -1/n1 * ones(n1, dtype=self.dtype)
        m_mu2 = 1/n2 * ones(n2, dtype=self.dtype)
        return cat((m_mu1, m_mu2), dim=0)

    def compute_gram(self, landmarks=False, new_obs=None):
        """
        Computes the Gram matrix of the data in question.

        The kernel used is the kernel stored in the attribute 'kernel'.

        The Gram matrix is not stored in memory because it is usually large
        and fast to compute.

        Parameters
        ----------
            landmarks : bool, optional
                False by default. If True, performs the computations on the
                the Nystrom dataset (landmarks).
            new_obs : torch.tensor, optional
                Unused by default. If not None, then the Gram matrix between
                data and `new_obs` is computed.

        Returns
        -------
            K : torch.Tensor,
                Gram matrix of interest.
        """
        if landmarks and self.data_ny is None:
            raise ValueError(
                "Cannot use landmarks, Nystrom approximation not provided."
            )
        data = self.data if not landmarks else self.data_ny
        D = cat([x for x in data.data.values()], axis=0)
        if new_obs is not None:
            K = self.kernel(D, new_obs)
        else:
            K = self.kernel(D, D)
        return K

    def compute_kmn(self, new_obs=None):
        """
        Computes an (nxlandmarks+nylandmarks)x(ndata) conversion gram matrix.

        Parameters
        ----------
            new_obs : torch.tensor, optional
                Unused by default. If not None, then the Gram matrix between
                landmarks and `new_obs` is computed.

        """
        if new_obs is not None:
            data = new_obs
        else:
            data = cat([x for x in self.data.data.values()], axis=0)
        landmarks = cat([x for x in self.data_ny.data.values()], axis=0)
        kmn = self.kernel(landmarks, data)
        return kmn

    def compute_nystrom_anchors(self):
        K = self.compute_gram(landmarks=True)
        P = self.compute_centering_matrix(landmarks=True)
        Kw = 1 / self.data_ny.ntot * multi_dot([P, K, P])
        return Statistics.ordered_eigsy(Kw, self.eps, self.clip_eigval)

    def compute_centered_gram(self, low_mem_footprint=False):
        """
        Compute the bicentered Gram matrix which shares its spectrum
        with the within covariance operator in the RKHS.

        Parameters
        ----------
            low_mem_footprint : bool, optional
                True by default. If True, a little trick is used to perform
                the computations without storing the full n x n
                centering matrix in the Nystrom case.
                If False, the default computations requiring the full
                n x n centering matrix is used in the Nystrom
                computation.
                Without effect when not using Nystrom.

        Returns
        -------
            Kw : bicentered Gram matrix.

        """

        # Nytrom version:
        if self.data_ny is not None:
            # Computing Nystrom anchors:
            self.sp_anchors, self.ev_anchors = self.compute_nystrom_anchors()
            assert sum(self.sp_anchors > 0) != 0, \
                'No anchors found, the dataset may have two many zeros.'
            if sum(self.sp_anchors > 0) < self.n_anchors:
                warning_message = '\tThe number of anchors is reduced from '
                warning_message += \
                    f'{self.n_anchors} to {sum(self.sp_anchors > 0)} ' + \
                    'for numerical stability'
                warnings.warn(warning_message)
                self.n_anchors = sum(self.sp_anchors > 0).item()
            self.sp_anchors, self.ev_anchors = (
                self.sp_anchors[: self.n_anchors],
                self.ev_anchors[:, : self.n_anchors]
            )

            # Calculating the centering matrix for the Nystrom approximation:
            Kmn = self.compute_kmn()
            Lp_inv_12 = diag(self.sp_anchors ** (-1/2))
            Pm = self.compute_centering_matrix(landmarks=True)

            # Calculating the matrix to diagonalise with Nystrom:
            if low_mem_footprint and self.anchor_basis == 'w':
                # trick to avoid storing a n x n centering matrix
                sample_size = list(self.data.nobs.values())

                # computing centered gram
                Kw = (
                    1 / (self.data.ntot * self.data_ny.ntot) *
                    multi_dot([
                        Lp_inv_12, self.ev_anchors.T, Pm,
                        multi_dot([
                            Kmn[:, :sample_size[0]],
                            Kmn[:, :sample_size[0]].T - 1/sample_size[0] *
                            to.sum(Kmn[:, :sample_size[0]], dim=1)
                            # use pytorch broadcasting
                        ]) +
                        multi_dot([
                            Kmn[:, -sample_size[1]:],
                            Kmn[:, -sample_size[1]:].T - 1/sample_size[1] *
                            to.sum(Kmn[:, -sample_size[1]:], dim=1)
                            # use pytorch broadcasting
                        ]),
                        Pm,
                        self.ev_anchors,
                        Lp_inv_12
                    ])
                )

            else:
                # original version
                # computing centering matrix
                P = self.compute_centering_matrix()
                # computing centered gram
                Kw = (
                    1 / (self.data.ntot * self.data_ny.ntot) *
                    multi_dot([
                        Lp_inv_12, self.ev_anchors.T, Pm, Kmn, P,
                        Kmn.T, Pm, self.ev_anchors, Lp_inv_12
                    ])
                )

        # Standard version (no Nystrom):
        else:
            # centering matrix
            P = self.compute_centering_matrix()
            # gram matrix
            K = self.compute_gram()
            # centered gram
            Kw = 1 / self.data.ntot * multi_dot([P, K, P])

        # output
        return Kw

    def diagonalize_centered_gram(self, low_mem_footprint=True):
        """
        Diagonalizes the bicentered Gram matrix which shares its spectrum
        with the within covariance operator in the RKHS.

        Parameters
        ----------
            low_mem_footprint : bool, optional
                True by default. If True, a little trick is used to perform
                the computations without storing the full n x n matrix
                centering matrix in the Nystrom case.
                If False, the default computations requiring the full
                n x n matrix centering matrix is used in the in Nystrom
                computation.
                Without effect when not using Nystrom.

        Returns
        -------
            Eigenvalues and eigenvectors, later stored in attributes 'sp' and
            'ev' respectively.

        """
        # Compute the bicentered Gram matrix
        Kw = self.compute_centered_gram(low_mem_footprint)

        # Diagonalisation with a function in C++:
        return Statistics.ordered_eigsy(Kw, self.eps, self.clip_eigval)

    def compute_pkm(self):
        """
        Computes the term corresponding to the matrix-matrix-vector
        product PK omega of the kFDA statistic.

        See the description of the method compute_kfdat() for a brief
        description of the computation of the KFDA statistic.

        Returns
        -------
        pkm : torch.tensor
            Corresponds to the product PK omega in the kFDA statistic.

        """
        # Calculating the bi-centering vector Omega
        omega = self.compute_omega()  # vector with 1/n1 and -1/n2
        if self.data_ny is not None:
            Lz12 = diag(self.sp_anchors**(-1/2))
            Kzx = self.compute_kmn()
            Pi = self.compute_centering_matrix(landmarks=True)
            pkm = (1 / np.sqrt(self.data_ny.ntot)
                   * mv(Lz12, mv(self.ev_anchors.T, mv(Pi, mv(Kzx, omega)))))
        else:
            Pbi = self.compute_centering_matrix()  # Centering by block matrix
            Kx = self.compute_gram()
            pkm = mv(Pbi, mv(Kx, omega))
        return pkm

    def compute_kfda(self):
        """
        Computes the kFDA truncated statistic of [Harchaoui 2009].

        Description
        -----------
        Here is a brief description of the computation of the statistic, for
        more details, refer to the article.

        Let k(·,·) denote the kernel function, K denote the Gram matrix of the
        two samples, and kx the vector of embeddings of the observations
        x1,...,xn1,y1,...,yn2:

                kx = (k(x1,·), ... k(xn1,·),k(y1,·),...,k(yn2,·))

        Let Sw denote the within covariance operator, and P denote the
        centering matrix such that

                Sw = 1/n (kx P)(kx P)^T

        Let Kw = 1/n (kx P)^T(kx P) denote the dual matrix of Sw, and (li) (ui)
        denote its eigenvalues (shared with Sw) and eigenvectors. We have:

                ui = 1/(lp * n)^{1/2} kx P up

        Let Swt denote the spectral truncation of Sw with t directions
        such that

                Swt = l1 (e1 (x) e1) + l2 (e2 (x) e2) + ... + lt (et (x) et)
                    = \\sum_{p=1:t} lp (ep (x) ep)

        where (li) and (ei) are the first t eigenvalues and eigenvectors of
        Sw ordered by decreasing eigenvalues, and (x) stands for the tensor
        product.

        Let d = mu2 - mu1 denote the difference of the two kernel mean
        embeddings of the two samples of sizes n1 and n2 (with n = n1 + n2)
        and omega the weights vector such that

                d = kx * omega

        The standard truncated KFDA statistic is given by :

                F   = n1*n2/n || Swt^{-1/2} d ||_H^2
                    = \\sum_{p=1:t} n1*n2 / ( lp*n) <ep,d>^2
                    = \\sum_{p=1:t} n1*n2 / ( lp*n)^2 up^T PK omega


        Projection
        ----------

        This statistic also defines a discriminant axis ht in the RKHS H.

            ht  = n1*n2/n Swt^{-1/2} d
                = \\sum_{p=1:t} n1*n2 / ( lp*n)^2 [up^T PK omega] kx P up

        To project the dataset on this discriminant axis, we compute :

            h^T kx =  \\sum_{p=1:t} n1*n2 / ( lp*n)^2 [up^T PK omega] up^T P K

        Returns
        -------
            kfdat : Pandas.Series
                Computed kFDA statistics. Indices correspond to truncations.

            kfdat_contrib : Pandas.Series
                Unidirectional statistic associated with each eigendirection
                of the within-group covariance operator. `kfdat` contains
                the cumulated sum of the values in `kfdat_contributions`.
                Indices correspond to truncations.

        """
        # Calculating eigenvalues and eigenvectors:
        self.sp, self.ev = self.diagonalize_centered_gram()

        # Maximal truncation identification:
        t = len(self.sp)

        # Calculating statistic for every truncation:
        pkm = self.compute_pkm()
        n1, n2 = self.data.nobs.values()
        exposant = 1 if self.data_ny is not None else 2

        # Calculating contributions of every truncation:
        kfda_contributions = ((n1 * n2) / (self.data.ntot ** exposant
                                           * self.sp[:t] ** exposant)
                              * mv(self.ev.T[:t], pkm) ** 2).numpy()
        # Sum of contributions produces the kFDA statistic:
        kfda = kfda_contributions.cumsum(axis=0)

        trunc = range(1, t+1)  # truncations

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kfdat = pd.Series(
                kfda, index=trunc, dtype=str(self.dtype).replace('torch.', '')
            )
            kfdat_contributions = pd.Series(
                kfda_contributions, index=trunc,
                dtype=str(self.dtype).replace('torch.', '')
            )
        return kfdat, kfdat_contributions

    def compute_mmd(self):
        """
        Computes the MMD (maximal mean discrepancy) test statistic.

        Returns
        -------
        float
            The value of MMD.

        """
        m = self.compute_omega()
        if self.data_ny is not None:
            # Computing Nystrom anchors:
            self.sp_anchors, self.ev_anchors = self.compute_nystrom_anchors()
            Lp12 = diag(self.sp_anchors ** (-1/2))
            Pm = self.compute_centering_matrix(landmarks=True)
            Kmn = self.compute_kmn()
            psi_m = mv(Lp12, mv(self.ev_anchors.T, mv(Pm, mv(Kmn, m))))
            mmd = dot(psi_m, psi_m) ** 2
        else:
            K = self.compute_gram()
            mmd = dot(mv(K, m), m) ** 2
        return mmd.item()

    def compute_upk(self, t, new_obs=None):
        """
        epk is an alias for the product ePK that appears when projecting the
        data on the discriminant axis. This functions computes the
        corresponding block with respect to the model parameters.

        warning: some work remains to be done to :
            - normalize the vectors with respect to n_anchors as in pkm
            - separate the different nystrom approaches
        FIXME: seems to be ok

        Parameters
        ----------
            t : int
                Maximal truncation.
            new_obs : torch.tensor, optional
                Unused by default. If not None, then the Gram matrix between
                landmarks and `new_obs` is computed.
        """
        if self.sp is None and self.ev is None:
            self.sp, self.ev = self.diagonalize_centered_gram()
        if self.data_ny is not None:
            Kzx = self.compute_kmn(new_obs=new_obs)
            if self.sp_anchors is None and self.ev_anchors is None:
                self.sp_anchors, self.ev_anchors = \
                    self.compute_nystrom_anchors()
            Lz12 = diag(self.sp_anchors**-(1/2))
            epk = 1 / self.data_ny.ntot**(1/2) * \
                multi_dot([
                    self.ev.T[:t], Lz12, self.ev_anchors.T, Kzx
                ]).T
        else:
            Pbi = self.compute_centering_matrix()
            Kx = self.compute_gram(new_obs=new_obs)
            epk = multi_dot([self.ev.T[:t], Pbi, Kx]).T
        return epk

    def compute_projections(self, stat, t=100, center=True, new_obs=None):
        """
        Computes the vector of projection of the embeddings on the discriminant
        axis corresponding to the KFDA statistic for every truncation up to t.

        The projection with a truncation t is given by the formula :

        h^T kx = C * \\sum_{p=1:t} n1*n2 / ( lp*n)^2 [up^T PK omega] up^T P K

        where C is a normalization constant.

        Parameters
        ----------

        stat : Pandas.Series
            kFDA statistics (same as the attribute `kfda_statistic` of class
            Ktest). Required for normalization.

        t : int, optional
            Maximal truncation, the default is 100.

        center : bool, optional
            If True (default), the projections are centered with respect to
            the mean embedding.

        new_obs : torch.tensor, optional
            Unused by default. If not None, then the projections for the
            `new_obs` data are computed.

        Returns
        -------

        proj_kfda : pandas.DataFrame
            Projections associated with every observation (rows) on every
            eigendirection (columns).

        proj_kpca : pandas.DataFrame
            Contributions of each eigendirection (columns) to projections
            associated with every observation (rows). 'proj_kfda' contains the
            cumulated sum of the values in 'proj_kpca'.

        """
        # diagonalize centered Gram matrix if needed
        if self.sp is None and self.ev is None:
            self.sp, self.ev = self.diagonalize_centered_gram()
        # fix truncation if needed
        t = min(t, len(self.sp))
        # compute intermediate quantities
        pkm = self.compute_pkm()
        upk = self.compute_upk(t, new_obs=new_obs)
        # number of observations in training data
        n1, n2 = self.data.nobs.values()
        n = self.data.ntot
        # number of observations in projected data
        if new_obs is not None:
            n_obs = new_obs.shape[0]
        else:
            n_obs = n
        # compute projections
        if center:
            # define centering matrix (identity if no centering)
            centering_mat = eye(n_obs, dtype=self.dtype) \
                - ones((n_obs, n_obs), dtype=self.dtype) / n_obs
            # project
            proj = (self.sp[:t]**(-2) * mv(self.ev.T[:t], pkm)
                    * matmul(centering_mat, upk)).numpy()
        else:
            # project
            proj = (self.sp[:t]**(-2) * mv(self.ev.T[:t], pkm)
                    * upk).numpy()
        # post-processing
        # (manage training data vs new obsercations differently)
        if new_obs is not None:
            proj_list = [proj]
        else:
            proj_list = [proj[:n1], proj[n1:]]
        # init output
        proj_kfda = {}
        proj_kpca = {}
        # group index (create a new one for new_obs if needed)
        if new_obs is not None:
            index_dict = {"new_obs": pd.Index(range(new_obs.shape[0]))}
        else:
            index_dict = self.data.index
        # iterate through groups
        for i, (name, ind) in enumerate(index_dict.items()):
            proj_kpca[name] = pd.DataFrame(
                proj_list[i], index=ind,
                columns=[str(t) for t in range(1, t+1)]
            )
            proj_kfda[name] = pd.DataFrame(
                proj_list[i].cumsum(axis=1), index=ind,
                columns=[str(t) for t in range(1, t+1)]
            )
            proj_kfda[name] /= np.sqrt(
                n ** 3 * stat.values[:t] / (n1 * n2)
            )
        return proj_kfda, proj_kpca

    def kfda_loss(self, t=100, new_obs=None, stat=None):
        """
        Compute the two compartments (one for each group) of the kFDA loss
        function associated to the prediction for each observations (either in
        training data or for provided new observations) depending on kFDA axis
        projection.

        The loss is the difference between the distance to each group mean
        embedding. The function computes each element of this difference.

        Parameters
        ----------

        t: int, optional
            Maximal truncation, the default is 100.

        new_obs: torch.tensor, optional
            Unused by default. If not None, then the loss function for the
            `new_obs` data are computed.

        stat : Pandas.Series, optional
            kFDA statistics (same as the attribute `kfda_statistic` of class
            Ktest). Required for projection normalization. Can be provided as
            input argument to avoid re-computing it. If `None` (default), then
            the kFDA statistics is re-computed.

        Returns
        -------

        distance_group1: dict
            dictionary of arrays (torch.Tensor) storing the distance from each
            observation to the mean embedding of group 1 for increasing
            truncation, either for each group in the training data or for the
            new observations.

        distance_group2: dict
            dictionary of arrays (torch.Tensor) storing the distance from each
            observation to the mean embedding of group 2 for increasing
            truncation, either for each group in the training data or for the
            new observations.
        """

        # get kFDA stat Value
        if stat is None:
            stat, _ = self.compute_kfda()

        # compute kfda projection for training data
        proj_kfda, _ = self.compute_projections(
            stat, t, center=False, new_obs=None
        )

        # compute mean embedding for training data (in the two groups)
        mean_embed = []
        for group in proj_kfda.keys():
            mean_embed.append(proj_kfda[group].mean(axis=0))
        mean_embed = pd.DataFrame(mean_embed)
        # cast back to torch object
        mean_embed = to.from_numpy(mean_embed.values)

        # if new observation, compute corresponding projection
        if new_obs is not None:
            proj_kfda, _ = self.compute_projections(
                stat, t, center=False, new_obs=new_obs
            )

        # init output
        distance_group1 = {}
        distance_group2 = {}

        # compute loss function compartments for each observation set
        # iterate through group (or new obs)
        for i, (group, proj_tab) in enumerate(proj_kfda.items()):

            # cast back to torch object
            proj_tab = to.from_numpy(proj_tab.values)

            # distance to each group
            distance_g1_val = to.abs(proj_tab - mean_embed[0,:])  # group 1
            distance_g2_val = to.abs(proj_tab - mean_embed[1,:])  # group 2

            distance_group1[group] = distance_g1_val
            distance_group2[group] = distance_g2_val

        # output
        return distance_group1, distance_group2

    def kfda_predict(self, t=100, new_obs=None, pred_threshold=0.5, stat=None):
        """
        Compute prediction for each observations according to kFDA, i.e.
        assign each observations to one of the two groups depending on
        kFDA axis projection.

        Parameters
        ----------

        stat : Pandas.Series
            kFDA statistics (same as the attribute `kfda_statistic` of class
            Ktest). Required for normalization.

        t : int, optional
            Maximal truncation, the default is 100.

        pred_threshold : float or Iterable, optional
            Number (or Iterable containing numbers) between `0` an 1 to bias
            prediction towards first group or second group (in appearence order
            in the data). `0` means predicting only first group and `1`
            predicting only second group. Default value is `0.5` and no bias is
            introduced. Useful for ROC curve and AUC computations.
            If Iterable, the prediction is computed for each threshold value.

        stat : Pandas.Series, optional
            kFDA statistics (same as the attribute `kfda_statistic` of class
            Ktest). Required for projection normalization. Can be provided as
            input argument to avoid re-computing it. If `None` (default), then
            the kFDA statistics is re-computed.

        Returns
        -------

        pred: dict or list of dict
            dictionary (or list of dictionaries) of arrays (np.ndarray) storing
            kFDA predictions for each observation and increasing truncation,
            either for each group in the training data or for the new
            observations. If `pred_threshold` input argument is an Iterable,
            then `pred` is a list corresponding to prediction dictionaries for
            each element in `pred_threshold`.

        loss: dict or list of dict
            dictionary (or list of dictionaries) of arrays (np.ndarray) storing
            kFDA loss function values for each observation and increasing
            truncation, either for each group in the training data or for the
            new observations. If `pred_threshold` input argument is an
            Iterable, then `pred` is a list corresponding to prediction
            dictionaries for each element in `pred_threshold`.
        """

        # check threshold input and convert to list if scalar
        msg = "'pred_threshold' should be a number or an iterable " + \
            "containing numbers between 0 and 1"
        if not isinstance(pred_threshold, (Real, Iterable)):
            raise TypeError(msg)
        if not isinstance(pred_threshold, Iterable):
            pred_threshold = [pred_threshold]
            if not all(
                isinstance(item, Real) and
                item >= 0 and item <= 1
                for item in pred_threshold
            ):
                raise ValueError(msg)

        # compute loss function associated to kFDA prediction
        distance_group1, distance_group2 = self.kfda_loss(t, new_obs, stat)

        # init output (list of dictionaries)
        pred = []
        loss = []

        # iterate through threshold value
        for thres_val in pred_threshold:

            # init intermediate results
            pred_inter = {}
            loss_inter = {}

            # compute prediction for each observation set
            # iterate through group (or new obs)
            for i, (group, dist_g1, dist_g2) in enumerate(zip(
                distance_group1.keys(),
                distance_group1.values(),
                distance_group2.values()
            )):

                # prediction loss
                loss_val = dist_g1 - dist_g2 - \
                    pred_threshold_fun(thres_val, dist_g2, dist_g1)
                # loss > 0 ?
                # loss < 0 -> group 1
                # loss > 0 -> group 2

                # convert loss to group name prediction
                group_name = np.array(list(self.data.index.keys()))
                pred_val = group_name[(1 - (loss_val < 0).int()).numpy()]

                # store prediction and loss for current set of observations
                pred_inter[group] = pred_val
                loss_inter[group] = loss_val.numpy()

            # store results
            pred.append(pred_inter)
            loss.append(loss_inter)

        if len(pred_threshold) == 1:
            pred = pred[0]
            loss = loss[0]

        # output
        return pred, loss
