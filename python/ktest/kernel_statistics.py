import numpy as np
import pandas as pd
from torch import cdist, cat, matmul, exp, mv, dot, diag, sqrt
from torch import ones, eye, zeros, tensor, float64
from torch.linalg import multi_dot, eigh
import warnings


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
        warnings.warn('The median is null. To avoid a kernel with zero bandwidth, we replace the median by the mean')
        mean = dtot.mean()
        if mean == 0:
            warnings.warn('warning: all your dataset is null')
        return mean
    else:
        return dtot.median()


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


class Statistics():
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

    """

    def __init__(
        self, data, kernel_function='gauss', bandwidth='median',
        median_coef=1, data_nystrom=None, n_anchors=None, anchor_basis='w'
    ):
        self.data = data

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
            self.kernel, self.computed_bandwidth = \
                gauss_kernel_median(
                    x=self.data.data[self.data.sample_names[0]],
                    y=self.data.data[self.data.sample_names[1]],
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
    def ordered_eigsy(matrix):
        # The matrix with column-wise eigenvectors
        sp, ev = eigh(matrix)
        order = sp.argsort()[::-1]
        ev = tensor(ev[:, order], dtype=float64)
        sp = tensor(sp[order], dtype=float64)
        return (sp, ev)

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
        data = self.data if not landmarks else self.data_ny
        if not landmarks or self.anchor_basis == 'w':
            In = eye(data.ntot)
            effectifs = list(data.nobs.values())

            cumul_effectifs = np.cumsum([0]+effectifs)

            # Computing a bloc diagonal matrix where the ith diagonal bloc is
            # J_ni, an (ni x ni) matrix full of 1/ni where ni is the size
            # of the ith group
            diag_Jn_by_n = cat([
                cat([
                    zeros(nprec, nell, dtype=float64),
                    1/nell*ones(nell, nell, dtype=float64),
                    zeros(data.ntot - nprec - nell, nell, dtype=float64)
                ], dim=0)
                for nell, nprec in zip(effectifs, cumul_effectifs)
            ], dim=1)
            return In - diag_Jn_by_n
        elif self.anchor_basis == 'k':
            return eye(data.ntot, dtype=float64)
        elif self.anchor_basis == 's':
            In = eye(data.ntot, dtype=float64)
            Jn = ones(data.ntot, data.ntot, dtype=float64)
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
        m_mu1 = -1/n1 * ones(n1, dtype=float64)
        m_mu2 = 1/n2 * ones(n2, dtype=float64)
        return cat((m_mu1, m_mu2), dim=0)

    def compute_gram(self, landmarks=False):
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

        Returns
        -------
            K : torch.Tensor,
                Gram matrix of interest.
        """
        data = self.data if not landmarks else self.data_ny
        D = cat([x for x in data.data.values()], axis=0)
        K = self.kernel(D, D)
        return K

    def compute_kmn(self):
        """
        Computes an (nxanchors+nyanchors)x(ndata) conversion gram matrix.

        """
        data = cat([x for x in self.data.data.values()], axis=0)
        landmarks = cat([x for x in self.data_ny.data.values()], axis=0)
        kmn = self.kernel(landmarks, data)
        return kmn

    def compute_nystrom_anchors(self):
        K = self.compute_gram(landmarks=True)
        P = self.compute_centering_matrix(landmarks=True)
        Kw = 1 / self.data_ny.ntot * multi_dot([P, K, P])
        return Statistics.ordered_eigsy(Kw)

    def diagonalize_centered_gram(self):
        """
        Diagonalizes the bicentered Gram matrix which shares its spectrum
        with the within covariance operator in the RKHS.

        Returns
        -------
            Eigenvalues and eigenvectors, later stored in attributes 'sp' and
            'ev' respectively.

        """
        # Computing centering matrix P:
        P = self.compute_centering_matrix()

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
            Kw = (1 / (self.data.ntot * self.data_ny.ntot)
                  * multi_dot([Lp_inv_12, self.ev_anchors.T, Pm, Kmn, P,
                               Kmn.T, Pm, self.ev_anchors, Lp_inv_12]))
        # Standard version:
        else:
            K = self.compute_gram()
            Kw = 1 / self.data.ntot * multi_dot([P, K, P])

        # Diagonalisation with a function in C++:
        return Statistics.ordered_eigsy(Kw)

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
        # Calculating the bi-centering vector Omega and the centering matrix Pbi
        omega = self.compute_omega()  # vector with 1/n1 and -1/n2
        Pbi = self.compute_centering_matrix()  # Centering by block matrix
        if self.data_ny is not None:
            Lz12 = diag(self.sp_anchors**(-1/2))
            Kzx = self.compute_kmn()
            Pi = self.compute_centering_matrix(landmarks=True)
            pkm = (1 / np.sqrt(self.data_ny.ntot)
                   * mv(Lz12, mv(self.ev_anchors.T, mv(Pi, mv(Kzx, omega)))))
        else:
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
            kfdat = pd.Series(kfda, index=trunc)
            kfdat_contributions = pd.Series(kfda_contributions, index=trunc)
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

    def compute_upk(self, t):
        """
        epk is an alias for the product ePK that appears when projecting the
        data on the discriminant axis. This functions computes the
        corresponding block with respect to the model parameters.

        warning: some work remains to be done to :
            - normalize the vectors with respect to n_anchors as in pkm
            - separate the different nystrom approaches
        """
        Pbi = self.compute_centering_matrix()
        if self.sp is None and self.ev is None:
            self.sp, self.ev = self.diagonalize_centered_gram()
        if self.data_ny is not None:
            Kzx = self.compute_kmn()
            if self.sp_anchors is None and self.ev_anchors is None:
                self.sp_anchors, self.ev_anchors = \
                    self.compute_nystrom_anchors()
            Lz12 = diag(self.sp_anchors**-(1/2))
            epk = 1 / self.data_ny.ntot**(1/2) * multi_dot([self.ev.T[:t],
                                                            Lz12,
                                                            self.ev_anchors.T,
                                                            Kzx]).T
        else:
            Kx = self.compute_gram()
            epk = multi_dot([self.ev.T[:t], Pbi, Kx]).T
        return epk

    def compute_projections(self, t=100):
        """
        Computes the vector of projection of the embeddings on the discriminant
        axis corresponding to the KFDA statistic for every truncation up to t.

        The projection with a truncation t is given by the formula :

            h^T kx = \\sum_{p=1:t} n1*n2 / ( lp*n)^2 [up^T PK omega] up^T P K

        Returns
        -------
        t : int, optional
            Maximal truncation, the default is 100.

        proj_kfda : pandas.DataFrame
            Projections associated with every observation (rows) on every
            eigendirection (columns).

        proj_kpca : pandas.DataFrame
            Contributions of each eigendirection (columns) to projections
            associated with every observation (rows). 'proj_kfda' contains the
            cumulated sum of the values in 'proj_kpca'.

        """
        if self.sp is None and self.ev is None:
            self.sp, self.ev = self.diagonalize_centered_gram()
        t = min(t, len(self.sp))
        pkm = self.compute_pkm()
        upk = self.compute_upk(t)
        n1, n2 = self.data.nobs.values()
        proj = (n1 * n2 * self.data.ntot ** (-2) * self.sp[:t] ** (-2)
                * mv(self.ev.T[:t], pkm) * upk).numpy()
        proj_list = [proj[:n1], proj[n1:]]
        proj_kfda = {}
        proj_kpca = {}
        for i, (name, ind) in enumerate(self.data.index.items()):
            proj_kpca[name] = pd.DataFrame(
                proj_list[i], index=ind,
                columns=[str(t) for t in range(1, t+1)]
            )
            proj_kfda[name] = pd.DataFrame(
                proj_list[i].cumsum(axis=1), index=ind,
                columns=[str(t) for t in range(1, t+1)]
            )
        return proj_kfda, proj_kpca
