import gzip
import warnings
import dill
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd
from scipy.stats import chi2, gaussian_kde
from sklearn.model_selection import RepeatedStratifiedKFold
from torch import float64
import torch as to
from tqdm import tqdm

from .data import Data
from .kernel_statistics import Statistics
from .utils import compute_accuracy


class Ktest(Statistics):
    """
    Class performing kernel tests, such as maximal mean discrepancy test (MMD)
    and a test based on kernel Fisher Discriminant Analysis (kFDA).

    Parameters
    ----------
        data : 1 or 2-dimensional array-like
            The data to test, containing two samples at the same time.
            Can be either Pandas.DataFrame, Pandas.Series (if univariate data),
            numpy.ndarray or torch.tensor. Rows correspond to observations,
            and columns correspond to features. If Pandas.DataFrame or
            Pandas.Series, and so is 'data', assigned indices are used to
            label observations. Otherwise, observations are treated with
            respect to the provided order.

        metadata : 1-dimensional array-like
            The metadata associated with the data to test. The values should
            indicate the labels of the samples in 'data'. At the moment, the
            library performs a two sample test only, thus two or more levels
            are expected. Can be either Pandas.DataFrame, Pandas.Series
            (if univariate data), numpy.ndarray or torch.tensor. If
            Pandas.DataFrame or Pandas.Series, ans so is 'data', indices should
            coincide. Otherwise, the first dimensions of 'data' and 'metadata',
            corresponding to observations, should coincide.

        sample_names : None or 1-dimensional array-like, optional
            If given, should containt exactly two strings, labeling the samples
            (overrides the metadata levels). If None, sample names are derived
            from the metadata levels (the first two detected levels).

        kernel_function : callable or str, optional
            Specifies the kernel function. Acceptable values in the form of a
            string are 'gauss' (default) and 'linear'. Pass a callable for a
            user-defined kernel function.

        kernel_bandwidth : 'median' or float, optional
            Value of the bandwidth for kernels using a bandwidth. If 'median'
            (default), the bandwidth will be set as the median or its multiple,
            depending on the value of the parameter `median_coef`. Pass a float
            for a user-defined value of the bandwidth.

        kernel_median_coef : float, optional
            Multiple of the median to compute bandwidth if bandwidth=='median'.
            The default is 1.

        nystrom : bool, optional
            If True, computes the Nystrom approximation, and uses it to compute
            the test statistics. The default if False.

        n_landmarks: int, optional
            Number of landmarks used in the Nystrom method. If unspecified, one
            fifth of the observations are selected as landmarks.

        n_anchors : int, optional
            Number of anchors used in the Nystrom method, by default equal to
            the number of landmarks.

        landmark_method : 'random' or 'kmeans++', optional
            Method of the landmarks selection, 'random '(default) corresponds
            to selecting landmarks among the observations according to the
            random uniform distribution.

        anchor_basis : str, optional
            Options for different ways of computing the covariance operator of
            the landmarks in the Nystrom method, of which the anchors are the
            eigenvalues. Possible values are 'w' (default),'s' and 'k'.

        dtype : torch.dtype, optional
            Floating point number type/precision used for data storage and
            computations. Default is `torch.float64`.

        eps : float, optional
            minimum threshold value to clip lower eigen values to zeros.
            If `None` (default), then machine precision (given by
            `torch.finfo()`) for specified dtype is used as threshold.

        clip_eigval : boolean,
            flag to enable/disable eigen value clipping.

        verbose : int, optional
            The higher the verbosity, the more messages keeping track of
            computations. The default is 1.
            - < 1: no messages,
            - 1: progress bar with computation time,
            - 2: warnings are printed once,
            - 3: warnings are printed every time they appear.

    Attributes
    ----------
        dataset: 1 or 2-dimensional array-like
            The data assigned to the parameter 'data' is stored here.

        data : instance of class Data
            Contains various information on the original dataset, see the
            documentation of the class Data for more details.

        kstat : instance of class Statistics
            Attribute used for statistics calculation, see the documentation
            of the class `ktest.kernel_statistics.Statistics` for more details.

        data_nystrom : None or instance of class Data
            None if nystrom==False. Otherwise, contains various information on
            the Nystrom dataset, see the documentation of the class Data
            for more details.

        stat : Pandas.Series or float or None
            Computed kFDA test statistic values (`Pandas.Series`)
            corresponding to increasing truncation values,
            or MMD test statistic value (float) if specifically requested.
            None if not computed.

        stat_type : str
            Type of statistics that was computed, either `"kfda"` (default) for
            kernel FDA-based 2 sample test or `"mmd"` for maximum mean
            discrepancy-based test.
            `None` if statistics values have not been computed yet.

        pval : Pandas.Series or float or None
            Computed p-values (`Pandas.Series`) associated to the kFDA test
            statistic values, corresponding to increasing truncation values,
            obtained from the asymptotic distribution (chi-square with T degree
            of freedom) or with a permutation approach,
            or p-value (float) associated to the MMD test statistic value,
            obtained with a permutation approach.
            None if not computed.

        stat_contrib : Pandas.Series or None
            Unidirectional statistic associated with each eigendirection of
            the within-group covariance operator for the kFDA statistics,
            corresponding to increasing truncation values.
            For kFDA, `self.stat` contains the cumulated sum of the values in
            `self.stat_contrib`.
            This quantity is irrelevant for MMD test statistics (see
            `self.stat_type`).

        pval_type : str
            Type of p-values that have been computed, can be `"asymp"` for
            "asymptotic" or `"perm"` for permutation. C.f.
            `self.pval` attribute doc. `None` if p-values have not been
            computed yet. Note: for MMD test statistics, p-values can only
            be estimated by permutations.

        proj : pandas.DataFrame
            Projections of the embeddings on the discriminant axis
            corresponding to the KFDA statistic, associated with every
            observation (rows) on every eigendirection (columns).

        proj_contrib : pandas.DataFrame
            Contributions of each eigendirection (columns) to projections of
            the embeddings on the discriminant axis corresponding to the
            KFDA statistic, associated with every observation (rows).
            'proj' contains the cumulated sum of the values in
            'proj_contrib'.

        rnd_gen : int, numpy.random.Generator instance,
                numpy.random.RandomState instance or None
            Determines random number generation for the Nystrom approximation
            and for the permutations. If None (default), the default numpy
            random generator is used.
            To ensure that the results are reproducible, pass an integer
            that will be used as seed for the random number generation, or
            a Numpy random Generator instance (recommended), or a Numpy
            RandomState instance (legacy).
    """

    def __init__(
        self, data, metadata, sample_names=None,
        kernel_function='gauss', kernel_bandwidth='median',
        kernel_median_coef=1, nystrom=False, n_landmarks=None,
        landmark_method='random', n_anchors=None, anchor_basis='w',
        random_state=None, dtype=float64, eps=None, clip_eigval=True,
        verbose=0
    ):
        with warnings.catch_warnings():
            if verbose < 2:
                warnings.simplefilter("ignore")
            elif verbose >= 3:
                warnings.simplefilter("always")

            self.dataset = data
            self.metadata = metadata
            self.data = Data(
                data=data, metadata=metadata, sample_names=sample_names,
                dtype=dtype
            )
            self.sample_names = self.data.sample_names

            self.dtype = dtype
            self.eps = eps
            self.clip_eigval = clip_eigval

            # random number generation
            if random_state is None:
                self.rnd_gen = np.random.default_rng()
            elif isinstance(random_state, int):
                self.rnd_gen = np.random.default_rng(random_state)
            else:
                assert isinstance(random_state, np.random.Generator) or \
                    isinstance(random_state, np.random.RandomState)
                self.rnd_gen = random_state

            ### Kernel:
            self.kernel_function = kernel_function
            self.kernel_bandwidth = kernel_bandwidth
            self.kernel_median_coef = kernel_median_coef

            ### Nystrom:
            self.data_nystrom = None
            self.n_landmarks = n_landmarks
            self.landmark_method = landmark_method
            self.n_anchors = n_anchors
            self.anchor_basis = anchor_basis

            if nystrom:
                self.data_nystrom = Data(
                    data=data, metadata=metadata,
                    sample_names=sample_names,
                    nystrom=True, n_landmarks=self.n_landmarks,
                    landmark_method=self.landmark_method,
                    random_state=self.rnd_gen,
                    dtype=dtype
                )

            ### define statistic object
            self.kstat = Statistics(
                self.data, kernel_function=self.kernel_function,
                bandwidth=self.kernel_bandwidth,
                median_coef=self.kernel_median_coef,
                data_nystrom=self.data_nystrom,
                n_anchors=self.n_anchors,
                anchor_basis=self.anchor_basis,
                eps=self.eps, clip_eigval=self.clip_eigval
            )

            ### Output statistics ###
            ## kFDA (or MMD) statistic
            self.stat = None
            self.stat_type = None
            self.pval = None
            self.pval_type = None
            self.stat_contrib = None

            ### Projections:
            self.proj = {}
            self.proj_contrib = {}

    def __str__(self):
        s = "An object of class Ktest."
        s += self.data.__str__()
        if self.data_nystrom is not None:
            s += "\nNystrom approximation with"
            s += f" {self.data_nystrom.ntot} landmarks."

        s += '\n___Multivariate test results___'
        ncs = 'not computed, run ktest.test().'
        if self.stat_type == 'mmd':
            mmd_s = f'stat={self.stat}, P-value={self.pval} (permutation)'
            s += '\nMMD: '
            s += f'{mmd_s}' if self.stat is not None else ncs
        elif self.stat_type == 'kfda':
            s += '\nkFDA:'
            if self.stat is None:
                s += '\n' + ncs
            else:
                for t in range(min(len(self.stat), 5)):
                    s += f'\nTruncation {t+1}: stat={self.stat.iloc[t]}, '
                    s += 'P-value='
                    s += f'{self.pval.iloc[t]}' if self.pval is not None else \
                        'not computed'
                    s += ' ('
                    s += 'asymptotic' if self.pval_type == "asymp" \
                        else "permuation"
                    s += ')'
        else:
            s += '\n' + ncs
        return s

    def __repr__(self):
        return self.__str__()

    def compute_test_statistic(self, stat='kfda', kstat=None, verbose=0):
        """
        Computes the kFDA or the MMD statistic.

        Parameters
        ----------
        stat : str
            The test statistic to use, can be either 'kfda' (default) or' mmd'.
        kstat : instance of Statistics or None
            If None (default), the comutations are performed on the 'kstat'
            attribute. If an instance of Statistics, it is used for
            computations, used to compute test statistics of the permuted
            samples.
        verbose : int, optional
            The higher the verbosity, the more messages keeping track of
            computations. The default is 1.
            - < 1: no messages,
            - 1: progress bar with computation time,
            - 2: warnings are printed once,
            - 3: warnings are printed every time they appear.

        Returns
        -------
        test statistics in different forms:
            - kFDA: two Pandas.Series with the kFDA statistic and the
            contributions for each truncation,
            - MMD: a single statistic value (float).

        """
        kstatistics = kstat if kstat is not None else self.kstat
        if stat == 'kfda':
            test_res = kstatistics.compute_kfda_stat()
        elif stat == 'mmd':
            test_res = kstatistics.compute_mmd()
        else:
            raise ValueError(
                f"Statistic '{stat}' not recognized. Possible values: " +
                "'kfda','mmd'"
            )
        return test_res

    def compute_pvalue(
        self, stat='kfda', permutation=False, n_permutations=500, verbose=1
    ):
        """
        Computes the p-value of the considered statistic.

        Parameters
        ----------
        stat : str
            The test statistic to use, can be either 'kfda' (default) or' mmd'.
        permutation : bool, optional
            If False (default), the asymptotic approach is applied to compute
            p-values. If True, a permutation test is performed.
        n_permutations : int, optional
            Number of permutations performed. The default is 500.
        verbose : int, optional
            The higher the verbosity, the more messages keeping track of
            computations. The default is 1.
            - < 1: no messages,
            - 1: progress bar with computation time,
            - 2: warnings are printed once,
            - 3: warnings are printed every time they appear.

        Returns
        -------
        p-values in different forms:
            - asymptotic kFDA: two Pandas.Series with p-values of kFDA and
            contributions for each truncation,
            - permutation kFDA: a Pandas.Series with p-values of kFDA
            statistics for each truncation,
            - MMD: a single p-value (float).

        """
        if stat == 'kfda' and not permutation:
            if verbose > 0:
                print('- Computing asymptotic p-values')
            pval = chi2.sf(self.stat, self.stat.index)
            return pd.Series(
                pval, index=self.stat.index,
                dtype=str(self.dtype).replace('torch.', '')
            )
        else:
            if verbose > 0:
                print('- Performing permutations to compute p-values:')
            stats_count = pd.Series(
                0, index=range(1, len(self.stat) + 1)
            ) if stat == 'kfda' else 0
            it = tqdm(range(n_permutations)) \
                if verbose > 0 else range(n_permutations)
            for i in it:
                meta_perm = self.metadata.copy()
                if isinstance(meta_perm, (pd.DataFrame, pd.Series)):
                    self.rnd_gen.shuffle(meta_perm.values)
                else:
                    self.rnd_gen.shuffle(meta_perm)
                data_perm = Data(data=self.dataset, metadata=meta_perm,
                                 sample_names=self.sample_names)
                data_perm_nystrom = None
                if self.data_nystrom is not None:
                    data_perm_nystrom = Data(
                        data=self.dataset, metadata=meta_perm,
                        sample_names=self.sample_names,
                        nystrom=True, n_landmarks=self.n_landmarks,
                        landmark_method=self.landmark_method,
                        random_state=self.rnd_gen
                    )

                kstat_perm = Statistics(
                    data_perm,
                    kernel_function=self.kernel_function,
                    bandwidth=self.kernel_bandwidth,
                    median_coef=self.kernel_median_coef,
                    data_nystrom=data_perm_nystrom,
                    n_anchors=self.n_anchors,
                    anchor_basis=self.anchor_basis,
                    eps=self.eps, clip_eigval=self.clip_eigval
                )

                perm_stats_res = self.compute_test_statistic(
                    stat=stat, kstat=kstat_perm, verbose=verbose-1
                )

                if stat == 'kfda':
                    stats_count += perm_stats_res[0].ge(self.stat)
                elif stat == 'mmd':
                    stats_count += (perm_stats_res >= self.stat)
            return stats_count / n_permutations

    def test(
        self, stat='kfda', permutation=False, n_permutations=500, verbose=1
    ):
        """
        Performs either the MMD or the kFDA test to compare the two samples of
        the considered dataset. Stores the results in the corresponding
        attributes, depending on the statistic and on the p-value approach.

        Parameters
        ----------
        stat : str
            The test statistic to use, can be either 'kfda' (default) or' mmd'.
        permutation : bool, optional
            If False (default), the asymptotic approach is applied to compute
            p-values. If True, a permutation test is performed.
        n_permutations : int, optional
            Number of permutations performed. The default is 500.
        verbose : int, optional
            The higher the verbosity, the more messages keeping track of
            computations. The default is 1.
            - < 1: no messages,
            - 1: progress bar with computation time,
            - 2: warnings are printed once,
            - 3: warnings are printed every time they appear.

        """
        with warnings.catch_warnings():
            if verbose < 2:
                warnings.simplefilter("ignore")
            elif verbose >= 3:
                warnings.simplefilter("always")

            if stat == 'kfda':
                if verbose > 0:
                    print('- Computing kFDA statistic')
                self.stat, self.stat_contrib = \
                    self.compute_test_statistic(verbose=verbose)
                if not permutation:
                    self.pval = self.compute_pvalue(verbose=verbose)
                    self.pval_type = "asymp"
                else:
                    self.pval = self.compute_pvalue(
                        permutation=permutation, n_permutations=n_permutations,
                        verbose=verbose
                    )
                    self.pval_type = "perm"
            elif stat == 'mmd':
                if verbose > 0:
                    print('- Computing MMD statistic')
                self.stat = self.compute_test_statistic(
                    stat=stat, verbose=verbose
                )
                self.pval = self.compute_pvalue(
                    stat=stat, verbose=verbose, n_permutations=n_permutations
                )
                self.pval_type = "perm"
            else:
                raise ValueError(
                    f"Statistic '{stat}' not recognized. " +
                    "Possible values : 'kfda','mmd'"
                )

            self.stat_type = stat

    def get_projections(self, n_trunc=100, center=True, new_obs=None, verbose=1):
        """
        Computes the vector of projection of the embeddings on the discriminant
        axis corresponding to the KFDA statistic for every truncation up to t.
        Assigns the values of projections and the contributions of every
        eigendirection to attributes 'proj' and 'proj_contrib'
        respectively.

        Parameters
        ----------
        n_trunc : int, optional
            Maximal truncation for projections calculation, the default is 100.

        center : bool, optional
            If True (default), the projections are centered with respect to
            the mean embedding.

        new_obs: array-like, pandas.DataFrame or torch.Tensor or numpy.array,
            optional
            Unused by default. If not None, then the projections for the
            `new_obs` data is computed.

        verbose : int, optional
            The higher the verbosity, the more messages keeping track of
            computations. The default is 1.
            - < 1: no messages,
            - 1: progress bar with computation time,
            - 2: warnings are printed once,
            - 3: warnings are printed every time they appear.

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
        with warnings.catch_warnings():
            if verbose < 2:
                warnings.simplefilter("ignore")
            elif verbose >= 3:
                warnings.simplefilter("always")

            if self.stat is None:
                self.test(stat='kfda')

            # check new_obs input convert and to torch.Tensor
            if new_obs is not None:
                if not (
                    isinstance(new_obs, pd.DataFrame) or
                    isinstance(new_obs, np.ndarray) or
                    isinstance(new_obs, to.Tensor)
                ) and new_obs.shape[1] != self.dataset.shape[1]:
                    msg = "'new_obs' should be an array-like object with " + \
                        "the same number of columns as the original " + \
                        "training data."
                    raise ValueError(msg)

                if isinstance(new_obs, pd.DataFrame):
                    new_obs = to.from_numpy(new_obs.to_numpy())
                elif isinstance(new_obs, np.ndarray):
                    new_obs = to.from_numpy(new_obs)

            # compute projections
            proj, proj_contrib = self.kstat.compute_projections(
                self.stat, n_trunc=n_trunc, center=center, new_obs=new_obs
            )

            # record projections only when projecting training data
            self.proj = proj
            self.proj_contrib = proj_contrib

            # output
            return proj, proj_contrib

    def predict(self, n_trunc=100, new_obs=None, pred_threshold=0.5, verbose=1):
        """
        Compute prediction for each observations according to kFDA and with
        increasing truncation values, i.e. assign each observations to one of
        the two groups using projections on kFDA axis.

        Parameters
        ----------

        n_trunc : int, optional
            Maximal truncation, the default is 100.

        new_obs: array-like, pandas.DataFrame or torch.Tensor or numpy.array,
            optional
            Unused by default. If not None, then the prediction for the
            `new_obs` data is computed.

        pred_threshold : float or Iterable, optional
            Number (or Iterable containing numbers) between `0` an 1 to bias
            prediction towards first group or second group (in appearence order
            in the data). `0` means predicting only first group and `1`
            predicting only second group. Default value is `0.5` and no bias is
            introduced. Useful for ROC curve and AUC computations.
            If Iterable, the prediction is computed for each threshold value.

        verbose : int, optional
            The higher the verbosity, the more messages keeping track of
            computations. The default is 1.
            - < 1: no messages,
            - 1: progress bar with computation time,
            - 2: warnings are printed once,
            - 3: warnings are printed every time they appear.

        Returns
        -------

        kfda_pred: dict or list of dict
            dictionary (or list of dictionaries) of arrays (np.ndarray) storing
            kFDA predictions for each observation and increasing truncation,
            either for each group in the training data or for the new
            observations. If `pred_threshold` input argument is an Iterable,
            then `pred` is a list corresponding to prediction dictionaries for
            each element in `pred_threshold`.

        kfda_loss: dict or list of dict
            dictionary (or list of dictionaries) of arrays (np.ndarray) storing
            kFDA loss function values for each observation and increasing
            truncation, either for each group in the training data or for the
            new observations. If `pred_threshold` input argument is an
            Iterable, then `pred` is a list corresponding to prediction
            dictionaries for each element in `pred_threshold`.

        kfda_res: loss: dict or list of dict
            dictionary (or list of dictionaries) of arrays (np.ndarray) storing
            kFDA residual values for each observation and increasing
            truncation, either for each group in the training data or for the
            new observations. If `pred_threshold` input argument is an
            Iterable, then `pred` is a list corresponding to prediction
            dictionaries for each element in `pred_threshold`.
        """

        with warnings.catch_warnings():
            if verbose < 2:
                warnings.simplefilter("ignore")
            elif verbose >= 3:
                warnings.simplefilter("always")

            # compute statistic if needed
            if self.stat is None:
                self.test(stat='kfda')

            # check new_obs input convert and to torch.Tensor
            if new_obs is not None:
                if not (
                    isinstance(new_obs, pd.DataFrame) or
                    isinstance(new_obs, np.ndarray) or
                    isinstance(new_obs, to.Tensor)
                ) and new_obs.shape[1] != self.dataset.shape[1]:
                    msg = "'new_obs' should be an array-like object with " + \
                        "the same number of columns as the original " + \
                        "training data."
                    raise ValueError(msg)

                if isinstance(new_obs, pd.DataFrame):
                    new_obs = to.from_numpy(new_obs.to_numpy())
                elif isinstance(new_obs, np.ndarray):
                    new_obs = to.from_numpy(new_obs)

            # compute prediction
            kfda_pred, kfda_loss, kfda_res = self.kstat.kfda_predict(
                n_trunc=n_trunc, new_obs=new_obs, pred_threshold=pred_threshold,
                stat=self.stat
            )

            # output
            return kfda_pred, kfda_loss, kfda_res

    def cv(
        self, n_trunc=100, pred_threshold=0.5, n_fold=5, n_repeat=1, ref=None,
        random_state=None, verbose=1
    ):
        """
        Compute prediction and prediction error by V-fold cross-validation,
        for each observations according to kFDA and with increasing truncation
        values.

        Parameters
        ----------

        n_trunc : int, optional
            Maximal truncation, the default is 100.

        pred_threshold : float or Iterable, optional
            Number (or Iterable containing numbers) between `0` an 1 to bias
            prediction towards first group or second group (in appearence order
            in the data). `0` means predicting only first group and `1`
            predicting only second group. Default value is `0.5` and no bias is
            introduced. Useful for ROC curve and AUC computations.
            If Iterable, the prediction is computed for each threshold value.

        n_fold : int, default=5
            Number of folds in cross-validation. Should be at least 2.

        n_repeat : int, default=1
            Number of times cross-validation will be repeated.

        ref : str or None, default=None
            Reference group/class/subsample for computing true positive rates.
            The other group/class/subsample will be used for computing true
            negative rate.
            `ref` should be among input metadata. If None, then the reference
            is chosen to be the first group by order of appearence in the data.

        random_state : int, numpy.random.RandomState or None, default=None
            Controls the generation of the random states for cross-validation
            subsampling. Pass an int for reproducible output across multiple
            function calls.

        verbose : int, optional
            The higher the verbosity, the more messages keeping track of
            computations. The default is 1.
            - < 1: no messages,
            - 1: progress bar with computation time,
            - 2: warnings are printed once,
            - 3: warnings are printed every time they appear.

        Returns
        -------

        accuracy : list of numpy.ndarray
            list of 1-D arrays of average accuracy over cross-validation
            for increasing truncation values, corresponding to each prediction
            threshold bias value(s) provided in input.

        true_pos : list of numpy.ndarray
            list of 1-D arrays of average true positive rates over
            cross-validation for increasing truncation values, corresponding
            to each prediction threshold bias value(s) provided in input.

        true_neg : list of numpy.ndarray
            list of 1-D arrays of average true negative rates over
            cross-validation for increasing truncation values, corresponding
            to each prediction threshold bias value(s) provided in input.

        residuals : list of numpy.ndarray
            list of 1-D arrays of average residual values over
            cross-validation for increasing truncation values, corresponding
            to each prediction threshold bias value(s) provided in input.
        """

        with warnings.catch_warnings():
            if verbose < 2:
                warnings.simplefilter("ignore")
            elif verbose >= 3:
                warnings.simplefilter("always")

            # check input
            if ref is not None and not ref in self.data.sample_names:
                msg = "`ref` is not a valid class/group/subsample " + \
                    "reference name."
                raise ValueError(msg)
            else:
                ref = self.data.sample_names[0]

            # compute test statistics if not done
            if self.stat is None:
                self.test(stat='kfda')

            # cross-validation sub-subsampling
            cv_setup = RepeatedStratifiedKFold(
                n_splits=n_fold, n_repeats=n_repeat, random_state=random_state
            )
            cv_split = cv_setup.split(self.dataset, self.metadata)

            # init result storage over fold
            fold_res = []

            # verbosity
            verbose >= 1 and print(f"Starting cross-validation...")

            # iterate trough data partitions
            for i, (train_index, test_index) in enumerate(cv_split):
                verbose >= 1 and print(f"Split {i}")

                # prepare training data
                train_data = Data(
                    data=self.dataset.iloc[train_index],
                    metadata=self.metadata.iloc[train_index],
                    sample_names=self.sample_names
                )

                # Nystrom subsampling if needed
                train_data_nystrom = None
                if self.data_nystrom is not None:
                    train_data_nystrom = Data(
                        data=self.dataset.iloc[train_index],
                        metadata=self.metadata.iloc[train_index],
                        sample_names=self.sample_names,
                        nystrom=True, n_landmarks=self.n_landmarks,
                        landmark_method=self.landmark_method,
                        random_state=self.rnd_gen
                    )

                # define statistic object for training data
                kstat_train = Statistics(
                    train_data,
                    kernel_function=self.kernel_function,
                    bandwidth=self.kernel_bandwidth,
                    median_coef=self.kernel_median_coef,
                    data_nystrom=train_data_nystrom,
                    n_anchors=self.n_anchors,
                    anchor_basis=self.anchor_basis,
                    eps=self.eps, clip_eigval=self.clip_eigval
                )

                # compute prediction
                kfda_pred, kfda_loss, kfda_res = kstat_train.kfda_predict(
                    n_trunc=n_trunc,
                    new_obs=to.from_numpy(
                        self.dataset.iloc[test_index].to_numpy()
                    ),
                    pred_threshold=pred_threshold,
                    stat=None
                )

                # compute accuracy
                accuracy_res, true_pos_res, true_neg_res = compute_accuracy(
                    kfda_pred["new_obs"],
                    self.metadata.iloc[test_index].to_numpy(),
                    ref=ref
                )

                # store prediction results
                fold_res.append({
                    "accuracy": accuracy_res,
                    "true_pos": true_pos_res,
                    "true_neg": true_neg_res,
                    "residuals": kfda_res["new_obs"],
                    "test_index": test_index
                })

            # verbosity
            verbose >= 1 and print(f"...Aggregating CV fold results")

            # reformat cv result storage
            # input: `fold_res` is a list of dictionaries containing list of
            # arrays storing a metric for each pred_threshold values
            # output: `<metric>_list` is a list of <metric> arrays for each
            # pred_threshold values
            # accuracy_list = [
            #     list(accuracy_tab) for accuracy_tab in
            #     zip(*[d["accuracy"] for d in fold_res])
            # ]
            accuracy_list = list(map(
                list, zip(*[d["accuracy"] for d in fold_res])
            ))
            true_pos_list = list(map(
                list, zip(*[d["true_pos"] for d in fold_res])
            ))
            true_neg_list = list(map(
                list, zip(*[d["true_neg"] for d in fold_res])
            ))
            res_list = list(map(
                list, zip(*[d["residuals"] for d in fold_res])
            ))

            # compute average accuracy, true pos rate and true neg rate over
            # all folds
            accuracy = [
                np.mean(np.vstack(accuracy_tab_list), axis=0)
                for accuracy_tab_list in accuracy_list
            ]
            true_pos = [
                np.mean(np.vstack(true_pos_tab_list), axis=0)
                for true_pos_tab_list in true_pos_list
            ]
            true_neg = [
                np.mean(np.vstack(true_neg_tab_list), axis=0)
                for true_neg_tab_list in true_neg_list
            ]
            residuals = [
                np.mean(np.vstack(res_tab_list), axis=0)
                for res_tab_list in res_list
            ]

            # output
            return accuracy, true_pos, true_neg, residuals

    def plot_density(
        self, trunc=None, trunc_max=100, colors=None, labels=None, alpha=.5,
        legend_fontsize=15, font_family='serif'
    ):
        """
        Plots a density of the projection on a discriminant axis of
        the kFDA statistic.

        Parameters
        ----------
        trunc : int, optional
            Axis to plot, by default equal to the maximal truncation.

        trunc_max : int, optional
            Maximal truncation for projections calculation, the default is 100.

        colors : dict or None
            Sample colors, should be a dictionary with keys corresponding to
            the attribute 'sample_names', and values to strings denoting
            colors. If not given, default colors are assigned.

        labels : dict or None
            Sample labels, should be a dictionary with keys corresponding to
            the attribute 'sample_names', and values to strings. If not given,
            samples are labeled with sample names.

        alpha : float, optional
            The alpha blending value, between 0 (transparent) and 1 (opaque).
            The default is 0.5.

        legend_fontsize : int, optional
            Legend font size. The default is 15.

        font_family : str, optional
            Legend and labels' font family name accepted by matplotlib
            (e.g., 'serif', 'sans-serif', 'monospace', 'fantasy' or
            'cursive'), the default is 'serif'.

        """
        t = trunc
        t_max = trunc_max

        if t is None:
            t = min(t_max, len(self.kstat.sp))
        if t > len(self.kstat.sp):
            raise ValueError(
                f"Value of t has to be at most {len(self.kstat.sp)}."
            )
        if t > t_max:
            t_max = t
        if not self.proj or \
                str(t) not in self.proj[self.sample_names[0]]:
            self.get_projections(n_trunc=t_max)

        if colors is None:
            colors = {
                self.sample_names[0]: 'indigo',
                self.sample_names[1]: 'turquoise'
            }

        rc('font', **{'family': font_family})
        fig, ax = plt.subplots(ncols=1, figsize=(12, 6))
        for name, df_proj in self.proj.items():
            dfxy = df_proj[str(t)]
            min_proj, max_proj = dfxy.min(), dfxy.max()
            min_scaled = min_proj - 0.1 * (max_proj - min_proj)
            max_scaled = max_proj + 0.1 * (max_proj - min_proj)
            x = np.linspace(min_scaled, max_scaled, 200)
            density = gaussian_kde(dfxy, bw_method=.2)
            y = density(x)
            label = labels[name] if labels is not None else name

            ax.plot(x, y, color=colors[name], lw=2)
            ax.fill_between(
                x, y, y2=0, color=colors[name], label=label, alpha=alpha
            )
            axis_label = f'DA{t}'
            ax.set_ylabel(axis_label, fontsize=25)
            ax.legend(fontsize=legend_fontsize)
        # plt.axvline(x=0, linestyle='--')
        ax.set_title('kFDA discriminant axis projection density', fontsize=25)
        return fig, ax

    def scatter_projection(
        self, trunc_x=1, trunc_y=2, proj_xy=['kfda', 'kfda_contrib'],
        trunc_max=100, colors=None, labels=None, alpha=.75,
        legend_fontsize=15, font_family='serif'
    ):
        """
        Plots a scatter of projections, where axes can represent either the
        discriminant axes of the kFDA statistic, or the corresponding
        eigenvector contributions.

        Parameters
        ----------
        trunc_x : int, optional
            Axis for the scatter with respect to axis x,
            the default is 1.

        trunc_y : int, optional
            Axis for the scatter with respect to axis y,
            the default is 1.

        trunc_max : int, optional
            Maximal truncation for projections calculation, the default is 100.

        proj_xy : pair of strings, optional
            Projections to scatter with respect to axes x and y respectively,
            possible values: ['kfda', 'kfda_contrib'] (default) and
            ['kfda_contrib', 'kfda_contrib']. In the first case, the truncation
            for the contribution is automatically assigned as the next one
            after the truncation for the discriminant axis (value of 't_x').

        colors : dict or None
            Sample colors, should be a dictionary with keys corresponding to
            the attribute 'sample_names', and values to strings denoting
            colors. If not given, default colors are assigned.

        labels : dict or None
            Sample labels, should be a dictionary with keys corresponding to
            the attribute 'sample_names', and values to strings. If not given,
            samples are labeled with sample names.

        alpha : float, optional
            The alpha blending value, between 0 (transparent) and 1 (opaque).
            The default is 0.5.

        legend_fontsize : int, optional
            Legend font size. The default is 15.

        font_family : str, optional
            Legend and labels' font family name accepted by matplotlib
            (e.g., 'serif', 'sans-serif', 'monospace', 'fantasy' or
            'cursive'), the default is 'serif'.

        """
        t_x = trunc_x
        t_y = trunc_y
        t_max = trunc_max

        max_t_xy = max(t_x, t_y)
        if max_t_xy > len(self.kstat.sp):
            raise ValueError(
                f"Value of t has to be at most {len(self.kstat.sp)}."
            )
        if max_t_xy > t_max:
            t_max = max_t_xy
        if not self.proj or \
                str(max_t_xy) not in self.proj[self.sample_names[0]]:
            self.get_projections(n_trunc=t_max)
        if proj_xy[0] == 'kfda' and proj_xy[1] == 'kfda_contrib':
            dict_proj_x = self.proj
            dict_proj_y = self.proj_contrib
            t_xy = [t_x, t_x + 1]
        elif proj_xy[0] == 'kfda_contrib' and proj_xy[1] == 'kfda_contrib':
            dict_proj_x = self.proj_contrib
            dict_proj_y = self.proj_contrib
            t_xy = [t_x, t_y]
        else:
            err_txt = "Possible values for 'proj_xy': "
            err_txt += "['kfda_contrib', 'kfda_contrib'], "
            err_txt += "['kfda', 'kfda_contrib']."
            raise ValueError(err_txt)

        if colors is None:
            colors = {
                self.sample_names[0]: 'indigo',
                self.sample_names[1]: 'turquoise'
            }
        rc('font', **{'family': font_family})
        fig, ax = plt.subplots(ncols=1, figsize=(12, 6))
        for name, df_proj in dict_proj_x.items():
            x = df_proj[str(t_xy[0])]
            y = dict_proj_y[name][str(t_xy[1])]
            label = labels[name] if labels is not None else name
            ax.scatter(x, y, s=30, alpha=alpha, facecolors='none',
                       edgecolors=colors[name], lw=2, label=label)
            xaxis_label = (
                f'DA{t_xy[0]}' if proj_xy[0] == 'kfda' else
                f'PC{t_xy[0]}' if proj_xy[0] == 'kfda_contrib' else None
            )
            ax.set_xlabel(xaxis_label, fontsize=25)
            yaxis_label = (
                f'DA{t_xy[1]}' if proj_xy[1] == 'kfda' else
                f'PC{t_xy[1]}' if proj_xy[1] == 'kfda_contrib' else None
            )
            ax.set_ylabel(yaxis_label, fontsize=25)
            ax.legend(fontsize=legend_fontsize)
        ax.set_title('Projection scatter', fontsize=25)
        return fig, ax

    def save(self, file_name, compress=True):
        """
        Save Ktest object to disk in binary format (pickle/dill) possibly
        with Gzip compression.

        If compression is enabled, a `.gz` extension is added to `file_name`
        if not already present.

        Parameters
        ----------
        file_name : str
            absolute or relatative path to file where to store the Ktest
            object.
        compress : bool
            flag to enable/disable compression when saving Ktest object to
            disk.
        """
        # compress result file?
        if compress:
            custom_open = gzip.open
            # add `.gz` extension if needed
            if not file_name.endswith('.gz'):
                file_name = file_name + '.gz'
        else:
            custom_open = open

        with custom_open(file_name, 'wb') as f:
            dill.dump(self, f)

    @staticmethod
    def load(file_name, compressed=True):
        """
        Load Ktest object that was saved on disk in binary format (pickle/dill)
        possibly with Gzip decompression.

        Parameters
        ----------
        file_name : str
            absolute or relatative path to file where the Ktest object is
            stored.
        compressed : bool
            flag to enable/disable decompression when loading Ktest object from
            disk.
        """
        if compressed:
            custom_open = gzip.open
        else:
            custom_open = open

        with custom_open(file_name, 'rb') as f:
            return dill.load(f)
