import numpy as np
import pandas as pd
import torch as to
from sklearn.cluster import kmeans_plusplus


class Data(object):
    """
    Class transforming two-sample data provided by the user into the quantities
    compatible with the kernel test statistic calculations performed by the
    class Statistics.

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

        nystrom : bool, optional
            If True, computes the Nystrom landmarks, in which case all
            attributes correspond to the landmarks and not the original data.
            The default is False.

        n_landmarks: int, optional
            Number of landmarks used in the Nystrom method. If unspecified, one
            fifth of the observations are selected as landmarks.

        landmark_method : 'random' or 'kmeans++', optional
            Method of the landmarks selection, 'random '(default) corresponds
            to selecting landmarks among the observations according to the
            random uniform distribution.

        random_state :  int, numpy.random.Generator instance,
                numpy.random.RandomState instance or None
            Determines random number generation for the landmarks selection.
            If None (default), the default numpy random generator is used.
            To ensure that the results are reproducible, pass an integer
            that will be used as seed for the random number generation, or
            a Numpy random Generator instance (recommended), or a Numpy
            RandomState instance (legacy).

        dtype : torch.dtype, optional
            Floating point number type/precision used for number storage and
            computations. Default is `torch.float64`.

        safe_subsample : bool
            Implement safe subsampling for Nystrom approximation, i.e. check
            that not all variables are constant in subsampled data. If not,
            subsampling is redone for at most `n_subsample_trial` times.
            Default is `True`.

        n_subsample_trial : integer, optional
            Number of times to retry subsampling in case all columns
            have constant values after subsampling. Default is `100`.

        verbose : int | bool, optional
            Verbosity level, see `ktest.utils.verbosity()` for more details.
            Default is `0` and no verbosity output.

    Attributes
    ----------
        sample_names : 1-dimensional array-like
            Identical to the corresponding parameter if provided by the user.
            Otherwise, the first two sample labels detected from metadata.

        data : dict
            Keys correspond to sample names, values are two torch.tensors
            corresponding to the data extracted for each sample.

        index : dict
            Keys correspond to sample names, values are two array-likes
            corresponding to observation labels either extracted from the data
            or asigned automatically.

        variables : 1-dimensional array-like
            Array-like of size nvar corresponding to variable (feature) labels
            either extracted from the data or asigned automatically.

        nobs : dict
            Keys correspond to sample names, values are two integers
            corresonding to the numbers of observations in each sample.

        nvar : int
            Number of variables in the dataset.

        ntot : int
            Total number of variables in the dataset (sum of nobs).

        dtype : torch.dtype
            Floating point number type/precision used for number storage and
            computations. Default is `torch.float64`.

        verbose : int | bool, optional
            Verbosity level, see `ktest.utils.verbosity()` for more details.
            Default is `0` and no verbosity output.

    """

    def __init__(
        self, data, metadata, sample_names=None, nystrom=False,
        n_landmarks=None, landmark_method='kmeans++', random_state=None,
        dtype=to.float64, safe_subsample=True, n_subsample_trial=100,
        verbose=0
    ):
        # init
        self.data = {}
        self.index = {}
        self.nobs = {}

        # dtype
        self.dtype = dtype

        # Nystrom subsampling setup
        self.verbose = verbose

        # process data
        self._process_data(
            data, metadata, sample_names, nystrom,
            n_landmarks, landmark_method, random_state,
            safe_subsample, n_subsample_trial
        )


    def __str__(self):
        s = f"\n{self.nvar} features across {self.ntot} observations"
        s += "\nComparison: "
        s += f"{self.sample_names[0]} "
        s += f"({self.nobs[self.sample_names[0]]} observations)"
        s += f" and {self.sample_names[1]} "
        s += f"({self.nobs[self.sample_names[1]]} observations)."
        return s

    def _process_data(
        self, data, metadata, sample_names, nystrom,
        n_landmarks, landmark_method, random_state,
        safe_subsample, n_subsample_trial
    ):
        """
        Process input data when initializing object.

        Note: internal function, see `ktest.data.Data` documentation for
        input argument description.
        """

        try:
            assert len(data.squeeze().shape) <= 2, \
                'Data has to be at most 2-dimensional'
            assert len(metadata.squeeze().shape) == 1, \
                'Metadata has to be 1-dimensional'
            if not (hasattr(data, 'index') or hasattr(metadata, 'index')):
                assrt_str_1 = 'If index is not provided, ' + \
                    'data and metadata have to have the same first dimension.'
                assert data.shape[0] == metadata.shape[0], assrt_str_1
            if isinstance(metadata, pd.DataFrame):
                meta_fmt = metadata[metadata.columns[0]]
            else:
                meta_fmt = metadata.copy()
            if sample_names is not None:
                levels = sample_names
                assrt_str_2 = 'Sample names must be present in metadata'
                assert np.isin(levels, meta_fmt.values).all(), assrt_str_2
                assrt_str_3 = 'Exactly 2 sample names are expected'
            else:
                levels = (meta_fmt.unique()[:2] if hasattr(meta_fmt, 'unique')
                          else np.unique(metadata)[:2])
                assrt_str_3 = 'Metadata has to include at least ' + \
                    '2 distinct values for 2-sample test.'
            assert len(levels) == 2, assrt_str_3

            self.sample_names = sample_names if sample_names is not None \
                else levels
            self.data[self.sample_names[0]] = data[meta_fmt == levels[0]]
            self.data[self.sample_names[1]] = data[meta_fmt == levels[1]]

            self.nvar = data.shape[1] if len(data.squeeze().shape) == 2 else 1
            self.variables = (data.columns if hasattr(data, 'columns')
                              else np.arange(self.nvar))

            for n, data_n in self.data.items():
                self.nobs[n] = data_n.shape[0]
                self.index[n] = (data_n.index if hasattr(data_n, 'index')
                                 else np.arange(self.nobs[n]))

                # Convert data to tensor:
                if isinstance(data_n, pd.Series):
                    self.data[n] = \
                        to.from_numpy(
                            data_n.to_numpy().reshape(-1, 1)
                        ).type(self.dtype)
                elif isinstance(data_n, pd.DataFrame):
                    self.data[n] = to.from_numpy(
                        data_n.to_numpy()
                    ).type(self.dtype)
                elif isinstance(data_n, to.Tensor):
                    self.data[n] = data_n.type(self.dtype)
                else:
                    X = data_n.to_numpy() \
                        if not isinstance(data_n, np.ndarray) else \
                        data_n.copy()
                    self.data[n] = to.from_numpy(X).type(self.dtype)

        except AttributeError as e:
            print(f'Unknown data type {type(data_n)}')
            raise e

        # variance check (if required)
        if (
            safe_subsample and
            not to.any(to.var(to.cat(list(self.data.values())), dim=0) > 0)
        ):
            msg = "All variables have constant values in your data."
            raise RuntimeError(msg)

        # total retained observations in selected subsamples
        self.ntot = sum(self.nobs.values())

        # Nystrom
        if nystrom:
            self._subsample(
                n_landmarks, landmark_method, random_state, safe_subsample,
                n_subsample_trial
            )

    def _subsample(
        self, n_landmarks, landmark_method, random_state, safe_subsample,
        n_subsample_trial
    ):
        """
        Subsampling data for Nystrom approximation.

        Note: internal function, see `ktest.data.Data` documentation for
        input argument description.
        """
        # random number generation
        if random_state is None:
            random_state = np.random.default_rng()
        elif isinstance(random_state, int):
            random_state = np.random.default_rng(random_state)
        else:
            assert isinstance(random_state, np.random.Generator) or \
                isinstance(random_state, np.random.RandomState)
            random_state = random_state

        # check variance in subsample to avoid any constant column
        # and re-do subsampling if needed
        for counter in range(n_subsample_trial):

            # iterate through samples
            for n, data_n in self.data.items():

                # number for landmarks in sample
                n_landmarks_n = (
                    min(
                        n_landmarks * self.nobs[n] // self.ntot,
                        self.nobs[n]
                    ) if n_landmarks is not None else self.nobs[n] // 5
                )

                if landmark_method == 'random':
                    ny_ind = random_state.choice(
                        self.nobs[n],
                        size=n_landmarks_n,
                        replace=False
                    )
                elif landmark_method == 'kmeans++':
                    rnd_st = random_state \
                        if isinstance(
                            random_state, (np.random.RandomState, int)
                        ) else None
                    _, ny_ind = kmeans_plusplus(
                        self.data[n].numpy(),
                        n_clusters=n_landmarks_n,
                        random_state=rnd_st
                    )
                else:
                    raise ValueError("unsupported 'landmark_method'")

                # save subsamping
                self.data[n] = self.data[n][ny_ind]
                self.nobs[n] = n_landmarks_n
                self.index[n] = self.index[n][ny_ind]

            # variance check (if required)
            if (
                not safe_subsample or
                to.any(to.var(to.cat(list(self.data.values())), dim=0) > 0)
            ):
                break
            elif counter == n_subsample_trial - 1:
                msg = " ".join([
                    "Subsampling failed after",
                    f"{n_subsample_trial} trials.",
                    "All variables have constant values."
                ])
                raise RuntimeError(msg)

        # update total retained observations in selected subsamples
        self.ntot = sum(self.nobs.values())
