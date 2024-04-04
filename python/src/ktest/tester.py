import numpy as np
import pandas as pd
from scipy.stats import chi2
import warnings
from tqdm import tqdm
from kernel_statistics import Statistics
from data import Data

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
            the number of landamarks.
    
        landmark_method : 'random' or 'kmeans++', optional
            Method of the landmarks selection, 'random '(default) corresponds 
            to selecting landmarks among the observations according to the 
            random uniform distribution.
    
        anchor_basis : str, optional
            Options for different ways of computing the covariance operator of 
            the landmarks in the Nystrom method, of which the anchors are the 
            eigenvalues. Possible values are 'w' (default),'s' and 'k'.
    
    Attributes 
    ----------
        dataset: 1 or 2-dimensional array-like
            The data assigned to the parameter 'data' is stored here.

        data : instance of class Data
            Contains various information on the original dataset, see the 
            documentation of the class Data for more details.
            
        kstat : instance of class Statistics
            Attribute used for statistics calculation, see the documentation 
            of the class Statistics for more details.
        
        data_nystrom : None or instance of class Data
            None if nystrom==False. Otherwise, contains various information on
            the Nystrom dataset, see the documentation of the class Data 
            for more details.
        
        kfdat : Pandas.Series or None
            None if not computed. Otherwise, stores the computed kFDA statistics.
            Indices correspond to truncations.
    
        pval_kfdat_asymp : Pandas.Series or None
            None if not computed. Otherwise, stores the p-values associated with
            the kFDA statistic (stored in 'kfdat'), obtained from the asymptotic 
            distribution (chi-square with T degree of freedom). Indices 
            correspond to truncations.
        
        pval_kfdat_perm : Pandas.Series or None
            None if not computed. Otherwise, stores the p-values associated with
            the kFDA statistic (stored in 'kfdat'), obtained with a permutation
            approach. Indices correspond to truncations.
                
        kfdat_contrib : Pandas.Series or None
            None if not computed. Otherwise, stores the unidirectional statistic
            associated with each eigendirection of the within-group covariance
            operator. `kfdat` contains the cumulated sum of the values 
            in `kfdat_contributions`. Indices correspond to truncations.
        
        pval_kfdat_contrib : Pandas.Series or None
            None if not computed. Otherwise, stores the asymptotic p-values 
            associated to each unidirectional statistic (chi-square with 1 
            degree of freedom). Indices correspond to truncations.
        
        mmd : float or None
            None if not computed. Otherwise, stores the MMD test statistics.
            
        pval_mmd : float or None
            None if not computed. Otherwise, stores the p-values 
            associated with each MMD statistic (only the permutation approach 
            is considered for MMD).
            
        rnd_gen :  int, RandomState instance or None
            Determines random number generation for the Nystrom approximation
            and for the permutations. If None (default), the generator is 
            the RandomState instance used by `np.random`. To ensure the results
            are reproducible, pass an int to instanciate the seed, or a 
            RandomState instance (recommended).
        
    """

    def __init__(self, data, metadata, sample_names=None,
                 kernel_function='gauss', kernel_bandwidth='median',
                 kernel_median_coef=1, nystrom=False, n_landmarks=None, 
                 landmark_method='random', n_anchors=None, anchor_basis='w', 
                 random_state=None):
        self.dataset = data
        self.metadata = metadata
        self.sample_names = sample_names
        self.data = Data(data=data, metadata=metadata, sample_names=sample_names)
        
        if isinstance(random_state, np.random.RandomState):
            self.rnd_gen = random_state
        elif isinstance(random_state, int):
            self.rnd_gen = np.random.RandomState(random_state)
        else:
            self.rnd_gen = np.random

        ### Kernel:
        self.kernel_function = kernel_function
        self.kernel_bandwidth = kernel_bandwidth
        self.kernel_median_coef = kernel_median_coef
        
        ### Nystrom:
        self.data_nystrom = None
        self.n_landmarks = n_landmarks
        self.landmark_method = landmark_method
        self.n_anchors = n_anchors
        self.anchor_basis=anchor_basis
        
        if nystrom:
            self.data_nystrom = Data(data=data, metadata=metadata, 
                                     sample_names=sample_names,
                                     nystrom=True, n_landmarks=self.n_landmarks, 
                                     landmark_method=self.landmark_method, 
                                     random_state=self.rnd_gen)
        self.kstat = Statistics(self.data, 
                                kernel_function=self.kernel_function,
                                bandwidth=self.kernel_bandwidth,
                                median_coef=self.kernel_median_coef,
                                data_nystrom=self.data_nystrom, 
                                n_anchors=self.n_anchors,
                                anchor_basis=self.anchor_basis)
        
        ### Output statistics ### 
        ## kFDA statistic
        self.kfdat = None
        self.pval_kfdat_asymp = None
        self.pval_kfdat_perm = None
        self.kfdat_contrib = None
        self.pval_kfdat_contrib = None
        
        ## MMD statistic
        self.mmd = None
        self.pval_mmd = None
        

    def __str__(self):
        s = "An object of class Ktest."
        s += self.data.__str__()
        if self.data_nystrom is not None: 
           s += "\nNystrom approximation with"
           s += f" {self.data_nystrom.ntot} landmarks."
           
        s += '\n___Multivariate test results___'
        ncs = 'not computed, run ktest.test.'
        mmd_s = f'{self.mmd}, pvalue (permutation test): {self.pval_mmd}.'
        s += '\nMMD:\n'
        s += f'{mmd_s}' if self.mmd is not None else ncs
        s += '\nkFDA:'
        if self.kfdat is None:
            s += '\n' + ncs
        else:
            for t in range(min(len(self.kfdat), 5)):
                s += f'\nTruncation {t+1}: {self.kfdat.iloc[t]}, pvalue: '
                s += 'asymptotic - '
                s += (f'{self.pval_kfdat_asymp.iloc[t]}' 
                      if self.pval_kfdat_asymp is not None else 'not computed')
                s += ',\n'
                s += 'permutation - '
                s += (f'{self.pval_kfdat_perm.iloc[t]}' 
                      if self.pval_kfdat_perm is not None else 'not computed')
                s += '.'
        return(s)

    def __repr__(self):
        return(self.__str__())
    
    def compute_test_statistic(self, stat='kfda', kstat=None, verbose=0):
        """
        Computes the kFDA or the MMD statistic.

        Parameters
        ----------
        stat : str
            The test statistic to use, can be either 'kfda' (default) or' mmd'. 
        kstat : instance of Statistics or None
            If None (default), the comutations are performed on the 'kstat' 
            attribute. If an instance of Statistics, it is used for computations,
            used to compute test statistics of the permuted samples.
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
        if verbose <= 0:
            warnings.simplefilter("ignore")
        if stat == 'kfda':
            test_res = kstatistics.compute_kfdat()
        elif stat == 'mmd':
            test_res = kstatistics.compute_mmd()
        else:
            raise ValueError(f"Statistic '{stat}' not recognized. Possible values: 'kfda','mmd'")
        return test_res
    
    def compute_pvalue(self, stat='kfda', permutation=False, n_permutations=500,
                       verbose=1):
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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pval = chi2.sf(self.kfdat, self.kfdat.index)
                pval_contrib = chi2.sf(self.kfdat_contrib, self.kfdat_contrib.index)
                return pd.Series(pval), pd.Series(pval_contrib)
        else:
            if verbose > 0:
                print('- Performing permutations to compute p-values:')
            stats_count = (pd.Series(0, index=range(1, len(self.kfdat) + 1)) 
                           if stat == 'kfda' else 0)
            it = tqdm(range(n_permutations)) if verbose > 0 else range(n_permutations)
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
                    data_perm_nystrom = Data(data=self.dataset, metadata=meta_perm, 
                                            sample_names=self.sample_names,
                                            nystrom=True, n_landmarks=self.n_landmarks, 
                                            landmark_method=self.landmark_method, 
                                            random_state=self.rnd_gen)
                kstat_perm = Statistics(data_perm,
                                        kernel_function=self.kernel_function,
                                        bandwidth=self.kernel_bandwidth,
                                        median_coef=self.kernel_median_coef,
                                        data_nystrom=data_perm_nystrom,  
                                        n_anchors=self.n_anchors,
                                        anchor_basis=self.anchor_basis)
                if verbose >= 3:
                    warnings.simplefilter("always")
                perm_stats_res = self.compute_test_statistic(stat=stat, 
                                                             kstat=kstat_perm,
                                                             verbose=verbose-1)
                if stat == 'kfda':
                    stats_count += perm_stats_res[0].ge(self.kfdat)
                elif stat == 'mmd' : 
                    stats_count += (perm_stats_res >= self.mmd)
            return stats_count / n_permutations
   
    def test(self, stat='kfda', permutation=False, n_permutations=500, verbose=1):
        """
        Performs either the MMD or the kFDA test to compare the two samples of
        the considered dataset. Stores the results in the corresponding attributes,
        depending on the statistic and on the p-value approach.

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
        if stat == 'kfda':
            if verbose > 0:
                print('- Computing kFDA statistic')
            (self.kfdat,
             self.kfdat_contrib) = self.compute_test_statistic(verbose=verbose)
            if not permutation:
                (self.pval_kfdat_asymp,
                 self.pval_kfdat_contrib) = self.compute_pvalue(verbose=verbose)
            else:
                self.pval_kfdat_perm = self.compute_pvalue(permutation=permutation, 
                                                           n_permutations=n_permutations,
                                                           verbose=verbose)
        elif stat == 'mmd':
            if verbose > 0:
                print('- Computing MMD statistic')
            self.mmd = self.compute_test_statistic(stat=stat, verbose=verbose)
            self.pval_mmd = self.compute_pvalue(stat=stat, verbose=verbose,
                                                n_permutations=n_permutations)
        else:
            if verbose >0:
                print(f"Statistic '{stat}' not recognized. Possible values : 'kfda','mmd'")

