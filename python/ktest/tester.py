import numpy as np
import pandas as pd
from scipy.stats import chi2
import warnings
from kernel_statistics import Statistics
from data import Data


# tracer l'evolution des corrélations par rapport aux gènes ordonnés

class Ktest(Statistics):
    """
    Ktest is a class that performs kernels tests such that MMD and the test based on Kernel Fisher Discriminant Analysis. 
    It also provides a range of visualisations based on the discrimination between two groups.  
    
    Attributes: 
    ------------------------
    data: instance of class Data
        data matrix initialized from data 
        
    kstat: instance of class Statistics

    df_kfdat: pandas.DataFrame
        stores the computed kfda statistics and
        rows: truncations
        columns: each performed test with specific parameters and samples 

    df_pval: pandas.DataFrame
        stores the p-values associated to the kfda statistic
        obtained from the asymptotic distribution (chi-square with T 
        degree of freedom) or a permutation approach. 
        rows: truncations
        columns: each performed test with specific parameters and samples
    
      
    df_kfdat_contributions: pandas.DataFrame
        stores the unidirectional statistic associated to each 
        eigendirection of the within-group covariance operator
        `df_kfdat` contains the cumulated sum of the values 
        in  `df_kfdat_contributions`
    
    df_pval_contributions: pandas.DataFrame
        stores the asymptotic or permutation p-values associated to each unidirectional 
        statistic (chi-square with 1 degree of freedom) 
    
    dict_mmd: dict 
        store the MMD statistics associated to each performed test. 
    """

    def __init__(self, data, metadata, sample_names=None,
                 kernel_function='gauss', kernel_bandwidth='median',
                 kernel_median_coef=1, verbose=0):
        """
        Generate a Ktest object and compute specified comparison

        Parameters
        ----------
            data : Pandas.DataFrame 
                the data dataframe
            
            metadata : Pandas.DataFrame 
                the metadata dataframe
                
            sample_names: 
                

            stat (default = 'kfda') : str in ['kfda','mmd']
                The test statistic to use. If `stat == mmd`, `permutation` is set to `True`
                    
            kernel_function (default = 'gauss') : function or str in ['gauss','linear','fisher_zero_inflated_gaussian','gauss_kernel_mediane_per_variable'] 
                if str : specifies the kernel function
                if function : kernel function specified by user

            kernel_bandwidth (default = 'median') : 'median' or float
                value of the bandwidth for kernels using a bandwidth
                if 'median' : the bandwidth will be set as the median or a multiple of it is
                    according to the value of parameter `median_coef`
                if float : value of the bandwidth

            kernel_median_coef (default = 1) : float
                multiple of the median to use as bandwidth if bandwidth=='median' 
        """
        self.dataset = data
        self.metadata = metadata
        self.sample_names = sample_names
        
        ### Kernel:
        self.kernel_function = kernel_function
        self.kernel_bandwidth = kernel_bandwidth
        self.kernel_median_coef = kernel_median_coef
        
        self.data = Data(data=data, metadata=metadata, sample_names=sample_names)
        self.kstat = Statistics(self.data, kernel_function=kernel_function,
                                bandwidth=kernel_bandwidth,
                                median_coef=kernel_median_coef)
        
        ### Output statistics ### 
        ## kFDA statistic
        self.kfdat = None
        self.pval_kfdat = None
        self.pval_kfdat_perm = None
        self.kfdat_contrib = None
        self.pval_kfdat_contrib = None
        
        ## MMD statistic
        self.mmd = None
        self.pval_mmd = None
        

    def compute_test_statistic(self, stat='kfda', kstat=None, unbiaised=False, 
                               verbose=0):
        """"
        Compute the kfda or the MMD statistic from scratch. 
        Compute every needed intermediate quantity. 
        Return the name of the column containing the statistics in dataframe `ktest.df_kfdat`.

        Parameters
        ----------
            verbose (default = 0) : int 
                The greater, the more verbose. 
        """        
        kstatistics = kstat if kstat is not None else self.kstat
        if stat == 'kfda':
            test_res = kstatistics.compute_kfdat(verbose=verbose)
        elif stat == 'mmd':
            test_res = kstatistics.compute_mmd(unbiaised=unbiaised,
                                                     verbose=verbose)
        return test_res
    
    def compute_pvalue(self, stat='kfda', permutation=False, n_permutations=500,
                       random_state=None):
        """
        Computes the p-value of the statistic of `stat`.         
        """
        if stat == 'kfda' and not permutation:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pval = chi2.sf(self.kfdat, self.kfdat.index)
                pval_contrib = chi2.sf(self.kfdat_contrib, self.kfdat_contrib.index)
                return pval, pval_contrib
        else:
            if isinstance(random_state, np.random.RandomState):
                rnd_gen = random_state
            elif isinstance(random_state, int):
                rnd_gen = np.random.RandomState(random_state)
            else:
                rnd_gen = np.random
            stats_count = (pd.Series(0, index=range(1, self.data.ntot + 1)) 
                           if stat == 'kfda' else 0)
            for i in range(n_permutations):
                meta_perm = self.metadata.copy()
                if isinstance(meta_perm, (pd.DataFrame, pd.Series)):
                    rnd_gen.shuffle(meta_perm.values)
                else:
                    rnd_gen.shuffle(meta_perm)
                data_perm = Data(data=self.dataset, metadata=meta_perm, 
                                 sample_names=self.sample_names)
                kstat_perm = Statistics(data_perm, 
                                        kernel_function=self.kernel_function,
                                        bandwidth=self.kernel_bandwidth,
                                        median_coef=self.kernel_median_coef)
                perm_stats_res = self.compute_test_statistic(stat=stat, 
                                                             kstat=kstat_perm)
                if stat == 'kfda':
                    stats_count += (perm_stats_res[0] >= self.kfdat)
                elif stat == 'mmd' : 
                    stats_count += (perm_stats_res >= self.mmd)
            return stats_count / n_permutations
   
    def multivariate_test(self, stat='kfda', permutation=False, 
                          n_permutations=500, verbose=0):
        if stat == 'kfda':
            (self.kfdat,
             self.kfdat_contrib) = self.compute_test_statistic(verbose=verbose)
            if not permutation:
                (self.pval_kfdat,
                 self.pval_kfdat_contrib) = self.compute_pvalue()
            else:
                self.pval_kfdat_perm = self.compute_pvalue(permutation=permutation, 
                                                           n_permutations=n_permutations)
        elif stat == 'mmd':
            self.mmd = self.compute_test_statistic(stat=stat, verbose=verbose)
            self.pval_mmd = self.compute_pvalue(stat=stat, 
                                                n_permutations=n_permutations)
        else:
            if verbose >0:
                print(f"Statistic '{stat}' not recognized. Possible values : 'kfda','mmd'")

        #if verbose>0:
        #    self.print_multivariate_test_results(long=False)
