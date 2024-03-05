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
        self.data = Data(data=data, metadata=metadata, sample_names=sample_names)
        self.kstat = Statistics(self.data, kernel_function=kernel_function,
                                bandwidth=kernel_bandwidth,
                                median_coef=kernel_median_coef)
        
        ### Output statistics ### 
        self.kfdat = None
        self.pval = None
        self.kfdat_contrib = None
        self.pval_contrib = None
        
        ## MMD statistic (and p-value...?)
        self.mmd = None
        

    def compute_test_statistic(self, stat='kfda', unbiaised=False, verbose=0):
        """"
        Compute the kfda or the MMD statistic from scratch. 
        Compute every needed intermediate quantity. 
        Return the name of the column containing the statistics in dataframe `ktest.df_kfdat`.

        Parameters
        ----------
            verbose (default = 0) : int 
                The greater, the more verbose. 
        """        
        if stat == 'kfda' and self.kfdat is None:
            self.kfdat, self.kfdat_contrib = self.kstat.compute_kfdat(verbose=verbose)
        elif stat == 'mmd' and self.mmd is None:
            self.mmd = self.kstat.compute_mmd(unbiaised=unbiaised, verbose=0)
        else:
            if verbose >0:
                print(f"Statistic '{stat}' not recognized. Possible values : 'kfda','mmd'")
    
    def compute_pvalue(self, stat='kfda'):
         """
         Computes the p-value of the statistic of `stat`.         
         """
         if stat == 'kfda':
             with warnings.catch_warnings():
                 warnings.simplefilter("ignore")
                 self.pval = chi2.sf(self.kfdat, self.kfdat.index)
                 self.pval_contrib = chi2.sf(self.kfdat_contrib, self.kfdat_contrib.index)
         else:
             raise NotImplementedError('MMD statistic p-value requires a permutation-based approach, which has not been fixed yet.')

    def multivariate_test(self, stat='kfda', verbose=0,):
        
        
        self.compute_test_statistic(stat=stat,verbose=verbose)
        self.compute_pvalue(stat=stat)

        #if verbose>0:
        #    self.print_multivariate_test_results(long=False)

