import numpy as np
import pandas as pd
from scipy.stats import chi2, gaussian_kde
import matplotlib.pyplot as plt
from matplotlib import rc
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
        
        kfda_statistic : Pandas.Series or None
            None if not computed. Otherwise, stores the computed kFDA statistics.
            Indices correspond to truncations.
    
        kfda_pval_asymp : Pandas.Series or None
            None if not computed. Otherwise, stores the p-values associated with
            the kFDA statistic (stored in 'kfda_statistic'), obtained from the asymptotic 
            distribution (chi-square with T degree of freedom). Indices 
            correspond to truncations.
        
        kfda_pval_perm  : Pandas.Series or None
            None if not computed. Otherwise, stores the p-values associated with
            the kFDA statistic (stored in 'kfda_statistic'), obtained with a permutation
            approach. Indices correspond to truncations.
                
        kfda_statistic_contrib : Pandas.Series or None
            None if not computed. Otherwise, stores the unidirectional statistic
            associated with each eigendirection of the within-group covariance
            operator. `kfda_statistic` contains the cumulated sum of the values 
            in `kfda_contributions`. Indices correspond to truncations.
        
        mmd_statistic : float or None
            None if not computed. Otherwise, stores the MMD test statistics.
            
        mmd_pval_perm : float or None
            None if not computed. Otherwise, stores the p-values 
            associated with each MMD statistic (only the permutation approach 
            is considered for MMD).
            
        kfda_proj : pandas.DataFrame
            Projections of the embeddings on the discriminant axis 
            corresponding to the KFDA statistic, associated with every 
            observation (rows) on every eigendirection (columns).
            
        within_kpca_proj : pandas.DataFrame
            Contributions of each eigendirection (columns) to projections of 
            the embeddings on the discriminant axis corresponding to the 
            KFDA statistic, associated with every observation (rows). 
            'kfda_proj' contains the cumulated sum of the values in 'kpca_proj'.
            
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
        self.data = Data(data=data, metadata=metadata, sample_names=sample_names)
        self.sample_names = self.data.sample_names
        
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
        self.kfda_statistic = None
        self.kfda_pval_asymp = None
        self.kfda_pval_perm  = None
        self.kfda_statistic_contrib = None
        
        ## MMD statistic
        self.mmd_statistic = None
        self.mmd_pval_perm = None
        
        ### Projections:
        self.kfda_proj = {}
        self.within_kpca_proj = {}
        

    def __str__(self):
        s = "An object of class Ktest."
        s += self.data.__str__()
        if self.data_nystrom is not None: 
           s += "\nNystrom approximation with"
           s += f" {self.data_nystrom.ntot} landmarks."
           
        s += '\n___Multivariate test results___'
        ncs = 'not computed, run ktest.test.'
        mmd_s = f'{self.mmd_statistic}, pvalue (permutation test): {self.mmd_pval_perm}.'
        s += '\nMMD:\n'
        s += f'{mmd_s}' if self.mmd_statistic is not None else ncs
        s += '\nkFDA:'
        if self.kfda_statistic is None:
            s += '\n' + ncs
        else:
            for t in range(min(len(self.kfda_statistic), 5)):
                s += f'\nTruncation {t+1}: {self.kfda_statistic.iloc[t]}. P-value:'
                s += '\n'
                s += 'asymptotic: '
                s += (f'{self.kfda_pval_asymp.iloc[t]}' 
                      if self.kfda_pval_asymp is not None else 'not computed')
                s += ', '
                s += 'permutation: '
                s += (f'{self.kfda_pval_perm .iloc[t]}' 
                      if self.kfda_pval_perm  is not None else 'not computed')
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
            test_res = kstatistics.compute_kfda()
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
                pval = chi2.sf(self.kfda_statistic, self.kfda_statistic.index)
                return pd.Series(pval)
        else:
            if verbose > 0:
                print('- Performing permutations to compute p-values:')
            stats_count = (pd.Series(0, index=range(1, len(self.kfda_statistic) + 1)) 
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
                    stats_count += perm_stats_res[0].ge(self.kfda_statistic)
                elif stat == 'mmd' : 
                    stats_count += (perm_stats_res >= self.mmd_statistic)
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
            (self.kfda_statistic,
             self.kfda_statistic_contrib) = self.compute_test_statistic(verbose=verbose)
            if not permutation:
                self.kfda_pval_asymp = self.compute_pvalue(verbose=verbose)
            else:
                self.kfda_pval_perm  = self.compute_pvalue(permutation=permutation, 
                                                           n_permutations=n_permutations,
                                                           verbose=verbose)
        elif stat == 'mmd':
            if verbose > 0:
                print('- Computing MMD statistic')
            self.mmd_statistic = self.compute_test_statistic(stat=stat, verbose=verbose)
            self.mmd_pval_perm = self.compute_pvalue(stat=stat, verbose=verbose,
                                                n_permutations=n_permutations)
        else:
            if verbose >0:
                print(f"Statistic '{stat}' not recognized. Possible values : 'kfda','mmd'")
                
    def plot_density(self, t=None, t_max=100, colors=None, labels=None, alpha=.5, 
                     legend_fontsize=15):
        """
        Plots a density of the projection on either the discriminant axes of 
        the kFDA statistic.

        Parameters
        ----------
        t : int, optional
            Truncation to plot, by default equal to the maximal truncation.
            
        t_max : int, optional
            Maximal truncation for projections calculation, the default is 100.
        
        colors : dict or None
            Sample colors, should be a dictionary with keys corresponding to the
            attribute 'sample_names', and values to strings denoting colors. 
            If not given, default colors are assigned.
        
        labels : dict or None
            Sample labels, should be a dictionary with keys corresponding to the
            attribute 'sample_names', and values to strings. If not given, 
            samples are labeled with sample names.
        
        alpha : float, optional
           The alpha blending value, between 0 (transparent) and 1 (opaque).
           The default is 0.5.
        
        legend_fontsize : int, optional
            Legend font size. The default is 15.

        """
        if t is None:
            t = t_max
        if t > len(self.kstat.sp):
            raise ValueError(f"Value of t has to be at most {len(self.kstat.sp)}.")
        if t > t_max:
            t_max = t
        if not self.kfda_proj or str(t) not in self.kfda_proj[self.sample_names[0]]:
            self.kfda_proj, self.within_kpca_proj = self.kstat.compute_projections(t_max)

        if colors is None:
            colors = {self.sample_names[0] : 'indigo', self.sample_names[1] : 'turquoise'}
        
        rc('font',**{'family':'serif','serif':['Times']})
        fig, ax = plt.subplots(ncols=1, figsize=(12,6))
        for name, df_proj in self.kfda_proj.items():
            dfxy = df_proj[str(t)]
            min_proj, max_proj = dfxy.min(), dfxy.max()
            min_scaled = min_proj - 0.1 * (max_proj - min_proj)
            max_scaled = max_proj + 0.1 * (max_proj - min_proj)
            x = np.linspace(min_scaled, max_scaled, 200)
            density = gaussian_kde(dfxy, bw_method=.2)
            y = density(x)
            label = labels[name] if labels is not None else name
            
            ax.plot(x, y, color=colors[name], lw=2)
            ax.fill_between(x, y, y2=0, color=colors[name], label=label, alpha=alpha)
            axis_label = f'DA{t}'
            ax.set_ylabel(axis_label, fontsize=25)
            ax.legend(fontsize=legend_fontsize)
        #plt.axvline(x=0, linestyle='--')
        ax.set_title('kFDA discriminant axis projection density', fontsize=25)
        return(fig,ax)
    
    def scatter_projection(self, t_x=1, t_y=2, proj_xy=['kfda', 'within_kpca'],
                           t_max=100, colors=None, labels=None, alpha=.75,
                           legend_fontsize=15):
        """
        Plots a scatter of projections, where axes can represent either the 
        discriminant axes of the kFDA statistic, or the corresponding 
        eigenvector contributions (kPCA).

        Parameters
        ----------
        t_x : int, optional
            Truncation for the scatter with respect to axis x, 
            the default is 1.
            
        t_y : int, optional
            Truncation for the scatter with respect to axis y, 
            the default is 1.
        
        t_max : int, optional
            Maximal truncation for projections calculation, the default is 100.
            
        proj_xy : pair of strings, optional
            Projections to scatter with respect to axes x and y respectively, 
            possible values: ['kfda', 'within_kpca'] (default) and 
            ['within_kpca', 'within_kpca']. In the first case, the truncation for
            the within kPCA is automatically assigned as the next one after 
            the truncation for the discriminant axis (value of 't_x').
        
        colors : dict or None
            Sample colors, should be a dictionary with keys corresponding to the
            attribute 'sample_names', and values to strings denoting colors. 
            If not given, default colors are assigned.
        
        labels : dict or None
            Sample labels, should be a dictionary with keys corresponding to the
            attribute 'sample_names', and values to strings. If not given, 
            samples are labeled with sample names.
        
        alpha : float, optional
           The alpha blending value, between 0 (transparent) and 1 (opaque).
           The default is 0.5.
        
        legend_fontsize : int, optional
            Legend font size. The default is 15.

        """
        max_t_xy = max(t_x, t_y)
        if max_t_xy > len(self.kstat.sp):
            raise ValueError(f"Value of t has to be at most {len(self.kstat.sp)}.")
        if max_t_xy > t_max:
            t_max = max_t_xy
        if not self.kfda_proj or str(max_t_xy) not in self.kfda_proj[self.sample_names[0]]:
            self.kfda_proj, self.within_kpca_proj = self.kstat.compute_projections(t_max)
        if proj_xy[0] == 'kfda' and proj_xy[1] == 'within_kpca':
            dict_proj_x = self.kfda_proj
            dict_proj_y = self.within_kpca_proj
            t_xy=[t_x, t_x + 1]
        elif proj_xy[0] == 'within_kpca' and proj_xy[1] == 'within_kpca':
            dict_proj_x = self.within_kpca_proj
            dict_proj_y = self.within_kpca_proj
            t_xy=[t_x, t_y]
        else:
            err_txt = "Possible values for 'proj_xy': "
            err_txt += "['within_kpca', 'within_kpca'], ['kfda', 'within_kpca']."
            raise ValueError(err_txt)
            
        if colors is None:
            colors = {self.sample_names[0] : 'indigo', self.sample_names[1] : 'turquoise'}
        rc('font',**{'family':'serif','serif':['Times']})
        fig, ax = plt.subplots(ncols=1, figsize=(12,6))
        for name, df_proj in dict_proj_x.items():
            x = df_proj[str(t_xy[0])]
            y = dict_proj_y[name][str(t_xy[1])]
            label = labels[name] if labels is not None else name
            ax.scatter(x, y, s=30, alpha=alpha, facecolors='none', 
                       edgecolors=colors[name], lw=2, label=label)
            xaxis_label = (f'DA{t_xy[0]}' if proj_xy[0] == 'kfda' else 
                           f'PC{t_xy[0]}' if proj_xy[0] == 'within_kpca' else None)
            ax.set_xlabel(xaxis_label, fontsize=25)
            yaxis_label = (f'DA{t_xy[1]}' if proj_xy[1] == 'kfda' else 
                           f'PC{t_xy[1]}' if proj_xy[1] == 'within_kpca' else None)
            ax.set_ylabel(yaxis_label, fontsize=25)
            ax.legend(fontsize=legend_fontsize)
        ax.set_title('Projection scatter', fontsize=25)
        return(fig,ax)

