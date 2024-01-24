from typing_extensions import Literal
from typing import Optional,Callable,Union,List

import numpy as np
from numpy.lib.function_base import kaiser
import pandas as pd
import torch
from torch import mv
import os

from scipy.linalg import svd
from numpy.random import permutation
from sklearn.model_selection import train_test_split


# CCIPL 
# from kernels import gauss_kernel_mediane
# Local 

from .kernel_statistics import Statistics
from .pvalues import Pvalues
from .save_data import SaveData
from .plots_univariate import Plot_Univariate
from .plots_summarized import Plot_Summarized
from .permutation import Permutation
from .correlation_operations import Correlations
from .hotelling_lawley import Hotelling_Lawley
from .utils import init_test_params
from .kernel_function import init_kernel_params

# tracer l'evolution des corrélations par rapport aux gènes ordonnés

class Ktest(Plot_Univariate,SaveData,Pvalues,Correlations,Permutation,Hotelling_Lawley):
    """
    Ktest is a class that performs kernels tests such that MMD and the test based on Kernel Fisher Discriminant Analysis. 
    It also provides a range of visualisations based on the discrimination between two groups.  
    """

    def __init__(self,
                data,
                metadata,
                condition,
                data_name='data',
                samples='all',
                var_metadata=None,
                stat='kfda',
                nystrom=False,
                permutation=False,
                n_permutations=500,
                seed_permutation=0,
                test_params=None,
                marked_obs_to_ignore=None,
                kernel=None,
                verbose=0):

        """
        Generate a Ktest object and compute specified comparison

        Parameters
        ----------
            data : Pandas.DataFrame 
                the data dataframe
                
            metadata : Pandas.DataFrame 
                the metadata dataframe

            condition : str
                column of the metadata dataframe containing the labels to test

            data_name (default = 'data'): str 
                Name of the data (examples : counts, normalized)

            samples (default = 'all') : str or iterable of str
                'all' : test the two categories contained in column condition (does not work for more than two yet)
                str : category of condition to compare to others as one vs all
                iterable of str : iterable of the two categories to compare to each other
                
            var_metadata (default = None): Pandas.DataFrame 
                the metainformation about the variables, index must correspond to data columns

            stat (default = 'kfda') : str in ['kfda','mmd']
                The test statistic to use. If `stat == mmd`, `permutation` is set to `True`

            permutation (default = False) : bool
                Whether to compute permutation or asymptotic p-values.
                Set to `True` if `stat == mmd`

            n_permutations (default = 500) : int
                Number of permutation needed to compute the permutation p-value.
                Ignored if permutation is False. 
            
            seed_permutation (default = 0) : int
                Seed of the first permutation. The other seeds are obtained by incrementation. 

            nystrom (default = False) : boolean
                if True, the test_params nystrom is set to True with default parameters

            test_params : object specifying the test to execute
                output of the function get_test_params()

            kernel : object specifying the kernel function to use
                output of the function init_kernel_params()

            marked_obs_to_ignore (default = None) : str
                column of the metadata containing booleans that are 
                True if the observation should be ignored from the analysis 
                and False otherwise.    
        """
        
        super(Ktest,self).__init__()
        if kernel is None:
            kernel = init_kernel_params()

        if test_params is None:
            test_params = init_test_params(
                                stat=stat,
                                nystrom=nystrom,
                                permutation=permutation,
                                n_permutations=n_permutations,
                                seed_permutation=seed_permutation,
                                )


        self.add_data_to_Ktest_from_dataframe(data,metadata,data_name=data_name,var_metadata=var_metadata,verbose=verbose)
        self.set_test_data_info(data_name=data_name,condition=condition,samples=samples,change_kernel=False,verbose=verbose)
        self.init_kernel(verbose=verbose,**kernel)
        self.kernel_specification = kernel
        self.set_test_params(verbose=verbose,**test_params)
        self.test_params_initial=test_params
        self.set_marked_obs_to_ignore(marked_obs_to_ignore=marked_obs_to_ignore,verbose=verbose)
        

    # Print outputs 
    def str_add_test_info(self,s,long=False):

        # if long:
        #     s+=f'\n'

        tp = self.get_test_params()
        stat = tp['stat']
        ny = tp['nystrom']
        s+=f'\nTest statistic : {stat}'
        
        if ny: 
            s+= f" with the nystrom approximation ({tp['n_anchors']} anchors)"

            if long:
                s += f"\n\t{tp['n_landmarks']} landmarks"
                lm = tp['landmark_method']
                ab = tp['anchor_basis']
                if lm == 'random':
                    s += f" (determined through random sampling)"
                if lm == 'kmeans':
                    s += f" (determined as kmeans centroids of each group)"
                
                s += f"\n\t{tp['n_anchors']} anchors"
                if ab == 'w':
                    s += " (the eigenvectors of the within group covariance operator of the landmarks)"
                elif ab == 'k':
                    s += " (the eigenvectors of the empirical second moment of the landmarks)"
                elif ab == 's':
                    s += " (the eigenvectors of the total covariance of the landmarks)"
        return(s)

    def str_add_data_info(self,s,long=False):

        nfeatures = self.data[self.data_name]['p']
        nobs = self.get_ntot()
        nassays = len(self.data)
        dn = self.get_nobs()
        dn.pop('ntot')

        # if long:
        #     s+='\n'
        s+=f"\n{nfeatures} features accros {nobs} observations within {nassays} assays (active : {self.data_name})"
        s+= f"\nComparison : "
        for k,n in list(dn.items())[:-1]:
            s+=f"{k} ({n} obs), "
        s=s[:-2]
        k,n = list(dn.items())[-1]
        s+= f" and {k} ({n} obs) "
        s+= f"(from column '{self.condition}' in metadata)"
        

        if long:
            if len(self.data)>1:
                assays = list(self.data.keys())
                s+= f"\nAssays : "
                for assay in assays[:-1]:
                    s+= "{self.data_name}, "
                s=s[:-2]
                s+=" and {assays[-1]}"

            if self.marked_obs_to_ignore is not None:
                s+=f"\nIgnoring cells in {self.marked_obs_to_ignore}"
        return(s)

    def str_add_multivariate_stat_and_pval(self,s,t):
        name_pval=self.get_pvalue_name()
        name_kfdat = self.get_kfdat_name()
        if name_pval in self.df_pval and name_kfdat in self.df_kfdat:
            pval = self.df_pval[name_pval]
            kfda = self.df_kfdat[name_kfdat]
            if t not in pval:
                s+= f"\n\t p-value and statistic not computed for t={t}"
            else:
                pval=pval[t]
                kfda=kfda[t]
                if pval>.01:
                    s+=f"\n\tp-value({t}) = {pval:.2f}"
                else:
                    s+=f"\n\tp-value({t}) = {pval:1.1e}"
                if kfda<1000:
                    s+=f" (kfda={kfda:.2f})"
                else :
                    s+=f" (kfda={kfda:.2e})"
        else:
            s+='p-value and KFDA statistic not computed'        
        return(s)

    def str_add_multivariate_test_results(self,s,long=False,t=10,ts=[1,5,10],stat=None,permutation=None):
        if long:
            s+='\n'        

        stat = self.stat if stat is None else stat 
        permutation = self.permutation if permutation is None else permutation
        pval_name = self.get_pvalue_name(stat=stat,permutation=permutation)
        nystr = ' with nystrom' if self.nystrom else ''


        s+=f'\n___Multivariate {stat}{nystr} test results___'

        if stat == 'mmd':
            pval_computed = pval_name in self.dict_pval_mmd
        elif stat == 'kfda':
            pval_computed = pval_name in self.df_pval

        if not pval_computed:
            s+="\nMultivariate test not performed yet, run ktest.multivariate_test()"
        else:
            sasymp = f"Permutation ({self.n_permutations} permutations)" if permutation else f"Asymptotic"
            
            if stat=='mmd':
                pval = self.dict_pval_mmd[pval_name]
                s+=f"\n{sasymp} p-value for multivariate testing : {pval}"
            else:
                strunc = "(truncation)" 
                s+= f"\n{sasymp} p-value{strunc} for multivariate testing : " 

                if long and stat == 'kfda':
                    for t in ts:
                        s = self.str_add_multivariate_stat_and_pval(s,t)
                else:
                    s = self.str_add_multivariate_stat_and_pval(s,t)
        return(s)

    def str_add_univariate_test_results(self,s,long=False,t=10,name=None,ntop=5,threshold=.05,log2fc=False):

        if long:
            s+='\n'
        if log2fc :
            if self.log2fc_data is None :
                if self.it_is_possible_to_compute_log2fc():
                    self.add_log2fc_to_var()
                else:
                    log2fc=False

        s+='\n___Univariate tests results___'
        ntested = self.get_ntested_variables()
        nvar = self.get_nvariables()
        if ntested == 0:
            s+=f'\nUnivariate test not performed yet, run ktest.univariate_test()'
        else:
            if ntested!=nvar:
                s+=f'\n{ntested} tested variables out of {nvar} variables'
            DEgenes = self.get_DE_genes(t=t,name=name,threshold=threshold)
            s+=f"\n{len(DEgenes)} DE genes for t={t} and threshold={threshold} (with BH correction)"
            
            if len(DEgenes)>0:
                topn = DEgenes.sort_values()[:ntop]
                s+= f'\nTop {ntop} DE genes (with BH correction): \n'
                for g in topn.index:
                    pval = topn[g]
                    log2fc_str = f"log2fc = {self.get_log2fc_of_variable(g):.2f}" if log2fc else '' 
                    if long:
                        pval_str = f"\tpval={pval:1.1e}" if pval <.01 else f"\tpval={pval:.2f}"
                        s+=f"\t{g} \t{pval_str} \t{log2fc_str}\n"
                    else:
                        log2fc_str=f' ({log2fc_str})' if log2fc else ''
                        s+=f'{g}{log2fc_str}, '
                        
                s= s[:-2]
        return(s)

    def print_test_info(self,long=False):
        s = ''
        s = self.str_add_test_info(s,long)
        print(s)

    def print_data_info(self,long=False):
        s = ''
        s = self.str_add_data_info(s,long)
        print(s)

    def print_multivariate_test_results(self,long=False,t=None,ts=[1,5,10]):
        """
        Print a summary of the multivariate test results. 

        Parameters 
        ----------

        long (default = False) : bool
            if True, the print is more detailled. 

        t (default = 10) : int
            considered if long == False
            truncation parameter of interest
        
        ts (default = [1,5,10]) : int
            considered if long == True
            truncation parameters of interest
            
        """


        s = ''
        s = self.str_add_multivariate_test_results(s,long,t,ts)
        print(s)

    def print_univariate_test_results(self,long=False,t=None,name=None,ntop=5,threshold=.05,log2fc=False):
        """
        Print a summary of the univariate test results. 

        Parameters 
        ----------

        long (default = False) : bool
            if True, the print is more detailled. 

        t (default = 10) : int
            truncation parameter of interest
        
        name (default = ktest.univariate_name): str
            Name of the set of univariate tests. 
            See ktest.univariate_test() for details.

        ntop (default = 5): int
            Number of top DE genes desplayed

        threshold (default = .05) : float in [0,1]
            test rejection threshold.

        log2fc (default = False) : bool
            print the log2fc of the displayed DE variables if possible.

        """
        s = ''
        s = self.str_add_univariate_test_results(s,long,t=t,name=name,ntop=ntop,threshold=threshold,log2fc=log2fc)
        print(s)

    def __str__(self,long=False):
        
        s="An object of class Ktest"
        
        
        s = self.str_add_data_info(s,long=long)  
        s = self.str_add_test_info(s,long=long)
        s = self.str_add_multivariate_test_results(s,long=long)
        s = self.str_add_univariate_test_results(s,long=long)
        return(s)

    def long_str(self):
        return(self.__str__(long=True))  

    def __repr__(self):
        return(self.__str__())
 
    def get_names(self):
        names = {'kfdat':[c for c in self.df_kfdat],
                'proj_kfda':[name for name in self.df_proj_kfda.keys()],
                'proj_kpca':[name for name in self.df_proj_kpca.keys()],
                'correlations':[name for name in self.corr.keys()]}
        return(names)

    def kfdat_statistic(self,verbose=0):
        """"
        Compute the kfda statistic from scratch. 
        Compute every needed intermediate quantity. 
        Return the name of the column containing the statistics in dataframe `ktest.df_kfdat`.

        Parameters
        ----------
            verbose (default = 0) : int 
                The greater, the more verbose. 
        """

        kn = self.get_kfdat_name()

        if kn in self.df_kfdat : 
            if verbose >0: 
                print(f'Statistic {kn} already computed')

        else:
            self.initialize_kfdat(verbose=verbose) # nystrom and eigenvectors. 
            self.compute_kfdat(verbose=verbose) # stat 
            
        return(kn)

    def mmd_statistic(self,unbiaised=False,verbose=0):
        """
        Compute the MMD statistic from scratch. 
        Compute every needed intermediate quantity. 
        Return the name of the key containing the statistic in dict `ktest.dict_mmd`. 
        """

        mn = self.get_mmd_name()
        if mn in self.dict_mmd :
            if verbose : 
                print(f'Statistic {mn} already computed')
        else:
            self.initialize_mmd(verbose=verbose)
            self.compute_mmd(unbiaised=unbiaised,verbose=0)
        return(mn)

    def kpca(self,t=None,verbose=0):
        
        proj_kpca_name = self.get_kfdat_name()
        if proj_kpca_name in self.df_proj_kpca :
            if verbose : 
                print(f'kfdat {proj_kpca_name} already computed')
        else:
            self.initialize_kfda(verbose=verbose)            
            self.compute_proj_kpca(t=t,verbose=verbose)

    def compute_test_statistic(self,
                            stat=None,
                            verbose=0):
        
        stat = self.stat if stat is None else stat 
        
        if stat == 'kfda':
            self.kfdat_statistic(verbose=verbose)
        elif stat == 'mmd':
            self.mmd_statistic(verbose=verbose)
        else:
            if verbose >0:
                print(f"Statistic '{stat}' not recognized. Possible values : 'kfda','mmd'")

    def multivariate_test(self,
                        stat=None,
                        permutation=None,
                        n_permutations=None,
                        seed_permutation=None,                   
                        n_jobs_permutation=1,
                        keep_permutation_statistics=False,
                        verbose=0,):
        
        
        self.compute_test_statistic(stat=stat,verbose=verbose)
        self.compute_pvalue(stat=stat,
                            permutation=permutation,
                            n_permutations=n_permutations,
                            seed_permutation=seed_permutation,
                            n_jobs_permutation=n_jobs_permutation,
                            keep_permutation_statistics=keep_permutation_statistics,
                            verbose=verbose)

        if verbose>0:
            self.print_multivariate_test_results(long=False)

    def get_spectrum(self,anchors=False,cumul=False,part_of_inertia=False,log=False,decreasing=False):
        sp,_ = self.get_spev(slot='anchors' if anchors else 'covw')        
        spp = (sp/sum(sp)) if part_of_inertia else sp
        spp = spp.cumsum(0) if cumul else spp
        spp = 1-spp if decreasing else spp
        spp = torch.log(spp) if log else spp

        return(spp)

    def get_pvalue_name(self,stat=None,permutation=None):
        if permutation is None:
            permutation = self.permutation
        if stat is None:
            stat = self.stat 
        if stat == 'mmd':
            return(self.permutation_mmd_name)
        elif stat == 'kfda':
            if self.permutation:
                return(self.permutation_kfda_name)
            else:
                return(self.get_kfdat_name())

    def get_pvalue_kfda(self,
                        name=None,
                        permutation=None,
                        contrib=False,
                        log=False,
                        verbose=0):
        
        name = self.get_pvalue_name(stat='kfda',permutation=permutation) if name is None else name 

        df_pval = self.df_pval_contributions if contrib else self.df_pval

        if name not in df_pval:
            if verbose>0:
                print(f'p-values associated to {name} not computed yet, run ktest.multivariate_test().')
            return(None)
        else:
            return(np.log(df_pval[name]) if log else df_pval[name])

    def get_pvalue_mmd(self,name=None):
        name = self.permutation_mmd_name if name is None else name 
            
        try:
            return(self.dict_pval_mmd[name])
        except KeyError:
            print(f"mmd name '{name}' not in dict_pval_mmd keys {self.dict_pval_mmd}")

    def get_pvalue(self,
                    stat=None,
                    name=None,
                    permutation=None,
                    contrib=False,
                    log=False,
                    verbose=0):
            
        if stat is None:
            stat = self.stat

        if stat == 'kfda':
            pval = self.get_pvalue_kfda(
                                name = name,
                                permutation=permutation,
                                contrib=contrib,
                                log=log,
                                verbose=verbose)
        elif stat =='mmd':
            pval = self.get_pvalue_mmd(name=name)

        return(pval) 

    def get_kfda(self,contrib=False,log=False,name=None,condition=None,samples=None,marked_obs_to_ignore=None):
        if name is None:
            name = self.get_kfdat_name(condition=condition,
                                   samples=samples,
                                   marked_obs_to_ignore=marked_obs_to_ignore) 
        df_kfda = self.df_kfdat_contributions if contrib else self.df_kfdat

        kfda = np.log(df_kfda[name]) if log else df_kfda[name]
        kfda = kfda[~kfda.isna()]
        return(kfda) 

    def get_mmd(self):
        mn = self.get_mmd_name()
        return(self.dict_mmd[mn])

    def get_statistic(self,stat=None):
        if stat is None:
            stat = self.stat

        if stat == 'kfda':
            return(self.get_kfda())
        elif stat == 'mmd':
            return(self.get_mmd())

    def get_diagnostics(self,
                        t=None,
                        diff=False,
                        var_within=False,
                        var_samples=False,
                        kfdr=False,
                        log=False,cumul=False,decreasing=False,):
        """
        Returns a dataframe containing information of the method with respect to the truncation parameters. 

        Parameters
        ----------
            t (default = None), int 
                Outputs are returned for truncations below `t`.
                If None, `t` is set to the maximum possible value 
                (order of the number of observations taking into account numerical errors)

            diff (default = False) : bool
                if True, adds the part of the difference between conditions 
                explained by each principal component to the output.
                
            var_within (default = False) : bool
                if True, adds the within group variability 
                explained by each principal component to the output.
            
            var_samples (default = False) : bool
                if True, adds the sample variabilities 
                explained by each principal component to the output.
                
            kfdr (default = False) : bool
                if True, adds the discrimination score 
                of each principal component to the output.
                (The score is returned as a ratio of the maximal discrimination
                 score obtained for the considered truncations to be in [0,1] ) 
                
            log (default = False) : bool
                if True, returns the log of the outputs. 
            
            cumul (default = False) : bool
                if True, returns the cumulated outputs, 
                i.e. for a truncation t, returns the outputs summed from 1 to t. 
                
            decreasing (default = False) : bool
                if True, returns 1-outputs.            
        """

        df_diff = pd.DataFrame()
        df_varw = pd.DataFrame()
        df_vars = pd.DataFrame()
        df_kfdr = pd.DataFrame()
        
        if diff: 
            x_diff = self.get_explained_difference(cumul=cumul,log=log,decreasing=decreasing).numpy()
            x_diff = np.concatenate([np.array([0]),x_diff])
            df_diff = pd.DataFrame(x_diff,columns=['difference'])
            df_diff.index = [int(i) for i in df_diff.index]
        
        if var_within:
            x_varw = self.get_explained_variability(within=True,cumul=cumul,log=log,decreasing=decreasing).numpy()
            x_varw = np.concatenate([np.array([0]),x_varw])
            df_varw = pd.DataFrame(x_varw,columns=['w-variability'])
            df_varw.index = [int(i) for i in df_varw.index]
        if var_samples:
            x_vars = self.get_explained_variability(within=False,cumul=cumul,log=log,decreasing=decreasing)
            x_vars = {f'{k}-variability':np.concatenate([np.array([0]),v.to_numpy()]) for k,v in x_vars.items()}
            df_vars = pd.DataFrame(x_vars)
            df_vars.index = [int(i) for i in df_vars.index]
            
        if kfdr: 
            x_kfdr = self.compute_directional_kfdr(t=t,cumul=cumul,log=log,decreasing=decreasing).numpy()
            x_kfdr = np.concatenate([np.array([0]),x_kfdr])
            df_kfdr = pd.DataFrame(x_kfdr,columns=['discrimination'])
            df_kfdr.index = [int(i) for i in df_kfdr.index]    
        df = pd.concat([df_diff,df_varw,df_vars,df_kfdr],axis=1)
        t = len(df) if t is None else t
        return(df[:t+1])

    def get_projections(self,
                        t,
                        discriminant_axis=False,
                        principal_components=False,
                        orthogonal=False,
                        mmd=False,
                        condition=None,
                        samples=None,
                        marked_obs_to_ignore=None,
                        in_dict = False 
                       ):
        """
        Returns the desired cell positions on directions of interest in the feature space.

        Parameters 
        ----------
            t : int or iterable of int
                truncations of interest

            discriminant_axis (default = False) : bool
                if True, add the cell projections on the discriminant direction to the output

            principal_components (default = False) : bool
                if True, add the cell projections on the principal components of the within group covariance to the output

            orthogonal (default = False): bool
                if True, add the cell projections on the orthogonal direction (orthogonal to the discriminant axis) to the output

            mmd (default = False) : bool
                if True, add the cell projections on the  MMD-withess function (axis supported by the mean embeddings difference)

            unidirectionall_mmd (default = None) : bool
                if True, add a projection similar to proj_kpca with a different normalization to the output

            condition (default = None): str
                    Column of the metadata that specify the dataset  

            samples (default = None): str 
                    List of values to select in the column condition of the metadata

            marked_obs_to_ignore (default = None): str
                    Column of the metadata specifying the observations to ignore

            in_dict (default = True) : bool
                if True : returns a dictionary of the outputs associated to each sample
                else : returns an unique object containing all the outputs     

        """


        df = pd.DataFrame(index=self.get_index(samples='all',in_dict=False))


        # truncation preprocessing, create two lists of int and str truncations.
        ts = [str(t)] if isinstance(t,int) else [str(t_) for t_ in t]  
        ts_int = [t] if isinstance(t,int) else [t_ for t_ in t] 


        # computes the asked projections if necessary 
        if any([discriminant_axis,principal_components]):
            self.projections(np.max(t))
        if any([mmd]):
            self.projections_MMD()
        if orthogonal:
            [self.orthogonal(t=t) for t in ts_int]


        for projection,proj_str,token in zip(
                                            ['proj_kfda','proj_kpca'],
                                            ['discriminant_axis','kpca'],
                                            [discriminant_axis,principal_components]):

            if token:
                df_proj = self.init_df_proj(projection)
                if len(ts)==1:
                    df[proj_str] = df_proj[ts]
                else:
                    for t in ts:
                        df[f'{proj_str}_{t}'] = df_proj[t]

        if orthogonal:
            if len(ts_int)==1:
                orthogonal_name=self.get_orthogonal_name(t=ts_int[0],center='w')
                df_proj = self.init_df_proj('proj_orthogonal',name=orthogonal_name)
                df['orthogonal'] = df_proj['1']
            else:
                for t in ts_int:
                    orthogonal_name=self.get_orthogonal_name(t=t,center='w')
                    df_proj = self.init_df_proj('proj_orthogonal',name=orthogonal_name)
                    df[f'orthogonal_{t}'] = df_proj['1']

        if mmd:
            df = pd.concat([df,self.init_df_proj('proj_mmd')],axis=1)

        return(self.stratify_output(df,samples=samples,
                                    condition=condition,
                                    marked_obs_to_ignore=marked_obs_to_ignore,
                                    in_dict=in_dict))

    def stratify_output(self,df,
                        condition=None,
                        samples=None,
                        marked_obs_to_ignore=None,
                        in_dict=False):
        
        index = self.get_index(samples=samples,
                               condition=condition,
                               marked_obs_to_ignore=marked_obs_to_ignore,
                               in_dict=in_dict)
        if in_dict:
            return({k:df.loc[i] for k,i in index.items()})
        else:
            return(df.loc[index])

    def get_metadata(self,
                     condition=None,
                     samples=None,
                     marked_obs_to_ignore=None,
                     in_dict=False):
        return(self.stratify_output(self.obs,
                                    condition=condition,
                                    samples=samples,
                                    marked_obs_to_ignore=marked_obs_to_ignore,
                                    in_dict=in_dict))



def ktest_from_xy(x,y,names='xy',kernel=None):
    if kernel is None:
        kernel = init_kernel_params()
    x = pd.DataFrame(x,columns = [str(i) for i in range(x.shape[1])])
    y = pd.DataFrame(y,columns = [str(i) for i in range(x.shape[1])])
    data = pd.concat([x,y])
    metadata = pd.DataFrame([names[0]]*len(x)+[names[1]]*len(y),columns=['condition'])
    t = Ktest(
            data,
            metadata,
            data_name='data',
            condition='condition',    
            kernel=kernel,)
    return(t)



