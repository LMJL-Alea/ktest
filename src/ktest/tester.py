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
from .base import init_kernel_params,init_test_params
 


def pytorch_eigsy(matrix):
    # j'ai codé cette fonction pour tester les approches de nystrom 
    # avec la diag de pytorch mais ça semble marcher moins bien que la fonction eigsy
    # cpdt je devrais comparer sur le meme graphique 
    sp,ev = torch.symeig(matrix,eigenvectors=True)
    order = sp.argsort()
    ev = ev[:,order]
    sp = sp[order]
    return(sp,ev)



# tracer l'evolution des corrélations par rapport aux gènes ordonnés
# test par permutation 

# plot proj :on a dfx et dfy pour tracer le result en fct d'un axe de la pca 
# on peut aussi vouloir tracer en fonction de ll'expression d'un gène 

# def des fonction type get pour les opérations d'initialisation redondantes
# acces facile aux noms des dict de dataframes. 
# faire en sorte de pouvoir calculer corr kfda et kpca 

# mettre une limite globale a 100 pour les tmax des projections (éviter d'enregistrer des structures de données énormes)
# mieux gérer la projection de gènes et le param color
 
# verbosity devient aussi un verificateur de code 

# repenser les plots et df_init_proj

    # trouver comment inserer un test sous H0 sans prendre trop de mémoire 
    # def split_one_sample_to_simulate_H0(self,sample='x'):
    #     z = self.x if sample =='x' else self.y
    #     nH0 = self.n1 if sample == 'x' else self.n2
    #     p = permutation(np.arange(nH0))
    #     self.x0,self.y0 = z[p[:nH0//2]],z[p[nH0//2:]]


class Ktest(Plot_Univariate,SaveData,Pvalues,Correlations,Permutation):
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
                npermutation=500,
                seed_permutation=0,
                test_params=None,
                center_by=None,
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

            npermutation (default = 500) : int
                Number of permutation needed to compute the permutation p-value.
                Ignored if permutation is False. 
            
            seed_permutation (default = 0) : int
                Seed of the first permutation. The other seeds are obtained by incrementation. 

            nystrom (default = False) : boolean
                if True, the test_params nystrom is set to True with default parameters

            test_params : object specifying the test to execute
                output of the function get_test_params()

            kernel : obkect specifying the kernel function to use
                output of the function init_kernel_params()

            center_by (default = None) : str
                column of the metadata containing a categorial variable to regress 

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
                                npermutation=npermutation,
                                seed_permutation=seed_permutation,
                                )


        self.add_data_to_Ktest_from_dataframe(data,metadata,data_name=data_name,var_metadata=var_metadata,verbose=verbose)
        self.set_test_data_info(data_name=data_name,condition=condition,samples=samples,verbose=verbose)
        self.kernel(verbose=verbose,**kernel)
        self.kernel_specification = kernel
        self.set_test_params(verbose=verbose,**test_params)
        self.test_params_initial=test_params
        self.set_center_by(center_by=center_by,verbose=verbose)
        self.set_marked_obs_to_ignore(marked_obs_to_ignore=marked_obs_to_ignore,verbose=verbose)
        

    def str_add_test_info(self,s,long=False):

        # if long:
        #     s+=f'\n'

        tp = self.get_test_params()
        stat = tp['stat']
        ny = tp['nystrom']
        s+=f'\nTest statistic : {stat}'
        
        if ny: 
            s+= f" with the nystrom approximation ({tp['nanchors']} anchors)"

            if long:
                s += f"\n\t{tp['nlandmarks']} landmarks"
                lm = tp['landmark_method']
                ab = tp['anchor_basis']
                if lm == 'random':
                    s += f" (determined through random sampling)"
                if lm == 'kmeans':
                    s += f" (determined as kmeans centroids of each group)"
                
                s += f"\n\t{tp['nanchors']} anchors"
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

            if self.center_by is not None:
                s+=f"\nEmbeddings centered by {self.center_by}"
            if self.marked_obs_to_ignore is not None:
                s+=f"\nIgnoring cells in {self.marked_obs_to_ignore}"
        return(s)

    def str_add_multivariate_stat_and_pval(self,s,t):
        pval = self.df_pval[self.get_pvalue_name()][t]
        kfda = self.df_kfdat[self.get_kfdat_name()][t]
        if pval>.01:
            s+=f"\n\tp-value({t}) = {pval:.2f}"
        else:
            s+=f"\n\tp-value({t}) = {pval:1.1e}"
        if kfda<1000:
            s+=f" (kfda={kfda:.2f})"
        else :
            s+=f" (kfda={kfda:.2e})"
        return(s)

    def str_add_multivariate_test_results(self,s,long=False,t=10,ts=[1,5,10],stat=None,permutation=None):
        
        stat = self.stat if stat is None else stat 
        permutation = self.permutation if permutation is None else permutation

        if long:
            s+='\n'        
        s+=f'\n___Multivariate {stat} test results___'

        if self.get_pvalue_name() not in self.df_pval:
            s+="\nMultivariate test not performed yet, run ktest.multivariate_test()"
        else:
            sasymp = f"Permutation ({self.npermutation} permutations)" if permutation else f"Asymptotic"
            strunc = "(truncation)" if stat == 'kfda' else ""
            s+= f"\n{sasymp} p-value{strunc} for multivariate testing : " 

            if long:
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

    def print_multivariate_test_results(self,long=False,t=10,ts=[1,5,10]):
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

    def print_univariate_test_results(self,long=False,t=10,name=None,ntop=5,threshold=.05,log2fc=False):
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
            
            # j'ai enlevé cette condition pour verifier que j'en ai plus besoin. 
            # if (self.nystrom and self.has_anchors) or not self.nystrom:
            #    self.compute_kfdat(verbose=verbose) # stat 
        return(kn)

    def mmd_statistic(self,shared_anchors=True,unbiaised=False,verbose=0):
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
            self.initialize_mmd(shared_anchors=shared_anchors,verbose=verbose)
            self.compute_mmd(shared_anchors=shared_anchors,unbiaised=unbiaised,verbose=0)
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
                        npermutation=None,
                        seed_permutation=None,                   
                        n_jobs=1,
                        keep_permutation_statistics=False,
                        verbose=0,
                        t_verbose=10):
        
        
        self.compute_test_statistic(stat=stat,verbose=verbose)
        self.compute_pvalue(stat=stat,
                            permutation=permutation,
                            npermutation=npermutation,
                            seed_permutation=seed_permutation,
                            n_jobs=n_jobs,
                            keep_permutation_statistics=keep_permutation_statistics,
                            verbose=verbose)

        if verbose>0:
            self.print_multivariate_test_results(t=t_verbose,long=False)


    def get_spectrum(self,anchors=False,cumul=False,part_of_inertia=False,log=False,decreasing=False):
        sp,_ = self.get_spev(slot='anchors' if anchors else 'covw')        
        spp = (sp/sum(sp)) if part_of_inertia else sp
        spp = spp.cumsum(0) if cumul else spp
        spp = 1-spp if decreasing else spp
        spp = torch.log(spp) if log else spp

        return(spp)

    def get_pvalue_name(self,permutation=None):
        if permutation is None:
            permutation = self.permutation

        if self.permutation:
            return(self.permutation_name)
        else:
            return(self.get_kfdat_name())


    def get_pvalue_kfda(self,
                        permutation=None,
                        contrib=False,
                        log=False,
                        verbose=0):
        

        pn = self.get_pvalue_name(permutation=permutation)
        df_pval = self.df_pval_contributions if contrib else self.df_pval

        if pn not in df_pval:
            if verbose>0:
                print(f'p-values associated to {pn} not computed yet, run ktest.multivariate_test().')
            return(None)
        else:
            return(np.log(df_pval[pn]) if log else df_pval[pn])



    def get_pvalue_mmd(self,):
        return(self.dict_pval_mmd[self.permutation_name])

    def get_pvalue(self,
                    stat=None,
                    permutation=None,
                    contrib=False,
                    log=False,
                    verbose=0):

        if stat is None:
            stat = self.stat

        if stat == 'kfda':
            pval = self.get_pvalue_kfda(
                                permutation=permutation,
                                contrib=contrib,
                                log=log,
                                verbose=verbose)
        elif stat =='mmd':
            pval = self.get_pvalue_mmd()

        return(pval) 

    def get_kfda(self,contrib=False,log=False,name=None):
        name = self.get_kfdat_name() if name is None else name
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




def ktest_from_xy(x,y,names='xy',kernel=init_kernel_params()):
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



