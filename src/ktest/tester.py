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


class Ktest(Plot_Univariate,SaveData,Pvalues,Correlations):
    """
    Ktest is a class that performs kernels tests such that MMD and the test based on Kernel Fisher Discriminant Analysis. 
    It also provides a range of visualisations based on the discrimination between two groups.  
    """

    def __init__(self):
        """\

        Returns
        -------
        :obj:`Ktest`
        """
        
        super(Ktest,self).__init__()



    def str_add_test_info(self,s,long=False):

        # if long:
        #     s+=f'\n'

        tp = self.get_test_params()
        test = tp['test']
        ny = tp['nystrom']
        s+=f'\nTest statistic : {test}'
        
        if tp['nystrom']: 
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
        pval = self.df_pval[self.get_kfdat_name()][t]
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

    def str_add_multivariate_test_results(self,s,long=False,t=10,ts=[1,5,10]):
        if long:
            s+='\n'        
        s+='\n___Multivariate test results___'

        if self.get_kfdat_name() not in self.df_pval:
            s+="\nMultivariate test not performed yet, run ktest.multivariate_test()"
        else:
            s+=f"\nAsymptotic p-value(truncation) for multivariate testing : "
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

        """
        s = ''
        s = self.str_add_univariate_test_results(s,long,t=t,ntop=ntop,threshold=threshold,log2fc=log2fc)
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

    def kfdat(self,t=None,verbose=0):
        """"
        This functions computes the truncated kfda statistic from scratch, if needed, it computes landmarks and 
        anchors for the nystrom approach and diagonalize the matrix of interest for the computation fo the statistic. 
        It also computes the asymptotic pvalues for each truncation and determine automatically a truncation of interest. 

        Parameters
        ----------
            self : Ktest,
            the model parameter attributes `approximation_cov`, `approximation_mmd` must be defined.
            if the nystrom method is used, the attribute `anchor_basis` should be defined and the anchors must have been computed. 

            t (default = None) : None or int,
            valeur maximale de troncature calculée. 
            Si None, t prend la plus grande valeur possible, soit n (nombre d'observations) pour la 
            version standard et nanchors (nombre d'ancres) pour la version nystrom  

            name (default = None) : None or str, 
            nom de la colonne des dataframe df_kfdat et df_kfdat_contributions dans lesquelles seront stockés 
            les valeurs de la statistique pour les différentes troncatures calculées de 1 à t 

            verbose (default = 0) : Dans l'idée, plus verbose est grand, plus la fonction détaille ce qu'elle fait

        """


        kfdat_name = self.get_kfdat_name()
        if kfdat_name in self.df_kfdat : # avoid computing twice the same stat 
            if verbose >0: 
                print(f'kfdat {kfdat_name} already computed')

        else:

            self.initialize_kfdat(verbose=verbose) # landmarks, ancres et diagonalisation    
            if (self.nystrom and self.has_anchors) or not self.nystrom:
                self.compute_kfdat(t=t,verbose=verbose) # caclul de la stat 
                # self.select_trunc() # selection automatique de la troncature 
                self.compute_pval() # calcul des troncatures asymptotiques 
                kfdat_name = self.get_kfdat_name()
                self.kfda_stat = self.df_kfdat[kfdat_name][10] # stockage de la valeur de la stat pour la troncature selectionnées 
                
        # les valeurs de la statistique ont été stockées dans une colonne de la dataframe df_kfdat. 
        # pour ne pas avoir à chercher le nom de cette colonne difficilement, il est renvoyé ici
        return(kfdat_name)

    def mmd(self,shared_anchors=True,name=None,unbiaised=False,verbose=0):
        """
        appelle la fonction initialize mmd puis la fonction compute_mmd si le mmd n'a pas deja ete calcule. 
        """
        approx = self.approximation_mmd
        mmd_name = self.get_mmd_name()

        if mmd_name in self.dict_mmd :
            if verbose : 
                print(f'mmd {mmd_name} already computed')
        else:
            self.initialize_mmd(shared_anchors=shared_anchors,verbose=verbose)
            self.compute_mmd(shared_anchors=shared_anchors,unbiaised=unbiaised,verbose=0)

    def kpca(self,t=None,verbose=0):
        
        proj_kpca_name = self.get_kfdat_name()
        if proj_kpca_name in self.df_proj_kpca :
            if verbose : 
                print(f'kfdat {proj_kpca_name} already computed')
        else:
            self.initialize_kfda(verbose=verbose)            
            self.compute_proj_kpca(t=t,verbose=verbose)

    def multivariate_test(self,t=10,verbose=1):
        n=self.get_ntot()
        self.kfdat(verbose=verbose-1)
        self.compute_pval(t=n)
        if verbose>0:
            self.print_multivariate_test_results(t=t,long=False)

def ktest(
    data,
    metadata,
    condition,
    data_name='data',
    samples='all',
    var_metadata=None,
    nystrom=False,
    test_params=None,
    center_by=None,
    marked_obs_to_ignore=None,
    kernel=None,
    verbose=0
            # df:data,meta:metadata,data_name,condition:test_condition,df_var:var_metadata,
            # test:test_params,viz:removed
    ):
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

    if kernel is None:
        kernel = init_kernel_params()

    if test_params is None:
        test_params = init_test_params(nystrom=nystrom)


    t = Ktest()
    t.add_data_to_Ktest_from_dataframe(data,metadata,data_name=data_name,var_metadata=var_metadata,verbose=verbose)
    t.set_test_data_info(data_name=data_name,condition=condition,samples=samples,verbose=verbose)
    t.kernel(verbose=verbose,**kernel)
    t.kernel_specification = kernel
    t.set_test_params(verbose=verbose,**test_params)
    t.test_params_initial=test_params
    t.set_center_by(center_by=center_by,verbose=verbose)
    t.set_marked_obs_to_ignore(marked_obs_to_ignore=marked_obs_to_ignore,verbose=verbose)
    return(t)


def ktest_from_xy(x,y,names='xy',kernel=init_kernel_params()):
    x = pd.DataFrame(x,columns = [str(i) for i in range(x.shape[1])])
    y = pd.DataFrame(y,columns = [str(i) for i in range(x.shape[1])])
    data = pd.concat([x,y])
    metadata = pd.DataFrame([names[0]]*len(x)+[names[1]]*len(y),columns=['condition'])
    t = ktest(
            data,
            metadata,
            data_name='data',
            condition='condition',    
            kernel=kernel,)
    return(t)



