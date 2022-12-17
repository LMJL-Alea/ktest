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


class Tester(Plot_Univariate,SaveData,Pvalues,Correlations):
    """
    Tester is a class that performs kernels tests such that MMD and the test based on Kernel Fisher Discriminant Analysis. 
    It also provides a range of visualisations based on the discrimination between two groups.  
    """

    def __init__(self):
        """\

        Returns
        -------
        :obj:`Tester`
        """
        
        super(Tester,self).__init__()

    def __str__(self):

        s = '##### Data ##### \n'
        
        if self.has_data:
            s+='Data : '
            for dn in self.data.keys():
                s+=f'{dn}, '
            s+=f'\t active : {self.data_name}'
            s+='\nObservations: '
            dict_nobs = self.get_nobs()
            for k,n in dict_nobs.items():
                s+=f'{k}:n={n} '
            # s += f'variables : {self.var[self.data_name][:3].tolist()}...\n'
            s += f'\nmeta : {self.obs.columns.tolist()}\n\n'
        else: 
            s += "View of Tester object with no data.\n" 
            s += "You can initiate the data with the class function 'init_data()'.\n\n"
        
        s += '##### Model #### \n'
        if self.has_model:

            ny = self.nystrom
            nys = 'nystrom' if ny else 'standard'
            s += f"Model: {nys}"

            if ny : 
                ab = self.anchors_basis
                lm = self.landmark_method
                m=self.m
                r=self.r
                s += f' {lm} (m={m}), {ab} (r={r})'
            s+= '\n\n'
        else: 
            s += "This Tester object has no model.\n"
            s += "You can initiate the model with the class function 'init_model()'.\n\n"

        
        s += '##### Kernel ####\n'
        if self.has_kernel:
            for dn in self.data.keys():
                if 'kernel_name' in self.data[dn]:
                    kernel_name = self.data[dn]['kernel_name']
                    s+=f'kernel {dn} : {kernel_name}\n'
            s+='\n'
        else:
            s+=f'This Tester object has no kernel.\n'
            s+=f"You can initiate the kernel function with the class function 'init_kernel()'. \n\n"
            


        s += '##### Results ##### \n'
        s += '--- Statistics --- \n'
        s += f"df_kfdat ({self.df_kfdat.columns})\n"
        s += f"df_kfdat_contributions ({self.df_kfdat_contributions.columns})\n"
        s += f"df_pval ({self.df_pval.columns})\n"
        s += f"df_pval_contributions ({self.df_pval_contributions.columns})\n"
        s += f"dict_mmd ({len(self.dict_mmd)})\n\n"


        s += '--- Projections --- \n'
        s += f"df_proj_kfda ({self.df_proj_kfda.keys()})\n"
        s += f"df_proj_kpca ({self.df_proj_kpca.keys()})\n"
        s += f"df_proj_mmd ({self.df_proj_mmd.keys()})\n"
        s += f"df_proj_residuals ({self.df_proj_residuals.keys()})\n\n"
        
        s += '--- Correlations --- \n'
        s += f"corr ({len(self.corr)})\n"
        s += f"corr ({len(self.corr)})\n\n"
        
        s += '--- Eigenvectors --- \n'
        s += 'A completer'
        # kx = self.spev['x'].keys()
        # ky = self.spev['y'].keys()
        # kxy = self.spev['xy'].keys()
        # kr = self.spev['residuals'].keys()
        # s+=f"spev['x']:({kx})\n"
        # s+=f"spev['y']:({ky})\n"
        # s+=f"spev['xy']:({kxy})\n"
        # s+=f"spev['residuals']:({kr})\n"

        return(s) 

    def __repr__(self):
        return(self.__str__())
 
    def get_names(self):
        names = {'kfdat':[c for c in self.df_kfdat],
                'proj_kfda':[name for name in self.df_proj_kfda.keys()],
                'proj_kpca':[name for name in self.df_proj_kpca.keys()],
                'correlations':[name for name in self.corr.keys()]}
        return(names)

    # def load_data(self,data_dict,):
    # def save_data():
    # def save_a_dataframe(self,path,which)
    # def get_dataframe_of_data(self):
    #     " a mettre à jour"
    #     x,y = self.get_xy()
    #     xindex = self.get_xy_index(sample='x')
    #     yindex = self.get_xy_index(sample='y')
    #     var = self.variables
        
    #     dfx = pd.DataFrame(x,index=xindex,columns=var)
    #     dfy = pd.DataFrame(y,index=yindex,columns=var)
    #     return(dfx,dfy)


    def kfdat(self,t=None,verbose=0):
        """"
        This functions computes the truncated kfda statistic from scratch, if needed, it computes landmarks and 
        anchors for the nystrom approach and diagonalize the matrix of interest for the computation fo the statistic. 
        It also computes the asymptotic pvalues for each truncation and determine automatically a truncation of interest. 

        Parameters
        ----------
            self : tester,
            the model parameter attributes `approximation_cov`, `approximation_mmd` must be defined.
            if the nystrom method is used, the attribute `anchor_basis` should be defined and the anchors must have been computed. 

            t (default = None) : None or int,
            valeur maximale de troncature calculée. 
            Si None, t prend la plus grande valeur possible, soit n (nombre d'observations) pour la 
            version standard et r (nombre d'ancres) pour la version nystrom  

            name (default = None) : None or str, 
            nom de la colonne des dataframe df_kfdat et df_kfdat_contributions dans lesquelles seront stockés 
            les valeurs de la statistique pour les différentes troncatures calculées de 1 à t 

            verbose (default = 0) : Dans l'idée, plus verbose est grand, plus la fonction détaille ce qu'elle fait

        """


        kfdat_name = self.get_kfdat_name()
        if kfdat_name in self.df_kfdat : # avoid computing twice the same stat 
            if verbose : 
                print(f'kfdat {kfdat_name} already computed')

        else:

            self.initialize_kfdat(verbose=verbose) # landmarks, ancres et diagonalisation           
            self.compute_kfdat(t=t,verbose=verbose) # caclul de la stat 
            self.select_trunc() # selection automatique de la troncature 
            self.compute_pval() # calcul des troncatures asymptotiques 
            kfdat_name = self.get_kfdat_name()
            self.kfda_stat = self.df_kfdat[kfdat_name][self.t] # stockage de la valeur de la stat pour la troncature selectionnées 
        
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


def create_and_fit_tester_for_two_sample_test_kfdat(df,meta,data_name,condition,df_var=None,nystrom=False,lm=None,ab=None,m=None,r=None,center_by=None,marked_obs_to_ignore=None,kernel='gauss_median',samples='all',viz=True):
    t = Tester()
    t.add_data_to_Tester_from_dataframe(df,meta,data_name=data_name,df_var=df_var)
    t.set_test_data_info(data_name=data_name,condition=condition,samples=samples)
    t.init_kernel(kernel='gauss_median' if kernel is None else kernel)
    t.init_model(nystrom=nystrom,landmark_method=lm,anchors_basis=ab,m=m,r=r)
    t.set_center_by(center_by=center_by)
    t.set_marked_obs_to_ignore(marked_obs_to_ignore=marked_obs_to_ignore)
    t.kfdat()

    if viz:
        t.projections()
        t.compute_pval(t=len(df))
        t.residuals()
    return(t)

