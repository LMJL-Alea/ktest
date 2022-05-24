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
from ktest.kernels import gauss_kernel_mediane






def pytorch_eigsy(matrix):
    # j'ai codé cette fonction pour tester les approches de nystrom 
    # avec la diag de pytorch mais ça semble marcher moins bien que la fonction eigsy
    # cpdt je devrais comparer sur le meme graphique 
    sp,ev = torch.symeig(matrix,eigenvectors=True)
    order = sp.argsort()
    ev = ev[:,order]
    sp = sp[order]
    return(sp,ev)

# Choix à faire ( trouver les bonnes pratiques )

# voir comment ils gèrent les plot dans scanpy

# initialiser le mask aussi au moment de tracer des figures

# tracer l'evolution des corrélations par rapport aux gènes ordonnés

# ecrire les docstring de chaque fonction 

# faire des fonctions compute kfdat et proj qui font appelle au scheduleur "compute and load" (a renommer) 
# le name de corr doit spécifier en plus du nom la projection auxquel il fiat référence (acp ou fda)

# enregistrer le coef de la projection kfda de chaque axe de l'acp 
# faire une fonction __print__ (?)
 
# calculer les p values par direction 
# liste de gène 
# lsmeans least square means (dans le cours de Franck pour ANOVA deux facteurs p32 test moyennes ajustées )
# mmd et test par permutation 

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

class Tester:
    """
    Tester is a class that performs kernels tests such that MMD and the test based on Kernel Fisher Discriminant Analysis. 
    It also provides a range of visualisations based on the discrimination between two groups.  
    """
    
# si j'ajoute la p-value, voilà le code 
# pval = [chi2.sf(test0R.df_kfdat['0h vs 48hrev'][t],t) for t in trunc]

    from .kernel_operations import \
        compute_gram, \
        center_gram_matrix_with_respect_to_some_effects, \
        compute_kmn, \
        diagonalize_centered_gram,\
        compute_within_covariance_centered_gram

    from .centering_operations import \
        compute_centering_matrix_with_respect_to_some_effects, \
        compute_omega, \
        compute_covariance_centering_matrix
    
    from .nystrom_operations import \
        compute_nystrom_anchors,\
        compute_nystrom_landmarks,\
        compute_quantization_weights,\
        reinitialize_landmarks,\
        reinitialize_anchors

    from .statistics import \
        get_trunc,\
        get_95variance_trunc,\
        get_explained_variance,\
        compute_kfdat,\
        compute_pval,\
        correct_BenjaminiHochberg_pval,\
        compute_pkm,\
        compute_upk,\
        initialize_kfdat,\
        kfdat,\
        kpca,\
        initialize_mmd,\
        mmd,\
        compute_mmd


    from .projection_operations import \
        compute_proj_kfda,\
        compute_proj_kpca,\
        init_df_proj,\
        compute_proj_mmd

    from .correlation_operations import \
        compute_corr_proj_var,\
        find_correlated_variables

    from .visualizations import \
        plot_kfdat,\
        init_plot_kfdat,\
        init_plot_pvalue,\
        plot_pvalue,\
        plot_kfdat_contrib,\
        plot_spectrum,\
        density_proj,\
        scatter_proj,\
        init_axes_projs,\
        density_projs,\
        scatter_projs,\
        set_color_for_scatter,\
        find_cells_from_proj,\
        plot_correlation_proj_var

    from .initializations import \
        init_data,\
        init_kernel,\
        set_center_by,\
        init_xy,\
        init_index_xy,\
        init_variables,\
        init_masks,\
        init_metadata,\
        init_model,\
        init_data_from_dataframe,\
        verbosity

    from .residuals import \
        compute_discriminant_axis_qh,\
        project_on_discriminant_axis,\
        compute_proj_on_discriminant_orthogonal,\
        compute_residual_covariance,\
        diagonalize_residual_covariance,\
        proj_residus

    def __init__(self):
        """\
        Parameters
        ----------
        x,y:                torch.Tensor or numpy.array of sizes n1 x p and n2 x p 
        kernel:             kernel function to apply on (x,y)
        x_index,y_index:    pd.Index or list of index to identify observations from x and y
        variables:          pd.Index or list of index to identify variables from x and y
        
        Returns
        -------
        :obj:`Tester`
        """
        self.has_data = False   
        self.has_model = False
        self.has_landmarks = False
        self.quantization_with_landmarks_possible = False
        
        # attributs initialisés 
        self.dict_data = {}
        # self.dict_model = {}
        self.df_kfdat = pd.DataFrame()
        self.df_pval = pd.DataFrame()
        self.df_proj_kfda = {}
        self.df_proj_kpca = {}
        self.df_proj_mmd = {}
        self.df_proj_residuals = {}
        self.corr = {}     
        self.dict_mmd = {}
        self.spev = {'x':{},'y':{},'xy':{},'residuals':{}} # dict containing every result of diagonalization
        # les vecteurs propres sortant de eigsy sont rangés en colonnes

        # for verbosity 
        self.start_times = {}

        # if x is not None and y is not None:
        #     self.init_data(x=x,y=y,kernel=kernel,x_index=x_index,y_index=y_index,variables=variables)
       

    def __str__(self):
        if self.has_data:
            s = f"View of Tester object with n1 = {self.n1}, n2 = {self.n2}\n"
            s += f"x ({self.x.shape}), y ({self.y.shape})\n"                       
        else: 
            s = "View of Tester object with no data\n"  
        cov = self.approximation_cov
        mmd = self.approximation_mmd
        ny = 'nystrom' in cov or 'nystrom' in mmd
        s += f"Model: {cov},{mmd}"
        if ny : 
            anchors_basis = self.anchors_basis
            m=self.m
            r=self.r
            s += f',{anchors_basis},m={m},r={r}'
        s+= '\n'
        
        s += f"df_kfdat ({self.df_kfdat.shape})\n"
        s += f"df_pval ({self.df_pval.shape})\n"
        s += f"df_proj_kfda ({len(self.df_proj_kfda)})\n"
        s += f"df_proj_kpca ({len(self.df_proj_kpca)})\n"
        s += f"df_proj_mmd ({len(self.df_proj_mmd)})\n"
        s += f"df_proj_residuals ({len(self.df_proj_residuals)})\n"
        s += f"corr ({len(self.corr)})\n"
        s += f"corr ({len(self.corr)})\n"
        s += f"dict_mmd ({len(self.dict_mmd)})\n\n"
        
        kx = self.spev['x'].keys()
        ky = self.spev['y'].keys()
        kxy = self.spev['xy'].keys()
        kr = self.spev['residuals'].keys()
        s+=f"spev['x']:({kx})\n"
        s+=f"spev['y']:({ky})\n"
        s+=f"spev['xy']:({kxy})\n"
        s+=f"spev['residuals']:({kr})\n"

        return(s) 

    def __repr__(self):
        return(self.__str__())
 
    def load_kfdat(self,path):
        df = pd.read_csv(path,header=0,index_col=0)
        for c in df.columns:
            if c not in self.df_kfdat.columns:
                self.df_kfdat[c] = df[c]
            else: 
                print(f'kfdat {c} already here')

    def load_proj_kfda(self,path,name,):
        df = pd.read_csv(path,header=0,index_col=0)
        if name not in self.df_proj_kfda:
            self.df_proj_kfda[name] = df
        else:
            print(f'proj kfda {name} already here')

    def load_proj_kpca(self,path,name):
        df = pd.read_csv(path,header=0,index_col=0)
        if name not in self.df_proj_kpca:
            self.df_proj_kpca[name] = df
        else:
            print(f'proj kpca {name} already here')

    def load_correlations(self,path,name):
        df = pd.read_csv(path,header=0,index_col=0)
        if name not in self.corr:
            self.corr[name] = df
        else:
            print(f'corr {name} already here')

    def save_kfdat(self,path):
        self.df_kfdat.to_csv(path,index=True)

    def save_proj_kfda(self,path,name):
        self.df_proj_kfda[name].to_csv(path,index=True)
    
    def save_proj_kpca(self,path,name):
        self.df_proj_kpca[name].to_csv(path,index=True)

    def save_correlations(self,path,name):
        self.corr[name].to_csv(path,index=True)

    def load_data(self,to_load):
        """
        to_load ={'kfdat':path_kfdat,
                    'proj_kfda':{name1:path1,name2,path2},
                    'proj_kpca':{name1:path1,name2,path2},
                    'correlations':{name1:path1,name2,path2}}
        """
        types_ref= {'proj_kfda':self.load_proj_kfda,
                    'proj_kpca':self.load_proj_kpca,
                    'correlations':self.load_correlations}
        
        for type in to_load:
            if type == 'kfdat':
                self.load_kfdat(to_load[type]) 
            else:
                for name,path in to_load[type].items():
                    types_ref[type](path,name)


    def get_names(self):
        names = {'kfdat':[c for c in self.df_kfdat],
                'proj_kfda':[name for name in self.df_proj_kfda.keys()],
                'proj_kpca':[name for name in self.df_proj_kpca.keys()],
                'correlations':[name for name in self.corr.keys()]}
        return(names)

    # def load_data(self,data_dict,):
    # def save_data():
    # def save_a_dataframe(self,path,which)

    def get_xy(self,landmarks=False):
        if landmarks: 
            x,y = self.xlandmarks,self.ylandmarks
        else:
            x,y = self.x[self.xmask,:],self.y[self.ymask,:]
        return(x,y)
      
    def name_generator(self,t=None,nystrom=0,nystrom_method='kmeans',r=None,obs_to_ignore=None):
        
        if obs_to_ignore is not None:
            name_ = f'~{obs_to_ignore[0]} n={len(obs_to_ignore)}'
        else:
            name_ = ""
            if t is not None:
                name_ += f"tmax{t}"
            if nystrom:
                name_ +=f'ny{nystrom}{nystrom_method}na{r}'
        return(name_)
    
    def ignore_obs(self,obs_to_ignore=None,reinitialize_ignored_obs=False):
        
        if self.ignored_obs is None or reinitialize_ignored_obs:
            self.ignored_obs = obs_to_ignore
        else:
            self.ignored_obs.append(obs_to_ignore)

        if obs_to_ignore is not None:
            x,y = self.get_xy()
            self.xmask = ~self.x_index.isin(obs_to_ignore)
            self.ymask = ~self.y_index.isin(obs_to_ignore)
            self.imask = ~self.index.isin(obs_to_ignore)
            self.n1 = x.shape[0]
            self.n2 = y.shape[0]

    def unignore_obs(self):
        
        self.xmask = [True]*len(self.x)
        self.ymask = [True]*len(self.y)
        self.imask = [True]*len(self.index)
        self.n1 = self.x.shape[0]
        self.n2 = self.y.shape[0]
    
    def infer_nobs(self,which ='proj_kfda',name=None):
        if not hasattr(self,'n1'):
            if name is None:
                name = self.get_names()[which][0]
            df_proj= self.init_df_proj(which,name)
            self.n1 =  df_proj[df_proj['sample']=='x'].shape[0]
            self.n2 =  df_proj[df_proj['sample']=='y'].shape[0]
#       
    def eval_nystrom_trace(self):
        """
        Returns the squared difference between the spectrum in ``self.sp`` and the nystrom spectrum in ``self.spny``
        """
        # je pourrais choisir jusqu'où je somme mais pour l'instant je somme sur l'ensemble du spectre 
        return((self.sp.sum() - self.spny.sum())**2)
    
    def eval_nystrom_spectrum(self):
        """
        Returns the squared difference between each eigenvalue of the spectrum in ``self.sp`` and each eigenvalue of the nystrom spectrum in ``self.spny``
        """
        return([ (lny.item() - l.item())**2 for lny,l in zip(self.spny,self.sp)])

    def eval_nystrom_discriminant_axis(self,nystrom=1,t=None,m=None):
        # a adapter au GPU
        # j'ai pas du tout réfléchi à la version test_data, j'ai juste fait en sorte que ça marche au niveau des tailles de matrices donc il y a peut-être des erreurs de logique

        n1,n2 = (self.n1,self.n2)
        ntot = n1+n2
        m1,m2 = (self.nxlandmarks,self.nylandmarks)
        mtot = m1+m2

        mtot=self.nxlandmarks + self.nylandmarks
        ntot= self.n1 + self.n2
        
        t = 60   if t is None else t # pour éviter un calcul trop long # t= self.n1 + self.n2
        m = mtot if m is None else m
        
        kmn   = self.compute_kmn(test_data=False)
        kmn_test   = self.compute_kmn(test_data=True)
        kny   = self.compute_gram(landmarks=True)
        k     = self.compute_gram(landmarks=False,test_data=False)
        Pbiny = self.compute_covariance_centering_matrix(sample='xy',quantization=True)
        Pbi   = self.compute_covariance_centering_matrix(sample='xy',quantization=False,test_data=False)
        
        Vny  = self.evny[:m]
        V    = self.ev[:t] 
        spny = self.spny[:m]
        sp   = self.sp[:t]

        mny1   = -1/m1 * torch.ones(m1, dtype=torch.float64) #, device=device) 
        mny2   = 1/m2 * torch.ones(m2, dtype=torch.float64) # , device=device)
        m_mtot = torch.cat((mny1, mny2), dim=0) # .to(device)
        

        mn1    = -1/n1 * torch.ones(n1, dtype=torch.float64) # , device=device)
        mn2    = 1/n2 * torch.ones(n2, dtype=torch.float64) # , device=device) 
        m_ntot = torch.cat((mn1, mn2), dim=0) #.to(device)
        
        # mn1_test    = -1/n1_test * torch.ones(n1_test, dtype=torch.float64) # , device=device)
        # mn2_test    = 1/n2_test * torch.ones(n2_test, dtype=torch.float64) # , device=device) 
        # m_ntot_test = torch.cat((mn1_test, mn2_test), dim=0) #.to(device)
        
        vpkm    = mv(torch.chain_matmul(V,Pbi,k),m_ntot)
        vpkm_ny = mv(torch.chain_matmul(Vny,Pbiny,kmn_test),m_ntot_test) if nystrom==1 else \
                  mv(torch.chain_matmul(Vny,Pbiny,kny),m_mtot)

        norm_h   = (ntot**-1 * sp**-2   * mv(torch.chain_matmul(V,Pbi,k),m_ntot)**2).sum()**(1/2)
        norm_hny = (mtot**-1 * spny**-2 * mv(torch.chain_matmul(Vny,Pbiny,kmn_test),m_ntot_test)**2).sum()**(1/2) if nystrom==1 else \
                   (mtot**-1 * spny**-2 * mv(torch.chain_matmul(Vny,Pbiny,kny),m_mtot)**2).sum()**(1/2)

        # print(norm_h,norm_hny)

        A = torch.zeros(m,t,dtype=torch.float64).addr(1/mtot*spny,1/ntot*sp) # A = outer(1/mtot*self.spny,1/ntot*self.sp)
        B = torch.zeros(m,t,dtype=torch.float64).addr(vpkm_ny,vpkm) # B = torch.outer(vpkm_ny,vpkm)
        C = torch.chain_matmul(Vny,Pbiny,kmn,Pbi,V.T)

        return(norm_h**-1*norm_hny**-1*(A*B*C).sum().item()) # <h_ny^* , h^*>/(||h_ny^*|| ||h^*||)

    def eval_nystrom_eigenfunctions(self,t=None,m=None):
        # a adapter au GPU


        n1,n2 = (self.n1,self.n2)
        ntot = n1+n2
        m1,m2 = (self.nxlandmarks,self.nylandmarks)
        mtot = m1+m2

        mtot=self.nxlandmarks + self.nylandmarks
        ntot= self.n1 + self.n2
        
        t = 60   if t is None else t # pour éviter un calcul trop long # t= self.n1 + self.n2
        m = mtot if m is None else m
        
        kmn   = self.compute_kmn()
        Pbiny = self.compute_covariance_centering_matrix(sample='xy',quantization=True)
        Pbi   = self.compute_covariance_centering_matrix(sample='xy',quantization=False)
        
        Vny  = self.evny[:m]
        V    = self.ev[:t] 
        spny = self.spny[:m]
        sp   = self.sp[:t]

        # print(((spny * mtot)**(-1/2))[:3],'\n',((sp*ntot)**-(1/2))[:3])
        return( ((((spny * mtot)**(-1/2))*((sp*ntot)**-(1/2)*torch.chain_matmul(Vny,Pbiny,kmn,Pbi, V.T)).T).T).diag())
        


        # cev=(ev[:trunc[-1]].t()*coefs).t()
        

        # hip_wny_w = torch.matmul() 





