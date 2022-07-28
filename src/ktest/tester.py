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
        center_kmn_matrix_with_respect_to_some_effects,\
        diagonalize_within_covariance_centered_gram,\
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
        get_explained_variance,\
        get_trace,\
        compute_kfdat,\
        compute_kfdat_with_different_order,\
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
        get_plot_properties,\
        get_color_for_scatter,\
        plot_correlation_proj_var,\
        plot_pval_with_respect_to_within_covariance_reconstruction_error,\
        plot_pval_with_respect_to_between_covariance_reconstruction_error,\
        plot_relative_reconstruction_errors,\
        plot_ratio_reconstruction_errors,\
        plot_within_covariance_reconstruction_error_with_respect_to_t,\
        plot_between_covariance_reconstruction_error_with_respect_to_t,\
        plot_pval_and_errors,\
        what_if_we_ignored_cells_by_condition,\
        what_if_we_ignored_cells_by_outliers_list,\
        prepare_visualization,\
        visualize_patient_celltypes_CRCL,\
        visualize_quality_CRCL,\
        visualize_effect_graph_CRCL

    from .initializations import \
        init_data,\
        init_kernel,\
        set_center_by,\
        init_xy,\
        init_index_xy,\
        init_variables,\
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
        proj_residus,\
        get_between_covariance_projection_error,\
        get_between_covariance_projection_error_associated_to_t,\
        get_ordered_spectrum_wrt_between_covariance_projection_error

    from .truncation_selection import \
        select_trunc_by_between_reconstruction_ratio,\
        select_trunc_by_between_reconstruction_ressaut,\
        select_trunc

    from .univariate_testing import \
        parallel_univariate_kfda,\
        univariate_kfda,\
        update_var_from_dataframe,\
        save_univariate_test_results_in_var,\
        load_univariate_test_results_in_var,\
        visualize_univariate_test_CRCL,\
        plot_density_of_variable,\
        get_zero_proportions_of_variable,\
        add_zero_proportions_to_var,\
        volcano_plot,\
        volcano_plot_zero_pvals_and_non_zero_pvals,\
        color_volcano_plot

    from .pvalues import \
        compute_pval,\
        correct_BenjaminiHochberg_pval,\
        correct_BenjaminiHochberg_pval_univariate,\
        get_rejected_variables_univariate


    def __init__(self):
        """\

        Returns
        -------
        :obj:`Tester`
        """
        self.has_data = False   
        self.has_model = False
        self.has_landmarks = False
        self.quantization_with_landmarks_possible = False
        
        # attributs initialisés 
        self.data = {'x':{},'y':{}}
        self.main_data=None

        # self.dict_model = {}
        self.df_kfdat = pd.DataFrame()
        self.df_kfdat_contributions = pd.DataFrame()
        self.df_pval = pd.DataFrame()
        self.df_pval_contributions = pd.DataFrame()
        self.df_proj_kfda = {}
        self.df_proj_kpca = {}
        self.df_proj_mmd = {}
        self.df_proj_residuals = {}
        self.corr = {}     
        self.dict_mmd = {}
        self.spev = {'x':{'anchors':{}},'y':{'anchors':{}},'xy':{'anchors':{}},'residuals':{}} # dict containing every result of diagonalization
        # les vecteurs propres sortant de eigsy sont rangés en colonnes

        # for verbosity 
        self.start_times = {}

        # if x is not None and y is not None:
        #     self.init_data(x=x,y=y,kernel=kernel,x_index=x_index,y_index=y_index,variables=variables)
       

    def __str__(self):

        s = '##### Data ##### \n'
        
        if self.has_data:
            n1,n2,n = self.get_n1n2n()
            x,y = self.get_xy()
            xindex,yindex = self.get_index(sample='x'),self.get_index(sample='y')
            s += f"View of Tester object with n1 = {n1}, n2 = {n2} (n={n})\n"

            s += f"x ({x.shape}), y ({y.shape})\n"
            s += f'kernel : {self.kernel_name}\n'                       
            s += f'x index : {xindex[:3].tolist()}... \n'
            s += f'y index : {yindex[:3].tolist()}... \n'
            s += f'variables : {self.variables[:3].tolist()}...\n'
            s += f'meta : {self.obs.columns.tolist()}\n'
        else: 
            s += "View of Tester object with no data.\n" 
            s += "You can initiate the data with the class function 'init_data()'.\n\n"
        
        s += '##### Model #### \n'
        if self.has_model: 
            cov = self.approximation_cov
            mmd = self.approximation_mmd
            ny = 'nystrom' in cov or 'nystrom' in mmd
            s += f"Model: {cov},{mmd}"
            if ny : 
                anchors_basis = self.anchors_basis
                m=self.m
                r=self.r
                s += f',{anchors_basis},m={m},r={r}'
            s+= '\n\n'
        else: 
            s += "This Tester object has no model.\n"
            s += "You can initiate the model with the class function 'init_model()'.\n\n"
        
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
    def get_dataframe_of_data(self,name_data=None):
        " a mettre à jour"
        x,y = self.get_xy(name_data=name_data)
        xindex = self.get_index(sample='x')
        yindex = self.get_index(sample='y')
        var = self.variables
        
        dfx = pd.DataFrame(x,index=xindex,columns=var)
        dfy = pd.DataFrame(y,index=yindex,columns=var)
        return(dfx,dfy)

    def get_xy(self,landmarks=False,outliers_in_obs=None,name_data=None):
        if name_data is None:
            name_data = self.main_data
        if landmarks: # l'attribut name_data n'a pas été adatpé aux landmarks car je n'en ai pas encore vu l'utilité 
            landmarks_name = 'landmarks' if outliers_in_obs is None else f'landmarks{outliers_in_obs}'
            x = self.data['x'][landmarks_name]['X'] 
            y = self.data['y'][landmarks_name]['X']
            
        else:
            if outliers_in_obs is None:
                x = self.data['x'][name_data]['X']
                y = self.data['y'][name_data]['X']
            else:         
                xindex = self.data['x']['index'] 
                yindex = self.data['y']['index']
                
                outliers    = self.obs[self.obs[outliers_in_obs]].index
                xmask       = ~xindex.isin(outliers)
                ymask       = ~yindex.isin(outliers)
                
                x = self.data['x'][name_data]['X'][xmask,:]
                y = self.data['y'][name_data]['X'][ymask,:]

        return(x,y)

    def get_index(self,sample='xy',landmarks=False,outliers_in_obs=None):
        if landmarks: 
            landmarks_name = 'landmarks' if outliers_in_obs is None else f'landmarks{outliers_in_obs}'
            xindex = self.obs[self.obs[f'x{landmarks_name}']].index
            yindex = self.obs[self.obs[f'y{landmarks_name}']].index
            
        else:
            if outliers_in_obs is None:
                xindex = self.data['x']['index'] 
                yindex = self.data['y']['index']
            else:
                xindex = self.data['x']['index'] 
                yindex = self.data['y']['index']
                
                outliers    = self.obs[self.obs[outliers_in_obs]].index
                xmask       = ~xindex.isin(outliers)
                ymask       = ~yindex.isin(outliers)

                xindex = self.data['x']['index'][xmask]
                yindex = self.data['y']['index'][ymask]

        return(xindex.append(yindex) if sample =='xy' else xindex if sample =='x' else yindex)
                
    def get_n1n2n(self,landmarks=False,outliers_in_obs=None):
        if landmarks: 
            landmarks_name = 'landmarks' if outliers_in_obs is None else f'landmarks{outliers_in_obs}'
            n1 = self.data['x'][landmarks_name]['n'] 
            n2 = self.data['y'][landmarks_name]['n']

        else:
            if outliers_in_obs is None:
                n1 = self.data['x']['n'] 
                n2 = self.data['y']['n']
            else:
                xindex = self.data['x']['index'] 
                yindex = self.data['y']['index']
                
                outliers    = self.obs[self.obs[outliers_in_obs]].index
                xmask       = ~xindex.isin(outliers)
                ymask       = ~yindex.isin(outliers)

                n1 = len(self.data['x']['index'][xmask])
                n2 = len(self.data['y']['index'][ymask])

        return(n1,n2,n1+n2)
        
    def determine_outliers_from_condition(self,threshold,which='proj_kfda',column_in_dataframe='standardstandard',t='1',orientation='>',outliers_in_obs=None):
        df = self.init_df_proj(which=which,name=column_in_dataframe,outliers_in_obs=outliers_in_obs)[str(t)]


        if orientation == '>':
            outliers = df[df>threshold].index
        if orientation == '<':
            outliers = df[df<threshold].index
        if orientation == '<>':
            outliers = df[df<threshold[0]].index
            outliers = outliers.append(df[df>threshold[1]].index)

        if outliers_in_obs is not None:
            df_outliers = self.obs[outliers_in_obs]
            old_outliers    = df_outliers[df_outliers].index
            outliers = outliers.append(old_outliers)

        return(outliers)

    def add_outliers_in_obs(self,outliers,name_outliers):
        index = self.get_index()
        self.obs[name_outliers] = index.isin(outliers)

# tout le reste est désuet. C'était des mesures de performances un peu bancales pour les calculs de nystrom. 
#       
    # def eval_nystrom_trace(self):
    #     """
    #     Returns the squared difference between the spectrum in ``self.sp`` and the nystrom spectrum in ``self.spny``
    #     """
    #     # je pourrais choisir jusqu'où je somme mais pour l'instant je somme sur l'ensemble du spectre 
    #     return((self.sp.sum() - self.spny.sum())**2)
    
    # def eval_nystrom_spectrum(self):
    #     """
    #     Returns the squared difference between each eigenvalue of the spectrum in ``self.sp`` and each eigenvalue of the nystrom spectrum in ``self.spny``
    #     """
    #     return([ (lny.item() - l.item())**2 for lny,l in zip(self.spny,self.sp)])

    # def eval_nystrom_discriminant_axis(self,nystrom=1,t=None,m=None):
    #     # a adapter au GPU
    #     # j'ai pas du tout réfléchi à la version test_data, j'ai juste fait en sorte que ça marche au niveau des tailles de matrices donc il y a peut-être des erreurs de logique

    #     n1,n2 = (self.n1,self.n2)
    #     ntot = n1+n2
    #     m1,m2 = (self.nxlandmarks,self.nylandmarks)
    #     mtot = m1+m2

    #     mtot=self.nxlandmarks + self.nylandmarks
    #     ntot= self.n1 + self.n2
        
    #     t = 60   if t is None else t # pour éviter un calcul trop long # t= self.n1 + self.n2
    #     m = mtot if m is None else m
        
    #     kmn   = self.compute_kmn(test_data=False)
    #     kmn_test   = self.compute_kmn(test_data=True)
    #     kny   = self.compute_gram(landmarks=True)
    #     k     = self.compute_gram(landmarks=False,test_data=False)
    #     Pbiny = self.compute_covariance_centering_matrix(sample='xy',quantization=True)
    #     Pbi   = self.compute_covariance_centering_matrix(sample='xy',quantization=False,test_data=False)
        
    #     Vny  = self.evny[:m]
    #     V    = self.ev[:t] 
    #     spny = self.spny[:m]
    #     sp   = self.sp[:t]

    #     mny1   = -1/m1 * torch.ones(m1, dtype=torch.float64) #, device=device) 
    #     mny2   = 1/m2 * torch.ones(m2, dtype=torch.float64) # , device=device)
    #     m_mtot = torch.cat((mny1, mny2), dim=0) # .to(device)
        

    #     mn1    = -1/n1 * torch.ones(n1, dtype=torch.float64) # , device=device)
    #     mn2    = 1/n2 * torch.ones(n2, dtype=torch.float64) # , device=device) 
    #     m_ntot = torch.cat((mn1, mn2), dim=0) #.to(device)
        
    #     # mn1_test    = -1/n1_test * torch.ones(n1_test, dtype=torch.float64) # , device=device)
    #     # mn2_test    = 1/n2_test * torch.ones(n2_test, dtype=torch.float64) # , device=device) 
    #     # m_ntot_test = torch.cat((mn1_test, mn2_test), dim=0) #.to(device)
        
    #     vpkm    = mv(torch.chain_matmul(V,Pbi,k),m_ntot)
    #     vpkm_ny = mv(torch.chain_matmul(Vny,Pbiny,kmn_test),m_ntot_test) if nystrom==1 else \
    #               mv(torch.chain_matmul(Vny,Pbiny,kny),m_mtot)

    #     norm_h   = (ntot**-1 * sp**-2   * mv(torch.chain_matmul(V,Pbi,k),m_ntot)**2).sum()**(1/2)
    #     norm_hny = (mtot**-1 * spny**-2 * mv(torch.chain_matmul(Vny,Pbiny,kmn_test),m_ntot_test)**2).sum()**(1/2) if nystrom==1 else \
    #                (mtot**-1 * spny**-2 * mv(torch.chain_matmul(Vny,Pbiny,kny),m_mtot)**2).sum()**(1/2)

    #     # print(norm_h,norm_hny)

    #     A = torch.zeros(m,t,dtype=torch.float64).addr(1/mtot*spny,1/ntot*sp) # A = outer(1/mtot*self.spny,1/ntot*self.sp)
    #     B = torch.zeros(m,t,dtype=torch.float64).addr(vpkm_ny,vpkm) # B = torch.outer(vpkm_ny,vpkm)
    #     C = torch.chain_matmul(Vny,Pbiny,kmn,Pbi,V.T)

    #     return(norm_h**-1*norm_hny**-1*(A*B*C).sum().item()) # <h_ny^* , h^*>/(||h_ny^*|| ||h^*||)

    # def eval_nystrom_eigenfunctions(self,t=None,m=None):
    #     # a adapter au GPU


    #     n1,n2 = (self.n1,self.n2)
    #     ntot = n1+n2
    #     m1,m2 = (self.nxlandmarks,self.nylandmarks)
    #     mtot = m1+m2

    #     mtot=self.nxlandmarks + self.nylandmarks
    #     ntot= self.n1 + self.n2
        
    #     t = 60   if t is None else t # pour éviter un calcul trop long # t= self.n1 + self.n2
    #     m = mtot if m is None else m
        
    #     kmn   = self.compute_kmn()
    #     Pbiny = self.compute_covariance_centering_matrix(sample='xy',quantization=True)
    #     Pbi   = self.compute_covariance_centering_matrix(sample='xy',quantization=False)
        
    #     Vny  = self.evny[:m]
    #     V    = self.ev[:t] 
    #     spny = self.spny[:m]
    #     sp   = self.sp[:t]

    #     # print(((spny * mtot)**(-1/2))[:3],'\n',((sp*ntot)**-(1/2))[:3])
    #     return( ((((spny * mtot)**(-1/2))*((sp*ntot)**-(1/2)*torch.chain_matmul(Vny,Pbiny,kmn,Pbi, V.T)).T).T).diag())
        


        # cev=(ev[:trunc[-1]].t()*coefs).t()
        

        # hip_wny_w = torch.matmul() 





