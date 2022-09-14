import torch
from torch import mv
import pandas as pd


from .kernel_trick import KernelTrick
"""
Ces fonctions calculent les coordonnées des projections des embeddings sur des sous-espaces d'intérêt 
dans le RKHS. Sauf la projection sur ce qu'on appelle pour l'instant l'espace des résidus, comme cette 
projection nécessite de nombreux calculs intermédiaires, elle a été encodée dans un fichier à part. 
"""

class ProjectionOps(KernelTrick):

    def __init__(self):
        super(ProjectionOps,self).__init__()

    def compute_proj_on_eigenvectors(self,t=None,verbose=0):
        
        self.verbosity(function_name='compute_proj_on_eigenvectors',
                    dict_of_variables={
                    't':t,
                    },
                    start=True,
                    verbose = verbose)
                    
        cov = self.approximation_cov
        sp,ev = self.get_spev('covw')
        
        tmax = 200
        t = tmax if (t is None and len(sp)>tmax) else len(sp) if (t is None or len(sp)<t) else t

        pkm=self.compute_pkm_new()
        upk=self.compute_upk_new(t)
        n1,n2,n = self.get_n1n2n()

        if cov == 'standard' or 'nystrom' in cov: 
            proj = (n1*n2*n**-2*sp[:t]**(-2)*mv(ev.T[:t],pkm)*upk).numpy()
            # proj = (n1*n2*n**-2*sp[:t]**(-3/2)*mv(ev.T[:t],pkm)*upk).cumsum(axis=1).numpy()
        if cov == 'quantization':
            proj = (sp[:t]**(-3/2)*mv(ev.T[:t],pkm)*upk).numpy()


        self.verbosity(function_name='compute_proj_on_eigenvectors',
                                    dict_of_variables={
                    't':t,
                    },
                    start=False,
                    verbose = verbose)
        return(proj,t)

    def compute_proj_kfda_new(self,t=None,verbose=0):
        # je n'ai plus besoin de trunc, seulement d'un t max 
        """ 
        Computes the vector of projection of the embeddings on the discriminant axis corresponding 
        to the KFDA statistic with a truncation parameter equal to t and stores the results as a column 
        of the attribute `df_proj_kfda`. 
        
        The projection is given by the formula :

                h^T kx =  \sum_{p=1:t} n1*n2 / ( lp*n)^2 [up^T PK omega] up^T P K   

        More details in the description of the method compute_kfdat(). 

        """

        proj_kfda_name = self.get_kfdat_name() 

        if proj_kfda_name in self.df_proj_kfda and str(t) in self.df_proj_kfda[proj_kfda_name]:
            if verbose : 
                print('Proj on discriminant axis Already computed')
        else:
            proj,t = self.compute_proj_on_eigenvectors(t=t)
            proj_kfda = proj.cumsum(axis=1)
            trunc = range(1,t+1) 
        
            if proj_kfda_name in self.df_proj_kfda:
                print(f"écrasement de {proj_kfda_name} dans df_proj_kfda")
            self.df_proj_kfda[proj_kfda_name] = pd.DataFrame(proj_kfda,index= self.get_xy_index(),columns=[str(t) for t in trunc])
        return(proj_kfda_name)


    def compute_proj_kpca_new(self,t=None,verbose=0):
        """ 
        
        """
        
        proj_kpca_name = self.get_kfdat_name() 

        if proj_kpca_name in self.df_proj_kpca and str(t) in self.df_proj_kpca[proj_kpca_name]:
            if verbose : 
                print('Proj on discriminant axis Already computed')
        else:
            proj,t = self.compute_proj_on_eigenvectors(t=t)
            proj_kfda = proj
            trunc = range(1,t+1) 
        
            if proj_kpca_name in self.df_proj_kpca:
                print(f"écrasement de {proj_kpca_name} dans df_proj_kpca")
            self.df_proj_kpca[proj_kpca_name] = pd.DataFrame(proj_kfda,index= self.get_xy_index(),columns=[str(t) for t in trunc])
        return(proj_kpca_name)

    def compute_proj_mmd_new(self,verbose=0):
        mmd_name = self.get_mmd_name()
        mmd = self.approximation_mmd
        if mmd_name in self.df_proj_mmd :
            if verbose : 
                print('Proj on discriminant axis Already computed')
        else:
            self.verbosity(function_name='compute_proj_mmd',
                    dict_of_variables={
                    'approximation':mmd,
                    },
                    start=True,
                    verbose = verbose)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            n1,n2,n = self.get_n1n2n()

            m = self.compute_omega_new(quantization=(mmd=='quantization'))
            if mmd == 'standard':
                K = self.compute_gram_new()
            
            
            proj = torch.matmul(K,m)
            if mmd_name in self.df_proj_mmd:
                print(f"écrasement de {mmd_name} dans df_proj_mmd")
            self.df_proj_mmd[mmd_name] = pd.DataFrame(proj,index=self.get_xy_index(),columns=['mmd'])
            # self.df_proj_mmd[name]['sample'] = ['x']*n1 + ['y']*n2
            
            self.verbosity(function_name='compute_proj_mmd',
                                    dict_of_variables={
                    'approximation':mmd,
                    },
                    start=False,
                    verbose = verbose)


