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

    def compute_proj_on_eigenvectors(self,t=None,condition=None,samples=None,marked_obs_to_ignore=None,verbose=0):
        
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

        pkm=self.compute_pkm()
        upk=self.compute_upk(t,proj_condition=condition,proj_samples=samples,proj_marked_obs_to_ignore=marked_obs_to_ignore)
        n1,n2,n = self.get_n1n2n()

        if cov == 'standard' or 'nystrom' in cov:  
            proj = (n1*n2*n**-2*sp[:t]**(-2)*mv(ev.T[:t],pkm)*upk).numpy()
        if cov == 'quantization':
            proj = (sp[:t]**(-3/2)*mv(ev.T[:t],pkm)*upk).numpy()


        self.verbosity(function_name='compute_proj_on_eigenvectors',
                                    dict_of_variables={
                    't':t,
                    },
                    start=False,
                    verbose = verbose)
        return(proj,t)

    def projections(self,t=None,condition=None,samples=None,marked_obs_to_ignore=None,verbose=0):

        """ 
        Computes the vector of projection of the embeddings on the discriminant axis corresponding 
        to the KFDA statistic with a truncation parameter equal to t and stores the results as a column 
        of the attribute `df_proj_kfda`. 
        
        The projection is given by the formula :

                h^T kx =  \sum_{p=1:t} n1*n2 / ( lp*n)^2 [up^T PK omega] up^T P K   

        More details in the description of the method compute_kfdat(). 

        """

        proj_name = self.get_kfdat_name(condition=condition,samples=samples,marked_obs_to_ignore=marked_obs_to_ignore) 

        if proj_name in self.df_proj_kfda and str(t) in self.df_proj_kfda[proj_name]:
            if verbose : 
                print('Proj on discriminant axis Already computed')
        else:
            proj,t = self.compute_proj_on_eigenvectors(t=t,condition=condition,samples=samples,marked_obs_to_ignore=marked_obs_to_ignore)
            proj_kpca = proj
            proj_kfda = proj.cumsum(axis=1)
            trunc = range(1,t+1) 
        
            if proj_name in self.df_proj_kfda:
                print(f"écrasement de {proj_name} dans df_proj_kfda")
            if proj_name in self.df_proj_kpca:
                print(f"écrasement de {proj_name} dans df_proj_kpca")
            index = self.get_index(condition=condition,samples=samples,marked_obs_to_ignore=marked_obs_to_ignore,in_dict=False)
            self.df_proj_kfda[proj_name] = pd.DataFrame(proj_kfda,index=index,columns=[str(t) for t in trunc])
            self.df_proj_kpca[proj_name] = pd.DataFrame(proj_kpca,index=index,columns=[str(t) for t in trunc])
        return(proj_name)
        
    def compute_proj_on_unidirectional_mmd(self,t=None,verbose=0):
        
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

        pkm=self.compute_pkm()
        upk=self.compute_upk(t)
        n1,n2,n = self.get_n1n2n()

        if cov == 'standard' or 'nystrom' in cov: 
            proj = (mv(ev.T[:t],pkm)*upk).numpy()
            # proj = (n1*n2*n**-2*sp[:t]**(-3/2)*mv(ev.T[:t],pkm)*upk).cumsum(axis=1).numpy()
        if cov == 'quantization':
            proj = (mv(ev.T[:t],pkm)*upk).numpy()


        self.verbosity(function_name='compute_proj_on_eigenvectors',
                                    dict_of_variables={
                    't':t,
                    },
                    start=False,
                    verbose = verbose)

        return(proj,t)

    def compute_proj_on_MMD(self,verbose=0):
        
        self.verbosity(function_name='compute_proj_on_eigenvectors',
                    dict_of_variables={
                    },
                    start=True,
                    verbose = verbose)
        
        mmd = self.approximation_mmd            
        m = self.compute_omega(quantization=(mmd=='quantization'))
        if mmd == 'standard':
            K = self.compute_gram()
        
        proj = torch.matmul(K,m)
            
        self.verbosity(function_name='compute_proj_on_eigenvectors',
                                    dict_of_variables={
                    },
                    start=False,
                    verbose = verbose)
        return(proj)

    def projections_MMD(self,t=None,verbose=0):
        """ 
        Computes the vector of projection of the embeddings on the discriminant axis corresponding 
        to the MMD statistic with a truncation parameter equal to t and with no truncation and stores 
        the results as column of the attribute `df_proj_mmd` et df_proj_unidirectional_mmd. 
        
        """

        mmd_name = self.get_mmd_name()
        
        if mmd_name in self.df_proj_mmd and (mmd_name in self.df_proj_unidirectional_mmd and str(t) in self.df_proj_unidirectional_mmd[mmd_name]):
            if verbose : 
                print('Proj on MMD discriminant axis Already computed')

        else:
            proj_unidirectional_mmd,t = self.compute_proj_on_unidirectional_mmd(t=t,verbose=verbose)
            proj_unidirectional_mmd = proj_unidirectional_mmd.cumsum(axis=1)
            proj_mmd = self.compute_proj_on_MMD(verbose=verbose)
            trunc = range(1,t+1) 
        
            if mmd_name in self.df_proj_mmd:
                print(f"écrasement de {mmd_name} dans df_proj_mmd")
            if mmd_name in self.df_proj_unidirectional_mmd:
                print(f"écrasement de {mmd_name} dans df_proj_unidirectional_mmd")
            self.df_proj_mmd[mmd_name] = pd.DataFrame(proj_mmd,index= self.get_index(in_dict=False),columns=['mmd'])
            self.df_proj_unidirectional_mmd[mmd_name] = pd.DataFrame(proj_unidirectional_mmd,index= self.get_index(in_dict=False),columns=[str(t) for t in trunc])
        return(mmd_name)





