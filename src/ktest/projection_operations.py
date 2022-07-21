import torch
from torch import mv
import pandas as pd

"""
Ces fonctions calculent les coordonnées des projections des embeddings sur des sous-espaces d'intérêt 
dans le RKHS. Sauf la projection sur ce qu'on appelle pour l'instant l'espace des résidus, comme cette 
projection nécessite de nombreux calculs intermédiaires, elle a été encodée dans un fichier à part. 
"""

def compute_proj_kfda(self,t=None,name=None,verbose=0,outliers_in_obs=None):
    # je n'ai plus besoin de trunc, seulement d'un t max 
    """ 
    Computes the vector of projection of the embeddings on the discriminant axis corresponding 
    to the KFDA statistic with a truncation parameter equal to t and stores the results as a column 
    of the attribute `df_proj_kfda`. 
    
    The projection is given by the formula :

            h^T kx =  \sum_{p=1:t} n1*n2 / ( lp*n)^2 [up^T PK omega] up^T P K   

    More details in the description of the method compute_kfdat(). 

    """

    cov,mmd = self.approximation_cov,self.approximation_mmd
    anchors_basis = self.anchors_basis
    

    suffix_nystrom = anchors_basis if 'nystrom' in cov else ''
    suffix_outliers = outliers_in_obs if outliers_in_obs is not None else ''
    name = name if name is not None else outliers_in_obs if outliers_in_obs is not None else f'{cov}{mmd}{suffix_nystrom}' 
    sp,ev = self.spev['xy'][f'{cov}{suffix_nystrom}{suffix_outliers}']['sp'],self.spev['xy'][f'{cov}{suffix_nystrom}{suffix_outliers}']['ev']
    
    tmax = 200
    t = tmax if (t is None and len(sp)>tmax) else len(sp) if (t is None or len(sp)<t) else t
        
    if name in self.df_proj_kfda and str(t) in self.df_proj_kfda[name]:
        if verbose : 
            print('Proj on discriminant axis Already computed')
    else:
        trunc = range(1,t+1) 
        self.verbosity(function_name='compute_proj_kfda',
                dict_of_variables={
                't':t,
                'approximation_cov':cov,
                'approximation_mmd':mmd,
                'name':name},
                start=True,
                verbose = verbose)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pkm=self.compute_pkm(outliers_in_obs=outliers_in_obs)
        upk=self.compute_upk(t,outliers_in_obs=outliers_in_obs)
        n1,n2,n = self.get_n1n2n(outliers_in_obs=outliers_in_obs)

        if cov == 'standard' or 'nystrom' in cov: 
            proj = (n1*n2*n**-2*sp[:t]**(-2)*mv(ev.T[:t],pkm)*upk).cumsum(axis=1).numpy()
            # proj = (n1*n2*n**-2*sp[:t]**(-3/2)*mv(ev.T[:t],pkm)*upk).cumsum(axis=1).numpy()
        if cov == 'quantization':
            proj = (sp[:t]**(-3/2)*mv(ev.T[:t],pkm)*upk).cumsum(axis=1).numpy()

        if name in self.df_proj_kfda:
            print(f"écrasement de {name} dans df_proj_kfda")
        self.df_proj_kfda[name] = pd.DataFrame(proj,index= self.get_index(outliers_in_obs=outliers_in_obs),columns=[str(t) for t in trunc])
        self.df_proj_kfda[name]['sample'] = ['x']*n1 + ['y']*n2
        
        self.verbosity(function_name='compute_proj_kfda',
                                dict_of_variables={
                't':t,
                'approximation_cov':cov,
                'approximation_mmd':mmd,
                'name':name},
                start=False,
                verbose = verbose)
    return(name)

def compute_proj_kpca(self,t=None,approximation_cov='standard',sample='xy',name=None,verbose=0,outliers_in_obs=None):
    # je n'ai plus besoin de trunc, seulement d'un t max 
    """ 
    
    """
    
    cov,mmd = self.approximation_cov,self.approximation_mmd
    name = name if name is not None else \
            outliers_in_obs if outliers_in_obs is not None else \
            f'{cov}{mmd}{sample}' 
    # name = name if name is not None else f'{cov}{mmd}' 

    anchors_basis = self.anchors_basis
    suffix_nystrom = anchors_basis if 'nystrom' in cov else ''
    suffix_outliers = '' if outliers_in_obs is None else outliers_in_obs

    sp = self.spev['xy'][f'{cov}{suffix_nystrom}{suffix_outliers}']['sp']
    ev = self.spev['xy'][f'{cov}{suffix_nystrom}{suffix_outliers}']['ev']
    
    if name in self.df_proj_kpca :
        if verbose : 
            print('Proj on variable directions Already computed')
    else:
        self.verbosity(function_name='compute_proj_kpca',
                dict_of_variables={
                't':t,
                # 'approximation_cov':approximation_cov,
                'sample':sample,
                'name':name},
                start=True,
                verbose = verbose)

        tmax = 200
        t = tmax if (t is None and len(sp)>tmax) else len(sp) if (t is None or len(sp)<t) else t
        trunc = range(1,t+1) 


        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pkm=self.compute_pkm(outliers_in_obs=outliers_in_obs)
        upk=self.compute_upk(t,outliers_in_obs=outliers_in_obs)
        n1,n2,n = self.get_n1n2n(outliers_in_obs=outliers_in_obs)

        if cov == 'standard' or 'nystrom' in cov: 
            proj = (n1*n2*n**-2*sp[:t]**(-2)*mv(ev.T[:t],pkm)*upk).numpy()
            # proj = (n1*n2*n**-2*sp[:t]**(-3/2)*mv(ev.T[:t],pkm)*upk).cumsum(axis=1).numpy()
        if cov == 'quantization':
            proj = (sp[:t]**(-3/2)*mv(ev.T[:t],pkm)*upk).numpy()



        if name in self.df_proj_kpca:
            print(f"écrasement de {name} dans df_proj_kpca")
        

        index = self.get_index(sample=sample,outliers_in_obs=outliers_in_obs)
        self.df_proj_kpca[name] = pd.DataFrame(proj,index=index,columns=[str(t) for t in trunc])
        self.df_proj_kpca[name]['sample'] = ['x']*n1*('x' in sample) + ['y']*n2*('y' in sample)
                
        self.verbosity(function_name='compute_proj_kpca',
                                dict_of_variables={
                't':t,
                'approximation_cov':approximation_cov,
                'sample':sample,
                'name':name},
                start=False,
                verbose = verbose)

def compute_proj_mmd(self,approximation='standard',name=None,verbose=0,outliers_in_obs=None):
    name = name if name is not None else outliers_in_obs if outliers_in_obs is not None else f'{approximation}' 
    if name in self.df_proj_mmd :
        if verbose : 
            print('Proj on discriminant axis Already computed')
    else:
        self.verbosity(function_name='compute_proj_mmd',
                dict_of_variables={
                'approximation':approximation,
                'name':name},
                start=True,
                verbose = verbose)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        n1,n2,n = self.get_n1n2n(outliers_in_obs=outliers_in_obs)

        m = self.compute_omega(quantization=(approximation=='quantization'),outliers_in_obs=outliers_in_obs)
        if approximation == 'standard':
            K = self.compute_gram(outliers_in_obs=outliers_in_obs)
        
        
        proj = torch.matmul(K,m)
        if name in self.df_proj_mmd:
            print(f"écrasement de {name} dans df_proj_mmd")
        self.df_proj_mmd[name] = pd.DataFrame(proj,index=self.get_index(outliers_in_obs=outliers_in_obs),columns=['mmd'])
        self.df_proj_mmd[name]['sample'] = ['x']*n1 + ['y']*n2
        
        self.verbosity(function_name='compute_proj_mmd',
                                dict_of_variables={
                'approximation':approximation,
                'name':name},
                start=False,
                verbose = verbose)

def init_df_proj(self,which,name=None,outliers_in_obs=None):
    # if name is None:
    #     name = self.main_name
    
    proj_options = {'proj_kfda':self.df_proj_kfda,
               'proj_kpca':self.df_proj_kpca,
               'proj_mmd':self.df_proj_mmd,
               'proj_residuals':self.df_proj_residuals # faire en sorte d'ajouter ça
               }
    if which in proj_options:
        dict_df_proj = proj_options[which]
        nproj = len(dict_df_proj)
        names = list(dict_df_proj.keys())
        if nproj == 0:
            print(f'{which} has not been computed yet')
        if nproj == 1:
            if name is not None and name != names[0]:
                print(f'{name} not corresponding to {names[0]}')
            else:
                df_proj = dict_df_proj[names[0]]
        if nproj >1:
            if name is not None and name not in names:
                print(f'{name} not found in {names}, default projection:  {names[0]}')
                df_proj = dict_df_proj[names[0]]
            elif name is None:
                print(f'projection not specified, default projection : {names[0]}') 
                df_proj = dict_df_proj[names[0]]
            else: 
                df_proj = dict_df_proj[name]
    elif which in self.variables:
        n1,n2,n = self.get_n1n2n(outliers_in_obs=outliers_in_obs)
        datax,datay = self.get_xy(outliers_in_obs=outliers_in_obs,name_data=name)
        loc_variable = self.variables.get_loc(which)
        index = self.get_index(outliers_in_obs=outliers_in_obs)
        df_proj = pd.DataFrame(torch.cat((datax[:,loc_variable],datay[:,loc_variable]),axis=0),index=index,columns=[which])
        # df_proj['sample']=['x']*n1 + ['y']*n2
    else:
        print(f'{which} not recognized')
        
    return(df_proj)

