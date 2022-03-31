import torch
from torch import mv
import pandas as pd

def compute_proj_kfda(self,t=None,name=None,verbose=0):
    # je n'ai plus besoin de trunc, seulement d'un t max 
    """ 
    Projections of the embeddings of the observation onto the discriminant axis
    9 methods : 
    approximation_cov in ['standard','nystrom','quantization']
    approximation_mmd in ['standard','nystrom','quantization']
    
    Stores the result as a column in the dataframe df_proj_kfda
    """

    cov,mmd = self.approximation_cov,self.approximation_mmd
    anchors_basis = self.anchors_basis
    
    name = name if name is not None else f'{cov}{mmd}' 

    suffix_nystrom = anchors_basis if 'nystrom' in cov else ''
    sp,ev = self.spev['xy'][cov+suffix_nystrom]['sp'],self.spev['xy'][cov+suffix_nystrom]['ev']
    
    if name in self.df_proj_kfda :
        if verbose : 
            print('Proj on discriminant axis Already computed')
    else:
        tmax = 200
        t = tmax if (t is None and len(sp)>tmax) else len(sp) if (t is None or len(sp)<t) else t
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
        pkm=self.compute_pkm()
        epk=self.compute_epk(t)
        n1,n2 = (self.n1,self.n2) 
        n = n1+n2

        if cov == 'standard' or 'nystrom' in cov:
            proj = (n**-1*sp[:t]**(-3/2)*mv(ev.T[:t],pkm)*epk).cumsum(axis=1).numpy()
        if cov == 'quantization':
            proj = (sp[:t]**(-3/2)*mv(ev.T[:t],pkm)*epk).cumsum(axis=1).numpy()

        if name in self.df_proj_kfda:
            print(f"écrasement de {name} dans df_proj_kfda")
        self.df_proj_kfda[name] = pd.DataFrame(proj,index= self.index[self.imask],columns=[str(t) for t in trunc])
        self.df_proj_kfda[name]['sample'] = ['x']*n1 + ['y']*n2
        
        self.verbosity(function_name='compute_proj_kfda',
                                dict_of_variables={
                't':t,
                'approximation_cov':cov,
                'approximation_mmd':mmd,
                'name':name},
                start=False,
                verbose = verbose)

def compute_proj_kpca(self,t=None,approximation_cov='standard',sample='xy',name=None,verbose=0,anchors_basis=None):
    # je n'ai plus besoin de trunc, seulement d'un t max 
    """ 
    
    """
    name = name if name is not None else f'{approximation_cov}{sample}' 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    quantization = approximation_cov =='quantization'
    sp,ev = self.spev[sample][approximation_cov]['sp'],self.spev[sample][approximation_cov]['ev']
    P = self.compute_centering_matrix(sample=sample,quantization=quantization)    
    n1,n2 = (self.n1,self.n2) 
    n = (n1*('x' in sample)+n2*('y' in sample))
    

    tmax = 200

    t = tmax if (t is None and len(sp)+1>tmax) else len(sp) if (t is None and len(sp)+1<=tmax) else t
    trunc = range(1,t+1) 

    self.verbosity(function_name='compute_proj_kpca',
            dict_of_variables={
            't':t,
            'approximation_cov':approximation_cov,
            'sample':sample,
            'name':name},
            start=True,
            verbose = verbose)


    if approximation_cov =='quantization':
        Kmn = self.compute_kmn(sample=sample)
        A_12 = self.compute_quantization_weights(sample=sample,power=1/2)                
        proj = ( sp[:t]**(-1/2)*torch.chain_matmul(ev.T[:t],A_12,P,Kmn).T)
    elif approximation_cov == 'nystrom':
        Kmn = self.compute_kmn(sample=sample)
        Up = self.spev[sample]['anchors'][anchors_basis]['ev']
        Lp_inv = torch.diag(self.spev[sample]['anchors'][anchors_basis]['sp']**-1)
        proj = (  n**(-1/2)*sp[:t]**(-1/2)*torch.chain_matmul(ev.T[:t],P,Kmn.T,Up,Lp_inv,Up.T,Kmn).T).cumsum(axis=1).numpy()
    elif approximation_cov == 'standard':
        K = self.compute_gram(sample=sample)
        proj = (  n**(-1/2)*sp[:t]**(-1/2)*torch.chain_matmul(ev.T[:t],P,K).T).numpy()


    if name in self.df_proj_kpca:
        print(f"écrasement de {name} dans df_proj_kpca")
    
    index = self.index[self.imask] if sample=='xy' else self.x_index[self.xmask] if sample =='x' else self.y_index[self.ymask]
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

def compute_proj_mmd(self,approximation='standard',name=None,verbose=0):
    name = name if name is not None else f'{approximation}' 
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
        n1,n2 = (self.n1,self.n2) 
        n = n1+n2

        m = self.compute_omega(quantization=(approximation=='quantization'))
        if approximation == 'standard':
            K = self.compute_gram()
        
        
        proj = torch.matmul(K,m)
        if name in self.df_proj_mmd:
            print(f"écrasement de {name} dans df_proj_mmd")
        self.df_proj_mmd[name] = pd.DataFrame(proj,index= self.index[self.imask],columns=['mmd'])
        self.df_proj_mmd[name]['sample'] = ['x']*n1 + ['y']*n2
        
        self.verbosity(function_name='compute_proj_mmd',
                                dict_of_variables={
                'approximation':approximation,
                'name':name},
                start=False,
                verbose = verbose)



def init_df_proj(self,which,name=None):
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
                print(f'{name} not found in {names}')
                df_proj = dict_df_proj[self.main_name]
            else: 
                df_proj = dict_df_proj[name]
    elif which in self.variables:
        datax,datay = self.get_xy()
        loc_variable = self.variables.get_loc(which)
        df_proj = pd.DataFrame(torch.cat((datax[:,loc_variable],datay[:,loc_variable]),axis=0),index=self.index[self.imask],columns=[which])
        df_proj['sample']=['x']*self.n1 + ['y']*self.n2
    else:
        print(f'{which} not recognized')
        
    return(df_proj)



# def init_df_proj(self,which,name=None):
#     # if name is None:
#     #     name = self.main_name
    
#     proj_options = {'proj_kfda':self.df_proj_kfda,
#                'proj_kpca':self.df_proj_kpca,
#                'proj_mmd':self.df_proj_mmd,}
#     if which in proj_options:
#         dict_df_proj = proj_options[which]
#         nproj = len(dict_df_proj)
#         names = list(dict_df_proj.keys())
#         if nproj == 0:
#             print(f'{which} has not been computed yet')
#         if nproj == 1:
#             if name is not None and name != names[0]:
#                 print(f'{name} not corresponding to {names[0]}')
#             else:
#                 df_proj = dict_df_proj[names[0]]
#         if nproj >1:
#             if name is not None and name not in names:
#                 print(f'{name} not found in {names}')
#             # if name is None and self.main_name not in names:
#             #     print("the default name {self.main_name} is not in {names} so you need to specify 'name' argument")
#             # if name is None and self.main_name in names:
#                 df_proj = dict_df_proj[self.main_name]
#             else: 
#                 df_proj = dict_df_proj[name]
#     elif which in self.variables:
#         datax,datay = self.get_xy()
#         loc_variable = self.variables.get_loc[which]
#         df_proj = pd.DataFrame(torch.cat((datax[:,loc_variable],datay[:,loc_variable]),axis=0),index=self.index[self.imask],columns=[which])
#         df_proj['sample']=['x']*self.n1 + ['y']*self.n2
#     else:
#         print(f'{which} not recognized')
        
#     return(df_proj)

