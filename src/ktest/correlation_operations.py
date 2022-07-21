import torch
import numpy as np
import pandas as pd

"""
On peut vouloir calculer la corrélation entre les variables d'origine et les directions déterminées par 
nos méthodes. Ces fonctions permettent de le faire simplement. 
"""

def compute_corr_proj_var(self,t=None,sample='xy',which='proj_kfda',name_corr=None,
                        name_proj=None,prefix_col='',verbose=0): 
        # df_array,df_proj,csvfile,pathfile,trunc=range(1,60)):
    
    self.verbosity(function_name='compute_corr_proj_var',
            dict_of_variables={'t':t,
                        'sample':sample,'which':which,'name_corr':name_corr,'name_proj':name_proj,'prefix_col':prefix_col},
            start=True,
            verbose = verbose)

    self.prefix_col=prefix_col

    df_proj= self.init_df_proj(which,name_proj)
    t = df_proj.shape[1] - 1 # -1 pour la colonne sample

    x,y = self.get_xy()

    array = torch.cat((x,y),dim=0).numpy() if sample == 'xy' else x.numpy() if sample=='x' else y.numpy()
    index = self.get_index(sample=sample)
  
    
    df_array = pd.DataFrame(array,index=index,columns=self.variables)
    for ti in range(1,t):
        df_array[f'{prefix_col}{ti}'] = pd.Series(df_proj[f'{ti}'])
    name_corr = name_corr if name_corr is not None else which.split(sep='_')[1]+name_proj if name_proj is not None else which.split(sep='_')[1] + 'covariance'
    self.corr[name_corr] = df_array.corr().loc[self.variables,[f'{prefix_col}{ti}' for ti in range(1,t)]]
    
    self.verbosity(function_name='compute_corr_proj_var',
            dict_of_variables={'t':t,'sample':sample,'which':which,'name_corr':name_corr,'name_proj':name_proj,'prefix_col':prefix_col},
            start=False,
            verbose = verbose)

def find_correlated_variables(self,name=None,nvar=1,t=1,prefix_col=''):
    if name is None:
        name = self.get_names()['correlations'][0]
    if nvar==0:
        return(np.abs(self.corr[name][f'{prefix_col}{t}']).sort_values(ascending=False)[:])
    else: 
        return(np.abs(self.corr[name][f'{prefix_col}{t}']).sort_values(ascending=False)[:nvar])

