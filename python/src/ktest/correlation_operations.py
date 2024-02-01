import torch
import numpy as np
import pandas as pd
from .base import Base
"""
On peut vouloir calculer la corrélation entre les variables d'origine et les directions déterminées par 
nos méthodes. Ces fonctions permettent de le faire simplement. 
"""
class Correlations(Base):
    def __init__(self,data,metadata=None,var_metadata=None):
        super(Correlations,self).__init__(data,metadata=metadata,var_metadata=var_metadata)

    def compute_corr_proj_var(self,t=None,proj='proj_kfda',verbose=0): 
            # df_array,df_proj,csvfile,pathfile,trunc=range(1,60)):
        
        self.verbosity(function_name='compute_corr_proj_var',
                dict_of_variables={'t':t,
                            'proj':proj},
                start=True,
                verbose = verbose)
        prefix_col = proj.split(sep='_')[1]
        variables = self.variables
        df_proj= self.init_df_proj(proj)
        t = 30 if t is None else t 
        # df = self.get_dataframe_of_all_data()
        df = self.get_data(in_dict=False,dataframe=True)

        
        for ti in range(1,t):
            df[f't{ti}'] = pd.Series(df_proj[f'{ti}'])
            
        name_corr = self.get_corr_name(proj)
        
        self.corr[name_corr] = df.corr().loc[variables,[f't{ti}' for ti in range(1,t)]]
        
        self.verbosity(function_name='compute_corr_proj_var',
                dict_of_variables={'t':t,'proj':proj},
                start=False,
                verbose = verbose)


    def find_correlated_variables(self,proj='proj_kfda',nvar=1,t=1):
        name = self.get_corr_name(proj)
        if nvar==0:
            return(np.abs(self.corr[name][f't{t}']).sort_values(ascending=False)[:])
        else: 
            return(np.abs(self.corr[name][f't{t}']).sort_values(ascending=False)[:nvar])


    def get_corr_of_variable(self,proj,variable,t):
        return(self.find_correlated_variables(proj=proj,nvar=0,t=t)[variable])
