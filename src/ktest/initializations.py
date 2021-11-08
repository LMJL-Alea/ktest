import torch
import numpy as np
import pandas as pd
from ktest.kernels import gauss_kernel_mediane
from time import time


def init_data(self,x,y,kernel=None, x_index=None, y_index=None,variables=None):
    # Tester works with torch tensor objects 
    self.x = torch.from_numpy(x).double() if (isinstance(x, np.ndarray)) else x
    self.y = torch.from_numpy(y).double() if (isinstance(y, np.ndarray)) else y

    self.n1_initial = x.shape[0]
    self.n2_initial = y.shape[0]
    
    self.n1 = x.shape[0]
    self.n2 = y.shape[0]

    # generates range index if no index
    self.x_index=pd.Index(range(1,self.n1+1)) if x_index is None else pd.Index(x_index) if isinstance(x_index,list) else x_index 
    self.y_index=pd.Index(range(self.n1,self.n1+self.n2)) if y_index is None else pd.Index(y_index) if isinstance(y_index,list) else y_index
    self.index = self.x_index.append(self.y_index) 
    self.variables = range(x.shape[1]) if variables is None else variables

    self.xmask = self.x_index.isin(self.x_index)
    self.ymask = self.y_index.isin(self.y_index)
    self.imask = self.index.isin(self.index)
    self.ignored_obs = None
    if kernel is None:
        self.kernel,self.mediane = gauss_kernel_mediane(x,y,return_mediane=True)        
    else:
        self.kernel = kernel
        
    # if self.df_kfdat.empty:
    #     self.df_kfdat = pd.DataFrame(index= list(range(1,self.n1+self.n2)))
    self.has_data = True        


def verbosity(self,function_name,dict_of_variables=None,start=True,verbose=0):
    if verbose >0:
        end = ' ' if verbose == 1 else '\n'
        if start:  # pour verbose ==1 on start juste le chrono mais écris rien     
            self.start_times[function_name] = time()
            if verbose >1 : 
                print(f"Starting {function_name} ...",end= end)
                if dict_of_variables is not None:
                    for k,v in dict_of_variables.items():
                        if verbose ==2:
                            print(f'\t {k}:{v}', end = '') 
                        else:
                            print(f'\t {k}:{v}')
                
        else: 
            start_time = self.start_times[function_name]
            print(f"Done {function_name} in  {time() - start_time:.2f}")

# def get_nobs(self,):
#     if hasattr(self,'n1'):
#         return(self.n1,self.n2)
#     else:
#         return(self)
