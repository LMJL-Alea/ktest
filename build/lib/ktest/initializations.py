import torch
import numpy as np
import pandas as pd
from ktest.kernels import gauss_kernel_mediane,mediane,gauss_kernel,linear_kernel
# from ktest._testdata import TestData
from time import time

from typing_extensions import Literal
from typing import Optional,Callable,Union,List


def init_testdata(self,x,y,x_index=None,y_index=None,variables=None,kernel=None,name=None):
    if name is None:
        name = 'data'+len(self.dict_data)
    if name in self.dict_data:
        print(f"{name} overwritten")
    self.dict_data[name] = TestData(x,y,x_index,y_index,variables,kernel)



def init_data_from_dataframe(self,dfx,dfy,kernel=None):
    x = dfx.to_numpy()
    y = dfy.to_numpy()
    x_index = dfx.index
    y_index = dfy.index
    variables = dfx.columns
    self.init_data(x,y,x_index,y_index,variables,kernel)

def init_data(self,
        x:Union[np.array,torch.tensor]=None,
        y:Union[np.array,torch.tensor]=None,
        x_index:List = None,
        y_index:List = None,
        variables:List = None,
        kernel:str='gauss_median'):
    """
    kernel : default 'gauss_median' for the gaussian kernel with median bandwidth
            'gauss_median_w' where w is a float for the gaussian kernel with a fraction of the median as the bandwidth 
            'gauss_x' where x is a float for the gaussian kernel with x bandwidth    
            'linear' for the linear kernel
            for a designed kernel, this parameter can be a function. 
    """
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
    
    assert(len(self.x_index) == self.n1)
    assert(len(self.y_index) == self.n2)

    self.index = self.x_index.append(self.y_index) 
    self.variables = range(x.shape[1]) if variables is None else variables

    self.xmask = self.x_index.isin(self.x_index)
    self.ymask = self.y_index.isin(self.y_index)
    self.imask = self.index.isin(self.index)
    self.ignored_obs = None

    if type(kernel) == str:
        kernel_params = kernel.split(sep='_')
        self.kernel_name = kernel
        if kernel_params[0] == 'gauss':
            if len(kernel_params)==2 and kernel_params[1]=='median':
                self.kernel,self.kernel_bandwidth = gauss_kernel_mediane(x,y,return_mediane=True)
            elif len(kernel_params)==2 and kernel_params[1]!='median':
                self.kernel_bandwidth = float(kernel_params[1])
                self.kernel = lambda x,y:gauss_kernel(x,y,self.kernel_bandwidth) 
            elif len(kernel_params)==3 and kernel_params[1]=='median':
                self.kernel_bandwidth = float(kernel_params[2])*mediane(x,y)
                self.kernel = lambda x,y:gauss_kernel(x,y,self.kernel_bandwidth) 
        if kernel_params[0] == 'linear':
            self.kernel = linear_kernel
    else:
        self.kernel = kernel
        self.kernel_name = 'specified by user'

    self.has_data = True        

def init_model(self,approximation_cov='standard',approximation_mmd='standard',
                m=None,r=None,landmark_method='random',anchors_basis='W'):

    self.approximation_cov = approximation_cov
    self.m = m
    self.r = r
    self.landmark_method = landmark_method
    self.anchors_basis = anchors_basis
    self.approximation_mmd = approximation_mmd


# def init_model(self,approximation_cov='standard',approximation_mmd='standard',
#                 m=None,r=None,landmark_method='random',anchors_basis='W',name=None):

#     if name is None:
#         name = 'model'+len(self.dict_model)
#     if name in self.dict_model:
#         print(f'{name} overwritten')
#     self.dict_model[name] = {
#                 'approximation_cov' : approximation_cov,
#                 'm' : m,
#                 'r' : r,
#                 'landmark_method' : landmark_method,
#                 'anchors_basis' : anchors_basis,
#                 'approximation_mmd' : approximation_mmd,
#                 }
    


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
