import torch
import numpy as np
import pandas as pd
from ktest.kernels import gauss_kernel_mediane,mediane,gauss_kernel,linear_kernel
# from ktest._testdata import TestData
from time import time

from typing_extensions import Literal
from typing import Optional,Callable,Union,List



# Désuet ? 
# def init_testdata(self,x,y,x_index=None,y_index=None,variables=None,kernel=None,name=None):
#     if name is None:
#         name = 'data'+len(self.dict_data)
#     if name in self.dict_data:
#         print(f"{name} overwritten")
#     self.dict_data[name] = TestData(x,y,x_index,y_index,variables,kernel)



def init_xy(self,x,y):
    '''
    This function initializes the attributes `x` and `y` of the Tester object 
    which contain the two datasets in torch.tensors format.


    Parameters
    ----------
        x : pandas.Series (univariate testing), pandas.DataFrame, numpy.ndarray or torch.Tensor
        The dataset corresponding to the first sample 

        y : pandas.Series (univariate testing), pandas.DataFrame, numpy.ndarray or torch.Tensor
        The dataset corresponding to the second sample

    Attributes Initialized
    ---------- ----------- 
        x : torch.Tensor, the first dataset
        y : torch.Tensor, the second dataset 
        n1_initial : int, the original size of `x`, in case we decide to rerun the test with a subset of the cells (deprecated ?)
        n2_initial : int, the original size of `y`, in case we decide to rerun the test with a subset of the cells (deprecated ?)
        n1 : int, size of `x`
        n2 : int, size of `y` 
        has_data : boolean, True if the Tester object has data (deprecated ?)

    '''
    # Tester works with torch tensor objects 

    for xy,sxy in zip([x,y],'xy'):
        token = True
        if isinstance(xy,pd.Series):
            xy = torch.from_numpy(xy.to_numpy().reshape(-1,1)).double()
        if isinstance(xy,pd.DataFrame):
            xy = torch.from_numpy(xy.to_numpy()).double()
        elif isinstance(xy, np.ndarray):
            xy = torch.from_numpy(xy).double()
        elif isinstance(xy,torch.Tensor):
            xy = xy
        else : 
            token = False
            print(f'unknown data type {type(xy)}')
        if token:
            if sxy == 'x':
                self.x = xy
                self.n1 = xy.shape[0]
            if sxy == 'y':
                self.y = xy
                self.n2 = xy.shape[0]
            self.has_data = True

         
def init_index_xy(self,x_index = None,y_index = None):
    '''
    This function initializes the attributes of the data indexes 
    `x_index` and `y_index` of `x` and `y` of the Tester object. 


    Parameters
    ----------
        x_index (default : None): None, list or pandas.Series
            if x_index is None, the index is a list of numbers from 1 to n1
            else, the list or pandas.Series should contain the ordered list 
            of index corresponding to the observations of x

        y_index (default : None): None, list or pandas.Series
            if y_index is None, the index is a list of numbers from n1+1 to n1+n2
            else, the list or pandas.Series should contain the ordered list 
            of index corresponding to the observations of y

    Attributes Initialized
    ---------- ----------- 
        
        x_index : pandas.Index, the indexes of the first dataset `x` 
        y_index : pandas.Index, the indexes of the second dataset `y`
        index : pandas.Index, the concatenation of `x_index` and `y_index`

    '''
    # generates range index if no index
    self.x_index=pd.Index(range(1,self.n1+1)) if x_index is None else pd.Index(x_index) if isinstance(x_index,list) else x_index 
    self.y_index=pd.Index(range(self.n1,self.n1+self.n2)) if y_index is None else pd.Index(y_index) if isinstance(y_index,list) else y_index
    assert(len(self.x_index) == self.n1)
    assert(len(self.y_index) == self.n2)
    self.index = self.x_index.append(self.y_index)

def init_variables(self,variables = None):
    '''
    Initializes the variables names in the attribute `variables`. 
    
    Parameters
    ----------
        variables (default : None) : None, list or pandas.Series
        An iterable containing the variable names,
        if None, the attribute variables is a list of numbers from 0 to the number of variables -1. 

    Attributes Initialized
    ---------- ----------- 
        variables : the list of variable names

    '''
    self.variables = range(self.x.shape[1]) if variables is None else variables


def init_data_from_dataframe(self,dfx,dfy,kernel='gauss_median',dfx_meta=None,dfy_meta=None,center_by=None,verbose=0):
    '''
    This function initializes all the information concerning the data of the Tester object.

    Parameters
    ----------
        dfx : pandas.Series (univariate testing) or pandas.DataFrame (univariate or multivariate testing)
            the columns correspond to the variables 
            the index correspond to the data indexes
            the dataframe contain the data
            dfx and dfy should have the same column names.    
            if pandas.Series, the variable name is set to 'univariate'
            
        dfy : pandas.Series (univariate testing) or pandas.DataFrame (univariate or multivariate testing)
            the columns correspond to the variables 
            the index correspond to the data indexes
            the dataframe contain the data        
            dfx and dfy should have the same column names.     
            if pandas.Series, the variable name is set to 'univariate'

        kernel : str or function (default : 'gauss_median') 
            if kernel is a string, it have to correspond to the following synthax :
                'gauss_median' for the gaussian kernel with median bandwidth
                'gauss_median_w' where w is a float for the gaussian kernel with a fraction of the median as the bandwidth 
                'gauss_x' where x is a float for the gaussian kernel with x bandwidth    
                'linear' for the linear kernel
            if kernel is a function, 
                it should take two torch.tensors as input and return a torch.tensor contaning
                 the kernel evaluations between the lines (observations) of the two inputs. 

        dfx_meta (optional): pandas.DataFrame,
            A dataframe containing meta information on the first dataset. 

        dfy_meta (optional): pandas.DataFrame,
            A dataframe containing meta information on the second dataset. 

        center_by (optional) : str, 
            either a column of self.obs or a combination of columns with the following syntax
            - starts with '#' to specify that it is a combination of effects
            - each effect should be a column of self.obs, preceded by '+' is the effect is added and '-' if the effect is retired. 
            - the effects are separated by a '_'
            exemple : '#-celltype_+patient'



    Attributes Initialized
    ---------- ----------- 
        x : torch.Tensor, the first dataset
        y : torch.Tensor, the second dataset 
        n1_initial : int, the original size of `x`, in case we decide to rerun the test with a subset of the cells (deprecated ?)
        n2_initial : int, the original size of `y`, in case we decide to rerun the test with a subset of the cells (deprecated ?)
        n1 : int, size of `x`
        n2 : int, size of `y` 
        has_data : boolean, True if the Tester object has data (deprecated ?)
        x_index : pandas.Index, the indexes of the first dataset `x` 
        y_index : pandas.Index, the indexes of the second dataset `y`
        index : pandas.Index, the concatenation of `x_index` and `y_index`
        variables : the list of variable names
        center_by : str,
            is set to None if center_by is a string but the Tester object doesn't have an `obs` dataframe. 
        obs : pandas.DataFrame, 
            Its index correspond to the attribute `index`
            It is the concatenation of dfx_meta and dfy_meta, 
            It contains at least one column 'sample' equal to 'x' if the observation comes from the 
            firs dataset and 'y' otherwise. 
        kernel : the kernel function to be used to compute Gram matrices. 

    '''
    if isinstance(dfx,pd.Series):
        dfx = dfx.to_frame(name='univariate')
        dfy = dfy.to_frame(name='univariate')
    
    self.verbose = verbose
    self.init_xy(dfx,dfy)
    self.init_index_xy(dfx.index,dfy.index)
    
    self.init_variables(dfx.columns)
    self.init_kernel(kernel)
    self.init_metadata(dfx_meta,dfy_meta) 
    self.init_masks()
    self.set_center_by(center_by)
    
def set_center_by(self,center_by=None):
    '''
    Initializes the attribute `center_by` which allow to automatically center the data with respect 
    to a stratification of the datasets informed in the meta information dataframe `obs`. 
    This centering is independant from the centering applied to the data to compute statistic-related centerings. 
    
    Parameters
    ----------
        center_by (default = None) : None or str, 
            if None, the attribute center_by is set to None 
            and no centering will be done during the computations of the Gram matrix. 
            else, either a column of self.obs or a combination of columns with the following syntax
            - starts with '#' to specify that it is a combination of effects
            - each effect should be a column of self.obs, preceded by '+' is the effect is added and '-' if the effect is retired. 
            - the effects are separated by a '_'
            exemple : '#-celltype_+patient'


    Attributes Initialized
    ---------- ----------- 
        center_by : str,
            is set to None if center_by is a string but the Tester object doesn't have an `obs` dataframe. 

    '''
    self.center_by = None
    if center_by is not None and hasattr(self,'obs'):
        self.center_by = center_by
        

def init_metadata(self,dfx_meta=None,dfy_meta=None):
    '''
    This function initializes the attribute `obs` containing metainformation on the data. 

    Parameters
    ----------
        dfx_meta (default = None): pandas.DataFrame,
            A dataframe containing meta information on the first dataset. 

        dfy_meta (default = None): pandas.DataFrame,
            A dataframe containing meta information on the second dataset. 
            

    Attributes Initialized
    ---------- ----------- 
        obs : pandas.DataFrame, 
            Its index correspond to the attribute `index`
            It is the concatenation of dfx_meta and dfy_meta, 
            It contains at least one column 'sample' equal to 'x' if the observation comes from the 
            firs dataset and 'y' otherwise. 

    '''
        
    if dfx_meta is not None :
        dfx_meta['sample'] = ['x']*len(dfx_meta)
        dfy_meta['sample'] = ['y']*len(dfy_meta)
        self.obs = pd.concat([dfx_meta,dfy_meta],axis=0)
        self.obs.index = self.index



def init_data(self,
        x:Union[np.array,torch.tensor]=None,
        y:Union[np.array,torch.tensor]=None,
        x_index:List = None,
        y_index:List = None,
        variables:List = None,
        kernel:str='gauss_median',
        dfx_meta:pd.DataFrame = None,
        dfy_meta:pd.DataFrame = None,
        center_by:str = None,
        verbose = 0):
    
    '''
    
    Parameters
    ----------

        kernel : str or function (default : 'gauss_median') 
            if kernel is a string, it have to correspond to the following synthax :
                'gauss_median' for the gaussian kernel with median bandwidth
                'gauss_median_w' where w is a float for the gaussian kernel with a fraction of the median as the bandwidth 
                'gauss_x' where x is a float for the gaussian kernel with x bandwidth    
                'linear' for the linear kernel
            if kernel is a function, 
                it should take two torch.tensors as input and return a torch.tensor contaning
                 the kernel evaluations between the lines (observations) of the two inputs. 

    Attributes Initialized
    ---------- ----------- 

    '''
    # remplacer xy_index par xy_meta

    self.verbose = verbose
    self.init_xy(x,y)
    self.init_index_xy(x_index,y_index) 
    self.init_variables(variables)
    self.init_kernel(kernel)
    self.init_masks()
    self.init_metadata(dfx_meta,dfy_meta)
    self.set_center_by(center_by)
    self.has_data = True        

def init_kernel(self,kernel):
    '''
    
    Parameters
    ----------

    Returns
    ------- 
    '''

    x = self.x
    y = self.y
    verbose = self.verbose

    if type(kernel) == str:
        kernel_params = kernel.split(sep='_')
        self.kernel_name = kernel
        if kernel_params[0] == 'gauss':
            if len(kernel_params)==2 and kernel_params[1]=='median':
                self.kernel,self.kernel_bandwidth = gauss_kernel_mediane(x,y,return_mediane=True,verbose=verbose)
            elif len(kernel_params)==2 and kernel_params[1]!='median':
                self.kernel_bandwidth = float(kernel_params[1])
                self.kernel = lambda x,y:gauss_kernel(x,y,self.kernel_bandwidth) 
            elif len(kernel_params)==3 and kernel_params[1]=='median':
                self.kernel_bandwidth = float(kernel_params[2])*mediane(x,y,verbose=verbose)
                self.kernel = lambda x,y:gauss_kernel(x,y,self.kernel_bandwidth) 
        if kernel_params[0] == 'linear':
            self.kernel = linear_kernel
    else:
        self.kernel = kernel
        self.kernel_name = 'specified by user'


def init_model(self,approximation_cov='standard',approximation_mmd='standard',
                m=None,r=None,landmark_method='random',anchors_basis='W'):
    '''
    
    Parameters
    ----------

    Returns
    ------- 
    It is not possible to use nystrom for small datasets (n<100)
    '''

    

    n1,n2 = self.n1,self.n2
    if "nystrom" in approximation_cov and (n1<100 or n2<100): 
        self.approximation_cov = 'standard'
    self.approximation_cov = approximation_cov
    self.m = m
    self.r = r
    self.landmark_method = landmark_method
    self.anchors_basis = anchors_basis
    self.approximation_mmd = approximation_mmd

def verbosity(self,function_name,dict_of_variables=None,start=True,verbose=0):
    '''
    
    Parameters
    ----------

    Returns
    ------- 
    '''
    
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

