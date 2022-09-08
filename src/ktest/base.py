import torch
import pandas as pd
import numpy as np
from typing_extensions import Literal
from typing import Optional,Callable,Union,List
from ktest.kernels import gauss_kernel_mediane,mediane,gauss_kernel,linear_kernel

class Model:
    def __init__(self):        
        super(Model, self).__init__()
        self.has_model = False
        

    def init_model(self,approximation_cov='standard',approximation_mmd='standard',
                    m=None,r=None,landmark_method='random',anchors_basis='w'):
        '''
        
        Parameters
        ----------
            approximation_cov : str in 'standard','nystrom1',nystrom2','nystrom3','quantization'. 
                                In practice, we only use 'standard' and 'nystrom3'. 
                                It is the method used to compute the covariance structures. 
            approximation_mmd : str (same as approximation_cov), the method used to compute
                                the difference between the mean embeddings. 
            m : int, the total number of landmarks. 
            r : int, the total number of anchors. 
            landmark_method : str in 'random','kmeans', the method to determine the landmarks. 
            anchors_basis : str in 'k','s','w'. The anchors are determined as the eigenvectors of:  
                            'k' : the gram matrix of the landmarks. 
                            's' : the centered covariance of the landmarks
                            'w' : the within group covariance of the landmarks (possible only if 'landmark_method' is 'random'.
        Returns
        ------- 
        It is not possible to use nystrom for small datasets (n<100)
        '''

        self.approximation_cov = approximation_cov
        self.m = m
        self.r = r
        self.landmark_method = landmark_method
        self.anchors_basis = anchors_basis
        self.approximation_mmd = approximation_mmd
        self.has_model = True

class Data:

    def __init__(self):        
        super(Data, self).__init__()
        self.center_by = None        
        self.has_data = False   
        self.has_landmarks = False
        self.quantization_with_landmarks_possible = False

        # attributs initialisés 
        self.data = {'x':{},'y':{}}
        self.main_data=None

        # Results
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

    def init_xy(self,x,y,data_name ='data',main=True ):
        '''
        This function initializes the attributes `x` and `y` of the Tester object 
        which contain the two datasets in torch.tensors format.

        A faire : mieux gérer les noms de variables des données secondaires

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
                xy = xy.double()
            else : 
                token = False
                print(f'unknown data type {type(xy)}')
            if token:
                if sxy == 'x':
                    self.data['x'][data_name] = {'X':xy,'p':xy.shape[1]}           
                    self.data['x']['n'] = xy.shape[0]                
                if sxy == 'y':
                    self.data['y'][data_name] = {'X':xy,'p':xy.shape[1]}           
                    self.data['y']['n'] = xy.shape[0]                
                    
                self.has_data = True
            if main:
                self.main_data = data_name

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
        n1,n2,n = self.get_n1n2n()
        self.data['x']['index']=pd.Index(range(1,n1+1)) if x_index is None else pd.Index(x_index) if isinstance(x_index,list) else x_index 
        self.data['y']['index']=pd.Index(range(n1,n)) if y_index is None else pd.Index(y_index) if isinstance(y_index,list) else y_index
        assert(len(self.data['x']['index']) == self.data['x']['n'])
        assert(len(self.data['y']['index']) == self.data['y']['n'])
        
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
        self.variables = range(self.data['x'][self.main_data]['p']) if variables is None else variables
        self.var = pd.DataFrame(index=self.variables)
        self.vard = {v:{} for v in self.variables}

    def init_data_from_dataframe(self,dfx,dfy,kernel='gauss_median',dfx_meta=None,dfy_meta=None,center_by=None,verbose=0,data_name='data'):
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

            data_name (default : 'data') : str,
                the name of the data in the structure data 

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
        self.init_xy(dfx,dfy,data_name=data_name)
        self.init_index_xy(dfx.index,dfy.index)
        
        self.init_variables(dfx.columns)
        self.init_kernel(kernel)
        self.init_metadata(dfx_meta,dfy_meta) 
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
            self.obs.index = self.get_index()
        else:
            self.obs= pd.DataFrame(index=self.get_index())

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

        x,y = self.get_xy()
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
        





        