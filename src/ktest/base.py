import torch
import pandas as pd
import numpy as np
from typing_extensions import Literal
from typing import Optional,Callable,Union,List
from ktest.kernels import gauss_kernel_mediane,mediane,gauss_kernel,linear_kernel,gauss_kernel_mediane_corrected_variance,gauss_kernel_mediane_log_corrected_variance
from torch import cat


def convert_to_torch_tensor(X):
    token = True
    if isinstance(X,pd.Series):
        X = torch.from_numpy(X.to_numpy().reshape(-1,1)).double()
    if isinstance(X,pd.DataFrame):
        X = torch.from_numpy(X.to_numpy()).double()
    elif isinstance(X, np.ndarray):
        X = torch.from_numpy(X).double()
    elif isinstance(X,torch.Tensor):
        X = X.double()
    else : 
        token = False
        print(f'unknown data type {type(X)}')            

    return(X)

def convert_to_pandas_index(index):
    if isinstance(index,list) or isinstance(index,range):
        return(pd.Index(index))
    else:
        return(index)

class Model:
    def __init__(self):        
        super(Model, self).__init__()
        self.has_model = False
        

    def init_model(self,nystrom=False,m=None,r=None,landmark_method='random',anchors_basis='w'):
        '''
        
        Parameters
        ----------
            nystrom (default = False) : bool
                Whether to use the nystrom approximation or not.
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

        self.nystrom = nystrom
        if nystrom:
            self.approximation_cov = 'nystrom3'
            self.approximation_mmd = 'standard'
        else:
            self.approximation_cov = 'standard'
            self.approximation_mmd = 'standard'

        # m_specified permet d'éviter une erreur d'arrondi sur m quand on initialise nystrom plusieurs fois et fait diminuer m
        self.m_initial = m 
        self.m = m
        self.r = r
        self.landmark_method = landmark_method
        self.anchors_basis = anchors_basis

        self.has_model = True
        self.nystrom_initialized = False

    def get_model(self):
        return(self.nystrom,self.landmark_method,self.anchors_basis,self.m_initial,self.r)


class Data:

    def __init__(self,verbose=0):        
        super(Data, self).__init__()
        self.center_by = None        
        self.has_data = False   
        self.has_landmarks = False
        self.has_kernel = False
        self.quantization_with_landmarks_possible = False

        # attributs initialisés 
        self.data = {}
        self.obs = pd.DataFrame(columns=['sample'])
        self.obs['sample'] = self.obs['sample'].astype('category')
        self.verbose=verbose
        self.var = {}
        self.vard = {}
        
        self.data_name = None
        self.condition = 'sample'
        self.samples = 'all'
        self.outliers_in_obs = None
        # self.data = {'x':{},'y':{}}
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
        self.spev = {'covw':{},'anchors':{},'residuals':{}} # dict containing every result of diagonalization
        # les vecteurs propres sortant de eigsy sont rangés en colonnes

        # for verbosity 
        self.start_times = {}

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
            self.obs.index = self.get_xy_index()
        else:
            self.obs= pd.DataFrame(index=self.get_xy_index())

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
        bandwidth = False
        if type(kernel) == str:
            kernel_params = kernel.split(sep='_')
            kernel_name = kernel
            if kernel_params[0] == 'gauss':
                if len(kernel_params)==2 and kernel_params[1]=='median': # ex: gauss_median
                    kernel_,kernel_bandwidth = gauss_kernel_mediane(x,y,return_mediane=True,verbose=verbose)
                    bandwidth = True
                elif len(kernel_params)==2 and kernel_params[1]!='median': # ex: gauss_3
                    kernel_bandwidth = float(kernel_params[1])
                    kernel_ = lambda x,y:gauss_kernel(x,y,kernel_bandwidth) 
                    bandwidth = True
                elif len(kernel_params)==3 and kernel_params[1]=='median': # ex: gauss_median_2
                    kernel_bandwidth = float(kernel_params[2])*mediane(x,y,verbose=verbose)
                    kernel_ = lambda x,y:gauss_kernel(x,y,kernel_bandwidth) 
                    bandwidth = True
                elif len(kernel_params)==4: 
                    if kernel_params[1]=='median' and kernel_params[2]=='variance': # ex : gauss_median_variance_var
                        variance_per_gene = self.get_var()[kernel_params[3]]
                        kernel_,kernel_bandwidth = gauss_kernel_mediane_corrected_variance(x,y,variance_per_gene,return_mediane=True,verbose=verbose)
                        bandwidth = True
                    
                    if kernel_params[1]=='median' and kernel_params[2]=='logvariance': # ex : gauss_median_logvariance_var
                        variance_per_gene = self.get_var()[kernel_params[3]]
                        kernel_,kernel_bandwidth = gauss_kernel_mediane_log_corrected_variance(x,y,variance_per_gene,return_mediane=True,verbose=verbose)
                        bandwidth = True


            if kernel_params[0] == 'linear':
                kernel_ = linear_kernel
        else:
            kernel_ = kernel
            kernel_name = 'specified by user'
        self.data[self.data_name]['kernel'] = kernel_
        self.data[self.data_name]['kernel_name'] = kernel_name
        if bandwidth:
            self.data[self.data_name]['kernel_bandwidth'] = kernel_bandwidth
        self.has_kernel = True

    # new version 

    def _update_dict_data(self,x,data_name,update_current_data_name=True): 
        '''
        Add the new data x to the torch.tensor of data `self.data[data_name]` after converting 
        x to a torch.tensor if needed.  

        Parameters
        ----------
            x : numpy.array, pandas.DataFrame, pandas.Series or torch.Tensor
                A table of any type containing the data
            
            data_name : str
                The data structure to update in the dict of data `self.data`.
                This name refers to the pretreatments and normalization steps applied to the data.


        Attributes Initialized
        ---------- ----------- 
            data : dict
                this dict structure contains data informations for each data_name
                    - 'X' : `torch.tensor` of data
                    - 'p' : dimension of the data, number of variables
                    - 'index' : `pandas.Index` of the observations
                    - 'variables' : `pandas.Index` of the variables
            
            current_data_name : this attributes take the value of the last `data_name` updated with this function.
                this attribute is used as default `data_name` when `data_name` is not informed in other functions.                    
        '''
        
        x = convert_to_torch_tensor(x)        
        if self.data_name is None:
            self.data_name = data_name

        if data_name not in self.data:
            self.data[data_name] = {'X':x,'index':pd.Index([]),'p':x.shape[1]}
        else:
            self.data[data_name]['X'] = cat((self.data[data_name]['X'],x),axis=0)
        
        if update_current_data_name:
            self.current_data_name = data_name
        self.has_data=True

    def _update_index(self,nobs,index,data_name):
        '''
        This function updates the value of the key 'index' in the attribute `self.data[data_name]`
        If there is no index, the index is set as a range of integer.

        Parameters
        ----------
            nobs : int
                number of observation to update the index
                this argument is used only if index is None to set the index as a range of integers

            index : list, pandas.index, iterable
                a list of value indexes which refers to the observations. 

            data_name : str
                The data structure to update in the dict of data `self.data`.
                This name refers to the pretreatments and normalization steps applied to the data.

        Attributes Initialized
        ---------- -----------
            data : Only the value of `self.data[data_name]['index']` is updated. 

        '''


        if index is None:
            i0 = 1 if len(self.data[data_name]['index'])==0 else self.data[data_name]['index'][-1] + 1
            index = range(i0,i0+nobs)

        index = convert_to_pandas_index(index)
        self.data[data_name]['index'] = self.data[data_name]['index'].append(index)

        return(index)

    def _update_meta_data(self,df_meta,):
        '''
        This function update the meta information about the data contained in the pandas.DataFrame 
        attribute `self.obs`

        The column 'sample' of `self.obs` is updated even if there is no meta data. 
        This column refers to the sample of origin of the data. 


        Parameters
        ----------
            nobs : int 
                number of observation concerned by the update

            df_meta : pandas.DataFrame
                table of metadata

            index : list, pandas.index, iterable
                a list of value indexes which refers to the observations. 

            sample: str
                a label for the observations corresponding to the observations updated


        Attributes Initialized
        ---------- -----------
            obs: pandas.DataFrame
                a structure containing every meta information of each observation

        '''
        # if df_meta is None:
        #     df_meta = pd.DataFrame([sample]*nobs,columns=['sample'],index=index)
        if self.obs.shape == (0,1):
            self.obs = df_meta
        else:
            # self.obs.update(df_meta)
            self.obs = pd.concat([self.obs,df_meta[~df_meta.index.isin(self.obs.index)]])


        for c in self.obs.columns:
            if len(self.obs[c].unique())<100:
                # print(c,self.obs[c].unique())
                self.obs[c] = self.obs[c].astype('category')

        # self.obs = pd.concat([self.obs,df_meta])
        # self.obs['sample'] = self.obs['sample'].astype('category')

    def _update_variables(self,variables,data_name):
        '''
        This function updates the variables information 

        Parameters
        ----------
            variables : list, pandas.index, iterable
                a list of value indexes which refers to the variables.

            data_name : str
                The data structure to update in the dict of data `self.data`.
                This name refers to the pretreatments and normalization steps applied to the data.

        Attributes Initialized
        ---------- -----------            
             data : dict
                 this dict structure contains data informations for each data_name
                    - 'X' : `torch.tensor` of data
                    - 'p' : dimension of the data, number of variables
                    - 'index' : `pandas.Index` of the observations
                    - 'variables' : `pandas.Index` of the variables    

            var : dict of pandas.DataFrame 
                a structure containing every meta information of each variable for each data_name

            vard : dict of dict
                a temporary structure containing every meta information of each variable for each data_name
                this structure is only used when it is not optimal to update the attribute `var`


        '''

        if variables is None:
            p = self.data[data_name]['p']
            variables = pd.Index(range(p))
        self.data[data_name]['variables']=variables
        self.var[data_name] = pd.DataFrame(index=variables)
        self.vard[data_name] = {v:{} for v in variables}

    def add_data_to_Tester(self,x,
                           data_name,
                           index=None,
                           variables=None,
                           df_meta=None,
                           df_var=None
                           ):
        nobs = len(x)
        if index is None and df_meta is not None:
            index = df_meta.index
        if data_name in self.data:
            old_index = self.data[data_name]['index']

            if len(index[index.isin(old_index)])==0:
                print('There is only new data')
                self._update_dict_data(x,data_name)
                index = self._update_index(nobs,index,data_name)
                self._update_meta_data(df_meta=df_meta)
                self._update_variables(variables,data_name)

            elif len(index[~index.isin(old_index)])==len(index):
                print('This dataset is already stored in tester.')
            else:
                print('There is some new data and already stored data, this situation is not implemented yet. ')
        else:
            self._update_dict_data(x,data_name)
            index = self._update_index(nobs,index,data_name)
            self._update_meta_data(df_meta=df_meta)
            self._update_variables(variables,data_name)
        if df_var is not None:
            self.update_var_from_dataframe(df_var)

    def add_data_to_Tester_from_dataframe(self,df,df_meta=None,df_var=None,data_name='data'):
        '''
        
        '''
        x = df.to_numpy()
        index = df.index
        variables = df.columns

        self.add_data_to_Tester(x,data_name,index,variables,df_meta,df_var)

    def get_index(self,landmarks=False):

        if landmarks:
            assert(self.has_landmarks)

        data_name,condition,samples,outliers_in_obs = self.get_data_name_condition_samples_outliers()        
        samples_list = self.obs[condition].cat.categories.to_list() if samples == 'all' else samples

        dict_index = {}

        for sample in samples_list: 
            ooi = self.obs[self.obs[condition]==sample].index
            if outliers_in_obs is not None:
                outliers = self.obs[self.obs[outliers_in_obs]].index             
                ooi = ooi[~ooi.isin(outliers)]
            if landmarks:            
                
                # When kmeans is used, the indexes do not refer to observations but to centroids
                if self.landmark_method == 'kmeans':
                    
                    kmeans_lm_name = self.get_kmeans_landmarks_name_for_sample(sample=sample)
                    try:
                        lm_index = self.data[kmeans_lm_name]['index']
                    except KeyError:
                        print(f'KeyError : {kmeans_lm_name} not in {list(self.data.keys())}')
                        
                    ooi = lm_index

                else:
                    landmarks_name = self.get_landmarks_name() 
                    try :
                        lm_index = self.obs[self.obs[f'{sample}_{landmarks_name}']==True].index
                    except KeyError:
                        print(f'KeyError : {sample}_{landmarks_name} not in {self.obs.columns.to_list()}')
                    ooi = ooi[ooi.isin(lm_index)]
            dict_index[sample] = ooi
        return(dict_index) 

    def get_kmeans_landmarks(self):
        
        condition = self.condition
        samples = self.samples
        
        samples_list = self.obs[condition].cat.categories.to_list() if samples == 'all' else samples
        dict_data = {}
        for sample in samples_list:
            kmeans_landmarks_name = self.get_kmeans_landmarks_name_for_sample(sample)
            dict_data[sample] = self.data[kmeans_landmarks_name]['X']
        return(dict_data)
    
    def get_data(self,landmarks=False):
        
        data_name = self.data_name
    
        if landmarks and self.landmark_method =='kmeans':
            dict_data = self.get_kmeans_landmarks()
            
        else:
            dict_index = self.get_index(landmarks=landmarks)
            if landmarks and self.landmark_method=='kmeans':
                dict_data = {k:self.data[data_name]['X'] for k in dict_index.keys()}
            else:
                dict_data = {k:self.data[data_name]['X'][self.obs.index.isin(v),:] for k,v in dict_index.items()}
        return(dict_data)
    
    def get_all_data(self,landmarks=False):
        if landmarks:
            print(f'get all data for landmarks is not mplemented yet')
        else: 
            return(self.data[self.data_name]['X'])

    def get_nobs(self,landmarks=False):
        
        dict_index = self.get_index(landmarks=landmarks)
        if landmarks and self.landmark_method == 'kmeans':
            dict_nobs = {k:len(v) for k,v in dict_index.items()}
        else:
            dict_nobs = {k:int(self.obs.index.isin(v).sum()) for k,v in dict_index.items()}
        dict_nobs['ntot'] = sum(list(dict_nobs.values()))
        return(dict_nobs)

    def get_ntot(self,landmarks=False):
        dict_nobs = self.get_nobs(landmarks=landmarks)
        return(dict_nobs['ntot'])

    def get_dataframes_of_data(self,landmarks=False):
        dict_data = self.get_data(landmarks=landmarks)
        dict_index = self.get_index(landmarks=landmarks)
        variables = self.data[self.data_name]['variables']

        dict_df = {}
        for s in dict_data.keys():
            x,i,v = dict_data[s],dict_index[s],variables
            dict_df[s] = pd.DataFrame(x,i,v)
        return(dict_df)
    
    def get_dataframe_of_all_data(self,landmarks=False):
        x = self.data[self.data_name]['X']
        i = self.obs.index
        v = self.data[self.data_name]['variables']
        return(pd.DataFrame(x,i,v))

    def get_variables(self):
        return(self.data[self.data_name]['variables'])

    def get_var(self):
        return(self.var[self.data_name])

    def get_vard(self):
        return(self.vard[self.data_name])   
        
    # Two sample 
    def init_xy(self,x,y,data_name,
                x_index=None,y_index=None,variables=None,df_var=None,dfx_meta=None,dfy_meta=None,kernel='gauss_median'):
        '''
        This function adds two dataset in the data_structure and name them 'x' and 'y'. 
        It is used when performing two-smaple test. 

        Parameters
        ----------
            x,y : numpy.array, pandas.DataFrame, pandas.Series or torch.Tensor
                A table of any type containing the data
            
            data_name : str
                The data structure to update in the dict of data `self.data`.
                This name refers to the pretreatments and normalization steps applied to the data.

            x_index,y_index : list, pandas.index, iterable
                a list of value indexes which refers to the observations. 

            variables : list, pandas.index, iterable
                a list of value indexes which refers to the variables.

            dfx_meta, dfy_meta : pandas.DataFrame
                table of metadata

            kernel : str
                Refers to the kernel function to use for testing

        Attributes Initialized
        ---------- ----------- 
            data : dict
                this dict structure contains data informations for each data_name
                    - 'X' : `torch.tensor` of data
                    - 'p' : dimension of the data, number of variables
                    - 'index' : `pandas.Index` of the observations
                    - 'variables' : `pandas.Index` of the variables
            
            obs: pandas.DataFrame
                a structure containing every meta information of each observation

            kernel : function
                kernel function to be used

        '''
            
        self.add_data_to_Tester(x,'x',data_name,x_index,variables,dfx_meta,df_var)
        self.add_data_to_Tester(y,'y',data_name,y_index,variables,dfy_meta)

        self.init_kernel(kernel)   

    def init_xy_from_dataframe(self,dfx,dfy,data_name,dfx_meta=None,dfy_meta=None,kernel='gauss_median',df_var=None):

        self.add_data_to_Tester_from_dataframe(df=dfx,sample='x',df_meta=dfx_meta,data_name=data_name,df_var=df_var)
        self.add_data_to_Tester_from_dataframe(df=dfy,sample='y',df_meta=dfy_meta,data_name=data_name)

        self.init_kernel(kernel)

    def get_n1n2n(self,landmarks=False):
        nobs = self.get_nobs(landmarks=landmarks)
        return(list(nobs.values()))

    def get_xy_index(self,sample='xy',landmarks=False):
        dict_index = self.get_index(landmarks=landmarks)
        indexes = list(dict_index.values())
        if sample=='xy':
            return(indexes[0].append(indexes[1]))
        if sample=='x':
            return(indexes[0])
        if sample=='y':
            return(indexes[1])

    def get_xy(self,landmarks=False):

        data = self.get_data(landmarks=landmarks)
        if len(data)>2:
            print('more than 2 groups',[f'{k}{len(v)}' for k,v in data.items()])
        return(list(data.values()))

    # L sample 
    def init_L_groups(self,data_list,data_name,sample_list=None,index_list=None,
    variables=None,df_meta_list=None,kernel='gauss_median',df_var=None):
        '''
        This function adds L dataset in the data_structure and name them according to
        `sample_list`. 
        It is used when performing L-sample test or kernel-MANOVA. 

        Parameters
        ----------
            data_list : list of numpy.array, pandas.DataFrame, pandas.Series or torch.Tensor
                A list of L tables of any type containing the data

            data_name : str
                The data structure to update in the dict of data `self.data`.
                This name refers to the pretreatments and normalization steps applied to the data.

            sample_list (default : None) : list of str
                The names to refer to each sample.
                If None, the samples are called 'x1', ... , 'xL'

            index_list : list of list, pandas.index, iterable
                a list of L lists of value indexes which refers to the observations. 

            variables : list, pandas.index, iterable
                a list of value indexes which refers to the variables.

            df_meta_list :  list of pandas.DataFrame
                list of L tables of metadata

            kernel : str
                Refers to the kernel function to use for testing

        Attributes Initialized
        ---------- ----------- 
            data : dict
                this dict structure contains data informations for each data_name
                    - 'X' : `torch.tensor` of data
                    - 'p' : dimension of the data, number of variables
                    - 'index' : `pandas.Index` of the observations
                    - 'variables' : `pandas.Index` of the variables

            obs: pandas.DataFrame
                a structure containing every meta information of each observation

            kernel : function
                kernel function to be used

        '''        
        
        L = len(data_list)
        if index_list is None:
            index_list = [None]*L
        if df_meta_list is None:
            df_meta_list = [None]*L
        if sample_list is None:
            sample_list = [f'x{l}' for l in range(1,L+1)]

        for x,sample,index,df_meta in zip(data_list,sample_list,index_list,df_meta_list):
            self.add_data_to_Tester(x,sample,data_name,index,variables,df_meta,df_var=df_var)
 
    def init_L_groups_from_dataframe(self,df_list,data_name,sample_list=None,df_meta_list=None,kernel='gauss_median',df_var=None):
        L = len(df_list)
        if df_meta_list is None:
            df_meta_list = [None]*L
        if sample_list is None:
            sample_list = [f'x{l}' for l in range(1,L+1)]
        for df,sample,df_meta in zip(df_list,sample_list,df_meta_list):
            self.add_data_to_Tester_from_dataframe(df,sample,df_meta,data_name,df_var=df_var)

    # Access data 
    def init_df_proj(self,proj,name=None,data_name=None):
        if data_name is None:
            data_name = self.current_data_name

        proj_options = {'proj_kfda':self.df_proj_kfda,
                'proj_kpca':self.df_proj_kpca,
                'proj_mmd':self.df_proj_mmd,
                'proj_residuals':self.df_proj_residuals # faire en sorte d'ajouter ça
                }

        if proj in proj_options:
            dict_df_proj = proj_options[proj]
            nproj = len(dict_df_proj)
            names = list(dict_df_proj.keys())
            if nproj == 0:
                print(f'{proj} has not been computed yet')
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
                    print(f'projection not specified with {proj}, default projection : {names[0]}') 
                    df_proj = dict_df_proj[names[0]]
                else: 
                    df_proj = dict_df_proj[name]
        elif proj in self.var[data_name].index:
            df_proj = pd.DataFrame(self.get_dataframe_of_all_data()[proj])
            # df_proj['sample']=['x']*n1 + ['y']*n2
        elif proj =='obs':
            df_proj = self.obs
        else:
            print(f'{proj} not recognized')

        return(df_proj)

    def make_groups_from_gene_presence(self,gene,data_name):

        dfg = self.init_df_proj(proj=gene,data_name=data_name)
        self.obs[f'pop{gene}'] = (dfg[gene]>=1).map({True: f'{gene}+', False: f'{gene}-'})
        self.obs[f'pop{gene}'] = self.obs[f'pop{gene}'].astype('category')

    def set_outliers_in_obs(self,outliers_in_obs=None):
        if outliers_in_obs in self.obs.columns or outliers_in_obs is None:
            self.outliers_in_obs = outliers_in_obs

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
        
        self.center_by = center_by      
           
    def set_test_data_info(self,data_name,condition,samples='all'):
        self.data_name = data_name
        self.condition = condition
        if condition in self.obs:
            self.obs[condition] = self.obs[condition].astype('category')
        self.samples = samples
        
    def get_data_name_condition_samples_outliers(self):
        return(self.data_name,self.condition,self.samples,self.outliers_in_obs)
        
    def get_spev(self,slot='covw',t=None,center=None):
        spev_name = self.get_covw_spev_name() if slot=='covw' else \
                    self.get_anchors_name() if slot=='anchors' else \
                    self.get_residuals_name(t=t,center=center)
        
        try:
            return(self.spev[slot][spev_name]['sp'],self.spev[slot][spev_name]['ev'])
        except KeyError:
            print(f'KeyError : spev {spev_name} not in {slot} {self.spev[slot].keys()}')
                    
    # Names 

    def get_data_to_test_str(self):
        dn = self.data_name
        c = self.condition
        smpl = '' if self.samples == 'all' else "".join(self.samples)
        cb = '' if self.center_by is None else f'_cb_{self.center_by}'    
        out = '' if self.outliers_in_obs is None else f'_{self.outliers_in_obs}'
        return(f'{dn}{c}{smpl}{cb}{out}')

    def get_model_str(self):
        ny = self.nystrom
        nys = 'ny' if self.nystrom else 'standard'
        ab = f'_basis{self.anchors_basis}' if ny else ''
        lm = f'_lm{self.landmark_method}_m{self.m}' if ny else ''
        return(f'{nys}{lm}{ab}')

    def get_landmarks_name(self):
        dtn = self.get_data_to_test_str()


        lm = self.landmark_method
        m = f'_m{self.m}'
        
        return(f'lm{lm}{m}_{dtn}')

    def get_kmeans_landmarks_name_for_sample(self,sample):
        landmarks_name = self.get_landmarks_name()
        return(f'{sample}_{landmarks_name}')


    def get_anchors_name(self,):
        dtn = self.get_data_to_test_str()

        lm = self.landmark_method
        ab = self.anchors_basis
        m = f'_m{self.m}'
        # r = f'_r{self.r}' # je ne le mets pas car il change en fonction des abérations du spectre
        return(f'lm{lm}{m}_basis{ab}_{dtn}')

    def get_covw_spev_name(self):
        dtn = self.get_data_to_test_str()
        mn = self.get_model_str()

        return(f'{mn}_{dtn}')

    def get_kfdat_name(self):
        dtn = self.get_data_to_test_str()
        mn = self.get_model_str()

        return(f'{mn}_{dtn}')

    def get_residuals_name(self,t,center):
        dtn = self.get_data_to_test_str()
        
        mn = self.get_model_str()

        c = center
        return(f'{c}{t}_{mn}_{dtn}')
        
    def get_mmd_name(self):
        dtn = self.get_data_to_test_str()
        mn = self.get_model_str()

        return(f'{mn}_{dtn}')

    def get_corr_name(self,proj):
        if proj in ['proj_kfda','proj_kpca']:
            name = f"{proj.split(sep='_')[1]}_{self.get_kfdat_name()}"
        else : 
            print(f'the correlation with {proj} is not handled yet.')
        return(name)
