import torch
import pandas as pd
import numpy as np
from typing_extensions import Literal
from typing import Optional,Callable,Union,List
from .kernels import mediane,gauss_kernel,linear_kernel
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

def get_kernel_name(function,bandwidth,median_coef):
    n = ''
    if function == 'gauss':
        n+=function
        if bandwidth == 'median':
            n+= f'_{median_coef}median' if median_coef != 1 else '_median' 
        else: 
            n+=f'_{bandwidth}'
        
        
    elif function == 'linear':
        n+=function
    else:
        n='user_specified'
    return(n)


def init_test_params(test='kfda',nystrom=False,m=None,r=None,landmark_method='random',
            anchors_basis='w'):
    return({'test':test,
            'nystrom':nystrom,
            'm':m,
            'r':r,
            'landmark_method':landmark_method,
            'anchors_basis':anchors_basis
    })


def init_kernel_params(function='gauss',bandwidth='median',median_coef=1,kernel_name=None):
    """
    Returns an object that defines the kernel
    """
    return(
        {'function':function,
            'bandwidth':bandwidth,
            'median_coef':median_coef,
            'kernel_name':kernel_name
            }
    )
    

class Model:
    def __init__(self):        
        super(Model, self).__init__()
        self.has_model = False
        
    def init_test_params(self,test='kfda',nystrom=False,m=None,r=None,landmark_method='random',anchors_basis='w'):
        '''
        
        Parameters
        ----------
            test : 'kfda' or 'mmd'
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

        self.test = test
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

    def get_test_params(self):
        
        return({'test':self.test,
            'nystrom':self.nystrom,
            'm':self.m_initial,
            'r':self.r,
            'landmark_method':self.landmark_method,
            'anchors_basis':self.anchors_basis
        })



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
        self.marked_obs_to_ignore = None
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
        self.df_proj_tmmd = {}
        self.df_proj_residuals = {}
        self.corr = {}     
        self.dict_mmd = {}
        self.spev = {'covw':{},'anchors':{},'residuals':{}} # dict containing every result of diagonalization
        # les vecteurs propres sortant de eigsy sont rangés en colonnes

        # for verbosity 
        self.start_times = {}
                
    def kernel(self,function='gauss',bandwidth='median',median_coef=1,kernel_name=None):
        '''
        
        Parameters
        ----------
            function (default = 'gauss') : str or function
                str in ['gauss','linear'] for gauss kernel or linear kernel. 
                function : kernel function specified by user

            bandwidth (default = 'median') : str or float
                str in ['median'] to use the median or a multiple of it as a bandwidth. 
                float : value of the bandwidth

            coef (default = 1) : float
                multiple of the median to use as bandwidth if kernel == 'gauss' and bandwidth == 'median' 

            correction (default = None) : None or str 
                if str : column of the metadata to correct the bandwidth   
        Returns
        ------- 
        '''

        x,y = self.get_xy()
        verbose = self.verbose
        has_bandwidth = False

        kernel_name = get_kernel_name(function=function,bandwidth=bandwidth,median_coef=median_coef) if kernel_name is None else kernel_name

        if function == 'gauss':
            if bandwidth == 'median':
                median = mediane(x,y,verbose=verbose)
                bandwidth = median_coef * median 
            kernel_ = lambda x,y:gauss_kernel(x,y,bandwidth) 
            has_bandwidth = True
            
            
        elif function == 'linear':
            kernel_ = linear_kernel
        else:
            kernel_ = function

        self.data[self.data_name]['kernel'] = kernel_
        self.data[self.data_name]['kernel_name'] = kernel_name
        if has_bandwidth:
            self.data[self.data_name]['kernel_bandwidth'] = bandwidth
        self.has_kernel = True
        self.kernel_params = init_kernel_params(function=function,bandwidth=bandwidth,median_coef=median_coef,kernel_name=kernel_name)

    def get_kernel_params(self):
        return(self.kernel_params)

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

    def _update_meta_data(self,metadata,):
        '''
        This function update the meta information about the data contained in the pandas.DataFrame 
        attribute `self.obs`

        The column 'sample' of `self.obs` is updated even if there is no meta data. 
        This column refers to the sample of origin of the data. 


        Parameters
        ----------
            nobs : int 
                number of observation concerned by the update

            metadata : pandas.DataFrame
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


        if metadata is not None:
            if self.obs.shape == (0,1):
                self.obs = metadata
            else:
                # self.obs.update(metadata)
                self.obs = pd.concat([self.obs,metadata[~metadata.index.isin(self.obs.index)]])


            for c in self.obs.columns:
                if len(self.obs[c].unique())<100:
                    # print(c,self.obs[c].unique())
                    self.obs[c] = self.obs[c].astype('category')

    def update_var_from_dataframe(self,df,verbose = 0):
        var = self.get_var()
        c_to_add = []
        for c in df.columns:
            if verbose>1:
                print(c,end=' ')
            token = False
            if 'univariate' in c and c in var:
                token = True
                nbef = sum(var[c]==1)
            if c not in var:
                c_to_add += [c]
                df[c] = df[c].astype('float64')
            else:
                if verbose>1:
                    print('update',end= '|')
                var[c].update(df[c].astype('float64'))
            if token:
                naft = sum(var[c]==1)
                if verbose >0:
                    print(f'\n tested from {nbef} to {naft}')
        df_to_add = df[c_to_add]
        self.var[self.data_name] = var.join(df_to_add)

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
                           metadata=None,
                           var_metadata=None,
                           update_current_data_name = True
                           ):
        nobs = len(x)
        if index is None and metadata is not None:
            index = metadata.index
        if data_name in self.data:
            old_index = self.data[data_name]['index']

            if len(index[index.isin(old_index)])==0:
                print('There is only new data')
                self._update_dict_data(x,data_name,update_current_data_name)
                index = self._update_index(nobs,index,data_name)
                self._update_meta_data(metadata=metadata)
                self._update_variables(variables,data_name)

            elif len(index[~index.isin(old_index)])==len(index):
                print('This dataset is already stored in tester.')
            else:
                print('There is some new data and already stored data, this situation is not implemented yet. ')
        else:
            self._update_dict_data(x,data_name)
            index = self._update_index(nobs,index,data_name)
            self._update_meta_data(metadata=metadata)
            self._update_variables(variables,data_name)
        if var_metadata is not None:
            self.update_var_from_dataframe(var_metadata)

    def add_data_to_Tester_from_dataframe(self,df,metadata=None,var_metadata=None,data_name='data',
                           update_current_data_name = True):
        '''
        
        '''
        x = df.to_numpy()
        index = df.index
        variables = df.columns

        self.add_data_to_Tester(x,data_name,index,variables,metadata,var_metadata,
                           update_current_data_name = update_current_data_name)

    def get_index(self,landmarks=False,condition=None,samples=None,marked_obs_to_ignore=None,in_dict=True):

        if landmarks:
            assert(self.has_landmarks)

        condition = self.condition if condition is None else condition
        samples = self.samples if samples is None else samples
        marked_obs_to_ignore = self.marked_obs_to_ignore if marked_obs_to_ignore is None else marked_obs_to_ignore
            
        samples_list = self.obs[condition].cat.categories.to_list() if samples == 'all' else samples

        index_output = {} if in_dict else pd.Index([])

        for sample in samples_list: 
            ooi = self.obs[self.obs[condition]==sample].index
            if marked_obs_to_ignore is not None:
                marked_obs = self.obs[self.obs[marked_obs_to_ignore]].index             
                ooi = ooi[~ooi.isin(marked_obs)]
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
            if in_dict:
                index_output[sample] = ooi
            else:
                
                index_output = index_output.append(ooi)
        return(index_output) 


    def get_kmeans_landmarks(self):
        
        condition = self.condition
        samples = self.samples
        
        samples_list = self.obs[condition].cat.categories.to_list() if samples == 'all' else samples
        dict_data = {}
        for sample in samples_list:
            kmeans_landmarks_name = self.get_kmeans_landmarks_name_for_sample(sample)
            dict_data[sample] = self.data[kmeans_landmarks_name]['X']
        return(dict_data)
    
    def get_data(self,landmarks=False,condition=None,samples=None,marked_obs_to_ignore=None):
        data_name = self.data_name
    
        if landmarks and self.landmark_method =='kmeans':
            dict_data = self.get_kmeans_landmarks()
            
        else:
            dict_index = self.get_index(landmarks=landmarks,condition=condition,samples=samples,marked_obs_to_ignore=marked_obs_to_ignore)
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

    def get_nobs(self,landmarks=False,condition=None,samples=None,marked_obs_to_ignore=None):
        
        dict_index = self.get_index(landmarks=landmarks,condition=condition,samples=samples,marked_obs_to_ignore=marked_obs_to_ignore)
        if landmarks and self.landmark_method == 'kmeans':
            dict_nobs = {k:len(v) for k,v in dict_index.items()}
        else:
            dict_nobs = {k:int(self.obs.index.isin(v).sum()) for k,v in dict_index.items()}
        dict_nobs['ntot'] = sum(list(dict_nobs.values()))
        return(dict_nobs)

    def get_ntot(self,landmarks=False):
        dict_nobs = self.get_nobs(landmarks=landmarks)
        return(dict_nobs['ntot'])

    def get_dataframes_of_data(self,landmarks=False,condition=None,samples=None,marked_obs_to_ignore=None):
        dict_data = self.get_data(landmarks=landmarks,condition=condition,samples=samples,marked_obs_to_ignore=marked_obs_to_ignore)
        dict_index = self.get_index(landmarks=landmarks,condition=condition,samples=samples,marked_obs_to_ignore=marked_obs_to_ignore)
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
    variables=None,metadata_list=None,kernel='gauss_median',var_metadata=None):
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

            metadata_list :  list of pandas.DataFrame
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
        if metadata_list is None:
            metadata_list = [None]*L
        if sample_list is None:
            sample_list = [f'x{l}' for l in range(1,L+1)]

        for x,sample,index,metadata in zip(data_list,sample_list,index_list,metadata_list):
            self.add_data_to_Tester(x,sample,data_name,index,variables,metadata,var_metadata=var_metadata)
 
    def init_L_groups_from_dataframe(self,df_list,data_name,sample_list=None,metadata_list=None,kernel='gauss_median',var_metadata=None):
        L = len(df_list)
        if metadata_list is None:
            metadata_list = [None]*L
        if sample_list is None:
            sample_list = [f'x{l}' for l in range(1,L+1)]
        for df,sample,metadata in zip(df_list,sample_list,metadata_list):
            self.add_data_to_Tester_from_dataframe(df,sample,metadata,data_name,var_metadata=var_metadata)


    def init_df_proj(self,proj,name=None,data_name=None):
        if data_name is None:
            data_name = self.current_data_name
        if proj == 'proj_kfda':
            df_proj = self.get_proj_kfda(name=name)
        elif proj == 'proj_kpca':
            df_proj = self.get_proj_kpca(name=name)
        elif proj == 'proj_mmd':
            df_proj = self.get_proj_mmd(name=name)
        elif proj == 'proj_tmmd':
            df_proj = self.get_proj_tmmd(name=name)
        elif proj == 'proj_residuals':
            df_proj = self.get_proj_residuals(name=name)

        elif proj in self.var[data_name].index:
            df_proj = pd.DataFrame(self.get_dataframe_of_all_data()[proj])
        elif proj =='obs':
            df_proj = self.obs
        else:
            print(f'{proj} not recognized')

        return(df_proj)




    def get_proj_kfda(self,name=None):
        if name is None:
            name = self.get_kfdat_name()
        if name in self.df_proj_kfda:
            return(self.df_proj_kfda[name])
        else:
            print(f"proj kfda '{name}' has not been computed yet")

    def get_proj_kpca(self,name=None):
        if name is None:
            name = self.get_kfdat_name()
        if name in self.df_proj_kpca:
            return(self.df_proj_kpca[name])
        else:
            print(f"proj kpca '{name}' has not been computed yet")

    def get_proj_mmd(self,name=None):
        if name is None:
            name = self.get_mmd_name()
        if name in self.df_proj_mmd:
            return(self.df_proj_mmd[name])
        else:
            print(f"proj mmd '{name}' has not been computed yet")

    def get_proj_tmmd(self,name=None):
        if name is None:
            name = self.get_mmd_name()
        if name in self.df_proj_mmd:
            return(self.df_proj_tmmd[name])
        else:
            print(f"proj tmmd '{name}' has not been computed yet")

    def get_proj_residuals(self,name=None):
        if name is None:
            name = self.get_residuals_name()
        if name in self.df_proj_residuals:
            return(self.df_proj_residuals[name])
        else:
            print(f"proj residuals '{name}' has not been computed yet")



    def make_groups_from_gene_presence(self,gene,data_name):

        dfg = self.init_df_proj(proj=gene,data_name=data_name)
        self.obs[f'pop{gene}'] = (dfg[gene]>=1).map({True: f'{gene}+', False: f'{gene}-'})
        self.obs[f'pop{gene}'] = self.obs[f'pop{gene}'].astype('category')

    def set_marked_obs_to_ignore(self,marked_obs_to_ignore=None):
        if marked_obs_to_ignore in self.obs.columns or marked_obs_to_ignore is None:
            self.marked_obs_to_ignore = marked_obs_to_ignore

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
        
    def get_data_name_condition_samples_marked_obs(self):
        return(self.data_name,self.condition,self.samples,self.marked_obs_to_ignore)
        
    def get_spev(self,slot='covw',t=None,center=None):
        """
        Return spectrum and eigenvectors.
        slot in ['covw','anchors','residuals']
        """

        spev_name = self.get_covw_spev_name() if slot=='covw' else \
                    self.get_anchors_name() if slot=='anchors' else \
                    self.get_residuals_name(t=t,center=center)
        
        try:
            return(self.spev[slot][spev_name]['sp'],self.spev[slot][spev_name]['ev'])
        except KeyError:
            print(f'KeyError : spev {spev_name} not in {slot} {self.spev[slot].keys()}')
                    
    # Names 

    def get_data_to_test_str(self,condition=None,samples=None,marked_obs_to_ignore=None):

        dn = self.data_name
        c = self.condition if condition is None else condition
        samples = self.samples if samples is None else samples
        mark = self.marked_obs_to_ignore if marked_obs_to_ignore is None else marked_obs_to_ignore


        # si les conditions et samples peuvent être mis en entrées, cente_by aussi
        smpl = '' if samples == 'all' else "".join(samples)
        cb = '' if self.center_by is None else f'_cb_{self.center_by}'    
        marking = '' if mark is None else f'_{mark}'
        return(f'{dn}{c}{smpl}{cb}{marking}')

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

    def get_kfdat_name(self,condition=None,samples=None,marked_obs_to_ignore=None):
        dtn = self.get_data_to_test_str(condition=condition,samples=samples,marked_obs_to_ignore=marked_obs_to_ignore)
        mn = self.get_model_str()

        return(f'{mn}_{dtn}')

    def get_residuals_name(self,t,center,condition=None,samples=None,marked_obs_to_ignore=None):
        dtn = self.get_data_to_test_str(condition=condition,samples=samples,marked_obs_to_ignore=marked_obs_to_ignore)
        
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
