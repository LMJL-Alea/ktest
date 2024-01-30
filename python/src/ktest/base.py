import torch
import pandas as pd
import numpy as np
from typing_extensions import Literal
from typing import Optional,Callable,Union,List
from torch import cat
from .utils import convert_to_torch_tensor,convert_to_pandas_index
from .names import Names    
from .kernel_function import Kernel_Function

class Base(Names,Kernel_Function):

    def __init__(self):   
        '''
        Attributes initialized : 
        ------------------------

        

        '''     
        super(Base, self).__init__()


        ### Input data ###
        self.data = {} 
        # is a dict containing the data matrix, indexes, variables names and the kernel function
        
        self.input_params={}
        # is a dict containing the parameters of the test to perform

        self.obs = pd.DataFrame(columns=['sample'])
        # is a pandas.DataFrame containing the meta information on the observations (cells)
        # the rows are the observations and each column correspond to a meta information 

        self.obs['sample'] = self.obs['sample'].astype('category')
        # this column of `self.obs` is still empty, it is initialized to be 
        # the default column containing the information on the groups to compare
        
        self.var = {}
        # is a dict containing a pandas.DataFrame of meta information on the variables (genes)
        # it is a dict because a specific dataframe is initialized for each key contained in 
        # the `self.data` structure
        # in the dataframes, the rows are the variables and each column is a meta information or 
        # an output of the method. This dataframe is used to store the results of univariate tests. 

        self.vard = {}
        # is dict containing a dict for each variable, 
        # it is used as an intermediate data structure before updating `self.var`
        # as it is easier to update dicts than dataframes. 

        self.hypotheses = {}
        # is a dict containing the matrices of interest associated to 
        # each general hypothesis that has been tested 


        ### Output statistics ### 
        
        self.df_kfdat = pd.DataFrame()
        self.df_pval = pd.DataFrame()
        # are two Pandas.DataFrames containing the computed kfda statistics and
        # associated asymptotic p-values (chi-square with T degree of freedom). The rows are the truncations and there is one 
        # columns per test performed with specific parameters and samples. 

        self.df_kfdat_contributions = pd.DataFrame()
        self.df_pval_contributions = pd.DataFrame()
        # are two pandas.DataFrames containing the unidirectional statistic 
        # associated to each eigendirection of the within-group covariance 
        # operator and associated asymptotic p-value (chi-square with one 
        # degree of freedom for each direction), 
        # `self.df_kfdat` is the cumulated sum of `self.df_kfdat_contributions`
        
        ## MMD statistic and p-value
        self.dict_mmd = {}
        self.dict_pval_mmd = {}
        # is a dict where each entry has the key corresponding to a specific 
        # set of parameters and the value is the associated MMD statistic
        # and associated permutation p-value respectively. 

        # Permutation statistics
        self.df_kfdat_perm = {}
        self.df_mmd_perm = pd.DataFrame()
        # These data structures contain the KFDA and MMD statistics associated to 
        # the permuted datasets that are used to estimate the quantile under H0 
        # with a permutation approach. 

        self.corr = {}     
        # is a dict where each entry is a pandas.DataFrame containing 
        # the correlations between the variables of interest with the 
        # directions of interest in the RKHS. 
        # These correlations are computed with functions stored in the file ./correlation_operations.py 

        ## Coordinates of the projections  
        self.df_proj_kfda = {} # on the discriminant axis  
        self.df_proj_kpca = {} # on the eigendirections of the within-group covariance operator 
        self.df_proj_mmd = {} # on the axis supported by \mu_1-\mu_2
        self.df_proj_orthogonal = {} # on the axis orthogonal to the discriminant axis that maximizes the variance
        self.df_proj_residuals = {} # on the eigendirections of the residual covariance operator of the kernel linear model
        
        # Eigenvectors
        self.spev = {'covw':{},'anchors':{},'residuals':{},'orthogonal':{}} 
        # is a dict containing every result of diagonalization 
        # (spectrums 'sp' and eigenvectors 'ev')
        # the eigenvectors are the columns of the matrix 'ev' and are computed with the function eigsy from apt
        

        ### Keys ###        
        # All the information stored in ktest is organized with respect to key strings. 
        # The keys stored in the following attributes correspond to the 'curent' info in use. 
        # Most of theses keys are updated automatically through functions. 
        self.data_name = None # current key to access to the slot of interest in `self.data`
        self.condition = 'sample' # current column of the metadata `self.obs` containing the samples to compare
        self.samples = 'all' # list of categories to compare from `self.obs[self.condition]`
        self.current_hyp = None # current key associated to the dict `self.hypotheses`
        self.log2fc_data = None # name of the dataframe on which are computed the log2 fold changes 
        self.univariate_name=None # user-customized prefix on the files of univariate results
        self.permutation_mmd_name=None
        self.permutation_kfda_name=None
        self.marked_obs_to_ignore = None

        # for verbosity 
        self.start_times = {}

        # tokens to assess global information on the ktest object :
        self.has_data = False # true if the ktest object contains data 
        self.has_landmarks = False # true if the nystrom landmarks have been determined
        self.has_anchors = False # true if the nystrom anchors have been determined
        self.has_model = False # true if the model parameters have been initiated with function set_test_params (not used)
        self.has_kernel = False # true if the kernel function has been initiated (not used)
        self.has_kfda_statistic = False # true if the kfda statistic has been computed (not used)

    def set_test_params(self,
                    stat='kfda',
                    nystrom=False,
                    n_landmarks=None,
                    n_anchors=None,
                    landmark_method='random',
                    anchor_basis='w',
                    permutation=False,
                    n_permutations=500,
                    seed_permutation=0,
                    verbose=0):
        '''
        It is not possible to use nystrom for small datasets (n<100)

        Parameters
        ----------
            stat (default=): str in ['kfda','mmd']
                Test statistic to be computed.

            nystrom (default = False) : bool
                Whether to use the nystrom approximation or not.

            n_landmarks : int, 
                the total number of landmarks. 

            n_anchors : int, 
                the total number of anchors. 

            landmark_method : str in 'random','kmeans', 
                the method to determine the landmarks. 

            anchor_basis (default = 'w') : str in ['k','s','w']. 
                The anchors are determined as the eigenvectors of:  
                            'k' : the gram matrix of the landmarks. 
                            's' : the centered covariance of the landmarks
                            'w' : the within group covariance of the landmarks (possible only if 'landmark_method' is 'random'.

            permutation (default = False): bool
                If `stat == 'kfda'`:
                    If True : a permutation p-value is computed with the function `ktest.multivariate_test()`
                    Else : an asymptotic p-value is computed
                If `stat == 'mmd'`, permutation is automatically set to True

            n_permutations (default = 500) : int
                Number of permutation wanted to obtain a permutation p-value
            
            seed_permutation (default=0) : int
                Random seed of the first permutation, each permutation seed is 
                then obtained by an unit incrementation of the previous seed. 

        '''

        input_params = {
            'stat':stat,
            'nystrom':nystrom,
            'n_landmarks':n_landmarks,
            'n_anchors':n_anchors,
            'landmark_method':landmark_method,
            'anchor_basis':anchor_basis,
            'permutation':permutation,
            'nperm':n_permutations,
            'seed_permutation':seed_permutation,
            }

        if input_params != self.input_params:
            self.input_params=input_params
            self.stat = stat
            self.nystrom = nystrom
            self.permutation = True if stat == 'mmd' else permutation
                
            if nystrom:
                self.nystrom_initialized = False
                self.n_landmarks_initial = n_landmarks 
                self.n_anchors = n_anchors
                self.landmark_method = landmark_method
                self.anchor_basis = anchor_basis
                self.approximation_cov = 'nystrom'
                self.approximation_mmd = 'standard'
                if self.has_data:
                    self.compute_nystrom_landmarks(verbose=verbose)
                    self.compute_nystrom_anchors(verbose=verbose)
            else:
                self.approximation_cov = 'standard'
                self.approximation_mmd = 'standard'

            if self.permutation:
                self.n_permutations = n_permutations
                self.seed_permutation = seed_permutation
                
            self.has_model = True

    def get_input_params(self):
        return(self.input_params)

    def get_test_params(self):
        test_params = {
                'stat':self.stat,
                'nystrom':self.nystrom,
                'permutation':self.permutation}

        if self.nystrom:
            test_params = {
                'n_landmarks':self.n_landmarks_initial,
                'n_anchors':self.n_anchors,
                'landmark_method':self.landmark_method,
                'anchor_basis':self.anchor_basis,
                **test_params
            }
        
        if self.permutation:
            test_params = {
                'n_permutations':self.n_permutations,
                'seed_permutation':self.seed_permutation,
                **test_params
            }
        return(test_params)
                
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
            token = False
            if 'univariate' in c and c in var:
                token = True
                nbef = sum(var[c]==1)
            if c not in var:
                c_to_add += [c]
                df[c] = df[c].astype('float64')
            else:
                var[c].update(df[c].astype('float64'))
            if token:
                naft = sum(var[c]==1)
        if verbose >0:
            if len(c_to_add)>0:
                s=f'- Update var'
                if verbose>1:
                    s+= f' {var.shape}'
                s+= f' with {len(c_to_add)} columns'
                print(s)
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
        self.initialize_vard(data_name=data_name,variables=variables)

    def initialize_vard(self,data_name=None,variables=None):
        if data_name is None: 
            data_name=self.data_name
        if variables is None:
            variables = self.get_variables(data_name=data_name)

        self.vard[data_name] = {v:{} for v in variables}

    def add_data_to_Ktest(self,x,
                           data_name,
                           index=None,
                           variables=None,
                           metadata=None,
                           var_metadata=None,
                           update_current_data_name = True,
                           verbose=0
                           ):
        
        if verbose>0:
            print(f"- Add data '{data_name}' to Ktest, dimensions={x.shape}")
        nobs = len(x)
        if index is None and metadata is not None:
            index = metadata.index
        if data_name in self.data:
            old_index = self.data[data_name]['index']

            if len(index[index.isin(old_index)])==0:
                if verbose>0:
                    print('\tThere is only new data')
                self._update_dict_data(x,data_name,update_current_data_name)
                index = self._update_index(nobs,index,data_name)
                self._update_meta_data(metadata=metadata)
                self._update_variables(variables,data_name)

            elif len(index[~index.isin(old_index)])==len(index):
                if verbose>0:
                    print('\tThis dataset is already stored in Ktest.')
            else:
                if verbose>0:
                    print('\tThere are index conflicts with new data and stored data, add only new data or change their index.')
        else:
            self._update_dict_data(x,data_name)
            index = self._update_index(nobs,index,data_name)
            self._update_meta_data(metadata=metadata)
            self._update_variables(variables,data_name)
        if var_metadata is not None:
            self.update_var_from_dataframe(var_metadata)

    def add_data_to_Ktest_from_dataframe(self,df,metadata=None,var_metadata=None,data_name='data',
                           update_current_data_name = True,verbose=0):
        '''
        
        '''
        x = df.to_numpy()
        index = df.index
        variables = df.columns

        self.add_data_to_Ktest(x,data_name,index,variables,metadata,var_metadata,
                           update_current_data_name = update_current_data_name,verbose=verbose)

    def get_samples_list(self,condition=None,samples=None,marked_obs_to_ignore=None):
        """
        Returns a list containing the names of the samples of interest. 

        Parameters
        ----------
            condition (default = None): str
                Column of the metadata that specify the dataset    
           
            samples (default = None): str 
                List of values to select in the column condition of the metadata
            
            marked_obs_to_ignore (default = None): str
                Column of the metadata specifying the observations to ignore

        """
        condition,samples,marked_obs_to_ignore = self.init_samples_condition_marked(condition=condition,
                                    samples=samples,
                                    marked_obs_to_ignore=marked_obs_to_ignore)
        marked_obs = self.obs[self.obs[marked_obs_to_ignore]].index if marked_obs_to_ignore is not None else []             
        
        return(self.obs[~self.obs.index.isin(marked_obs)][condition].cat.categories.to_list() if samples == 'all' else samples)    

    def init_samples_condition_marked(self,
                                      condition=None,
                                      samples=None,
                                      marked_obs_to_ignore=None):
        

        if condition is None:
            condition = self.condition 
            
        if samples is None:
            if condition == self.condition:
                samples = self.samples 
            else : 
                samples = 'all'

        if marked_obs_to_ignore is None:
            marked_obs_to_ignore = self.marked_obs_to_ignore 
            
        return(condition,samples,marked_obs_to_ignore)
    
    def get_index(self,landmarks=False,condition=None,samples=None,marked_obs_to_ignore=None,in_dict=True):
        """
        Returns the index of the observations of the Ktest object.

        Parameters
        ----------
            landmarks (default = False) : bool
                if True, focuses on the nystrom landmarks
                else, focuses on the observations. 
      
            condition (default = None): str
                Column of the metadata that specify the dataset  
  
            samples (default = None): str 
                List of values to select in the column condition of the metadata
            
            marked_obs_to_ignore (default = None): str
                Column of the metadata specifying the observations to ignore

            in_dict (default = True) : bool
                if True : returns a dictionary of the outputs associated to each sample
                else : returns an unique object containing all the outputs     

        """
        if landmarks:
            assert(self.has_landmarks)

        condition,samples,marked_obs_to_ignore = self.init_samples_condition_marked(condition=condition,
                                           samples=samples,
                                           marked_obs_to_ignore=marked_obs_to_ignore)
            
        samples_list = self.get_samples_list(condition,samples)
        
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

    def get_kmeans_landmarks(self,in_dict=True):
        
        condition = self.condition
        samples = self.samples
        
        samples_list = self.get_samples_list(condition,samples)
        data = {}
        for sample in samples_list:
            kmeans_landmarks_name = self.get_kmeans_landmarks_name_for_sample(sample)
            data[sample] = self.data[kmeans_landmarks_name]['X']
        if in_dict:
            return(data)
        else:
            return(torch.cat(list(data.values())))

    def get_data(self,
                 landmarks=False,
                 condition=None,
                 samples=None,
                 marked_obs_to_ignore=None,
                 data_name=None,
                 in_dict=True,
                 dataframe=False):
        """
        Returs the selected data in the desired format

        Parameters
        ----------
            landmarks (default = False) : bool
                if True, focuses on the nystrom landmarks
                else, focuses on the observations. 
      
            condition (default = None): str
                Column of the metadata that specify the dataset  
  
            samples (default = None): str 
                List of values to select in the column condition of the metadata
            
            marked_obs_to_ignore (default = None): str
                Column of the metadata specifying the observations to ignore

            in_dict (default = True) : bool
                if True : returns a dictionary of the outputs associated to each sample
                else : returns an unique object containing all the outputs  
            
            dataframe (default = False) : bool
                if True, the output tables are in pandas.DataFrame format. 
                else, the output tables are in torch.Tensor format. 

        """

        if data_name is None:    
            data_name = self.data_name

        if dataframe:
            v = self.get_variables(data_name=data_name)

        if landmarks and self.landmark_method =='kmeans':
            data = self.get_kmeans_landmarks(in_dict=in_dict)

        else:
            index = self.get_index(landmarks=landmarks,condition=condition,samples=samples,marked_obs_to_ignore=marked_obs_to_ignore,in_dict=in_dict)

            if in_dict:
                data = {}
                for k,i in index.items():
                    x = self.data[data_name]['X'][self.obs.index.isin(i),:]
                    data[k] = pd.DataFrame(x,i,v) if dataframe else x
            else:
                indexer_ = self.obs.index.get_indexer(index)
                # x = self.data[data_name]['X'][self.obs.index.isin(index),:]
                x = self.data[data_name]['X'][indexer_,:]
                data = pd.DataFrame(x,index,v) if dataframe else x  

        return(data)
    
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

    def get_ntot(self,landmarks=False,condition=None,samples=None,marked_obs_to_ignore=None):
        dict_nobs = self.get_nobs(landmarks=landmarks,condition=condition,samples=samples,marked_obs_to_ignore=marked_obs_to_ignore)
        return(dict_nobs['ntot'])

    def get_dataframe_of_means(self,landmarks=False,condition=None,samples=None,marked_obs_to_ignore=None,data_name=None):
        ddf = self.get_data(
            landmarks=landmarks,
            condition=condition,
            samples=samples,
            marked_obs_to_ignore=marked_obs_to_ignore,
            data_name=data_name,
            dataframe=True,
            in_dict=True
        )
        return(pd.DataFrame({sample:ddf[sample].mean() for sample in ddf}))

    def get_variables(self,data_name=None):
        """
        Returns the list of variables of the data

        Parameters
        ----------
            data_name (default = None): str
                Refers to the array of interest if needed
        
        """
        if data_name is None:
            data_name = self.data_name
        return(self.data[data_name]['variables'])

    def get_nvariables(self,data_name=None):
        if data_name is None:
            data_name = self.data_name
        return(self.data[data_name]['p'])

    def get_var(self,data_name=None):
        if data_name is None:
            data_name = self.data_name
        return(self.var[data_name])

    def get_vard(self,data_name=None):
        if data_name is None:
            data_name = self.data_name
        return(self.vard[data_name])   
        
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
            print('more than 2 groups',[f'{k} ({len(v)})' for k,v in data.items()])
        return(list(data.values()))

    def init_df_proj(self,proj,name=None,data_name=None):
        '''
        Returns the desired dataframe  
        
        Parameters
        ----------
            proj : str 
                if proj in ['proj_kfda','proj_kpca','proj_mmd','proj_orthogonal']
                    returns a dataframe containing the position of each cell on the corresponding axis.
                    - proj_kfda : discriminant axes 
                    - proj_kpca : principal components of the within group covariance
                    - proj_mmd : projection on the MMD-withess function (axis supported by the mean embeddings difference)
                    - proj_orthogonal : (`name` has to be specified) projection on the principal components of the within group covariance computed on the space orthogonal to the discriminant axis. 
                if proj in variables list (self.get_variables()):
                    returns the value of this variable for each observation
                if proj in metadata (self.obs.columns)
                    return the value of this metainformation for each observation

            name (default = None) : str
                specify the projection asked.
                Set automatically to the last version of the projection computed.

            data_name (default = None) : str
                Refers to the considered data assay on which computations have been made. 

        '''


        if data_name is None:
            data_name = self.current_data_name
        if proj == 'proj_kfda':
            df_proj = self.get_proj_kfda(name=name)
        elif proj == 'proj_kpca':
            df_proj = self.get_proj_kpca(name=name)
        elif proj == 'proj_mmd':
            df_proj = self.get_proj_mmd(name=name)
        elif proj == 'proj_orthogonal':
            df_proj = self.get_proj_orthogonal(name=name)

        elif proj in self.get_variables(data_name):
            # df_proj = pd.DataFrame(self.get_dataframe_of_all_data()[proj])
            df_proj = pd.DataFrame(self.get_data(samples='all',in_dict=False,dataframe=True)[proj])

        elif proj in self.obs:
            df_proj = self.obs[proj]
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

    def get_proj_orthogonal(self,name=None):
        if name is None:
            name = self.get_orthogonal_name()
        if name in self.df_proj_orthogonal:
            return(self.df_proj_orthogonal[name])
        else:
            print(f"proj orthogonal '{name}' has not been computed yet")

    def set_marked_obs_to_ignore(self,marked_obs_to_ignore=None,verbose=0):
        if marked_obs_to_ignore is None:
            self.marked_obs_to_ignore = marked_obs_to_ignore
        elif marked_obs_to_ignore in self.obs.columns:
            if verbose>0:
                print(f'- Observations in column {marked_obs_to_ignore} will be ignored from the analysis')
            self.marked_obs_to_ignore = marked_obs_to_ignore
        else:
            print(f"- marked_obs_to_ignore={marked_obs_to_ignore} is not recognised. \n\t Use function 'mark_observations' to mark observations")
            self.marked_obs_to_ignore = None
           
    def set_test_data_info(self,samples='all',condition=None,data_name=None,change_kernel=True,verbose=0):
        """
        Set the necessary information to define which test to perform. 

        Parameters
        ----------
            samples (default = 'all') : 'all' or list of str
                List of samples to compare

            condtion : 
                Column of the metadata containing the samples labels

            data_name : 
                dataset assay 

            change_kernel (default = True) : bool
                Recompute the kernel parameters associated to the specific comparison being performed. 

            verbose (default = 0): int
                The higher, the more verbose.  
        """


        condition,samples,_ = self.init_samples_condition_marked(condition=condition,
                                    samples=samples,
                                    marked_obs_to_ignore=None)

        
        data_name = self.data_name if data_name is None else data_name


        if verbose>0:
            s=f"- Set test data info"
            if verbose == 1:
                if samples == 'all':
                    s+=f' (condition={condition})'
                else:
                    s+=' ('+",".join(samples)+f' from {condition})'
            if verbose >1:
                s+=f"\n\tdata : {data_name}\n\tcondition : {condition}\n\tsamples : {samples}"
            print(s)

        self.data_name = data_name
        self.condition = condition
        try:
            self.obs[condition] = self.obs[condition].astype('category')
        except KeyError:
            print(f"KeyError : condition {condition} not in obs")
        self.samples = samples
        if change_kernel:
            kernel = self.kernel_specification 
            self.init_kernel(verbose=verbose,**kernel)
        
    def get_test_data_info(self):
        return({'dataset':self.data_name,
                'condition':self.condition,
                'samples':self.samples,
                'marked_obs_to_ignore':self.marked_obs_to_ignore})
        
    def get_spev(self,slot='covw',t=None,center=None):
        """
        Return spectrum and eigenvectors.
        slot in ['covw','anchors','residuals','orthogonal]
        """

        spev_name = self.get_covw_spev_name() if slot=='covw' else \
                    self.get_anchors_name() if slot=='anchors' else \
                    self.get_orthogonal_name(t=t,center=center)
        
        try:
            return(self.spev[slot][spev_name]['sp'],self.spev[slot][spev_name]['ev'])
        except KeyError:
            print(f'KeyError : spev {spev_name} not in {slot} {self.spev[slot].keys()}')
   