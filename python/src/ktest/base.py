import torch
import pandas as pd
import numpy as np
from typing_extensions import Literal
from typing import Optional,Callable,Union,List
from .kernels import gauss_kernel,linear_kernel,gauss_kernel_mediane,fisher_zero_inflated_gaussian_kernel,gauss_kernel_weighted_variables,gauss_kernel_mediane_per_variable
from torch import cat
from .utils import get_kernel_name,init_kernel_params,convert_to_torch_tensor,convert_to_pandas_index
    

class Base:

    def __init__(self,verbose=0):        
        super(Base, self).__init__()
        self.truncation = 10
        self.center_by = None        
        self.has_data = False 
        self.has_model = False  
        self.has_landmarks = False
        self.has_anchors = False
        self.has_kernel = False
        self.has_kfda_statistic = False
        self.univariate_name=None
        self.permutation_mmd_name=None
        self.permutation_kfda_name=None
        self.quantization_with_landmarks_possible = False

        # attributs initialisés 
        self.data = {}
        self.input_params={}
        self.obs = pd.DataFrame(columns=['sample'])
        self.obs['sample'] = self.obs['sample'].astype('category')
        self.verbose=verbose
        self.var = {}
        self.vard = {}
        
        self.data_name = None
        self.log2fc_data = None
        self.condition = 'sample'
        self.samples = 'all'
        self.marked_obs_to_ignore = None
        self.main_data=None

        # Statistics 
        self.df_kfdat = pd.DataFrame()
        self.df_kfdat_contributions = pd.DataFrame()
        self.dict_mmd = {}

        # p-values
        self.df_pval = pd.DataFrame()
        self.df_pval_contributions = pd.DataFrame()
        self.dict_pval_mmd = {}

        # Permutation statistics
        self.df_kfdat_perm = {}
        self.df_mmd_perm = pd.DataFrame()

        # Projections and Correlations
        self.corr = {}     
        self.df_proj_kfda = {}
        self.df_proj_kpca = {}
        self.df_proj_mmd = {}
        self.df_proj_unidirectional_mmd = {}
        self.df_proj_residuals = {}
        
        # Eigenvectors
        self.spev = {'covw':{},'anchors':{},'residuals':{}} # dict containing every result of diagonalization
        # les vecteurs propres sortant de eigsy sont rangés en colonnes

        # for verbosity 
        self.start_times = {}


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
                self.approximation_cov = 'nystrom3'
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
                
    def set_truncation(self,t):
        self.truncation=t

    def kernel(self,function='gauss',
               bandwidth='median',
               median_coef=1,
               kernel_name=None,
               weights = None,
               weights_power = 1,
               pi1=None,
               pi2=None,
               verbose=0):
        '''
        
        Parameters
        ----------
            function (default = 'gauss') : str or function
                str in ['gauss','linear','fisher_zero_inflated_gaussian','gauss_kernel_mediane_per_variable'] for gauss kernel or linear kernel. 
                function : kernel function specified by user

            bandwidth (default = 'median') : str or float
                str in ['median'] to use the median or a multiple of it as a bandwidth. 
                float : value of the bandwidth

            median_coef (default = 1) : float
                multiple of the median to use as bandwidth if kernel == 'gauss' and bandwidth == 'median' 

            pi1,pi2 (default = None) : None or str 
                if function == 'fisher_zero_inflated_gaussian' : columns of the metadata containing 
                the proportions of zero for the two samples   
        Returns
        ------- 
        '''

        if verbose >0:
            s=f'- Define kernel function'
            if verbose ==1:
                s+=f' ({function})'
            else:
                s+=f'\n\tfunction : {function}'
                if function == 'gauss':
                    s+=f'\n\tbandwidth : {bandwidth}'
                if bandwidth == 'median' and median_coef != 1:
                    s+=f'\n\tmedian_coef : {median_coef}'
                if kernel_name is not None:
                    s+=f'\n\tkernel_name : {kernel_name}'
            print(s)
        
        x,y = self.get_xy()
        has_bandwidth = False

        kernel_name = get_kernel_name(function=function,bandwidth=bandwidth,median_coef=median_coef) if kernel_name is None else kernel_name
        if verbose>1:
            print("kernel_name:",kernel_name)
        if function == 'gauss':
            has_bandwidth = True
            if weights is not None:
                if isinstance(weights,str):
                    if weights in self.get_var():
                        weights_ = self.get_var()[weights]
                    elif weights in ['median','variance']:
                        weights_=weights
                    else: 
                        print(f"kernel weights '{weights}' not recognized.")
                else:
                    weights_ = weights
                kernel_,computed_bandwidth = gauss_kernel_weighted_variables(x=x,y=y,
                                                                           weights=weights_,
                                                                           weights_power=weights_power,
                                                                           bandwidth=bandwidth,
                                                                          median_coef=median_coef,
                                                                          return_mediane=True,
                                                                          verbose=verbose)

            else:
                kernel_,computed_bandwidth = gauss_kernel_mediane(x=x,y=y,      
                                                bandwidth=bandwidth,  
                                               median_coef=median_coef,
                                               return_mediane=True,
                                               verbose=verbose)



        elif function == 'linear':
            kernel_ = linear_kernel
        elif function == 'fisher_zero_inflated_gaussian':
            has_bandwidth = True
            kernel_,computed_bandwidth = fisher_zero_inflated_gaussian_kernel(x=x,y=y,
                                                                    pi1=pi1,pi2=pi2,
                                                                    bandwidth=bandwidth,
                                                                    median_coef=median_coef,
                                                                    return_mediane=True,
                                                                    verbose=verbose)
        # elif function == 'gauss_kernel_mediane_per_variable':
        #     has_bandwidth = True
        #     kernel_,computed_bandwidth = gauss_kernel_mediane_per_variable(x=x,y=y,
        #                                                                    bandwidth=bandwidth,
        #                                                                   median_coef=median_coef,
        #                                                                   return_mediane=True,
        #                                                                   verbose=verbose)



        else:
            kernel_ = function


        if verbose>1:
            print("kernel",kernel_)
        self.data[self.data_name]['kernel'] = kernel_
        self.data[self.data_name]['kernel_name'] = kernel_name
        if has_bandwidth:
            self.data[self.data_name]['kernel_bandwidth'] = computed_bandwidth
        self.has_kernel = True
        self.kernel_params = init_kernel_params(function=function,
                                                bandwidth=bandwidth,
                                                median_coef=median_coef,
                                                weights=weights,
                                                weights_power=weights_power,
                                                kernel_name=kernel_name,
                                                pi1=pi1,pi2=pi2)

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
                x = self.data[data_name]['X'][self.obs.index.isin(index),:]
                data = pd.DataFrame(x,index,v) if dataframe else x  

        return(data)
    
        # def get_data(self,landmarks=False,condition=None,samples=None,marked_obs_to_ignore=None,data_name=None):
        
        # if data_name is None:    
        #     data_name = self.data_name
    
        # if landmarks and self.landmark_method =='kmeans':
        #     dict_data = self.get_kmeans_landmarks()
            
        # else:
        #     dict_index = self.get_index(landmarks=landmarks,condition=condition,samples=samples,marked_obs_to_ignore=marked_obs_to_ignore)
        #     if landmarks and self.landmark_method=='kmeans':
        #         dict_data = {k:self.data[data_name]['X'] for k in dict_index.keys()}
        #     else:
        #         dict_data = {k:self.data[data_name]['X'][self.obs.index.isin(v),:] for k,v in dict_index.items()}
        # return(dict_data)
    

        # def get_dataframes_of_data(self,landmarks=False,condition=None,samples=None,marked_obs_to_ignore=None,data_name=None):
            
        #     if data_name is None:
        #         data_name = self.data_name

        #     dict_data = self.get_data(landmarks=landmarks,condition=condition,samples=samples,marked_obs_to_ignore=marked_obs_to_ignore,data_name=data_name)
        #     dict_index = self.get_index(landmarks=landmarks,condition=condition,samples=samples,marked_obs_to_ignore=marked_obs_to_ignore)
        #     variables = self.data[data_name]['variables']

        #     dict_df = {}
        #     for s in dict_data.keys():
        #         x,i,v = dict_data[s],dict_index[s],variables
        #         dict_df[s] = pd.DataFrame(x,i,v)
        #     return(dict_df)

        # def get_dataframe_of_all_data(self,landmarks=False,data_name=None):
        #     if data_name is None:
        #         data_name = self.data_name
        #     if landmarks:
        #         print('Warning : this function is not implemented for landmarks yet')
        #     x = self.data[data_name]['X']
        #     i = self.obs.index
        #     v = self.data[data_name]['variables']

        #     return(pd.DataFrame(x,i,v))


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
            self.add_data_to_Ktest(x,sample,data_name,index,variables,metadata,var_metadata=var_metadata)
 
    def init_L_groups_from_dataframe(self,df_list,data_name,sample_list=None,metadata_list=None,kernel='gauss_median',var_metadata=None):
        L = len(df_list)
        if metadata_list is None:
            metadata_list = [None]*L
        if sample_list is None:
            sample_list = [f'x{l}' for l in range(1,L+1)]
        for df,sample,metadata in zip(df_list,sample_list,metadata_list):
            self.add_data_to_Ktest_from_dataframe(df,sample,metadata,data_name,var_metadata=var_metadata)


    def init_df_proj(self,proj,name=None,data_name=None):
        '''
        Returns the desired dataframe  
        
        Parameters
        ----------
            proj : str 
                if proj in ['proj_kfda','proj_kpca','proj_mmd','proj_unidirectional_mmd','proj_residuals']
                    returns a dataframe containing the position of each cell on the corresponding axis.
                    - proj_kfda : discriminant axes 
                    - proj_kpca : principal components of the within group covariance
                    - proj_mmd : projection on the MMD-withess function (axis supported by the mean embeddings difference)
                    - proj_unidirectional_mmd : similar to proj_kpca with a different normalization
                    - proj_residuals : (`name` has to be specified) projection on the principal components of the within group covariance computed on the space orthogonal to the discriminant axis. 
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
        elif proj == 'proj_unidirectional_mmd':
            df_proj = self.get_proj_unidirectional_mmd(name=name)
        elif proj == 'proj_residuals':
            df_proj = self.get_proj_residuals(name=name)

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

    def get_proj_unidirectional_mmd(self,name=None):
        if name is None:
            name = self.get_mmd_name()
        if name in self.df_proj_mmd:
            return(self.df_proj_unidirectional_mmd[name])
        else:
            print(f"proj unidirectional mmd '{name}' has not been computed yet")

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

    def set_center_by(self,center_by=None,verbose=0):
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
                is set to None if center_by is a string but the Ktest object doesn't have an `obs` dataframe. 

        '''
        if verbose>0:
            if center_by is not None:
                print(f"- Using '{center_by}' to center the embeddings in the feature space.")
        self.center_by = center_by      
           
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
            self.kernel(verbose=verbose,**kernel)
        
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
        c,samples,mark = self.init_samples_condition_marked(condition=condition,
                                           samples=samples,
                                           marked_obs_to_ignore=marked_obs_to_ignore)

        # si les conditions et samples peuvent être mis en entrées, cente_by aussi
        smpl = '' if samples == 'all' else "".join(samples)
        cb = '' if self.center_by is None else f'_cb_{self.center_by}'    
        marking = '' if mark is None else f'_{mark}'
        return(f'{dn}{c}{smpl}{cb}{marking}')

    def get_model_str(self):
        ny = self.nystrom
        nys = 'ny' if self.nystrom else 'standard'
        ab = f'_basis{self.anchor_basis}' if ny else ''
        lm = f'_lm{self.landmark_method}_m{self.get_n_landmarks()}' if ny else ''
        return(f'{nys}{lm}{ab}')

    def get_landmarks_name(self):
        dtn = self.get_data_to_test_str()


        lm = self.landmark_method
        n_landmarks = f'_m{self.get_n_landmarks()}'
        
        return(f'lm{lm}{n_landmarks}_{dtn}')

    def get_kmeans_landmarks_name_for_sample(self,sample):
        landmarks_name = self.get_landmarks_name()
        return(f'{sample}_{landmarks_name}')


    def get_anchors_name(self,):
        dtn = self.get_data_to_test_str()

        lm = self.landmark_method
        ab = self.anchor_basis
        n_landmarks = f'_m{self.get_n_landmarks()}'
        # r = f'_r{self.r}' # je ne le mets pas car il change en fonction des abérations du spectre
        return(f'lm{lm}{n_landmarks}_basis{ab}_{dtn}')

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
