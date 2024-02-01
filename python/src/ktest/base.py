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

    def __init__(self,data,
                 metadata=None,
                 var_metadata=None,
                 ):   
        '''
        Attributes initialized : 
        ------------------------
        self.data : torch.tensor
            data matrix initialized from data 
        
        self.obs : pandas.DataFrame
            dataframe of meta information about the observations
            initialized from metadata

        self.var : pandas.DataFrame
            dataframe of meta information about the variables           
            `self.var` is used to store the results of univariate tests. 
            initialized from var
         
        self.index : pandas.index 
            indexes of the observations
        
        self.variables : pandas.index 
            indexes of the variables

        self.nobs : int
            number of observations 

        self.nvar : int 
            number of variables

        self.vard : dict of dicts
            contains a dict for each variable of 
            `self.variables`, it is used as an intermediate data storage

        self.hypotheses : dict 
            contains a slot for each hypothesis to test in general 
            hypothesis testing (see `self.set_hypothesis()`)

        self.stat (default = 'kfda') : str in ['kfda','mmd']
            the test statistic to use for two-sample testing

        self.nystrom (default = False) : bool
            wether to use the Nystrom approximation or not. 

        self.nystrom_initialized (default = False) : bool
            set to True when the Nystrom parameters are initialized

        self.n_landmarks_initial : int
            number of landmarks used in the Nystrom method

        self.n_anchors : int
            number of anchors used in the Nystrom method

        self.landmark_method (default = 'random') : str in ['random','kmeans']
            method to use to select the landmarks             
        
        self.kmeans_landmarks : dict 
            stores the kmeans centroids to use as landmarks in the nystrom method

        self.anchor_basis (default = 'w') : str in ['w','s','k']
            matrix of the landmark to use to define the anchors in the Nystrom method
        
        self.permutation (default = False) : bool
            wether or not to use a permutation approach to compute the p-value 
            of the test. 
        
        self.n_permutations : int 
            number of permutations to use for the permutation approach
        
        self.seed_permutation : int 
            random seed to initialize the permutation 
        
        self.df_kfdat : pandas.DataFrame
            stores the computed kfda statistics and
            rows : truncations
            columns : each performed test with specific parameters and samples 

        self.df_pval : pandas.DataFrame
            stores the p-values associated to the kfda statistic
            obtained from the asymptotic distribution (chi-square with T 
            degree of freedom) or a permutation approach. 
            rows : truncations
            columns : each performed test with specific parameters and samples
        
          
        self.df_kfdat_contributions : pandas.DataFrame
            stores the unidirectional statistic associated to each 
            eigendirection of the within-group covariance operator
            `self.df_kfdat` contains the cumulated sum of the values 
            in  `self.df_kfdat_contributions`
        
        self.df_pval_contributions : pandas.DataFrame
            stores the asymptotic or permutation p-values associated to each unidirectional 
            statistic (chi-square with 1 degree of freedom) 
        
        self.dict_mmd : dict 
            store the MMD statistics associated to each performed test. 
        
        self.dict_pval_mmd : dict 
            stores the permutation p-values associated to each MMD statistic 
            
        self.df_kfdat_perm : dict of pandas.DataFrames 
            stores the KFDA statistics computed from permuted samples in the 
            permutation approach. 

        self.df_mmd_perm : pandas.DataFrame
            stores the MMD statistics computed from permuted samples. 

        self.df_proj_kfda : dict of pandas.DataFrames 
            one entry per column of `self.df_kfdat`
            contains the positions of the observations projected on the 
            discriminant axis. 
            rows : observations
            columns : truncations

        self.df_proj_kpca : dict of pandas.DataFrames 
            one entry per column of `self.df_kfdat`
            contains the positions of the observations projected on the 
            eigendirections of the within-group covariance operator.  
            rows : observations
            columns : truncations

        self.df_proj_mmd : dict of pandas.DataFrames 
            one entry per entry of `self.dict_mmd`
            contains the positions of the observations projected on the 
            axis supported by the difference of the kernel mean embeddings 
            of the two compared probability distributions $\mu_1 - \mu_2$. 
            rows : observations
            columns : truncations
 
        self.df_proj_orthogonal : dict of pandas.DataFrames 
            contains the positions of the observations projected on the 
            axis orthogonal to the discriminant axis that maximizes the variance
            rows : observations
            columns : truncations

        self.df_proj_residuals : dict of pandas.DataFrames 
            contains the positions of the observations projected on the 
            eigendirections of the residual covariance operator of the 
            kernel linear model. 
            rows : observations
            columns : truncations

        self.corr : dict 
            stores pandas.DataFrames of corellations between variables and positions
            of the observations projected on directions of interest in the feature 
            space. see the file ./correlation_operations.py for more details
       
        self.spev ({'covw':{},'anchors':{},'residuals':{},'orthogonal':{}}) : dict of dicts 
            each entry correspond to a specific type of operator to diagonalize
            and will be a dict containing a list of eigenvalues ('sp' for spectrum)
            and a matrix of which the columns are eigenvectors ('ev' for eigenvectors). 
            both associated to the diagonalization of the corresponding operator using 
            the kernel trick.  
            all these quantities are obtained from the function eigsy from the package apt
        
        self.condition : str 
            column of `self.obs` containing the samples to compare
        
        self.samples (default = 'all') 
            list of categories of `self.obs[self.condition]` to compare
        
        self.current_hyp : str
            current key associated to the dict `self.hypotheses`
        
        self.univariate_name : str 
            user-customized prefix on the univariate test results

        self.permutation_mmd_name : str 
            current column of interest in `self.df_mmd_perm`

        self.permutation_kfda_name : str 
            current entry of interest in `self.df_kfdat_perm`

        self.marked_obs_to_ignore : str 
            column of booleans of `self.obs` indicating the observations to ignore 
            when performing computations. 
            
        self.start_times : dict 
            stores the start time of functions that take time to run for measurements. 
            
        self.has_landmarks : bool 
            True if the nystrom landmarks have been determined

        self.has_anchors : bool 
            True if the nystrom anchors have been determined

        self.log2fc_computed : bool 
            True when the log2 fold changes have been computed
        


        '''     
        super(Base, self).__init__()

        

        nobs,nvar = data.shape
        x = convert_to_torch_tensor(data.to_numpy())        
        self.data = x
        self.index = data.index
        self.variables = data.columns
        self.vard = {v:{} for v in self.variables}
        self.nvar = nvar
        self.nobs = nobs
        
        if metadata is not None:
            assert(nobs == len(metadata))
            assert(nobs == data.index.isin(metadata.index).sum())
            # the columns that could be used as conditions as set to categorical 
            # if not already
            self.obs = metadata.copy()
            for column in self.obs.columns:
                if len(self.obs[column].unique())<100:
                        self.obs[column] = self.obs[column].astype('category')
        else: 
            self.obs = pd.DataFrame(index=self.index)


        if var_metadata is not None:
            assert(nvar == len(var_metadata))
            assert(nvar == data.columns.isin(var_metadata.index).sum())
            self.var = var_metadata.copy()
        else:
            self.var = pd.DataFrame(index=self.variables)



        self.hypotheses = {}
        self.stat = 'kfda'
        self.nystrom = False
        self.nystrom_initialized = False
        self.n_landmarks_initial = None
        self.n_anchors = None
        self.landmark_method = 'random'
        self.anchor_basis = 'w'
        self.permutation = False
        self.n_permutations = 500
        self.seed_permutation = 20231124


        ### Output statistics ### 
        
        self.df_kfdat = pd.DataFrame()
        self.df_pval = pd.DataFrame()
        self.df_kfdat_contributions = pd.DataFrame()
        self.df_pval_contributions = pd.DataFrame()
        
        ## MMD statistic and p-value
        self.dict_mmd = {}
        self.dict_pval_mmd = {}

        # Permutation statistics
        self.df_kfdat_perm = {}
        self.df_mmd_perm = pd.DataFrame()


        ## Coordinates of the projections  
        self.df_proj_kfda = {} 
        self.df_proj_kpca = {} 
        self.df_proj_mmd = {} 
        self.df_proj_orthogonal = {} 
        self.df_proj_residuals = {}
        
        # Eigenvectors
        self.spev = {'covw':{},'anchors':{},'residuals':{},'orthogonal':{}} 
        

        ### Keys ###        
        self.condition = None
        self.samples = 'all' 
        self.current_hyp = None 
        self.log2fc_computed = False 
        self.univariate_name=None 
        self.permutation_mmd_name=None
        self.permutation_kfda_name=None
        self.marked_obs_to_ignore = None


        ### Tokens ### 
        self.has_landmarks = False
        self.has_anchors = False
        self.has_kfda_statistic = False 


        ### Others ###
        self.start_times = {}
        self.corr = {}     




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

        new_params = {
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
        
        current_params = self._get_test_params()

        if new_params != current_params:
            self.stat = stat
            self.nystrom = nystrom
            self.permutation = True if stat == 'mmd' else permutation
                
            self.nystrom_initialized = False
            self.n_landmarks_initial = n_landmarks 
            
            self.n_anchors = n_anchors
            self.landmark_method = landmark_method
            self.anchor_basis = anchor_basis
            
            self.n_permutations = n_permutations
            self.seed_permutation = seed_permutation
            
            if nystrom:
                self.compute_nystrom_landmarks(verbose=verbose)
                self.compute_nystrom_anchors(verbose=verbose)

             
    def _get_test_params(self):
        test_params = {
            'stat':self.stat,
            'nystrom':self.nystrom,
                'n_landmarks':self.n_landmarks_initial,
                'n_anchors':self.n_anchors,
                'landmark_method':self.landmark_method,
                'anchor_basis':self.anchor_basis,
            'permutation':self.permutation,
                'n_permutations':self.n_permutations,
                'seed_permutation':self.seed_permutation,
                }
        return(test_params)
              
    def set_test_data_info(self,samples='all',condition=None,change_kernel=True,verbose=0):
        """
        Set the necessary information to define which test to perform. 

        Parameters
        ----------
            samples (default = 'all') : 'all' or list of str
                List of samples to compare

            condtion : 
                Column of `self.obs` containing the samples labels 

            change_kernel (default = True) : bool
                Recompute the kernel parameters associated to the specific comparison being performed. 

            verbose (default = 0): int
                The higher, the more verbose.  
        """


        condition,samples,_ = self.init_samples_condition_marked(condition=condition,
                                    samples=samples,
                                    marked_obs_to_ignore=None)

        if verbose>0:
            s=f"- Set test data info"
            if verbose == 1:
                if samples == 'all':
                    s+=f' (condition={condition})'
                else:
                    s+=' ('+",".join(samples)+f' from {condition})'
            if verbose >1:
                s+=f"\n\tcondition : {condition}\n\tsamples : {samples}"
            print(s)

        self.condition = condition
        try:
            self.obs[condition] = self.obs[condition].astype('category')
        except KeyError:
            print(f"KeyError : condition {condition} not in obs")
        self.samples = samples
        if change_kernel:
            self.init_kernel(
                function=self.kernel_function,
                bandwidth=self.kernel_bandwidth,
                median_coef=self.kernel_median_coef,
                kernel_name=self.kernel_name,)
            
    def get_test_data_info(self):
        return({'condition':self.condition,
                'samples':self.samples,
                'marked_obs_to_ignore':self.marked_obs_to_ignore})
   
    def update_var_from_dataframe(self,df,verbose = 0):
        var = self.var
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
        self.var = var.join(df_to_add)



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
            assert(self.landmark_method != 'kmeans') # kmeans centroids have no index

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
            data[sample] = self.kmeans_landmarks[kmeans_landmarks_name]
        if in_dict:
            return(data)
        else:
            return(torch.cat(list(data.values())))

    def get_data(self,
                 landmarks=False,
                 condition=None,
                 samples=None,
                 marked_obs_to_ignore=None,
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


        if dataframe:
            v = self.variables
            
        if landmarks and self.landmark_method =='kmeans':
            data = self.get_kmeans_landmarks(in_dict=in_dict)

        else:
            index = self.get_index(landmarks=landmarks,condition=condition,samples=samples,marked_obs_to_ignore=marked_obs_to_ignore,in_dict=in_dict)

            if in_dict:
                data = {}
                for k,i in index.items():
                    # x = self.data[data_name]['X'][self.obs.index.isin(i),:]
                    x = self.data[self.index.isin(i),:]
                    data[k] = pd.DataFrame(x,i,v) if dataframe else x
            else:
                indexer_ = self.obs.index.get_indexer(index)
                # x = self.data[data_name]['X'][indexer_,:]
                x = self.data[indexer_,:]
                data = pd.DataFrame(x,index,v) if dataframe else x  

        return(data)
    
    def get_nobs(self,landmarks=False,condition=None,samples=None,marked_obs_to_ignore=None):
        
        if landmarks and self.landmark_method == 'kmeans':
            centroids = self.get_kmeans_landmarks()
            dict_nobs = {k:len(v) for k,v in centroids.items()}
        else:
            dict_index = self.get_index(landmarks=landmarks,condition=condition,samples=samples,marked_obs_to_ignore=marked_obs_to_ignore)
            dict_nobs = {k:int(self.obs.index.isin(v).sum()) for k,v in dict_index.items()}
        dict_nobs['ntot'] = sum(list(dict_nobs.values()))
        return(dict_nobs)

    def get_ntot(self,landmarks=False,condition=None,samples=None,marked_obs_to_ignore=None):
        dict_nobs = self.get_nobs(landmarks=landmarks,condition=condition,samples=samples,marked_obs_to_ignore=marked_obs_to_ignore)
        return(dict_nobs['ntot'])

    def get_dataframe_of_means(self,landmarks=False,condition=None,samples=None,marked_obs_to_ignore=None):
        ddf = self.get_data(
            landmarks=landmarks,
            condition=condition,
            samples=samples,
            marked_obs_to_ignore=marked_obs_to_ignore,
            dataframe=True,
            in_dict=True
        )
        return(pd.DataFrame({sample:ddf[sample].mean() for sample in ddf}))


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

    def init_df_proj(self,proj,name=None):
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
                if proj in variables list (self.variables):
                    returns the value of this variable for each observation
                if proj in metadata (self.obs.columns)
                    return the value of this metainformation for each observation

            name (default = None) : str
                specify the projection asked.
                Set automatically to the last version of the projection computed.


        '''


        if proj == 'proj_kfda':
            df_proj = self.get_proj_kfda(name=name)
        elif proj == 'proj_kpca':
            df_proj = self.get_proj_kpca(name=name)
        elif proj == 'proj_mmd':
            df_proj = self.get_proj_mmd(name=name)
        elif proj == 'proj_orthogonal':
            df_proj = self.get_proj_orthogonal(name=name)

        elif proj in self.variables:
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
   