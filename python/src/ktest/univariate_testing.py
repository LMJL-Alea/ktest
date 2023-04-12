import pandas as pd

from time import time
from joblib import Parallel, delayed, parallel_backend
import os
import numpy as np
from .utils_univariate import filter_genes_wrt_pval
from .pvalues import correct_BenjaminiHochberg_pval_of_dfcolumn
'''
La question du test univarié sur des données univariées est difficile notamment par rapport à ce qu'on veut
retenir de chaque test univarié unique. 
Ces fonctions permettent de simplifier l'approche de test univariée à partir de données multivariées et 
la visualisation des résultats.  
'''

class Univariate:
    
    def __init__(self):        
        super(Univariate, self).__init__()

    def compute_zero_proportions_of_variable(self,variable):
        """
        Returns a dict object with the zero information of the variable with respect to the tested conditions.
        
        Parameters
        ----------
            variable : str, 
                name of the variable to compute zero information on 
            
        Returns
        -------
            a dict containing for each sample 's' corresponding to a tested condition : 
                - 's_pz' : proportion of zero of the variable in the sample
                - 's_nz' : number of zero observation of the variable in the sample
                - 's_n' : total number of observations in the sample 
        """

        proj = self.init_df_proj(variable)
        dict_index = self.get_index()
        dict_zp = {}
        for s,index_sample in dict_index.items():
            df = proj[proj.index.isin(index_sample)][variable]
            n = len(df)
            nz = len(df[df==0])
            dict_zp[f'{s}_pz'] = nz/n
            dict_zp[f'{s}_nz'] = nz
            dict_zp[f'{s}_n'] = n
        return(dict_zp)

    def add_zero_proportions_to_var(self,data_name=None):
        '''
        Fills the attribute var of the active dataset with columns corresponding to the 
        zero information of the variables with respect to the tested conditions. 
        
        Updated attributes 
        ------- ----------
            The var DataFrame of the active dataset is updated for each sample 's' corresponding to a 
            tested condition with the following columns :
                - 's_pz' : proportion of zero of the variables in the sample
                - 's_nz' : number of zero observation of the variables in the sample
                - 's_n' : total number of observations in the sample 
                - 's_pct_expression' : proportion of non-zero of the variables in the sample 
                - 's_mean' : mean expression of the variables in the sample. 
        '''
        if data_name is None:
            data_name = self.data_name

        dict_df = self.get_dataframes_of_data(data_name=data_name)
        dfzp = pd.DataFrame()
        for s,df in dict_df.items():
            dfzp[f'{s}_nz'] = (df==0).sum()
            dfzp[f'{s}_n'] = len(df)
            dfzp[f'{s}_pz'] = dfzp[f'{s}_nz']/len(df)
            dfzp[f'{s}_pct_expression'] = 1 - dfzp[f'{s}_pz']
            dfzp[f'{s}_mean'] = df.mean()


        self.update_var_from_dataframe(dfzp)
    
    def it_is_possible_to_compute_log2fc(self,data_name=None):
        df = self.get_dataframe_of_all_data(data_name)
        npositive = (df>=0).sum().sum()
        ntotal = df.shape[0]*df.shape[1]
        return(npositive == ntotal)

    def add_log2fc_to_var(self,data_name=None):

        if data_name is None:
            data_name = self.data_name

        if self.it_is_possible_to_compute_log2fc(data_name):

            dfs = self.get_dataframes_of_data (data_name=data_name)     
            s1,s2 = dfs.keys()
            dffc = pd.DataFrame()
            cols = []

            for sample,df in dfs.items():
                col = f'{data_name}_{sample}_mean'
                cols += [col]
                dffc[col] = df.mean()
            log2fc = np.log(dffc[cols[0]]/dffc[cols[1]])/np.log(2)
            dffc[f'{data_name}_log2fc_{s1}/{s2}'] = log2fc
            self.update_var_from_dataframe(dffc)
            self.log2fc_data = data_name

        # ancienne version
        # def add_log2fc_to_var(self,data_name=None):
        #     if data_name is not None:
        #         current_data_name = self.data_name
        #         self.data_name = data_name
        #         dn = f'{data_name}_'
        #     else :
        #         dn = ''
        #     dict_df = self.get_dataframes_of_data()
        #     dfzp = pd.DataFrame()
        #     for s,df in dict_df.items():
        #         dfzp[f'{dn}{s}_mean'] = df.mean()
        #     s1,s2 = dict_df.keys()
        #     dfzp[f'{dn}log2fc_{s1}/{s2}'] = np.log(dfzp[f'{dn}{s1}_mean']/dfzp[f'{dn}{s2}_mean'])/np.log(2)
        # #     print(dfzp)
        #     if data_name is not None: 
        #         self.data_name = current_data_name
        #     self.update_var_from_dataframe(dfzp)

    def compute_log2fc_from_another_dataframe(self,df,data_name='raw'):
        
        index = self.get_dataframe_of_all_data().index
        df = df[df.index.isin(index)]
        self.add_data_to_Ktest_from_dataframe(df,data_name=data_name,update_current_data_name=False)            
        self.add_log2fc_to_var(data_name=data_name)

    def get_log2fc_of_variable(self,variable):
        s1,s2 = self.get_samples_list()
        col = f'{self.log2fc_data}_log2fc_{s1}/{s2}'
        var = self.get_var()
        return(var[col][variable])

    def get_zero_proportions_of_variable(self,variable):
        """
        Returns a dict of the ero proportions 
        """
        var = self.get_var()
        sl = list(self.get_index().keys())
        dict_zp = {sample:var[f'{sample}_pz'][variable] for sample in sl}
        return(dict_zp)

    def variable_eligible_for_univariate_testing(self,variable,zero_threshold=.85):
        """
        Returns a boolean, whether or not a variable has a large enough proportion of 
        non-zero data in at least one of the tested conditions to be tested with an univariate test
        in the active dataset.
        
        Parameters
        ----------
            variable : str, 
                    name of the variable to test eligibility on 
            
            zero_threshold : float in [0,1],
                    the variable is eligible if at least one condition among the tested conditions
                    has a smaller proportion of zero  than zero_threshold.

        Returns
        -------
            eligible : boolean
                    True if the variable has a large enough proportion of non-zero data in at least
                     one of the tested conditions to be tested.
                    False otherwise
         
        """
        dict_zp = self.compute_zero_proportions_of_variable(variable)
        eligible = any([zero_proportion <= zero_threshold for k,zero_proportion in dict_zp.items() if '_pz' in k])
        return(eligible)

    def compute_eligible_variables_for_univariate_testing(self,max_dropout=.85):
        """
        Returns the list of variables that have a large enough proportion of 
        non-zero data in at least one of the tested conditions to be tested with an univariate test 
        in the active dataset.

        Parameters
        ----------
            
            zero_threshold : float in [0,1],
                    a variable is eligible for univariate testing if at least one condition among the tested conditions
                    has a smaller proportion of zero than zero_threshold.

        Returns
        -------
            the list of variable that have a large enough proportion of non-zero data in at least
            one of the tested conditions to be tested.
        """
        # on suppose que la fonction add_zero_proportions_to_var_new a deja tourné

        sl = list(self.get_index().keys())
        dfe = pd.DataFrame()
        var = self.get_var()
        for s in sl:
            dfe[f'{s}_eligible'] = var[f'{s}_pz']<=max_dropout 
        dfe[f'eligible'] = (dfe[f'{sl[0]}_eligible']) | (dfe[f'{sl[1]}_eligible'])
        neligible = dfe[f'eligible'].sum()
        print(f'{neligible} eligible variables out of {len(dfe)} with max_dropout = {max_dropout}')
        return(dfe[dfe['eligible']].index)

    def compute_variables_that_are_non_zero_in_more_than_k_observation(self,k=3):
        """
        Returns the list of variables that are non zero in more than 'k' observations in the active dataset. 

        Parameters
        ----------
            k (default=3): int
                The minimum number of cells in which a variable must be non-zeri to be returned

        Returns
        -------
            the list of variables that are non zero in more than 'k' observations
        """

        cs = list(self.get_index().keys())
        var = self.get_var()
        n = self.get_ntot()
        var['ncells_expressed'] = n - (var[f'{cs[0]}_nz'] + var[f'{cs[1]}_nz'])
        return(var[var['ncells_expressed']>=k].index)

    def compute_k_variables_with_higher_variance(self,k=2000):
        """
        Returns the list of 'k' variables that have the higher variance of the active dataset. 
        
        Parameters
        ----------
            k (default=2000): int
                The number of variables to be returned
        Returns
        -------
            the list of 'k' variables that have the higher variance of the active dataset.
        """

        df = self.get_dataframe_of_all_data()
        return(df.var().sort_values(ascending=False)[:k].index)
 
    def get_genes_presents_in_more_than_k_cells(self,k=3,data_name=None):
        if data_name is None:
            data_name = self.data_name

        cs = list(self.get_data().keys())
        var = self.get_var(data_name)
        n = self.get_ntot()
        var['ncells_expressed'] = n - (var[f'{cs[0]}_nz'] + var[f'{cs[1]}_nz'])
        return(var[var['ncells_expressed']>=k].index)


    def has_non_zero_obs_in_both_conditions(self,variable,verbose=0):
        data = self.init_df_proj(variable)
        meta = self.obs.copy()
        samples_list = self.get_samples_list()
        mask =  data[variable]!=0
        
        meta = meta[mask]

        return(all([s in meta[self.condition].unique() for s in samples_list]))        

    def get_data_and_metadata_with_ignored_zeros(self,variable,verbose=0):

        assert(self.has_non_zero_obs_in_both_conditions(variable))
        data = self.init_df_proj(variable)
        meta = self.obs.copy()
        mask =  data[variable]!=0
        
        data = data[mask]
        meta = meta[mask]

        if verbose >0:
            s=''
            n_nz = len(data)
            ntot = self.get_ntot()
            s+=f'\t{len(data)}/{self.get_ntot()} non-zero observations considered\n\t'
            for sample in self.get_samples_list():
                n_sample = len(meta[meta[self.condition]==sample])
                s += f'{sample}:{n_sample}, '
            s = s[:-2]
            print(s)

        return(data,meta)
       
    def initialize_univariate_kernel(self,variable,kernel,kernel_info):
        if kernel is None:
            return(self.get_kernel_params())
        if kernel == 'fisher_zero_inflated_gaussian':
            kernel_params = self.get_kernel_params()
            kernel_params['function'] = kernel
            var = self.get_var()
            kernel_params['pi1'] = var[kernel_info[0]][variable]
            kernel_params['pi2'] = var[kernel_info[1]][variable]
            return(kernel_params)
        else:
            return(kernel)


    def ktest_for_one_variable(self,variable,kernel=None,kernel_info=None,inform_var=True,ignore_zeros=False,verbose=0):
        # Get testing info from global object 
        from .tester import Ktest

        if verbose >0:
            print(f'- Defining ktest for variable {variable}')
        if verbose >1:
            print(f'\tdata_name : {self.data_name}')
            print(f'\tcondition : {self.condition}')
            print(f'\tsamples : {self.samples}')
            print(f'\ttest_params : {self.get_test_params()}')
            print(f'\tcenter_by : {self.center_by}')
            print(f'\tkernel : {kernel} {kernel_info}')
            print(f'\tignore_zeros : {ignore_zeros}')

        # Initialize univariate Ktest with common covariance 
        # var_metadata = pd.DataFrame(self.get_var().loc[variable]).T if inform_var else None

        kernel_params = self.initialize_univariate_kernel(variable,kernel,kernel_info)
        if verbose>1:
            print(kernel_params)
        data = self.init_df_proj(variable)
        meta = self.obs.copy()
        test_params=self.test_params_initial.copy()
        
        if ignore_zeros:
            data,meta = self.get_data_and_metadata_with_ignored_zeros(variable,verbose=verbose)
            n_nz = len(data)
            if n_nz < 300 and test_params['nystrom'] == True: 
                test_params['nystrom'] = False
                if verbose >0:
                    print(f'\tNot using nystrom because <300 non-zero observations')

            
        

        t = Ktest(
                    data =data,
                    metadata =meta,
                    data_name=  self.data_name,
                    condition= self.condition,
                    samples=self.samples,
                    var_metadata=pd.DataFrame(self.get_var().loc[variable]).T if inform_var else None,
                    test_params=test_params,
                    center_by=self.center_by,
                    marked_obs_to_ignore=self.marked_obs_to_ignore,
                    kernel=kernel_params,
                    verbose=verbose
                    )
        return(t)

    def compute_univariate_kfda(self,variable,kernel=None,kernel_info=None,ignore_zeros=False,verbose=0):
        # récupérer la donnée

        t = self.ktest_for_one_variable(variable=variable,kernel=kernel,kernel_info=kernel_info,ignore_zeros=ignore_zeros,verbose=verbose)
        t.multivariate_test(verbose=verbose)

        # ccovw = 'ccovw' if common_covariance else ''
        if t.has_kfda_statistic:
            kfdat_name = t.get_kfdat_name()
            pvalue_name = t.get_pvalue_name()
            try:
                t.df_kfdat[variable] = t.df_kfdat[kfdat_name]
            except KeyError:
                print(f'{kfdat_name} not in df_kfdat ({t.df_kfdat.columns})')        

            try:
                t.df_pval[variable] = t.df_pval[pvalue_name]
            except KeyError:
                print(f'{pvalue_name} not in df_pval ({t.df_pval.columns})')        
        return(t)
        
    def add_results_univariate_kfda_in_vard(self,vtest,variable,name='',tmax=31,
                truncations_of_interest=list(range(1,31)),diagnostic=False,t_diagnostic=31,verbose=0):
        if verbose >0:
            print('- Add results univariate kfda in vard')
        
        vard = self.get_vard()
        not_tested = vtest.df_kfdat.empty
        ### stat and pvals 
        if not_tested:
            if verbose >0 : 
                print(f'\tAdding NA because variable {variable} was not tested')

        tmax = np.min((np.max(truncations_of_interest),len(vtest.get_explained_difference())))

        for t in truncations_of_interest:
            if t<tmax:
                pvals = vtest.df_pval[variable]   
                stats = vtest.df_kfdat[variable]
                if t not in pvals and not_tested == False:
                    not_tested = True
                    
                colpval = self.get_column_name_in_var(t=t,corrected=False,name=name,output='pval')
                colkfda = self.get_column_name_in_var(t=t,corrected=False,name=name,output='kfda')
                
                
                pval = np.NaN if not_tested else pvals[t]
                kfda = np.NaN if not_tested else stats[t]
                
                vard[variable][colpval]=pval
                vard[variable][colkfda]=kfda

        #### diagnostics
        if diagnostic:
            tmax = np.min((t_diagnostic,len(vtest.get_explained_difference())))
            ts_diagnostic = list(range(1,tmax))
            for t in ts_diagnostic:
                colB = self.get_column_name_in_var(t=t,corrected=False,name=name,output='errB')
                colW = self.get_column_name_in_var(t=t,corrected=False,name=name,output='errW')
                
                errB = np.NaN if not_tested else vtest.get_explained_difference_of_t(t)
                errW = np.NaN if not_tested else vtest.get_explained_variability_of_t(t)

                vard[variable][colB] = errB
                vard[variable][colW] = errW

        col_univariate = self.get_column_name_in_var(name=name,output='univariate')
        vard[variable][col_univariate] = True

    def add_results_univariate_kfda_in_vard_old(self,vtest,variable,name='',tmax=31,
                truncations_of_interest=list(range(1,31)),diagnostic=False,t_diagnostic=31,verbose=0):
        if verbose >0:
            print('- Add results univariate kfda in var')
        
        vard = self.get_vard()
        
        ts_test = truncations_of_interest
        ts_diagnostic = list(range(1,t_diagnostic))

        ts = list(range(1,tmax)) 
        tnames = [str(t) for t in ts]
        tnames += ['r1','r2','r3','rmax']
        
        
        if vtest.df_kfdat.empty:
            if verbose >0: 
                print(f'\tAdding NA because variable {variable} could not be tested')
            for tname in tnames:

                    
                    col = self.get_column_name_in_var(t=tname,corrected=False,name=name,output='')
                    vard[variable][f'{col}pval'] = np.NaN
                    vard[variable][f'{col}kfda'] = np.NaN 

                    if diagnostic :
                        vard[variable][f'{col}errB'] = np.NaN
                        vard[variable][f'{col}errW'] = np.NaN
                    if 'r' in tname:
                        vard[variable][f'{col}t'] = np.NaN
        else:
            
            tmax = len(vtest.get_explained_difference())
            
            tr1 = vtest.select_trunc_by_between_reconstruction_ressaut(which_ressaut='first')
            tr2 = vtest.select_trunc_by_between_reconstruction_ressaut(which_ressaut='second')
            tr3 = vtest.select_trunc_by_between_reconstruction_ressaut(which_ressaut='third')
            trm = vtest.select_trunc_by_between_reconstruction_ressaut(which_ressaut='max')
            ts += [tr1,tr2,tr3,trm]

            for t,tname in zip(ts,tnames):
                if t<tmax:
                    pval = vtest.df_pval[variable][t]
                    kfda = vtest.df_kfdat[variable][t]

                    errB = vtest.get_explained_difference_of_t(t)
                    errW = vtest.get_explained_variability_of_t(t)
                    col = self.get_column_name_in_var(t=tname,corrected=False,name=name,output='')
        
                    vard[variable][f'{col}pval'] = pval
                    vard[variable][f'{col}kfda'] = kfda 
                    vard[variable][f'{col}errB'] = errB
                    vard[variable][f'{col}errW'] = errW
                    if tname not in [str(t) for t in ts]:
                        vard[variable][f'{col}t'] = t
        col_univariate = self.get_column_name_in_var(name=name,output='univariate')
        vard[variable][col_univariate] = True
                
        # if verbose>1:
        #     tab = '\t' if len(str(variable))>6 else '\t\t'
        #     zp = self.get_zero_proportions_of_variable(variable)
        
        #     tr1 = vard[variable][f'{dname}_tr1_t']
        #     tr2 = vard[variable][f'{dname}_tr2_t']
        #     errB1 = vard[variable][f'{dname}_tr1_errB']
        #     errB2 = vard[variable][f'{dname}_tr2_errB']
        #     pval1 = vard[variable][f'{dname}_tr1_pval']
        #     pval2 = vard[variable][f'{dname}_tr2_pval']

            # zp_string = "zp: "+" ".join([f'{s}{zp[s]:.2f}' for s in zp.keys()])
            # string = f'{variable} {tab} {zp_string} \t pval{tr1:1.0f}:{pval1:1.0e}  r{tr1:1.0f}:{errB1:.2f} \t pval{tr2:1.0f}:{pval2:1.0e}  r{tr2:1.0f}:{errB2:.2f}'
            # print(string)



    def univariate_test_of_variable(self,
                                    variable,
                                    name=None,
                                    kernel=None,
                                    kernel_info=None,
                                    parallel=False,
                                    truncations_of_interest=list(range(1,31)),
                                    ignore_zeros=False,
                                    diagnostic=False,
                                    t_diagnostic=31,
                                    verbose=0):

        col = self.get_column_name_in_var(name=name,output='univariate')
        var = self.get_var()
        vard = self.get_vard()

        if col in var and var[col][variable]==1:
            if verbose>1:
                print(f'\t{variable} already tested')
        else:
            t=self.compute_univariate_kfda(variable,kernel=kernel,kernel_info=kernel_info,ignore_zeros=ignore_zeros,verbose=verbose)

        self.add_results_univariate_kfda_in_vard(t,variable,name=name,
                    truncations_of_interest=truncations_of_interest,diagnostic=diagnostic,t_diagnostic=t_diagnostic,verbose=verbose)
        
        if parallel:
            return({'v':variable,**vard[variable]})
        return(t) 
 
    def univariate_test(self,
                        variables_to_test=None,
                        fromto=[0,-1],
                        name=None,
                        save_path=None,
                        overwrite=False,
                        truncations_of_interest=list(range(1,31)),
                        n_jobs=1,
                        lots=None,
                        ignore_zeros=False,
                        diagnostic=False,
                        t_diagnostic=31,
                        kernel=None,
                        kernel_info=None,
                        verbose=0):
        """
        Perform univariate testing or Differential Expression Analysis on each variable.
        The results are stored in the var dataframe accessible through `ktest.get_var()`
        and saved in `save_path` if specified. 
        The results stored are the kfda statistics and p-values associated to each truncation present in
        `truncations_of_interest`. 
        If `diagnostic` is True, the explained variabilities and explained differences 
            are added to the result dataframe for truncations from 1 to `t_diagnostic`            

        Parameters
        ----------
            variables_to_test (default = None) : iterable of variables
                List of variables of interest to be tested. 
                Has priority over `fromto`.
            
            fromto (default = [0,-1]) : list of two int 
                Position in the object variables list of the first and last variable to test
                If the second entry is -1, the last variable to test is the last one in the variables list.     
                Warning : not designed to test a specific set of variables (e.g. genes pathway), 
                        parameter `variables_to_test` is.   
                Warning : ignored if `variables_to_test` is specified. 

            name (default = None) : int 
                A name to refer to this set of tests if you aim at testing different groups
                 of variables separately (e.g. different gene pathways)

            save_path (default = None) : str
                path of the directory in which to save the resulting .csv file. 
                the file is automatically named with respect to test info. 
                if None : the results are not written in a .csv file
                
            overwrite (default = False) : bool
                ignored if `save_path` is not specified
                if True : the written .csv file will contain current results only
                if False :if the result file to be saved already exists in the directory `save_path`, 
                    its results are loaded and added to the current results so that all the results
                    are saved. 
            
            truncations_of_interest (default = list(range(1,31)) : List of int
                List of truncations to compute the results of. 

            n_jobs (default = 1) : int 
                Number of CPUs to use for parallel computing. 
                
            lots (default = 100) : int
                Number of tested variables between each save of intermediate results.

            ignore_zeros (default = False) : bool
                Whether or not considers null observations for univariate testing. 

            diagnostic (default = False) : bool
                if True, the diagnostic quantities (explained variabilities and explained differences)
                are added to the results. 

            t_diagnostic (default = 31) : int
                ignored if `diagnostic` is False
                the diagnostic quantities are added to the result dataframe 
                for truncations from 1 to `t_diagnostic`  
            
            kernel_bandwidth_col_in_var (default = None) : str 
                Specify the kernel bandwidth to use for each variable with a column of the var dataframe (e.g. a column containing the variances) 
                Ignored if the kernel is not gaussian

            verbose (default = 0) : int
                The higher, the more verbose is the function.     
        """

        if name is None:
            name = '' if self.univariate_name is None else self.univariate_name
        self.univariate_name=name

        if not overwrite:
            self.load_univariate_test_results_in_var_if_possible(save_path,name=name,verbose=verbose) # load data if necessary
        self.update_var_from_vard(verbose=verbose)
        voi = self.determine_variables_to_test(fromto=fromto,variables_to_test=variables_to_test,
                                                name=name,ignore_zeros=ignore_zeros,verbose=verbose) # determine variables to test
        t00 = time()
        if len(voi)==0:
            if verbose>0:
                print(f'- No variable to test')
        else:
            if lots is None:
                lots = len(voi)
            vss = [voi[h*lots:(h+1)*lots] for h in range(len(voi)//lots)]
            vss += [voi[(len(voi)//lots)*lots:]]
            i=0

            for vs in vss: 
                if len(vs)>0:
                    i+=len(vs)
                    if verbose>0:
                        v0,v1 = vs[0],vs[-1]
                        print(f'- Testing {len(vs)} variables from {v0} to {v1} with n_jobs={n_jobs}\n\t...')

                    t0 = time()  
                    if n_jobs == 1:
                        results=[]
                        for v in vs:
                            if verbose==1:
                                print(v,end=' ')
                            elif verbose >1:
                                print('\n',v)
                            results+=[
                                self.univariate_test_of_variable(
                                                v,name=name,
                                                kernel=kernel,
                                                kernel_info=kernel_info,
                                                parallel=True,
                                                truncations_of_interest=truncations_of_interest,
                                                ignore_zeros=ignore_zeros,
                                                diagnostic=diagnostic,
                                                t_diagnostic=t_diagnostic,
                                                verbose=verbose-1
                                    )]
                        print('')
                    elif n_jobs >1:
                        with parallel_backend('loky'):
                            results = Parallel(n_jobs=n_jobs)(
                                delayed(self.univariate_test_of_variable)(
                                    v,name=name,
                                    kernel=kernel,
                                    kernel_info=kernel_info,
                                    truncations_of_interest=truncations_of_interest,
                                    diagnostic=diagnostic,
                                    ignore_zeros=ignore_zeros,
                                    t_diagnostic=t_diagnostic,
                                    parallel=True,
                                    verbose=verbose-1
                                ) for v  in  vs)
                    
                    if verbose>0:
                        print(f'\tDone in {time() - t0}')
                        print(f' \n\t{i}/{len(voi)} variables tested in {time() - t00}')
                    self.update_vard_from_parallel_univariate_kfda_results(results=results,tested_variables=vs,verbose=verbose) # update vard attribute with the results of vs genes tested
                    self.save_univariate_results(save_path=save_path,name=name,overwrite=overwrite,verbose=verbose) # save intermediate results in file in case of early interuption

        self.save_univariate_results(save_path,name,verbose=verbose) # save intermediate results in file in case of early interuption

    # functions for parallel univariate testing        
    def load_univariate_test_results_in_var_if_possible(self,save_path,name=None,verbose=0):
        
        if name is None:
            name=self.univariate_name

        if save_path is None:
            if verbose>1 : 
                print("- You can load previously computed univariate test results with parameter 'save_path'.")
        else:
            file_name = self.get_column_name_in_var(name=name,output='univariate')+'.csv'
            file_exists = os.path.isfile(save_path+file_name)
            
            if file_exists :
                if verbose > 0:
                    print(f'- Load univariate test results from \n\tdir: {save_path}\n\tfile: {file_name}')
                self.load_univariate_test_results_in_var(save_path,file_name,verbose=verbose)
            else:
                if verbose > 0:
                    print(f'- File to load univariate test results not found')
                    print(f'\tdir: {save_path} \n\tfile: {file_name} ')
                
    def update_var_from_vard(self,verbose=0):
        self.update_var_from_dataframe(pd.DataFrame(self.get_vard()).T,verbose=verbose)
        self.initialize_vard()

    def save_univariate_results(self,save_path,name,overwrite=False,verbose=0):
        """
        Save univariate tests results in a .csv file

        Parameters 
        ----------
            save_path : str
                path of the directory in which to save the resulting .csv file. 
                the file is automatically named with respect to test info. 
                
            name : str
                Refers to the set of test results to save

            overwrite (default = False) : bool
                if True : the written .csv file will contain current results only
                if False :if the result file to be saved already exists in the directory `save_path`, 
                    its results are loaded and added to the current results so that all the results
                    are saved. 
                        
        """

        self.update_var_from_vard(verbose=verbose)
        if save_path is not None:
            if verbose>1:
                print('- Saving intermediate results')
            if not overwrite:   
                self.load_univariate_test_results_in_var_if_possible(save_path,name,verbose=verbose)
            self.save_univariate_test_results_in_var(save_path,name=name,verbose=verbose)

    def determine_variables_to_test(self,fromto,name,variables_to_test=None,ignore_zeros=False,verbose=0): 
        """
        Selects the variable to perform an univariate test on 

        Parameters
        ----------
        fromto : list of two elements
            Ignored if variables_to_test is not None
            First and last index of the variables to test in the variable list. 
        
        name : str
            This name will be present in each column of the `var` dataframe containing the outputs of these tests. 

        variables_to_test : iterable of variables to test 
            Has priority on `fromto`

        ignore_zeros (default = False) : bool
            Whether or not considers null observations for univariate testing. 


        
        """
        var = self.get_var()
        col = self.get_column_name_in_var(name=name,output='univariate')
        variables = self.get_variables()

        if variables_to_test is None:
            fromto[1] = len(variables) if fromto[1]==-1 else fromto[1]      
            voi = variables[fromto[0]:fromto[1]]
        else: 
            voi = variables[variables.isin(variables_to_test)]

        nvtot = len(variables)
        if col in var:
            tested = var[var[col]==1].index
            voi = voi[~voi.isin(tested)]
        
        if ignore_zeros:
            voi = pd.Index([v for v in voi if self.has_non_zero_obs_in_both_conditions(v)])

        # verbose stuff
        if verbose >0:
            if len(voi)>0 :
                s=f'- Determined {len(voi)}/{nvtot} variables to test'
                if verbose == 1:
                    if variables_to_test is None:
                        v0,v1 = voi[0],voi[-1]            
                        s+= f'(from {v0} to {v1})'
                else:
                    voi_str = [str(i) for i in voi.tolist()]
                    s+= ":\n"+" ".join(voi_str)
                print(s)

            else:

                s=f'-No variable to test'
                if variables_to_test is None:
                    v0,v1 = fromto[0],fromto[1]
                    s+=f' in [{v0},{v1}]'

                    

        return(voi)

    def update_vard_from_parallel_univariate_kfda_results(self,results,tested_variables,verbose=0):
        if verbose>0:
            s=f'- Update vard with {len(tested_variables)} tested variables'
            if verbose >1:
                v0,v1 = tested_variables[0],tested_variables[-1]
                s+= f'({v0} to {v1})'
            print(s)
        for a_,v in zip(results,tested_variables):
            if a_ is None : 
                print(f'{v} was not tested')
            if a_ is not None:
                assert(a_['v'] == v)
                for key,value in a_.items():
                    if key != 'v':
                        vard = self.get_vard()
                        vard[v][key] = value

    def save_univariate_test_results_in_var(self,path,name=None,verbose=0):
        if name is None:
            name = self.univariate_name
        dn = self.data_name
        print(f'- Saving univariate test results of {name} in \n\tdir: {path}')   
        for cu in self.var[dn].columns:
            if 'univariate' in cu and name in cu:

                dname = cu.split(sep='_univariate')[0]
                cols = [c for c in self.var[dn].columns if f'{dname}_' in c]
                df = self.var[dn][cols]
                df = df[df[cu]==True]
                if verbose>0:
                    print(f'\tfile:{cu}.csv ({df.shape})')
                df.to_csv(f'{path}{cu}.csv')

    def load_univariate_test_results_in_var(self,path,file,verbose=0):
        df = pd.read_csv(f'{path}{file}',index_col=0)
        df = df[~df.index.duplicated(keep='first')]
        if verbose >0:
            print(f'- Loaded univariate test results : {df.shape}')
        self.update_var_from_dataframe(df,verbose=verbose)
    
    # pvalue related
    def get_rejected_genes_for_toi(self,name=None):
        if name is None:
            name = self.univariate_name
        kfdat_name = self.get_kfdat_name()
        tnames = ['1','r1','r2','r3','rmax']
        dict_rejected = {}
        for t in tnames:
            for corrected in [True,False]:
                corr_str = 'BHc' if corrected else ''
                pval_col = f'{name}_{kfdat_name}_t{t}_pval{corr_str}'
                pvals = self.var[self.data_name][f'{pval_col}']
                pvals = filter_genes_wrt_pval(pvals,threshold=.05)
                print(pval_col,len(pvals))
                dict_rejected[f'{pval_col}'] = pvals
        return(dict_rejected)

    def get_rejected_genes(self,name=None,tname='r1',corrected=False,threshold=.05):
        if name is None:
            name = self.univariate_name
        kfdat_name = self.get_kfdat_name()
        corr_str = 'BHc' if corrected else ''
        pval_col = f'{name}_{kfdat_name}_t{tname}_pval{corr_str}'
        pvals = self.get_var()[f'{pval_col}']
        pvals = filter_genes_wrt_pval(pvals,threshold=threshold)
        df_output = pd.DataFrame()
        df_output['pvals'] = pvals
        if tname.isdigit():
            df_output['trunc'] = int(tname)
        else:
            df_output['trunc'] = self.get_var()[f'{name}_{kfdat_name}_t{tname}_t']
        return(df_output)
        
    def get_tested_variables(self,name=None,verbose=0):
        var = self.get_var()
        col = self.get_column_name_in_var(name=name,output='univariate')
        
        if col not in var:
            if verbose>0:
                print(f'- Specified univariate tests not performed, you can run ktest.univariate_test(name={name})')
            return([])
        
        else:
            var_tested = var[var[col]==1].index
            ntested = len(var_tested)
            nvar = self.get_nvariables()
            if verbose>0:
                if ntested != nvar:
                    print(f'Warning : only {ntested} variables were tested out of {nvar}')
            return(var_tested)

    def get_ntested_variables(self,name=None,verbose=0):
        return(len(self.get_tested_variables(name=name,verbose=verbose)))

    def get_pvals_univariate(self,t=None,name=None,corrected=True,verbose=0):
        """
        Returns the pandas.Series of pvalues.

        Parameters 
        ----------

        t (default = 10): int
            Truncation parameter 
        
        name (default = ktest.univariate_name):
            Name given to the set of univariate tests. 
            See ktest.univariate_test() for details.

        corrected (default = True) : bool 
            whether correct the pvalues for multiple testing or not.

        verbose (default = 0) ; int 
            The greater, the more verbose is the output.
        """
        
        if t is None:
            t = self.truncation

        tested = self.get_tested_variables(name,verbose=verbose)
        ntested = len(tested)
        
        if ntested>0:
            if corrected :
                self.correct_BenjaminiHochberg_pval_univariate(t=t,name=name,verbose=verbose)
            var = self.get_var()
            col = self.get_column_name_in_var(t=t,name=name,corrected=corrected,output='pval')
            return(var[col][var.index.isin(tested)].sort_values())

    def get_kfda_univariate(self,t=None,name=None,verbose=0):
        """
        Returns the pandas.Series of pvalues.

        Parameters 
        ----------

        t (default = 10): int
            Truncation parameter 
        
        name (default = ktest.univariate_name):
            Name given to the set of univariate tests. 
            See ktest.univariate_test() for details.

        corrected (default = True) : bool 
            whether correct the pvalues for multiple testing or not.

        verbose (default = 0) ; int 
            The greater, the more verbose is the output.
        """
        
        if t is None:
            t = self.truncation

        tested = self.get_tested_variables(name,verbose=verbose)
        ntested = len(tested)
        
        if ntested>0:
            var = self.get_var()
            col = self.get_column_name_in_var(t=t,name=name,output='kfda')
            return(var[col][var.index.isin(tested)].sort_values())
        

    def get_DE_genes(self,t=None,name=None,corrected=True,threshold=.05,verbose=0):
        """
        Returns the pandas.Series of pvalues lower than 'threshold'.

        Parameters 
        ----------

        t (default = 10): int
            Truncation parameter 
        
        name (default = ktest.univariate_name):
            Name given to the set of univariate tests. 
            See ktest.univariate_test() for details.

        corrected (default = True) : bool 
            whether correct the pvalues for multiple testing or not. 

        threshold (default = .05) : float in [0,1]
            test rejection threshold.

        verbose (default = 0) ; int 
            The greater, the more verbose is the output.
        """

        pvals = self.get_pvals_univariate(t=t,name=name,corrected=corrected,verbose=verbose)
        return(pvals[pvals<threshold])
        
    def get_column_name_in_var(self,t=1,name=None,output='univariate',corrected=True):
        if name is None:
            name = self.univariate_name
        
        col = f"{name}_{self.get_kfdat_name()}_"

        if output=='univariate':
            col+=f'univariate'
        
        else:
            col+=f't{t}_{output}'

        if output == 'pval' and corrected:
            col+='BHc'
        
        return(col)
            
    def get_corrected_variables(self,t,name=None,verbose=0):
        var = self.get_var()
        col = self.get_column_name_in_var(t=t,name=name,output='corrected')
        
        if col not in var:
            if verbose>0:
                print(f'- Not any variable has been corrected yet')
            return([])
        
        else:
            var_corrected = var[var[col]==1].index
            ncorrected = len(var_corrected)
            ntested = self.get_ntested_variables(name=name)
            if verbose>0:
                if ncorrected != ntested:
                    print(f'Warning : only {ncorrected} variables were corrected out of {ntested} tested variables')
            return(var_corrected)

    def get_ncorrected_variables(self,t,name=None,verbose=0):
        return(len(self.get_corrected_variables(t=t,name=name,verbose=verbose)))

    def correct_BenjaminiHochberg_pval_univariate(self,t,name=None,
                        exceptions=[],focus=None,
                        verbose=0):
        
        ncorrected = self.get_ncorrected_variables(t=t,name=name,verbose=verbose)
        nvar = self.get_nvariables()
        if ncorrected == nvar:
            if verbose:
                print(f'All the {nvar} variables are already corrected for multiple testing')
        else:

        
            col = self.get_column_name_in_var(t=t,
                                                corrected=False,
                                                name=name,
                                                output='pval') 

            pval = self.var[self.data_name][col]
            pval = pval if focus is None else pval[pval.index.isin(focus)]
            pval = pval[~pval.index.isin(exceptions)]
            pval = pval[~pval.isna()]
            ngenes_to_correct = len(pval)

            if ngenes_to_correct > ncorrected:
                if verbose >0:
                    print(f"- Updating corrected pvals with {ngenes_to_correct - ncorrected} tested variables out of {ngenes_to_correct}.")
                dfc = pd.DataFrame(index=self.get_variables())
                dfc[col+'BHc'] = correct_BenjaminiHochberg_pval_of_dfcolumn(pval)
                colc = self.get_column_name_in_var(t=t,name=name,output='corrected')

                corrected_genes = pval.index       
                dfc[colc] = False
                series = dfc[colc].copy()
                series[corrected_genes] = True
                dfc[colc] = series
        
                self.update_var_from_dataframe(dfc)
