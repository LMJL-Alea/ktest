import pandas as pd

from time import time
from joblib import Parallel, delayed, parallel_backend
import os
import numpy as np
from .utils_univariate import filter_genes_wrt_pval

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

    def ktest_for_one_variable(self,variable,kernel=None,inform_var=True,verbose=0):
        # Get testing info from global object 
        from .tester import ktest

        # Initialize univariate Ktest with common covariance 
        # var_metadata = pd.DataFrame(self.get_var().loc[variable]).T if inform_var else None
        kernel = self.get_kernel_params() if kernel is None else kernel
        if verbose >0:
            print(f'- Defining ktest for variable {variable}')

        if verbose >1:
            print(f'\tdata_name : {self.data_name}')
            print(f'\tcondition : {self.condition}')
            print(f'\tsamples : {self.samples}')
            print(f'\ttest_params : {self.get_test_params()}')
            print(f'\tcenter_by : {self.center_by}')
            print(f'\tkernel : {self.get_kernel_params}')


        t = ktest(
                    data =self.init_df_proj(variable),
                    metadata =self.obs.copy(),
                    data_name=  self.data_name,
                    condition= self.condition,
                    samples=self.samples,
                    var_metadata=pd.DataFrame(self.get_var().loc[variable]).T if inform_var else None,
                    test_params=self.get_test_params(),
                    center_by=self.center_by,
                    marked_obs_to_ignore=self.marked_obs_to_ignore,
                    kernel=kernel,
                    verbose=verbose
                    )
        return(t)

    def compute_univariate_kfda(self,variable,kernel=None,verbose=0):
        # récupérer la donnée
        

        t = self.ktest_for_one_variable(variable=variable,kernel=kernel,verbose=verbose)
        t.multivariate_test(verbose=verbose)

        # ccovw = 'ccovw' if common_covariance else ''
        if t.has_kfda_statistic:
            kfdat_name = t.get_kfdat_name()
            t.df_kfdat[variable] = t.df_kfdat[kfdat_name]
            t.df_pval[variable] = t.df_pval[kfdat_name]        
        return(t)
        
    def add_results_univariate_kfda_in_var(self,vtest,variable,name='',tmax=31,verbose=0):
        if verbose >0:
            print('- Add results univariate kfda in var')
        
        vard = self.get_vard()
        
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

    def univariate_kfda(self,variable,name=None,kernel_bandwidths_col_in_var=None,parallel=False,verbose=0):

        col = self.get_column_name_in_var(name=name,output='univariate')
        var = self.get_var()
        vard = self.get_vard()

        if col in var and var[col][variable]==1:
            if verbose>1:
                print(f'\t{variable} already tested')
        else:
            kernel = self.get_kernel_params()
            if kernel_bandwidths_col_in_var is not None:
                kernel['bandwidth'] = var[kernel_bandwidths_col_in_var].loc[variable]
            t=self.compute_univariate_kfda(variable,kernel=kernel,verbose=verbose)

        self.add_results_univariate_kfda_in_var(t,variable,name=name,verbose=verbose)
        
        if parallel:
            return({'v':variable,**vard[variable]})
        return(t) 
 

    def univariate_test(self,n_jobs=1,lots=100,fromto=[0,-1],variables_to_test=None,save_path=None,name='',kernel_bandwidths_col_in_var=None,verbose=0):
        """
        parallel univariate testing.
        Variables are tested by groups of `lots` variables in order to save intermediate results if the 
        procedure is interupted before the end. 
        """

        self.univariate_name=name
        self.load_univariate_test_results_in_var_if_possible(save_path,name=name,verbose=verbose) # load data if necessary
        self.update_var_from_vard(verbose=verbose)
        voi = self.determine_variables_to_test(fromto=fromto,variables_to_test=variables_to_test,
                                                name=name,verbose=verbose) # determine variables to test
        t00 = time()
        # not parallel testing
        if len(voi)==0:
            if verbose>0:
                print(f'- No variable to test')
        else:
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
                            results+=[self.univariate_kfda(v,name=name,
                                                    kernel_bandwidths_col_in_var=kernel_bandwidths_col_in_var,
                                                    parallel=True,
                                                    verbose=verbose-1)]
                        print('')
                    elif n_jobs >1:
                        with parallel_backend('loky'):
                            results = Parallel(n_jobs=n_jobs)(delayed(self.univariate_kfda)(v,name=name,
                            kernel_bandwidths_col_in_var=kernel_bandwidths_col_in_var,
                            parallel=True,
                            verbose=verbose-1) for v  in  vs)
                    
                    if verbose>0:
                        print(f'\tDone in {time() - t0}')
                        print(f' \n\t{i}/{len(voi)} variables tested in {time() - t00}')
                    self.update_vard_from_parallel_univariate_kfda_results(results=results,tested_variables=vs,verbose=verbose) # update vard attribute with the results of vs genes tested
                    self.save_intermediate_results(save_path,name,verbose=verbose) # save intermediate results in file in case of early interuption



    # functions for parallel univariate testing        
    def load_univariate_test_results_in_var_if_possible(self,save_path,name=None,verbose=0):
        
        if name is None:
            name=self.univariate_name

        if save_path is None:
            if verbose>0 : 
                print("- No test results loaded. Inform 'save_path' to load univariate test results")
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

    def save_intermediate_results(self,save_path,name,verbose=0):
        self.update_var_from_vard(verbose=verbose)
        if save_path is not None:
            if verbose>1:
                print('- Saving intermediate results')
            self.load_univariate_test_results_in_var_if_possible(save_path,name,verbose=verbose)
            self.save_univariate_test_results_in_var(save_path,verbose=verbose)

    def determine_variables_to_test(self,fromto,name,variables_to_test=None,verbose=0): 
        """
        Selects the variable to perform an univariate test on 

        Parameters
        ----------
        fromto : list of two elements
            Ignored if variables_to_test is not None
            First and last index of the variables to test in the variable list. 
        
        variables_to_test : iterable of variables to test 
            Has priority on `fromto`

        name : str
            This name will be present in each column of the `var` dataframe containing the outputs of these tests. 

        
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
        
        if verbose >0:
            if len(voi)>0 and variables_to_test is None:
                v0,v1 = voi[0],voi[-1]
                print(f'- Determined {len(voi)}/{nvtot} variables to test (from {v0} to {v1})')
            elif len(voi)>0 and variables_to_test is not None:
                print(f'- Determined {len(voi)}/{nvtot} variables to test')

            else:
                v0,v1 = fromto[0],fromto[1]
                print(f'- Nothing to test in [{v0},{v1}]')
        return(voi)

    def update_vard_from_parallel_univariate_kfda_results(self,results,tested_variables,verbose=0):
        if verbose>0:
            v0,v1 = tested_variables[0],tested_variables[-1]
            print(f'\tUpdate {len(tested_variables)} tested variables in vard (from {v0} to {v1})')
        for a_,v in zip(results,tested_variables):
            if a_ is None : 
                print(f'{v} was not tested')
            if a_ is not None:
                assert(a_['v'] == v)
                for key,value in a_.items():
                    if key != 'v':
                        vard = self.get_vard()
                        vard[v][key] = value

    def save_univariate_test_results_in_var(self,path,verbose=0):
        dn = self.data_name
        print(f'- Saving univariate test results in \n\tdir: {path}')   
        for cu in self.var[dn].columns:
            if 'univariate' in cu:

                dname = cu.split(sep='_univariate')[0]
                cols = [c for c in self.var[dn].columns if f'{dname}_' in c]
                df = self.var[dn][cols]
                df = df[df[cu]==True]
                if verbose>0:
                    print(f'\tfile:{cu}.csv ({df.shape})')
                df.to_csv(f'{path}{cu}.csv')

    def load_univariate_test_results_in_var(self,path,file,verbose=0):
        df = pd.read_csv(f'{path}{file}',index_col=0)
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

        # def get_ntested_variables_(self,name=None,verbose=0):

        #     var = self.get_var()
        #     col = self.get_column_name_in_var(name=name,output='univariate')
            
        #     if col not in var:
        #         if verbose>0:
        #             print(f'- Specified univariate tests not performed, you can run ktest.univariate_test(name={name})')
        #         return(0)

        #     else:
        #         nvar = self.get_nvariables()
        #         ntested = int(var[col].sum())
        #         if verbose>0:
        #             if nvar != ntested:
        #                 print(f'Warning : only {ntested} variables were tested out of {nvar}')
        #         return(ntested)


    def get_pvals_univariate(self,t,name=None,corrected=True,verbose=0):
        """
        Returns the pandas.Series of pvalues.

        Parameters 
        ----------

        t : int
            Truncation parameter 
        
        name (default = ktest.univariate_name):
            Name given to the set of univariate tests. 
            See ktest.univariate_test() for details.

        corrected (default = True) : bool 
            whether correct the pvalues for multiple testing or not.

        verbose (default = 0) ; int 
            The greater, the more verbose is the output.
        """
        tested = self.get_tested_variables(name,verbose=verbose)
        ntested = len(tested)
        
        if ntested>0:

            var = self.get_var()
            col = self.get_column_name_in_var(t=t,name=name,corrected=corrected,output='pval')
            
            if corrected :
                self.correct_BenjaminiHochberg_pval_univariate(trunc=t,name=name,verbose=verbose)

            return(var[col][var.index.isin(tested)])
        

    def get_DE_genes(self,t,name=None,corrected=True,threshold=.05,verbose=0):
        """
        Returns the pandas.Series of pvalues lower than 'threshold'.

        Parameters 
        ----------

        t : int
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
            
