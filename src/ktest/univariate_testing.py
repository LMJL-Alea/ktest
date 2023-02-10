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

    def get_univariate_results_in_var(self,trunc,name):
        return(f'{name}_{self.get_kfdat_name()}_t{trunc}')

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

    def add_zero_proportions_to_var(self):
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
        dict_df = self.get_dataframes_of_data()
        dfzp = pd.DataFrame()
        for s,df in dict_df.items():
            dfzp[f'{s}_nz'] = (df==0).sum()
            dfzp[f'{s}_n'] = len(df)
            dfzp[f'{s}_pz'] = dfzp[f'{s}_nz']/len(df)
            dfzp[f'{s}_pct_expression'] = 1 - dfzp[f'{s}_pz']
            dfzp[f'{s}_mean'] = df.mean()

#                 dfzp[f'{s}_{g1[:-1]}_diff_pct'] = dfzp[f'{s}_{g1}_pct_expression'] - dfzp[f'{s}_{g2}_pct_expression'] 
#                 dfzp[f'sanity_{s}_{g1[:-1]}_log2fc'] = np.log(dfzp[f'sanity_{s}_{g1}_mean']/dfzp[f'sanity_{s}_{g2}_mean'])/np.log(2)
#                 dfzp[f'count_{s}_{g1[:-1]}_log2fc'] = np.log(dfzp[f'count_{s}_{g1}_mean']/dfzp[f'count_{s}_{g2}_mean'])/np.log(2)
#       
        self.update_var_from_dataframe(dfzp)
    
    def add_log2fc_to_var(self,data_name=None):
        if data_name is not None:
            current_data_name = self.data_name
            self.data_name = data_name
            dn = f'{data_name}_'
        else :
            dn = ''
        dict_df = self.get_dataframes_of_data()
        dfzp = pd.DataFrame()
        for s,df in dict_df.items():
            dfzp[f'{dn}{s}_mean'] = df.mean()
        s1,s2 = dict_df.keys()
        dfzp[f'{dn}log2fc_{s1}/{s2}'] = np.log(dfzp[f'{dn}{s1}_mean']/dfzp[f'{dn}{s2}_mean'])/np.log(2)
    #     print(dfzp)
        if data_name is not None: 
            self.data_name = current_data_name
        self.update_var_from_dataframe(dfzp)

    def compute_log2fc_from_another_dataframe(self,df,data_name='raw'):
        
        index = self.get_dataframe_of_all_data().index
        df = df[df.index.isin(index)]
        self.add_data_to_Tester_from_dataframe(df,data_name=data_name,update_current_data_name=False)            
        self.add_log2fc_to_var(data_name=data_name)


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
 
    def get_genes_presents_in_more_than_k_cells(self,k=3):
        cs = list(self.get_data().keys())
        var = self.var[self.data_name]
        n = self.get_ntot()
        var['ncells_expressed'] = n - (var[f'{cs[0]}_nz'] + var[f'{cs[1]}_nz'])
        return(var[var['ncells_expressed']>=k].index)

    def ktest_for_one_variable(self,variable,kernel=None,inform_var=True):
        # Get testing info from global object 
        from .tester import ktest_multivariate

        # Initialize univariate tester with common covariance 
        # var_metadata = pd.DataFrame(self.get_var().loc[variable]).T if inform_var else None
        
        t = ktest_multivariate(
                    data =self.init_df_proj(variable),
                    metadata =self.obs.copy(),
                    data_name=  self.data_name,
                    condition= self.condition,
                    samples=self.samples,
                    var_metadata=pd.DataFrame(self.get_var().loc[variable]).T if inform_var else None,
                    test_params=self.get_test_params(),
                    center_by=self.center_by,
                    marked_obs_to_ignore=self.marked_obs_to_ignore,
                    kernel=self.get_kernel_params() if kernel is None else kernel,)

        return(t)

    def compute_univariate_kfda(self,variable,kernel=None):
        # récupérer la donnée
        
        t = self.ktest_for_one_variable(variable=variable,kernel=kernel)
        t.initialize_kfdat()
        kfdat_name = t.kfdat()

        # ccovw = 'ccovw' if common_covariance else ''
        t.df_kfdat[variable] = t.df_kfdat[kfdat_name]
        t.df_pval[variable] = t.df_pval[kfdat_name]
        
        return(t)
        
    def add_results_univariate_kfda_in_var(self,vtest,variable,name='',verbose=False):
        kfdat_name = self.get_kfdat_name()
        dname = f'{name}_{kfdat_name}'
        vard = self.get_vard()

        tmax = len(vtest.get_explained_difference())
        ts = list(range(1,31)) if tmax >30 else list(range(1,tmax))

        tnames = [str(t) for t in ts]
        tr1 = vtest.select_trunc_by_between_reconstruction_ressaut(which_ressaut='first')
        tr2 = vtest.select_trunc_by_between_reconstruction_ressaut(which_ressaut='second')
        tr3 = vtest.select_trunc_by_between_reconstruction_ressaut(which_ressaut='third')
        trm = vtest.select_trunc_by_between_reconstruction_ressaut(which_ressaut='max')

        ts += [tr1,tr2,tr3,trm]
        tnames += ['r1','r2','r3','rmax']
        for t,tname in zip(ts,tnames):
            if t<100:

                col = f'{dname}_t{tname}'
                pval = vtest.df_pval[variable][t]
                kfda = vtest.df_kfdat[variable][t]

                errB = vtest.get_explained_difference_of_t(t)
                errW = vtest.get_explained_variability_of_t(t)

                vard[variable][f'{col}_pval'] = pval
                vard[variable][f'{col}_kfda'] = kfda 
                vard[variable][f'{col}_errB'] = errB
                vard[variable][f'{col}_errW'] = errW
                if tname not in [str(t) for t in ts]:
                    vard[variable][f'{col}_t'] = t

        vard[variable][f'{dname}_univariate'] = True
                
        if verbose:
            tab = '\t' if len(variable)>6 else '\t\t'
            zp = self.get_zero_proportions_of_variable(variable)
        
            tr1 = vard[variable][f'{dname}_tr1_t']
            tr2 = vard[variable][f'{dname}_tr2_t']
            errB1 = vard[variable][f'{dname}_tr1_errB']
            errB2 = vard[variable][f'{dname}_tr2_errB']
            pval1 = vard[variable][f'{dname}_tr1_pval']
            pval2 = vard[variable][f'{dname}_tr2_pval']

            zp_string = "zp: "+" ".join([f'{s}{zp[s]:.2f}' for s in zp.keys()])
            string = f'{variable} {tab} {zp_string} \t pval{tr1:1.0f}:{pval1:1.0e}  r{tr1:1.0f}:{errB1:.2f} \t pval{tr2:1.0f}:{pval2:1.0e}  r{tr2:1.0f}:{errB2:.2f}'
            print(string)

    def univariate_kfda(self,variable,name,kernel_bandwidths_col_in_var=None,parallel=False):
        

        univariate_name = f'{self.get_kfdat_name}_univariate'
        dn = self.data_name


        if univariate_name in self.var[dn] and self.var[dn][univariate_name][variable]:
            print(f'{univariate_name} already computed for {variable}')
        else:
            kernel = self.get_kernel_params()
            if kernel_bandwidths_col_in_var is not None:
                kernel['bandwidth'] = self.get_var()[kernel_bandwidths_col_in_var].loc[variable]
            t=self.compute_univariate_kfda(variable,kernel=kernel)

        self.add_results_univariate_kfda_in_var(t,variable,name=name)
        
        if parallel:
            return({'v':variable,**self.vard[dn][variable]})
        return(t) 
 
    def ktest_univariate(self,n_jobs=1,lots=100,fromto=[0,-1],save_path=None,name='',kernel_bandwidths_col_in_var=None,verbose=0):
        """
        parallel univariate testing.
        Variables are tested by groups of `lots` variables in order to save intermediate results if the 
        procedure is interupted before the end. 
        """

        
        self.load_univariate_test_results_in_var_if_possible(save_path,name,verbose=verbose) # load data if necessary
        voi = self.determine_variables_to_test(fromto=fromto,name=name,verbose=verbose) # determine variables to test

        # not parallel testing
        if n_jobs==1:
            results=[]
            for v in voi:
                if verbose>0:
                    print(v)
                results+=[self.univariate_kfda(v,name,kernel_bandwidths_col_in_var=kernel_bandwidths_col_in_var,parallel=True)]
        
        # parallel testing 
        elif n_jobs >1:
            vss = [voi[h*lots:(h+1)*lots] for h in range(len(voi)//lots)]
            vss += [voi[(len(voi)//lots)*lots:fromto[1]]]
            i=0
            for vs in vss: 

                i+=len(vs)
                if verbose>0:
                    print(f'testing {i}/{len(voi)}')
                t0 = time()  
                with parallel_backend('loky'):
                    results = Parallel(n_jobs=n_jobs)(delayed(self.univariate_kfda)(v,name,
                    kernel_bandwidths_col_in_var=kernel_bandwidths_col_in_var,parallel=True) for v  in  vs)
                
                if verbose>0:
                    print(f'Tested in {time() - t0} \n\n')
                self.update_vard_from_parallel_univariate_kfda_results(results=results,tested_variables=vs,verbose=verbose) # update vard attribute with the results of vs genes tested
                self.save_intermediate_results(save_path,name,verbose=verbose) # save intermediate results in file in case of early interuption

    # functions for parallel univariate testing        
    def load_univariate_test_results_in_var_if_possible(self,save_path,name,verbose=0):
        file_name=f'{name}_{self.get_kfdat_name()}_univariate.csv'
        if verbose >0:
            print(f'{file_name} in dir :{file_name in os.listdir(save_path)}')
        if save_path is not None and file_name in os.listdir(save_path):
            if verbose >0:
                print(f'loading {file_name}')
            self.load_univariate_test_results_in_var(save_path,file_name,verbose=verbose)

    def update_var_from_vard(self):
        self.update_var_from_dataframe(pd.DataFrame(self.get_vard()).T)

    def save_intermediate_results(self,save_path,name,verbose=0):
        if save_path is not None:
            if verbose>0:
                print('saving')
            file_name=f'{name}_{self.get_kfdat_name()}_univariate.csv'
            self.update_var_from_vard()
            self.load_univariate_test_results_in_var_if_possible(save_path,file_name)
            self.save_univariate_test_results_in_var(save_path,verbose=verbose)

    def determine_variables_to_test(self,fromto,name,verbose=0): 
         
        var = self.get_var()
        dname = f'{name}_{self.get_kfdat_name()}'
        variables = self.get_variables()
        fromto[1] = len(variables) if fromto[1]==-1 else fromto[1]      
        voi = variables[fromto[0]:fromto[1]]
        if verbose >0:
            print(f'{len(voi)} variable to test')
        
        if f'{dname}_univariate' in var:
            tested = var[var[f'{dname}_univariate']==1].index
            voi = voi[~voi.isin(tested)]
        if verbose>0:
            print(f'{len(voi)} not already tested among them')
        return(voi)

    def update_vard_from_parallel_univariate_kfda_results(self,results,tested_variables,verbose=0):
        for a_,v in zip(results,tested_variables):
            if a_ is None : 
                print(f'{v} was not tested')
            if a_ is not None:
    #                 print(a_)
                assert(a_['v'] == v)
                for key,value in a_.items():
                    if key != 'v':
                        vard = self.get_vard()
                        vard[v][key] = value

    def save_univariate_test_results_in_var(self,path,verbose=0):
        dn = self.data_name
        for cu in self.var[dn].columns:
            if 'univariate' in cu:
                if verbose>0:
                    print(cu)
                dname = cu.split(sep='_univariate')[0]
                cols = [c for c in self.var[dn].columns if f'{dname}_' in c]
                df = self.var[dn][cols]
                df = df[df[cu]==True]
                if verbose>0:
                    print(df.shape)
                df.to_csv(f'{path}{cu}.csv')

    def load_univariate_test_results_in_var(self,path,file,verbose=0):
        df = pd.read_csv(f'{path}{file}',index_col=0)
        if verbose >0:
            print(file,len(df))
        self.update_var_from_dataframe(df,verbose=verbose)
    

    # pvalue related

    def get_rejected_genes_for_toi(self,name):
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

    def get_rejected_genes(self,name,tname='r1',corrected=False,threshold=.05):
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
        
