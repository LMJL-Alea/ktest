import pandas as pd
# from .tester import Ktest
from joblib import Parallel, delayed, parallel_backend
import numpy as np


class Permutation:
    def __init__(self):        
        super(Permutation, self).__init__()

    def create_permuted_ktest(self,seed,verbose=0):
        from .tester import Ktest
        metadata_perm = self.obs.copy()
        condition = self.obs[self.condition].copy().values.tolist()
        

        # Permute labels (bientôt une fonction à part)
        np.random.seed(seed=seed)
        np.random.shuffle(condition)
        metadata_perm[self.condition] = condition
        
        kt_perm = Ktest(
            # data= self.get_dataframe_of_all_data(),
            data= self.get_data(dataframe=True,in_dict=False),
            metadata=metadata_perm,
            condition = self.condition,
            samples=self.samples,
            stat = self.stat,
            nystrom = self.nystrom,
            n_landmarks = self.n_landmarks_initial,
            n_anchors = self.n_anchors,
            landmark_method = self.landmark_method,
            anchor_basis = self.anchor_basis,
            marked_obs_to_ignore=self.marked_obs_to_ignore,
            verbose=verbose-1,

            kernel_function=self.kernel_function,
            kernel_bandwidth=self.kernel_bandwidth,
            kernel_median_coef=self.kernel_median_coef,
            kernel_name=self.kernel_name,
        )
        return(kt_perm)

    def compute_permutation_statistic(self,stat,seed,parallel=False,verbose=0):
        
        kt_perm = self.create_permuted_ktest(seed=seed,verbose=verbose)
        kt_perm.compute_test_statistic(stat=stat,verbose=verbose-1)
                
        if parallel == False:
            return(kt_perm)
        else:
            statistic = kt_perm.get_statistic(stat=stat)
            if stat == 'kfda':
                return(statistic)
            else: 
                return({'mmd':statistic})

    def get_permutation_name(self,n_permutations,seed):
        self.get_kfdat_name()
        mn = self.get_model_str()
        dtn = self.get_data_to_test_str()
        return(f'{mn}_perm{n_permutations}_seed{seed}_{dtn}')

    def compute_nperm_permutation_statistics(self,stat,n_permutations,seed,n_jobs=1,verbose=0):
        seeds = range(seed,seed+n_permutations)
        pn = self.get_permutation_name(n_permutations=n_permutations,seed=seed)
        if n_jobs == 1:
            results=[]
            for s in seeds:
                if verbose>1:
                    print(f's={seed}',end=' ')
                results+= [self.compute_permutation_statistic(
                                        stat=stat,
                                        seed=s,
                                        parallel=True,
                                        verbose=verbose)]
        else:
            results = Parallel(n_jobs=n_jobs)(delayed(self.compute_permutation_statistic)(
                                        stat=stat,
                                        seed=s,
                                        parallel=True,
                                        verbose=verbose,
                            ) for s in seeds)
        if stat == 'mmd':
            df = pd.DataFrame(results).rename(columns={'mmd':pn},inplace=False)
            return(df)
        else:
            return(pd.concat(results,axis=1))

    def compute_permutation_pvalue(self,stat,n_permutations,seed,perm_stats):
        
        self.compute_test_statistic(stat=stat)
        true_stat = self.get_statistic(stat=stat)
        pn = self.get_permutation_name(n_permutations=n_permutations,seed=seed)
        
        if stat == 'kfda':
            perm_pval = perm_stats.ge(true_stat,axis=0).sum(axis=1)/n_permutations
            self.df_pval[pn] = perm_pval
        elif stat == 'mmd' : 
            perm_pval = perm_stats.ge(true_stat).sum(axis=0)/n_permutations

            self.dict_pval_mmd[pn] = perm_pval.values[0]
        
    def store_permutation_statistics_in_ktest(self,stat,n_permutations,seed,perm_stats):
        pn = self.get_permutation_name(n_permutations=n_permutations,seed=seed)
        if stat == 'kfda':
            self.df_kfdat_perm[pn] = perm_stats
        elif stat == 'mmd' : 
            self.df_mmd_perm[pn] = perm_stats
            
    def permutation_pvalue(self,stat,n_permutations=None,seed=None,n_jobs=1,keep_permutation_statistics=False,verbose=0):
        
        if n_permutations is None: 
            n_permutations = self.n_permutations
        else:
            self.n_permutations = n_permutations

        if seed is None : 
            seed = self.seed_permutation
        else :
            self.seed_permutation = seed 
        

        pn = self.get_permutation_name(n_permutations=n_permutations,seed=seed)

        if verbose>0:
            print(f'- Permutation statistic n_permutations={n_permutations} seeds from {seed} to {seed+n_permutations} with {n_jobs} jobs')

        if pn in self.df_pval:
            if verbose>0:
                print(f'- This permutation p-value has already been computed')
        else:

            perm_stats = self.compute_nperm_permutation_statistics(
                                                stat=stat,
                                                 n_permutations=n_permutations,
                                                 seed=seed,
                                                 n_jobs=n_jobs,
                                                 verbose=verbose
                                                )

            self.compute_permutation_pvalue(
                                                stat=stat,
                                                n_permutations=n_permutations,
                                                seed=seed,
                                                perm_stats=perm_stats
                                                )
            
            if keep_permutation_statistics:
                self.store_permutation_statistics_in_ktest(
                                                stat=stat,
                                                n_permutations=n_permutations, 
                                                seed=seed,
                                                perm_stats=perm_stats)
            if stat == 'mmd':
                self.permutation_mmd_name = pn
            elif stat == 'kfda':
                self.permutation_kfda_name = pn
        return(pn)



    # Permutation pour la MANOVA 

    def compute_permutation_kernel_hotelling_lawley_statistic(
            self,seed,parallel=False,verbose=0):
        
        kt_perm = self.create_permuted_ktest(seed=seed,verbose=verbose)
        
        kt_perm.set_design(self.design_cols)
        kt_perm._diagonalize_residual_covariance()
        
        kt_perm.set_hypothesis(L = self.hypotheses[self.current_hyp]['L'],
                            hypothesis_name = self.current_hyp,
                            Tmax=self.hypotheses[self.current_hyp]['Tmax'])
        kt_perm.compute_kernel_Hotelling_Lawley_test_statistic(hypothesis_name=self.current_hyp)
        statistic = pd.DataFrame(kt_perm.hypotheses[self.current_hyp]['Hotellin-Lawley'],columns=[seed])
        
        if parallel == False:
            return(kt_perm)
        else:
            
            return(statistic)


    def compute_nperm_permutation_HL_statistics(self,n_permutations,seed,n_jobs=1,verbose=0):
        seeds = range(seed,seed+n_permutations)
        
        pn = self.get_permutation_name(n_permutations=n_permutations,seed=seed)
        if n_jobs == 1:
            results=[]
            for s in seeds:
                if verbose>1:
                    print(f's={seed}',end=' ')
                results+= [self.compute_permutation_kernel_hotelling_lawley_statistic(
                                        seed=s,
                                        parallel=True,
                                        verbose=verbose)]
        else:
            results = Parallel(n_jobs=n_jobs)(delayed(self.compute_permutation_kernel_hotelling_lawley_statistic)(
                                        seed=s,
                                        parallel=True,
                                        verbose=verbose,
                            ) for s in seeds)
        return(pd.concat(results,axis=1))




    def compute_permutation_pvalue_for_HL(self,n_permutations,seed,perm_stats):
        self._diagonalize_residual_covariance()
        self.compute_kernel_Hotelling_Lawley_test_statistic(hypothesis_name=self.current_hyp)
        true_stat = pd.DataFrame(self.hypotheses[self.current_hyp]['Hotellin-Lawley'],columns=['HL'])
        pn = self.get_permutation_name(n_permutations=n_permutations,seed=seed)

        perm_pval = perm_stats.ge(true_stat,axis=0).sum(axis=1)/n_permutations
        self.df_pval[f'HL_{pn}'] = perm_pval
    
    def permutation_HL_pvalue(self,
                        n_permutations=None,
                        seed=None,
                        n_jobs=1,
                        keep_permutation_statistics=False,
                        verbose=0):

        if n_permutations is None: 
            n_permutations = self.n_permutations
        else:
            self.n_permutations = n_permutations

        if seed is None : 
            seed = self.seed_permutation
        else :
            self.seed_permutation = seed 


        pn = self.get_permutation_name(n_permutations=n_permutations,seed=seed)

        if verbose>0:
            print(f'- Permutation statistic n_permutations={n_permutations} seeds from {seed} to {seed+n_permutations} with {n_jobs} jobs')

        if pn in self.df_pval:
            if verbose>0:
                print(f'- This permutation p-value has already been computed')
        else:

            perm_stats = self.compute_nperm_permutation_HL_statistics(
                                                n_permutations=n_permutations,
                                                seed=seed,
                                                n_jobs=n_jobs,
                                                verbose=verbose
                                                )

            self.compute_permutation_pvalue_for_HL(
                                                n_permutations=n_permutations,
                                                seed=seed,
                                                perm_stats=perm_stats
                                                )

    #         if keep_permutation_statistics:
    #             self.store_permutation_statistics_in_ktest(
    #                                             stat=stat,
    #                                             n_permutations=n_permutations, 
    #                                             seed=seed,
    #                                             perm_stats=perm_stats)
            self.permutation_HL_name = f'HL_{pn}'
        return(f'HL_{pn}')









