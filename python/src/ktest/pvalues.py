
from scipy.stats import chi2

import pandas as pd
from joblib import parallel_backend
from joblib import Parallel, delayed
import warnings
"""
Ces fonctions gèrent tout ce qui est relatif aux pvaleurs. 
Je n'ai jamais pris le temps d'ajouter des fonctions spécifiques à des pvaleurs obtenues par permutation.
"""

def correct_BenjaminiHochberg_pval_of_dfcolumn(df):
    df = pd.concat([df,df.rank()],keys=['pval','rank'],axis=1)
    df['pvalc'] = df.apply(lambda x: len(df) * x['pval']/x['rank'],axis=1) # correction
    df['rankc'] = df['pvalc'].rank() # calcul du nouvel ordre
    corrected_pvals = []
    # correction des pval qui auraient changé d'ordre
    l = []
    if not df['rankc'].equals(df['rank']):
        first_rank = df['rank'].sort_values().values[0] # égal à 1 sauf si égalité 
        pvalc_prec = df.loc[df['rank']==first_rank,'pvalc'].iat[0]
        df['rank'] = df['rank'].fillna(10000)
        for rank in df['rank'].sort_values().values[1:]: # le 1 est déjà dans rank prec et on prend le dernier 
            # if t >=8:
            #     print(rank,end=' ')
            pvalc = df.loc[df['rank']==rank,'pvalc'].iat[0]
            if pvalc_prec >= 1 : 
                pvalue = 1 # l += [1]
            elif pvalc_prec > pvalc :
                pvalue = pvalc # l += [pvalc]
            elif pvalc_prec <= pvalc:
                pvalue = pvalc_prec # l+= [pvalc_prec]
            else: 
                print('error pval correction',f'rank{rank} pvalc{pvalc} pvalcprec{pvalc_prec}')
                print(df.loc[df['rank']==rank].index)
            pvalc_prec = pvalc
            l += [pvalue]
        # dernier terme 
        pvalue = 1 if pvalc >1 else pvalc
        l += [pvalue]
#             corrected_pvals[t] = pd.Series(l,index=ranks)#df['rank'].sort_values().index)
    if len(l)>0: 
        return(pd.Series(l,index=df['rank'].sort_values().index))
    else: 
        return(pd.Series(df['pvalc'].values,index=df['rank'].sort_values().index))

def correct_BenjaminiHochberg_pval_of_dataframe(df_pval,t=20):
    """
    Benjamini Hochberg correction of a dataframe containing the p-values where the rows are the truncations.    
    """
    trunc = range(1,t+1)
    corrected_pvals = []
    for t in trunc:
        # print(t)
        corrected_pvals += [correct_BenjaminiHochberg_pval_of_dfcolumn(df_pval.T[t],t=t)]   
    return(pd.concat(corrected_pvals,axis=1).T)

# je sais pas si cette fonction est encore utile 
def parallel_BH_correction(dict_of_df_pval,stat,t=20,n_jobs=6):
    iter_param = list(dict_of_df_pval.keys())
    if stat == 'mmd':        
        with parallel_backend('loky'):
            a = Parallel(n_jobs=n_jobs)(delayed(correct_BenjaminiHochberg_pval_of_dfcolumn)(dict_of_df_pval[param]['0']) for param  in  iter_param)
    else: 
        with parallel_backend('loky'):
            a = Parallel(n_jobs=n_jobs)(delayed(correct_BenjaminiHochberg_pval_of_dataframe)(dict_of_df_pval[param],t=t) for param  in  iter_param)
    return({k:df for k,df in zip(iter_param,a)})


class Pvalues:
        
    def __init__(self):        
        super(Pvalues, self).__init__()

    def compute_kfda_asymptotic_pvalue(self):
        """
        Computes the asymptotic pvalues of the kfda statistic. 
        """

        kn = self.get_kfdat_name()
        kfda = self.get_kfda(contrib=False).to_frame()
        kfda_contrib = self.get_kfda(contrib=True).to_frame()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.df_pval[kn] = kfda.apply(lambda x: chi2.sf(x[kn],int(x.name)),axis=1) 
            self.df_pval_contributions[kn] = kfda_contrib.apply(lambda x: chi2.sf(x[kn],int(x.name)),axis=1) 

    def compute_pvalue(self,
                    stat=None,
                    permutation=None,
                    n_permutations=None,
                    seed_permutation=None,
                    n_jobs_permutation=1,
                    keep_permutation_statistics=False,
                    verbose=0
                    ):
        """
        Computes the p-value of the statistic of `stat`.         
        """

        if permutation is None:
            permutation = self.permutation
        else:
            self.permutation = permutation

        if stat is None:
            stat = self.stat 
        else:
            self.stat = stat



        if stat == 'kfda' and not permutation:
            self.compute_kfda_asymptotic_pvalue()
        else:
            self.permutation_pvalue(stat=stat, 
                                    n_permutations=n_permutations,
                                    seed=seed_permutation,
                                    n_jobs=n_jobs_permutation,
                                    keep_permutation_statistics=keep_permutation_statistics,
                                    verbose=verbose)

    def correct_BenjaminiHochberg_pval(self,t=20):
        """
        Correction of the p-values of df_pval according to Benjamini and Hochberg 1995 approach.
        This is to use when the different tests correspond to multiple testing. 
        The results are stored in self.df_BH_corrected_pval 
        The pvalues are adjusted for each truncation lower or equal to t. 
        """
        
        self.df_pval_BH_corrected = correct_BenjaminiHochberg_pval_of_dataframe(self.df_pval,t=t)

  

    # def get_rejected_variables_univariate(self,var_prefix,BH=False):
    #     BH_str = 'BHc' if BH else ''
    #     pval = self.var[f'{var_prefix}_pval{BH_str}']
    #     pval = pval[~pval.isna()]
    #     pval = pval[pval<.05]
    #     return(pval.sort_values())

