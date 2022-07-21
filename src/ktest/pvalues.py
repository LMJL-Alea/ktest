
from scipy.stats import chi2
import pandas as pd
from joblib import parallel_backend
from joblib import Parallel, delayed
from .univariate_testing import filter_genes_wrt_pval

"""
Ces fonctions gèrent tout ce qui est relatif aux pvaleurs. 
Je n'ai jamais pris le temps d'ajouter des fonctions spécifiques à des pvaleurs obtenues par permutation.
"""

def compute_pval(self,t=None):
    """
    Computes the asymptotic pvalues of the kfda statistic. 
    
    Calcul des pvalue asymptotique d'un df_kfdat pour chaque valeur de t. 
    Attention, la présence de Nan augmente considérablement le temps de calcul. 
    """
    pvals = {}
    pvals_contrib = {}
    t = min(100,len(self.df_kfdat)) if t is None else min(t,len(self.df_kfdat))
    trunc=range(1,t+1)

    for t_ in trunc:
        pvals[t_] = self.df_kfdat.T[t_].apply(lambda x: chi2.sf(x,int(t_)))
        pvals_contrib[t_] = self.df_kfdat_contributions.T[t_].apply(lambda x: chi2.sf(x,1))

    self.df_pval = pd.DataFrame(pvals).T 
    self.df_pval_contributions = pd.DataFrame(pvals_contrib).T

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

def correct_BenjaminiHochberg_pval(self,t=20):
    """
    Correction of the p-values of df_pval according to Benjamini and Hochberg 1995 approach.
    This is to use when the different tests correspond to multiple testing. 
    The results are stored in self.df_BH_corrected_pval 
    The pvalues are adjusted for each truncation lower or equal to t. 
    """
    
    self.df_pval_BH_corrected = correct_BenjaminiHochberg_pval_of_dataframe(self.df_pval,t=t)

def correct_BenjaminiHochberg_pval_univariate(self,var_prefix,exceptions=[],focus=None,add_to_prefix=''):
    pval = self.var[f'{var_prefix}_pval']
    pval = pval if focus is None else pval[pval.index.isin(focus)]
    pval = pval[~pval.index.isin(exceptions)]
    pval = pval[~pval.isna()]
    pvalBH = correct_BenjaminiHochberg_pval_of_dfcolumn(pval)
    self.var[f'{var_prefix}{add_to_prefix}_pvalBHc'] = pvalBH

def get_rejected_variables_univariate(self,var_prefix,BH=False):
    BH_str = 'BHc' if BH else ''
    pval = self.var[f'{var_prefix}_pval{BH_str}']
    pval = pval[~pval.isna()]
    pval = pval[pval<.05]
    return(pval.sort_values())

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


