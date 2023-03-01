import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from .tester import Ktest
# from .pvalues import correct_BenjaminiHochberg_pval_of_dfcolumn,correct_BenjaminiHochberg_pval_of_dataframe
# from .functions import compute_standard_kfda,compute_standard_mmd
from .utils_pandas import pd_select_df_from_index


from joblib import parallel_backend,Parallel, delayed
from joblib.externals.loky import set_loky_pickler


def permute_and_compute_kfda(x,y,seed,gene,kernel='gauss_median',params_model={}):
    xb,yb = pd_permutation_ccdf(x,y,seed,gene)
    return(compute_standard_kfda(xb,yb,name=seed,pval=False,kernel=kernel,params_model=params_model))
    
def permute_and_compute_mmd(x,y,seed,gene,kernel='gauss_median',params_model={}):
    xb,yb = pd_permutation_ccdf(x,y,seed,gene)
    return(compute_standard_mmd(xb,yb,name=seed,kernel=kernel,params_model=params_model))
    
def non_parallel_permutation_from_dataframe(x,y,c,seeds,stat='kfda',kernel='gauss_median',params_model={}):
    xc,yc = x[c],y[c]
    if stat == 'kfda':
        
        test_orig = Ktest()
        test_orig.df_kfdat = compute_standard_kfda(xc,yc,name='orig',pval=False,kernel=kernel,params_model=params_model)
        
        outputs=[]
        for  seed in seeds:
            outputs += [permute_and_compute_kfda(xc,yc,seed,c,kernel=kernel,params_model=params_model)]
        kfda_perm = pd.concat(outputs,axis=1)
        return((kfda_perm.ge(test_orig.df_kfdat['orig'],axis=0).sum(axis=1)/len(seeds)).to_frame(name=c))
    
    
    elif stat == 'mmd':
        
        orig_mmd = compute_standard_mmd(xc,yc,name='orig',kernel=kernel,params_model=params_model)['orig']
        outputs={}
        for  seed in seeds:
            outputs[seed] = permute_and_compute_mmd(xc,yc,seed,c,kernel=kernel,params_model=params_model)[seed]
        mmd_perm = pd.Series(outputs)
        pval = mmd_perm.ge(orig_mmd).sum()/len(seeds)
        return({c:pval})

def parallel_permutation_gene_level_from_dataframes(x,y,columns,seeds,stat='kfda',n_jobs=7,kernel='gauss_median',params_model={}):

    # appelle une fonction en parallele qui itère sur les gènes
    
    with parallel_backend('loky'):
        a = Parallel(n_jobs=n_jobs)(delayed(non_parallel_permutation_from_dataframe)(x,y,c,seeds,stat=stat,kernel=kernel,params_model=params_model) for c  in  columns)
        # a = Parallel(n_jobs=n_jobs)(delayed(non_parallel_permutation_from_dataframe)(x,y,c,seeds,stat=stat,kernel=kernel) for c  in  columns)
    
    if stat == 'kfda':
        return(pd.concat(a,axis=1))
    else:
        d = a[0]
        for d_ in a[1:]:
            d.update(d_)
#         print(d)
        return(d)

def pd_permutation_ccdf(dfx,dfy,seed,gene):
    """
    return two permuted dataframes from a dataframe df
    seed : random seed for perputation
    n1 : nobs in first sample. if None, df is separated in two equal parts
    """
        
    n1 = len(dfx)
    df = pd.concat((dfx,dfy),axis=0)
    
    if gene in range(1,1001):
        
    
        df_index = df.index.tolist()
        np.random.seed(seed=seed)
        np.random.shuffle(df_index)
        xindex = df_index[:n1]
        yindex = df_index[n1:]
    

    
    elif gene in range(1001,5501):
        dfx_dropout = dfx[:n1//10]
        dfx_nonzero = dfx[n1//10:]

        dfy_dropout = dfy[:n1//10]
        dfy_nonzero = dfy[n1//10:]

        index_dropout = pd.concat((dfx_dropout,dfy_dropout),axis=0).index.tolist()
        index_nonzero = pd.concat((dfx_nonzero,dfy_nonzero),axis=0).index.tolist()

        np.random.seed(seed=seed*2)
        np.random.shuffle(index_dropout)
        np.random.seed(seed=seed*2+1)
        np.random.shuffle(index_nonzero)

        xindex = index_dropout[:n1//10]+index_nonzero[:9*(n1//10)]
        yindex = index_dropout[n1//10:]+index_nonzero[9*(n1//10):]

    elif gene in range(5501,10001):
        
        dfx_mod1 = dfx[:n1//2]
        dfx_mod2 = dfx[n1//2:]

        dfx_dropout = pd.concat((dfx_mod1[:n1//10],dfx_mod2[:n1//10]),axis=0)
        dfx_nzmod1 = dfx_mod1[n1//10:]
        dfx_nzmod2 = dfx_mod2[n1//10:]

        dfy_mod1 = dfy[:n1//2]
        dfy_mod2 = dfy[n1//2:]

        dfy_dropout = pd.concat((dfy_mod1[:n1//10],dfy_mod2[:n1//10]),axis=0)
        dfy_nzmod1 = dfy_mod1[n1//10:]
        dfy_nzmod2 = dfy_mod2[n1//10:]


        index_dropout = pd.concat((dfx_dropout,dfy_dropout),axis=0).index.tolist()
        index_nzmod1 = pd.concat((dfx_nzmod1,dfy_nzmod1),axis=0).index.tolist()
        index_nzmod2 = pd.concat((dfx_nzmod2,dfy_nzmod2),axis=0).index.tolist()

        np.random.seed(seed=seed*3)
        np.random.shuffle(index_dropout)
        np.random.seed(seed=seed*3+1)
        np.random.shuffle(index_nzmod1)
        np.random.seed(seed=seed*3+2)
        np.random.shuffle(index_nzmod2)

        xindex = index_dropout[:n1//5]+index_nzmod1[:2*n1//5]+index_nzmod2[:2*n1//5]
        yindex = index_dropout[n1//5:]+index_nzmod1[2*n1//5:]+index_nzmod2[2*n1//5:]

    return(pd_select_df_from_index(df,xindex),
            pd_select_df_from_index(df,yindex))


def FDR(x,truth):
    return 0 if sum(x<0.05)==0 else sum(x[truth==0]<0.05)/sum(x<.05)
def typeI_error(x,truth):
    return(sum(x[truth==0]<0.05)/sum(truth==0))
def TDR(x,truth):
    return(0 if sum(x<0.05)==0 else sum(x[truth==1]<.05)/sum(x<.05))
def TDR_per_alternative(x,truth):
    x = x[truth==1]
    return(sum(x<.05)/len(x))
def stat_power(x,truth):
    return(sum(x[truth==1]<.05)/sum(truth==1))


def post_traitement_ccdf(self,t,dict_hyperparams={}):
    if t == 0: # pour  MMD
        dict_hyperparams['t']=0
        results = post_traitement_ccdf_of_column(x=self.df_pval['0'],
                                       xBH=self.df_pval_BH_corrected,
                                       dict_hyperparams=dict_hyperparams)
    else:
        results = post_traitement_ccdf_of_dataframes(
            dfpval=self.df_pval,
            dfBH=self.df_pval_BH_corrected,
            t=t,
            dict_hyperparams=dict_hyperparams)
    return(results)
   
def post_traitement_ccdf_of_dataframes(dfpval,dfBH,t,dict_hyperparams={}):
    """
    In dfpval and dfBH, rows correspond to the truncations and columns correspond to the genes
    """
    results = []

    dfpval.columns = dfpval.columns.astype(int)
    dfBH.columns = dfBH.columns.astype(int)
    
    for t_ in range(1,t+1):
        x = dfpval.T[t_].sort_index()
        xBH = dfBH.T[t_].sort_index()
        dict_hyperparams['t']=t_
        results += post_traitement_ccdf_of_column(x,xBH,dict_hyperparams=dict_hyperparams)

    return(results)
    
def post_traitement_ccdf_of_column(x,xBH,dict_hyperparams={}):
    """
    Computes the performances of an univariate two sample test applied to a dataset corresponding to ccdf 
    Attention il a peut-être un problème avec les calculs de DE,DM et DP 
    """
    results = []
    truth = np.array([1]*1000 + [0]*9000)
    truth_DE = np.array([1]*250 + [0]*9750)
    truth_DM = np.array([0]*250 + [1]*250 + [0]*9500)
    truth_DP = np.array([0]*500 + [1]*250 + [0]*9250)
    truth_DB = np.array([0]*750 + [1]*250 + [0]*9000)
    # les H0 sont en inversé pour pouvoir prendre 1- TDR per alternative 
    truth_EE = np.array([0]*1000 + [1]*4500 + [0]*4500)
    truth_EB = np.array([0]*1000 + [0]*4500 + [1]*4500)
    
    perfs = {'fdr':FDR(xBH,truth),'tpr':TDR(xBH,truth),'pwr':stat_power(x,truth),'ti_err':typeI_error(x,truth),
         'DE':TDR_per_alternative(xBH,truth_DE),'DM':TDR_per_alternative(xBH,truth_DM),
         'DP':TDR_per_alternative(xBH,truth_DP),'DB':TDR_per_alternative(xBH,truth_DB),
         'EE':1-TDR_per_alternative(x,truth_EE),'EB':1-TDR_per_alternative(x,truth_EB)
        } 
    return([{**perfs,**dict_hyperparams}])

def parallel_post_traitement_ccdf(dict_of_dfs_pval,dict_of_dfs_BH,stat,t,fixed_dict_params,n_jobs=6):
    params = list(dict_of_dfs_pval.keys())
    if stat == 'mmd':        
        with parallel_backend('loky'):
            a = Parallel(n_jobs=n_jobs)(delayed(post_traitement_ccdf_of_column)(
                dict_of_dfs_pval[param]['0'],
                dict_of_dfs_BH[param]['0'],
                dict_hyperparams={**fixed_dict_params,**{'s':param,'t':0}}) for param  in  params)
    else: 
        with parallel_backend('loky'):
            a = Parallel(n_jobs=n_jobs)(delayed(post_traitement_ccdf_of_dataframes)(
                dict_of_dfs_pval[param],
                dict_of_dfs_BH[param],
                t=t,
                dict_hyperparams={**fixed_dict_params,**{'s':param}}) for param  in  params)
    if stat == 'mmd':
        return([i[0] for i in a])
    else:
        return([i for i in a[0]])

def post_traitement_ccdf_of_dataframes(dfpval,dfBH,t,dict_hyperparams={}):
    """
    In dfpval and dfBH, rows correspond to the truncations and columns correspond to the genes
    """
    results = []

    dfpval.columns = dfpval.columns.astype(int)
    dfBH.columns = dfBH.columns.astype(int)
    
    for t_ in range(1,t+1):
        x = dfpval.T[t_].sort_index()
        xBH = dfBH.T[t_].sort_index()
        dict_hyperparams['t']=t_
        results += post_traitement_ccdf_of_column(x,xBH,dict_hyperparams=dict_hyperparams)

    return(results)
    
def post_traitement_ccdf_of_column(x,xBH,dict_hyperparams={}):
    """
    Computes the performances of an univariate two sample test applied to a dataset corresponding to ccdf 
    Attention il a peut-être un problème avec les calculs de DE,DM et DP 
    """
    results = []
    truth = np.array([1]*1000 + [0]*9000)
    truth_DE = np.array([1]*250 + [0]*9750)
    truth_DM = np.array([0]*250 + [1]*250 + [0]*9500)
    truth_DP = np.array([0]*500 + [1]*250 + [0]*9250)
    truth_DB = np.array([0]*750 + [1]*250 + [0]*9000)
    # les H0 sont en inversé pour pouvoir prendre 1- TDR per alternative 
    truth_EE = np.array([0]*1000 + [1]*4500 + [0]*4500)
    truth_EB = np.array([0]*1000 + [0]*4500 + [1]*4500)
    
    perfs = {'fdr':FDR(xBH,truth),'tpr':TDR(xBH,truth),
            'pwr':stat_power(x,truth),'ti_err':typeI_error(x,truth),
         'DE':TDR_per_alternative(x,truth_DE),'DM':TDR_per_alternative(x,truth_DM),
         'DP':TDR_per_alternative(x,truth_DP),'DB':TDR_per_alternative(x,truth_DB),
         'EE':TDR_per_alternative(x,truth_EE),'EB':TDR_per_alternative(x,truth_EB)
        } 
    return([{**perfs,**dict_hyperparams}])

def parallel_post_traitement_ccdf(dict_of_dfs_pval,dict_of_dfs_BH,stat,t,fixed_dict_params,n_jobs=6):
    params = list(dict_of_dfs_pval.keys())
    if stat == 'mmd':        
        with parallel_backend('loky'):
            a = Parallel(n_jobs=n_jobs)(delayed(post_traitement_ccdf_of_column)(
                dict_of_dfs_pval[param]['0'],
                dict_of_dfs_BH[param]['0'],
                dict_hyperparams={**fixed_dict_params,**{'s':param,'t':0}}) for param  in  params)
    else: 
        with parallel_backend('loky'):
            a = Parallel(n_jobs=n_jobs)(delayed(post_traitement_ccdf_of_dataframes)(
                dict_of_dfs_pval[param],
                dict_of_dfs_BH[param],
                t=t,
                dict_hyperparams={**fixed_dict_params,**{'s':param}}) for param  in  params)
    if stat == 'mmd':
        return([i[0] for i in a])
    else:
        return([i for i in a[0]])

def load_and_post_treat_ccdf(stat,n,kernel,seed,dirpval,dirBH,t=19,nystrom={}):
    loaded=False
    if len(nystrom)>0:
        ny=nystrom['ny']
        r=nystrom['r']
        m=nystrom['m']
        nystrom_str = f'_nystrom{ny}_r{r}_m{m}'
        fpval = f'{stat}_B1000_n{n}_g1_10001_k{kernel}{nystrom_str}_s{seed}.csv'
        fBH = f'{stat}B1000_n{n}_g1_10001_k{kernel}{nystrom_str}_s{seed}_BH.csv'
        if fpval in os.listdir(dirpval) and fBH in os.listdir(dirBH):
            dfpval = pd.read_csv(f'{dirpval}{fpval}',index_col=0)
            dfBH = pd.read_csv(f'{dirBH}{fBH}',index_col=0)
            loaded = True
    else:
        fpval = f'{stat}_B1000_n{n}_g1_10001_k{kernel}_s{seed}.csv'
        fBH = f'{stat}_B1000_n{n}_g1_10001_k{kernel}_s{seed}_BH.csv'
        if fpval in os.listdir(dirpval) and fBH in os.listdir(dirBH):
            dfpval = pd.read_csv(f'{dirpval}{fpval}',index_col=0)
            dfBH = pd.read_csv(f'{dirBH}{fBH}',index_col=0)
            loaded = True
        
    if loaded:
        dict_params = {'stat':stat,'n':n,'kernel':kernel,'s':seed}
        if nystrom:            
            dict_params = {**dict_params,**nystrom}
        if stat == 'mmd':
            dict_params['t']=0
            return(post_traitement_ccdf_of_column(dfpval['0'],dfBH['0'],dict_params))
        else:
            return(post_traitement_ccdf_of_dataframes(dfpval,dfBH,t,dict_params))
    else:
        return([])
         
def plot_performances_ccdf(df_res,show_perf_from_ccdf=False):
    """
    This function reproduces the figure containing type I error, FDR,TDR and power of ccdf and others.
    show_perf_from_ccdf == True : show the curves from ccdf 
    """
    fix,axes = plt.subplots(ncols=4,figsize=(25,12))
    if show_perf_from_ccdf:
        path = '/home/anthony/These/Implementations/data/2021_12_Simu_Univariate/'
        df_resM = pd.read_csv(path+'results.csv',sep=' ')
        

        for method in df_resM['method'].unique():
            df_resM.loc[df_resM['method']==method].plot(x='n',y='ti_err',ax=axes[0],label=method)
            df_resM.loc[df_resM['method']==method].plot(x='n',y='fdr',ax=axes[1],label=method)
            df_resM.loc[df_resM['method']==method].plot(x='n',y='pwr',ax=axes[2],label=method)
            df_resM.loc[df_resM['method']==method].plot(x='n',y='tpr',ax=axes[3],label=method)

    # df_res = self.df_results_ccdf
    df_res.T['ti_err'].plot(ax=axes[0],label='kfda')
    df_res.T['fdr'].plot(ax=axes[1],label='kfda')
    df_res.T['pwr'].plot(ax=axes[2],label='kfda')
    df_res.T['tpr'].plot(ax=axes[3],label='kfda')
    axes[0].legend()
    for ax in axes:
        ax.set_ylim(-.05,1.05)

def plot_performances_ccdf(df_res,show_perf_from_ccdf=False):
    
    fix,axes = plt.subplots(ncols=4,figsize=(25,12))
    if show_perf_from_ccdf:
        path = '/home/anthony/These/Implementations/data/2021_12_Simu_Univariate/'
        df_resM = pd.read_csv(path+'results.csv',sep=' ')
        

        for method in df_resM['method'].unique():
            df_resM.loc[df_resM['method']==method].plot(x='n',y='ti_err',ax=axes[0],label=method)
            df_resM.loc[df_resM['method']==method].plot(x='n',y='fdr',ax=axes[1],label=method)
            df_resM.loc[df_resM['method']==method].plot(x='n',y='pwr',ax=axes[2],label=method)
            df_resM.loc[df_resM['method']==method].plot(x='n',y='tpr',ax=axes[3],label=method)

    df_res.plot(ax=axes[0],x='n',y='ti_err',label='kfda')
    df_res.plot(ax=axes[1],x='n',y='fdr',label='kfda')
    df_res.plot(ax=axes[2],x='n',y='pwr',label='kfda')
    df_res.plot(ax=axes[3],x='n',y='tpr',label='kfda')
    axes[0].legend()
    for ax in axes:
        ax.set_ylim(-.05,1.05)


