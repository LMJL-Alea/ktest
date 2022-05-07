from scipy.stats import chi2
import pandas as pd
from ktest.tester import Tester
from ktest.statistics import correct_BenjaminiHochberg_pval_of_dfcolumn,correct_BenjaminiHochberg_pval_of_dataframe
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from joblib import parallel_backend
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler


def compute_standard_kfda(x,y,name,pval=True,kernel='gauss_median',params_model={}):

    test = Tester()
    test.init_data(x,y,kernel=kernel)
    test.init_model(**params_model)
    test.kfdat(name=name,pval=pval)
    if pval:
        return(test.df_kfdat,test.df_pval)
    else:
        return(test.df_kfdat)
    

def compute_standard_mmd(x,y,name,kernel='gauss_median',params_model={}):
    if isinstance(x,pd.DataFrame) or isinstance(x,pd.Series):
        if len(x.shape)==1:
            x = x.to_numpy().reshape(-1,1)
            y = y.to_numpy().reshape(-1,1)
        else:
            x = x.to_numpy()
            y = y.to_numpy()
    test = Tester()
    test.init_data(x,y,kernel=kernel)
    test.init_model(**params_model)
    test.mmd(name=name)
    return(test.dict_mmd)

def pd_select_df_from_index(df,index):
    """select a dataframe given an index"""
    return(df.loc[df.index.isin(index)])

# Pour chaque simu, un param bouge tandis que les autres sont fixés. 
# Pour un format uniforme, les param fixés sont rangés dans un dict 

def permute_and_compute_kfda(x,y,seed,gene,kernel='gauss_median',params_model={}):
    xb,yb = pd_permutation_ccdf(x,y,seed,gene)
    return(compute_standard_kfda(xb,yb,name=seed,pval=False,kernel=kernel,params_model=params_model))
    
def permute_and_compute_mmd(x,y,seed,gene,kernel='gauss_median',params_model={}):
    xb,yb = pd_permutation_ccdf(x,y,seed,gene)
    return(compute_standard_mmd(xb,yb,name=seed,kernel=kernel,params_model=params_model))
    
def non_parallel_permutation_from_dataframe(x,y,c,seeds,stat='kfda',kernel='gauss_median',params_model={}):
    xc,yc = x[c],y[c]
    if stat == 'kfda':
        
        test_orig = Tester()
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

def concat_kfda_and_pval_from_parallel_simu(output_parallel_simu):
    lkfda,lpval = [],[]
    for l in output_parallel_simu: 
        lkfda += [l[0]]
        lpval += [l[1]]
    
    return(pd.concat(lkfda,axis=1),pd.concat(lpval,axis=1))

def parallel_kfda_univariate(x,y,kernel='gauss_median',params_model={},n_jobs=3):
    """
    A function to analyse each gene in parallel using the univariate approach
    n_job=1 : parallel code is not used, it is usefull for debugging
    """

    outputs_test = Tester()
    outputs_test.df_power = pd.DataFrame()
    outputs_test.infos = {}
    # non parallel 
    if n_jobs==1:
        a=[]
        for c in x.columns:
            xc, yc = x[c].to_numpy().reshape(-1,1),y[c].to_numpy().reshape(-1,1)
            a+=[compute_standard_kfda(xc,yc,c,kernel=kernel,params_model=params_model)]
    elif n_jobs >1:
        # with parallel_backend('loky'):
        with parallel_backend('loky'):
            a = Parallel(n_jobs=n_jobs)(delayed(compute_standard_kfda)(
                x[variable].to_numpy().reshape(-1,1),
                y[variable].to_numpy().reshape(-1,1),
                variable,
                kernel = kernel,
                params_model=params_model,
                ) for variable  in  x.columns)
    kfda0,pval0 = concat_kfda_and_pval_from_parallel_simu(a)
    outputs_test.df_kfdat = kfda0
    outputs_test.df_pval = pval0
    return(outputs_test)

    
def parallel_BH_correction(dict_of_df_pval,stat,t=20,n_jobs=6):
    iter_param = list(dict_of_df_pval.keys())
    if stat == 'mmd':        
        with parallel_backend('loky'):
            a = Parallel(n_jobs=n_jobs)(delayed(correct_BenjaminiHochberg_pval_of_dfcolumn)(dict_of_df_pval[param]['0']) for param  in  iter_param)
    else: 
        with parallel_backend('loky'):
            a = Parallel(n_jobs=n_jobs)(delayed(correct_BenjaminiHochberg_pval_of_dataframe)(dict_of_df_pval[param],t=t) for param  in  iter_param)
    return({k:df for k,df in zip(iter_param,a)})

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
         'DE':TDR_per_alternative(x,truth_DE),'DM':TDR_per_alternative(x,truth_DM),
         'DP':TDR_per_alternative(x,truth_DP),'DB':TDR_per_alternative(x,truth_DB),
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
    
    perfs = {'fdr':FDR(xBH,truth),'tpr':TDR(xBH,truth),'pwr':stat_power(x,truth),'ti_err':typeI_error(x,truth),
         'DE':TDR_per_alternative(x,truth_DE),'DM':TDR_per_alternative(x,truth_DM),
         'DP':TDR_per_alternative(x,truth_DP),'DB':TDR_per_alternative(x,truth_DB),
         'EE':TDR_per_alternative(x,truth_EE),#'EB':TDR_per_alternative(x,truth_EB)
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
        
def plot_xrowcol(res,xs,rows,cols,cats,d):
    for row in rows:
        fig,axes = plt.subplots(ncols=len(cols),figsize=(len(cols)*6,6))
        for col,ax in zip(cols,axes):
            for cat in cats:
                ls='--' if cat[0] == 'E' else '-'
                l=[]
                for x in xs:
                    l+=[res.loc[res[d['x']]==x].loc[res[d['col']]==col].loc[res[d['row']]==row][cat].iat[0]]
    #                 print(l)
                ax.plot(xs,l,label=cat,ls=ls)   
            ax.set_ylim(-.05,1.05)
            ax.set_xlabel(d['x'],fontsize=20)
            ax.set_ylabel('power',fontsize=20)
            title=d['row']+f'={row:.2f}'+d['col']+f'={col:.2f}'
            ax.set_title(title,fontsize=30)
            ax.axhline(.05,ls='--',c='crimson')
        axes[0].legend()
        
def plot_xrepcol(res,xs,reps,cols,cats,d): # pas de row mais rep
    for cat in cats:
        fig,axes = plt.subplots(ncols=len(cols),figsize=(len(cols)*6,6))
        
        iterator = zip(cols,[0]) if len(cols) == 1 else zip(cols,axes)
        for col,ax in iterator:
            if ax == 0:
                ax = axes
            for i,rep in enumerate(reps):
#                 c = colorFader(c1,c2,i/len(reps))
                l=[]
                for x in xs:
                    l+=[res.loc[res[d['x']]==x].loc[res[d['col']]==col].loc[res[d['rep']]==rep][cat].iat[0]]
                ax.plot(xs,l,label=d['rep']+f'={rep:.3f}')#,c=c)
                ax.set_ylim(-.05,1.05)
                ax.set_xlabel(d['x'],fontsize=20)
                ax.set_ylabel('power',fontsize=20)
                title=f'{cat} '+d['col']+f'={col:.2f}'
                ax.set_title(title,fontsize=30)
                if cat == 'ti_err':
                    ax.axhline(.05,ls='--',c='crimson')            
                else:
                    ax.axhline(1,ls='--',c='crimson')
        if len(cols) == 1:
            axes.legend()
        else:
            axes[0].legend()


##### Functions specific to my notebooks for CRCL data and ccdf illustrations 
from time import time
from scipy.cluster.hierarchy import dendrogram, linkage

def reduce_category(c,ct):
    return(c.replace(ct,"").replace(' ','').replace('_',''))


def get_color_from_color_col(color_col):
    if color_col is not None:
        color = {}
        for cat in color_col.unique():
            color[cat] = color_col.loc[color_col == cat].index
    else :
        color = None
    return(color)

def get_cat_from_names(names,dict_data):        
    cat = []
    for name in names: 
        cat1 = name.split(sep='_')[0]
        cat2 = name.split(sep='_')[1]

        df1 = pd.concat([dict_data[c] for c in [c for c in cat1.split(sep=',')]],axis=0) if ',' in cat1 else dict_data[cat1]
        df2 = pd.concat([dict_data[c] for c in [c for c in cat2.split(sep=',')]],axis=0) if ',' in cat2 else dict_data[cat2]

        if len(df1)>10:
            cat += [cat1]
        if len(df2)>10:
            cat += [cat2]
    return(list(set(cat)))



def get_dist_matrix_from_dict_test_and_names(names,dict_tests,dict_data):
    cat = get_cat_from_names(names,dict_data)
    dist = {c:{} for c in cat}
    for name in names:
        cat1 = name.split(sep='_')[0]
        cat2 = name.split(sep='_')[1]
        if name in dict_tests:
            test = dict_tests[name]
            t = test.get_trunc()
            stat = test.df_kfdat[name][t]
            dist[cat1][cat2] = stat
            dist[cat2][cat1] = stat
    return(pd.DataFrame(dist).fillna(0).to_numpy())

def add_tester_to_dict_tests_from_name_and_dict_data(name,dict_data,dict_tests,dict_meta=None,params_model={},center_by=None,free_memory=True):
    if name not in dict_tests:
        cat1 = name.split(sep='_')[0]
        cat2 = name.split(sep='_')[1]

        df1 = pd.concat([dict_data[c] for c in [c for c in cat1.split(sep=',')]],axis=0) if ',' in cat1 else dict_data[cat1]
        df2 = pd.concat([dict_data[c] for c in [c for c in cat2.split(sep=',')]],axis=0) if ',' in cat2 else dict_data[cat2]



        if dict_meta is not None:
            df1_meta = pd.concat([dict_meta[c] for c in [c for c in cat1.split(sep=',')]],axis=0) if ',' in cat1 else dict_meta[cat1]
            df2_meta = pd.concat([dict_meta[c] for c in [c for c in cat2.split(sep=',')]],axis=0) if ',' in cat2 else dict_meta[cat2]
        else:
            df1_meta = pd.DataFrame()    
            df2_meta = pd.DataFrame()    
           
        colcat1 = []
        colcat2 = []
        colsexe1 = []
        colsexe2 = []
        for cat,colcat,colsexe in zip([cat1,cat2],[colcat1,colcat2],[colsexe1,colsexe2]):
            for c in cat.split(sep=','):
                colcat += [c]*len(dict_data[c])
                colsexe += ['M' if c[0]=='M' else 'W']*len(dict_data[c])
                
                
        df1_meta['cat'] = colcat1
        df2_meta['cat'] = colcat2
        df1_meta['sexe'] = colsexe1
        df2_meta['sexe'] = colsexe2
        df1_meta['cat'] = df1_meta['cat'].astype('category')
        df2_meta['cat'] = df2_meta['cat'].astype('category')
        df1_meta['sexe'] = df1_meta['sexe'].astype('category')
        df2_meta['sexe'] = df2_meta['sexe'].astype('category')
        
        
        if len(df1)>10 and len(df2)>10:
            t0=time()
            print(name,len(df1),len(df2),end=' ')
            test = Tester()
#             center_by = 'cat' if center_by_cat else None
            test.init_data_from_dataframe(df1,df2,dfx_meta = df1_meta,dfy_meta=df2_meta,center_by=center_by)
            test.obs['cat']=test.obs['cat'].astype('category')
            test.obs['sexe']=test.obs['sexe'].astype('category')
            test.init_model(**params_model)
            test.kfdat(name=name)
            t = test.t
            
                        
            test.compute_proj_kfda(t=t+1)
            
            kfda = test.df_kfdat[name][t]
            test.compute_pval(t=t)
            pval = test.df_pval[name][t]
            if free_memory:
                test.x = np.array(0)
                test.y = np.array(0) 
                del(test.spev['xy']['standard']['ev'])
#                 test.spev = {'x':{},'y':{},'xy':{},'residuals':{}}
            
            dict_tests[name] = test
            print(f'{time()-t0:.3f} t={t} kfda={kfda:.4f} pval={pval}')
            
 
def plot_discriminant_and_kpca_of_chosen_truncation_from_name(name,dict_tests,color_col=None):

    cat1 = name.split(sep='_')[0]
    cat2 = name.split(sep='_')[1]
    infos = name.split(sep='_')[2] if len(name.split(sep='_'))>2 else ""
    fig,axes = plt.subplots(ncols=2,figsize=(12*2,6))
    test = dict_tests[name]
    t = test.t
    pval = test.df_pval.loc[t].values[0]
    test.density_projs(fig=fig,axes=axes[0],projections=[t],labels=[cat1,cat2],)
    color = get_color_from_color_col(color_col)
    test.scatter_projs(fig=fig,axes=axes[1],projections=[[t,1]],labels=[cat1,cat2],color=color)
    fig.suptitle(f'{cat1} vs {cat2} {infos}: pval={pval:.5f}',fontsize=30,y=1.04)
    fig.tight_layout()
    return(fig,axes)


def plot_density_of_chosen_truncation_from_names(name,dict_tests,fig=None,ax=None,t=None,labels=None):
    if fig is None:
        fig,ax = plt.subplots(ncols=1,figsize=(12,6))
    cat1 = name.split(sep='_')[0]
    cat2 = name.split(sep='_')[1]
    infos = name.split(sep='_')[2] if len(name.split(sep='_'))>2 else ""
    test = dict_tests[name]
    trunc = test.t if t is None else t
    pval = test.df_pval.loc[trunc].values[0]
    test.density_projs(fig=fig,axes=ax,projections=[trunc],labels=[cat1,cat2] if labels is None else labels,)
    fig.suptitle(f'{cat1} vs {cat2} {infos}: pval={pval:.5f}',fontsize=30,y=1.04)
    fig.tight_layout()
    return(fig,ax)


def plot_scatter_of_chosen_truncation_from_names(name,dict_tests,color_col=None,fig=None,ax=None,t=None):
    if fig is None:
        fig,ax = plt.subplots(ncols=1,figsize=(12,6))
    cat1 = name.split(sep='_')[0]
    cat2 = name.split(sep='_')[1]
    infos = name.split(sep='_')[2] if len(name.split(sep='_'))>2 else ""
    test = dict_tests[name]
    trunc = test.t if t is None else t
    pval = test.df_pval.loc[trunc].values[0]
    color = get_color_from_color_col(color_col)
    test.scatter_projs(fig=fig,axes=ax,projections=[[trunc,1]],labels=[cat1,cat2],color=color)
    fig.suptitle(f'{cat1} vs {cat2} {infos}: pval={pval:.5f}',fontsize=30,y=1.04)
    fig.tight_layout()
    return(fig,ax)

def plot_density_of_univariate_data_from_name_and_dict_data(name,dict_data,fig=None,ax=None):
    if fig is None:
        fig,ax = plt.subplots(ncols=1,figsize=(12,6))
    ax.set_title('observed data',fontsize=20)
    infos = name.split(sep='_')[2] if len(name.split(sep='_'))>2 else ""
    for iname,xy,color in zip([0,1],'xy',['blue','orange']):
        cat = name.split(sep='_')[iname]
        x = dict_data[cat]
        bins=int(np.floor(np.sqrt(len(x))))
        ax.hist(x,density=True,histtype='bar',label=f'{xy}({len(x)})',alpha=.3,bins=bins,color=color)
        ax.hist(x,density=True,histtype='step',bins=bins,lw=3,edgecolor='black')

    cat1 = name.split(sep='_')[0]
    cat2 = name.split(sep='_')[1]
#     fig.suptitle(f'{cat1} vs {cat2} {infos}: pval={pval:.5f}',fontsize=30,y=1.04)
    fig.tight_layout()
    return(fig,ax)

def plot_dendrogram_from_names(names,dict_tests,dict_data,rotlab=0,ct=''):
    cat = get_cat_from_names(names,dict_data)
    cat_plot = []
    for c in cat:
        cat_plot += [reduce_category(c,ct)]

    dists = get_dist_matrix_from_dict_test_and_names(names,dict_tests,dict_data)


    fig,ax = plt.subplots(figsize=(7,7))
    dendrogram(linkage(dists,'single'),
               ax=ax, 
               orientation='top',
                labels=cat_plot,
                distance_sort='descending',
        
                show_leaf_counts=True)
    ax.set_title(ct,fontsize=20)
    ax.tick_params(axis='x', labelrotation=rotlab,labelsize=20 )
    return(fig,ax)

def reduce_labels_and_add_ncells(cats,ct,dict_data):
    cats_labels=[]
    for cat in cats: 
        if ',' in cat:
            ncells = np.sum([len(dict_data[c]) for c in cat.split(sep=',') ])
            rcats = []
            for c in cat.split(sep=','):
                rcats+=[reduce_category(c,ct)]
            rcat = ','.join(rcats)
        else:
            ncells = len(dict_data[cat])
            rcat =  reduce_category(cat,ct)
        cats_labels+= [f'{rcat}({ncells})']
    return cats_labels


####################### Residual of Discrimination





################# Simple visualization 

def plot_density(x,y,fig=None,ax=None):
    if fig== None:
        fig,ax = plt.subplots(figsize=(10,6))    
    for x,xy,color in zip([x,y],'xy',['xkcd:cerulean','xkcd:light orange']):
        bins=int(np.floor(np.sqrt(len(x))))
        ax.hist(x,density=True,histtype='bar',label=f'{xy}({len(x)})',alpha=.3,bins=bins,color=color)
        ax.hist(x,density=True,histtype='step',bins=bins,lw=3,edgecolor='black')
    return(fig,ax)

def plot_scatter(xh,xv,yh,yv,fig=None,ax=None):
    if fig== None:
        fig,ax = plt.subplots(figsize=(10,6))    
    for hv,xy,color,m in zip([[xh,xv],[yh,yv]],'xy',['xkcd:cerulean','xkcd:light orange'],['x','+']):
        h,v = hv
        ax.scatter(h,v,c=color,s=30,alpha=.8,marker =m,label=f'{xy}({len(h)})')
    for hv,xy,color in zip([[xh,xv],[yh,yv]],'xy',['xkcd:cerulean','xkcd:light orange']):         
        h,v = hv
        mh = h.mean()
        mv = v.mean()
        ax.scatter(mh,mv,edgecolor='black',linewidths=3,s=200,color=color)
    return(fig,ax)



#### custom dendrogram 

from scipy.stats import chi2
from torch import zeros
from scipy.cluster.hierarchy import dendrogram


def get_kfda_from_name_and_dict_tests(name,dict_tests):
    if name in dict_tests:
        test = dict_tests[name]
        t = test.t
        kfda = test.df_kfdat[name][t]
    else:
        kfda = np.inf
    return(kfda)

def get_cat_argmin_from_similarities(similarities,dict_tests):
    s = similarities
    s = s[s>0]
    c1 = s.min().sort_values().index[0]
    c2 = s[c1].sort_values().index[0]
    kfda = s.min().min()
    return(c1,c2,kfda)


def update_cats_from_similarities(cats,similarities,dict_tests):
    c1,c2,_ = get_cat_argmin_from_similarities(similarities,dict_tests)
    cats = [c for c in cats if c not in [c1,c2]]
    cats += [f'{c1},{c2}' if c1<c2 else f'{c2},{c1}']
    return(cats)

def kfda_similarity_of_datasets_from_cats(cats,dict_data,dict_tests,kernel='gauss_median',params_model={},):
    n = len(cats)
    similarities = {}
    names = []
    for c1 in cats:
        if c1 not in similarities:
            similarities[c1] = {c1:0}
        for c2 in cats:
            if c2 not in similarities:
                similarities[c2] = {c2:0}
                
            if c1<c2:
                name = f'{c1}_{c2}'
                add_tester_to_dict_tests_from_name_and_dict_data(name,dict_data,dict_tests,params_model=params_model)
                kfda = get_kfda_from_name_and_dict_tests(name,dict_tests)
                
                similarities[c1][c2] = kfda
                similarities[c2][c1] = kfda
    return(pd.DataFrame(similarities,index=cats,columns=cats))


def custom_linkage2(cats,dict_data,dict_tests,kernel='gauss_median',params_model={}):
    cats = cats.copy()
    n = len(cats)
    Z = np.empty((n - 1, 4))
    
    similarities = kfda_similarity_of_datasets_from_cats(cats,dict_data,dict_tests,kernel=kernel,params_model=params_model,)
#     D = squareform(similarities.to_numpy())

    id_map = list(range(n))
    x=0
    y=0
    
    indexes = list(range(n-1))
    cats_ = cats.copy()
    ordered_comparisons = []
    for k in range(n - 1):
#         print(f'\n#####k = {k}')
        
        # my find two closest clusters x, y (x < y)
        c1,c2,kfda = get_cat_argmin_from_similarities(similarities,dict_tests)
        x,y = np.min([cats.index(c1),cats.index(c2)]),np.max([cats.index(c1),cats.index(c2)])
#         print(x,y)
        id_x = id_map.index(x)
        id_y = id_map.index(y)
        catx = cats[x]
        caty = cats[y]
        catxy = f'{c1},{c2}' if c1<c2 else f'{c2},{c1}'
        cats += [catxy]
        nx = 1 if id_x < n else Z[id_x - n, 3]
        ny = 1 if id_y < n else Z[id_y - n, 3]

#         print(f'x={x} id_x={id_x} catx={c1} \n y={y} id_y={id_y} caty={c2} \n kfda={kfda}')
        cats_ = update_cats_from_similarities(cats_,similarities,dict_tests)
        ordered_comparisons += [f'{c1}_{c2}' if c1<c2 else f'{c2}_{c1}']
        similarities = kfda_similarity_of_datasets_from_cats(cats_,dict_data,dict_tests,kernel=kernel,params_model=params_model)

        #         # my record the new node
        Z[k, 0] = min(x, y)
        Z[k, 1] = max(y, x)
        Z[k, 2] = kfda
        Z[k, 3] = nx + ny # nombre de clusters ou nombre de cells ?  
        id_map[id_x] = -1  # cluster x will be dropped
        id_map[id_y] = n + k  # cluster y will be replaced with the new cluster
        
#         print(f'id_map={id_map} \n ')
    return(Z,ordered_comparisons)



def plot_custom_dendrogram_from_cats(cats,dict_data,dict_tests,y_max=None,fig=None,ax=None,cats_labels=None,params_model={}):
    if fig is None:
        fig,ax = plt.subplots(figsize= (10,6))
    if cats_labels==None:
        cats_labels = cats
    
    linkage,comparisons = custom_linkage2(cats,dict_data,dict_tests,params_model = params_model)
    kfda_max = np.max(linkage[:,2])
#     cats_with_ncells = []
#     for cat in cats: 
#         if ',' in cat:
#             ncells = np.sum([len(dict_data[c]) for c in cat.split(sep=',') ])
#         else:
#             ncells = len(dict_data[cat])
#         cats_with_ncells+= [f'{cat}({ncells})']
    d = dendrogram(linkage,labels=cats_labels,ax=ax)

    abscisses = d['icoord']
    ordinates = d['dcoord']
    
        
    icomp = 0
    for x,y in zip(abscisses,ordinates):
        
        if np.abs(y[1]-kfda_max)<10e-6:
            dot_of_test_result_on_dendrogram(1/2*(x[1]+x[2]),y[1],comparisons[-1],dict_tests,ax)
        for i in [0,-1]:
            if y[i] > 10e-6:
                dot_of_test_result_on_dendrogram(x[i],y[i],comparisons[icomp],dict_tests,ax)
                icomp+=1
    print(comparisons[icomp])

    if y_max is not None:
        ax.set_ylim(0,y_max)
    ax.tick_params(axis='x', labelrotation=90,labelsize=20 )
#     return(d)

    return(fig,ax)


def dot_of_test_result_on_dendrogram(x,y,name,dict_tests,ax):
    test = dict_tests[name]
    t = test.t
    pval = test.df_pval[name][t]
    kfda = test.df_kfdat[name][t]
    c = 'green' if pval >.05 else 'red'
    yaccept = chi2.ppf(.95,t)
    ax.scatter(x,yaccept,s=500,c='red',alpha=.5,marker='_',)
    ax.scatter(x,y,s=300,c=c,edgecolor='black',alpha=1,linewidths=3,)


### Comparaison M vs W 

def get_name_MvsF(ct,data_type,cts,dict_data):
    mwi = [f'{mw}{i}' for i in '123' for mw in 'MW']
    name = ''
    virgule = 0 
    for m in mwi:
        if 'M' in m:
            for celltype in cts[ct]:
                cat = f'{m}{celltype}{data_type}'
                if cat in dict_data:
                    name += f'{cat},'
                    
    name = name[:-1]
    name += '_'
    virgule = 0
    for m in mwi:
        if 'W' in m:
            for celltype in cts[ct]:
                cat = f'{m}{celltype}{data_type}'
                if cat in dict_data:
                    name += f'{cat},'
    return(name[:-1])


