
import matplotlib.pyplot as plt
import pandas as pd
from .tester import Ktest
from .base import init_kernel_params,init_test_params
import numpy as np


def get_meta_from_df(df):
    meta = pd.DataFrame()
    meta['index'] = df.index
    meta.index = df.index
    meta['batch'] = meta['index'].apply(lambda x : x.split(sep='.')[0])
    meta['condition'] = meta['index'].apply(lambda x : x.split(sep='.')[1])
    return(meta)

def get_Ktest_from_df_and_comparaison(df,comparaison,name,kernel=init_kernel_params(),nystrom=False,center_by=None,condition='condition',meta=None):
    if meta is None:
        meta = get_meta_from_df(df)
    cstr = "_".join(comparaison)
    cells = meta[meta[condition].isin(comparaison)].index
    data = df[df.index.isin(cells)] 
    metadata = meta[meta.index.isin(cells)] 
    null_genes = list(data.sum()[data.sum()==0].index)
    if len(null_genes)>0:
        print('there are null genes')
        print(null_genes)
        data = data.drop(columns=null_genes)
    
    t = Ktest(
                data,
                metadata.copy(),
                data_name=f'{name}_{cstr}',
                condition=condition,
                nystrom=nystrom,
                center_by=center_by,
                kernel=kernel,
                        # df:data,meta:metadata,data_name,condition:test_condition,df_var:var_metadata,
                        # test:test_params,viz:removed
                )
    t.multivariate_test()
    t.projections()
    return(t)
    


def get_name_in_dict_data(k):
    name = 'corrected_counts' if 'corrected_counts' in k else 'residuals'
    name += '_batch_corrected' if '_batch_corrected' in k else ''
    return(name)

def figures_outliers_of_reversion_from_tq(ktest,t,q_,df,focus=None,color=None,marker=None,return_outliers=False,contrib=False):
    # suptitle = f'{cstr} {name} t{trunc} q{q_}'
    
    str_focus = '' if focus is None else f'_{focus}'
    outliers_name= f't{t}_q{q_}{str_focus}'
    
    dfproj = ktest.init_df_proj(proj='proj_kfda',name=ktest.get_kfdat_name())[str(t)]#.sort_values(ascending=False)
    meta = get_meta_from_df(df)
    if focus is not None:
        dfproj = dfproj[dfproj.index.isin(meta[meta['condition']==focus].index)]
    q = dfproj.quantile(q_)
    if q_<.5:
        outliers_list = dfproj[dfproj<q].index
    else:
        outliers_list = dfproj[dfproj>q].index
    
    fig,axes = figures_outliers_of_reversion(
        ktest=ktest,
        t=t,
        df=df,
        outliers_list=outliers_list,
        outliers_name=outliers_name,
        color=color,
        marker=marker,
        contrib=contrib)
    
    ax = axes[1]
    ax.axvline(q,color='crimson',ls='--')
    ax.set_title(f't{t} q{q_}',fontsize=30)
    
    ax = axes[2]
    ax.axvline(q,color='crimson',ls='--')

    if return_outliers:
        return(fig,axes,outliers_list)
    else:
        return(fig,axes)


def figures_outliers_of_reversion_from_tq2(ktest,t,q_,df,focus=None,color=None,marker=None,return_outliers=False,contrib=False):
    # suptitle = f'{cstr} {name} t{trunc} q{q_}'
    
    str_focus = '' if focus is None else f'_{focus}'
    outliers_name= f't{t}_q{q_}{str_focus}'
    
    dfproj = ktest.init_df_proj(proj='proj_kfda',name=ktest.get_kfdat_name())[str(t)]#.sort_values(ascending=False)
    meta = get_meta_from_df(df)
    if focus is not None:
        dfproj = dfproj[dfproj.index.isin(meta[meta['condition']==focus].index)]
    q = dfproj.quantile(q_)
    if q_<.5:
        outliers_list = dfproj[dfproj<q].index
        remaining = dfproj[dfproj>q].index
    else:
        outliers_list = dfproj[dfproj>q].index
        remaining = dfproj[dfproj<q].index
    
    fig,axes = figures_outliers_of_reversion2(
        ktest=ktest,
        t=t,
        df=df,
        outliers_list=outliers_list,
        outliers_name=outliers_name,
        color=color,
        marker=marker,
        contrib=contrib)
    
    fig,axes = figures_outliers_of_reversion2(
        ktest=ktest,
        t=t,
        df=df,
        outliers_list=remaining,
        outliers_name=outliers_name,
        color=color,
        marker=marker,
        contrib=contrib)

    ax = axes[1]
    ax.axvline(q,color='crimson',ls='--')
    ax.set_title(f't{t} q{q_}',fontsize=30)
    
    ax = axes[2]
    ax.axvline(q,color='crimson',ls='--')

 
    fig,axes = figures_outliers_of_reversion2(
        ktest=ktest,
        t=t,
        df=df,
        outliers_list=remaining,
        outliers_name=outliers_name,
        color=color,
        marker=marker,
        contrib=contrib)



    if return_outliers:
        return(fig,axes,outliers_list)
    else:
        return(fig,axes)


def get_dict_ktests_outliers_vs_each_condition(df,outliers_list,outliers_name):
    dict_output = {}

    dfout = df[df.index.isin(outliers_list)]
    meta = get_meta_from_df(df)
    metaout = get_meta_from_df(dfout)
    metaout['population'] = [f'out']*len(metaout)
    
    for condition in ['0H','24H','48HDIFF','48HREV']:
        dfr = df[meta['condition']==condition]
        dfr = dfr[~dfr.index.isin(outliers_list)]
        metar = get_meta_from_df(dfr)
        metar['population'] = [condition]*len(metar)

        dfoutvscond = pd.concat([dfout,dfr])
        metaoutvscond = pd.concat([metaout,metar])
        toutvscondition = Ktest(data=dfoutvscond,
                                metadata=metaoutvscond.copy(),
                                data_name=f'{outliers_name}_{condition}',
                                condition='population',
                                nystrom=False,
                                center_by=None,)
        toutvscondition.multivariate_test()
        dict_output[condition] = toutvscondition
    return(dict_output)

def get_dict_ktests_condition_vs_each_other_condition(df,condition_of_interest='48HREV'):
    dict_output = {}

    meta = get_meta_from_df(df)
    dfc = df[meta['condition']==condition_of_interest]
    metac = get_meta_from_df(dfc)
    metac['population'] = [condition_of_interest]*len(metac)
    
    for condition in ['0H','24H','48HDIFF','48HREV']:
        if condition != condition_of_interest:
            dfr = df[meta['condition']==condition]
            metar = get_meta_from_df(dfr)
            metar['population'] = [condition]*len(metar)

            dfout = pd.concat([dfc,dfr])
            metaout = pd.concat([metac,metar])
            tout = Ktest(
                        data=dfout,
                        metadata=metaout.copy(),
                        data_name=f'{condition_of_interest}_{condition}',
                        condition='population',
                        nystrom=False,
                        center_by=None,)
            tout.multivariate_test()
            dict_output[condition] = tout
    return(dict_output)


def get_ktest_outliers_vs_all(df,outliers_list,outliers_name):

    dfout = df[df.index.isin(outliers_list)]
    metaout = get_meta_from_df(dfout)
    metaout['population'] = [f'out']*len(metaout)
    
    dfothers = df[~df.index.isin(outliers_list)]
    metaothers = get_meta_from_df(dfothers)
    metaothers['population'] = [f'others']*len(metaothers)
    
    dfoutvsothers = pd.concat([dfout,dfothers])
    metaoutvsothers = pd.concat([metaout,metaothers])
    toutvsothers = Ktest(
                    data=dfoutvsothers,
                    metadata=metaoutvsothers.copy(),
                    data_name=f'{outliers_name}_all_other_cells',
                    condition='population',
                    nystrom=False,
                    center_by=None,)
    toutvsothers.multivariate_test()
    return(toutvsothers)

def figures_outliers_of_reversion(ktest,t,df,outliers_list,outliers_name,color=None,marker=None,contrib=False):
    
    fig,axes = plt.subplots(ncols=4,figsize=(28,7)) 
                
    ax = axes[0]
    ktest.plot_pval_and_errors(fig=fig,ax=ax,truncations_of_interest=[1,3,5],t=20,pval_contrib=contrib)

    ax = axes[1]
    ktest.hist_discriminant(t=t,fig=fig,ax=ax,)


    ax = axes[2]
    ktest.plot_residuals(t=t,fig=fig,ax=ax,highlight=outliers_list,color=color,marker=marker)
    ax.set_title(outliers_name,fontsize=30)

    ax = axes[3]
    ktest.fit_Ktest_with_ignored_observations(list_of_observations_to_ignore = outliers_list,
                                                list_name=outliers_name)
    ktest.plot_pval_and_errors(fig=fig,ax=ax,truncations_of_interest=[1,3,5],t=20,marked_obs_to_ignore=outliers_name,pval_contrib=contrib)
    

    if color is not None:
        true_condition = ktest.condition
        ktest.condition = color
    effectifs_outliers = " ".join([f'{k}:{len(v[v.isin(outliers_list)])}' for k,v in ktest.get_index().items()])
    ax.set_title(f'without\n{effectifs_outliers}', fontsize=30)
    if color is not None:
        ktest.condition = true_condition

    if len(outliers_list)>0:

        dtest = get_dict_ktests_outliers_vs_each_condition(df,outliers_list,outliers_name)
        fig_,axes_ = plt.subplots(ncols=4,figsize=(28,7))     
        for condition,ax in zip(['0H','24H','48HDIFF','48HREV'],axes_):

            dtest[condition].plot_pval_and_errors(
                fig=fig_,ax=ax,truncations_of_interest=[1,3,5],t=20,pval_contrib=contrib,var_conditions=False,diff=False)

            ax.set_title(f'out vs {condition}',fontsize=30)
    else:
        axes[3].set_title('no outliers',fontsize=30)
    fig.tight_layout()
    return(fig,axes)


def figures_outliers_of_reversion2(ktest,t,df,outliers_list,outliers_name,color=None,marker=None,contrib=False):
    
    
    fig,axes = plt.subplots(ncols=3,figsize=(28,7)) 
                
    ax = axes[0]
    ktest.plot_pvalue(fig=fig,ax=ax,t=20,contrib=contrib,color_agg=None,log=True,label_agg='all',)

    ax = axes[1]
    ktest.hist_discriminant(t=t,fig=fig,ax=ax,)

    ax = axes[2]
    ktest.plot_residuals(t=t,fig=fig,ax=ax,highlight=outliers_list,color=color,marker=marker)
    ax.set_title(outliers_name,fontsize=30)

    ax = axes[0]
    ktest.fit_Ktest_with_ignored_observations(list_of_observations_to_ignore = outliers_list,
                                                list_name=outliers_name)
    ktest.set_marked_obs_to_ignore(marked_obs_to_ignore=outliers_name)
    ktest.plot_pvalue(fig=fig,ax=ax,t=20,contrib=contrib,color_agg=None,log=True,label_agg='without pop',)
    ktest.set_marked_obs_to_ignore()
    

    if color is not None:
        true_condition = ktest.condition
        ktest.condition = color
    effectifs_outliers = " ".join([f'{k}:{len(v[v.isin(outliers_list)])}' for k,v in ktest.get_index().items()])
    ax.set_title(f'without\n{effectifs_outliers}', fontsize=30)
    if color is not None:
        ktest.condition = true_condition

    if len(outliers_list)>0:

        dtest = get_dict_ktests_outliers_vs_each_condition(df,outliers_list,outliers_name)
        fig,ax = plt.subplots(figsize=(12,6))     
        for condition in ['0H','24H','48HDIFF','48HREV']:
            dtest[condition].plot_pvalue(
                fig=fig,ax=ax,t=20,contrib=contrib,color_agg=None,log=True,label_agg=f'out vs {condition}')

            # ax.set_title(f'out vs {condition}',fontsize=30)
    # else:
    #     axes[3].set_title('no outliers',fontsize=30)
    fig.tight_layout()
    return(fig,axes)


def plot_densities_of_conditions_for_a_variable_reversion(df,meta,condition,g,
        datasets=['0H','48HREV0','48HREV24','24H','48HDIFF'],
        colors= ['xkcd:blue','xkcd:sky blue','xkcd:sea green','xkcd:kelly green','xkcd:red orange'],
        label_datasets=False,means=True,hist=True,fig=None,axes=None):
    
    if fig is None:
        fig,axes = plt.subplots(5,1,figsize=(4,7))

    axes[0].set_title(g,fontsize=30)
    emin,emax = 0,0
    for k,dataset,c in zip(range(5),datasets,colors):
        ax = axes[k]
        cells = meta[meta[condition].isin([dataset])].index
        expr = df[df.index.isin(cells)][g] 
        emin = expr.min() if expr.min()<emin else emin
        emax = expr.max() if expr.max()>emax else emax
        if hist:
            bins = int(np.floor(3*np.sqrt(len(expr))))
            ax.hist(expr,density=True,histtype='step',lw=3,bins=bins,color=c)
            ax.hist(expr,density=True,histtype='bar',alpha=.5,bins=bins,color=c)
        if means:
            ax.axvline(expr.mean(),color=c,lw=2)
        if label_datasets:
            ax.set_ylabel(dataset,fontsize=15)
    for k in range(5):
        axes[k].set_xlim(emin,emax)

def plot_violins_of_conditions_for_a_variable_reversion(df,meta,condition,g,
        datasets=['0H','48HREV0','48HREV24','24H','48HDIFF'],
        colors = ['xkcd:blue','xkcd:sky blue','xkcd:sea green','xkcd:kelly green','xkcd:red orange'],
        fig=None,ax=None):    
    
    if fig is None:
        fig,ax = plt.subplots()
        
    exprs = []
    for d in datasets:
        cells = meta[meta[condition].isin([d])].index
        expr =df[df.index.isin(cells)][g]
        exprs += [expr] 
        
    violons = ax.violinplot(exprs,showmeans=True)
    ax.set_xticks(range(1,len(datasets)+1))
    ax.set_xticklabels(datasets,rotation=90)
#     ax.xticks(rotation=90)
    for v,c in zip(violons['bodies'],colors):
        v.set_facecolor(c)
#                                     ax.boxplot(exprs,labels=datasets)
    ax.set_title(g)
      
def plot_time_trajectory_for_a_variable_reversion(df,meta,condition,g,
        colors = ['xkcd:sky blue','xkcd:sea green','xkcd:red orange'],
        fig=None,ax=None):    
    
    if fig is None:
        fig,ax = plt.subplots()
    tdiff = []
    trev = []
    tstop = []



    for dataset,trajectory,color in zip(['48HREV0','48HREV24','48HDIFF'],[trev,tstop,tdiff],
                                        ['xkcd:sky blue','xkcd:sea green','xkcd:red orange']):

        for d in ['0H','24H']:
            cells = meta[meta[condition].isin([d])].index
            expr =df[df.index.isin(cells)][g]
            trajectory += [expr.mean()]
        cells = meta[meta[condition].isin([dataset])].index
        expr =df[df.index.isin(cells)][g]
        trajectory += [expr.mean()]
          
        ax.plot([1,2,3],trajectory,color=color,label=dataset,alpha=.8)
        ax.scatter([1,2,3],trajectory,color=color,s=10)
    ax.set_xticks([1,2,3])
    ax.set_xticklabels(['0H','24H','48H'])
    ax.legend(fontsize=7)
    ax.set_title(g)
    
def center_dataframe_by_effect(df,effect):
    df['effect'] = effect
    mean = df.groupby('effect').mean()
    
    first = True
    for e in df['effect'].unique():
        dfe = df[df['effect']==e]
        dfe = dfe.drop(columns=['effect'])
        dfec = dfe - mean.loc[e]
        if first:
            first=False
            dfc = dfec
        else:
            dfc = pd.concat([dfc,dfec])
    return(dfc)

