
import matplotlib.pyplot as plt
import pandas as pd
from .tester import create_and_fit_tester_for_two_sample_test_kfdat

def get_meta_from_df(df):
    meta = pd.DataFrame()
    meta['index'] = df.index
    meta.index = df.index
    meta['batch'] = meta['index'].apply(lambda x : x.split(sep='.')[0])
    meta['condition'] = meta['index'].apply(lambda x : x.split(sep='.')[1])
    return(meta)



def get_name_in_dict_data(k):
    name = 'corrected_counts' if 'corrected_counts' in k else 'residuals'
    name += '_batch_corrected' if '_batch_corrected' in k else ''
    return(name)

def figures_outliers_of_reversion_from_tq(tester,trunc,q_,df,focus=None,color=None,marker=None,return_outliers=False):
    # suptitle = f'{cstr} {name} t{trunc} q{q_}'
    
    str_focus = '' if focus is None else f'_{focus}'
    outliers_name= f't{trunc}_q{q_}{str_focus}'
    
    dfproj = tester.init_df_proj(proj='proj_kfda',name=tester.get_kfdat_name())[str(trunc)]#.sort_values(ascending=False)
    meta = get_meta_from_df(df)
    if focus is not None:
        dfproj = dfproj[dfproj.index.isin(meta[meta['condition']==focus].index)]
    q = dfproj.quantile(q_)
    if q_<.5:
        outliers_list = dfproj[dfproj<q].index
    else:
        outliers_list = dfproj[dfproj>q].index
    
    fig,axes = figures_outliers_of_reversion(
        tester=tester,
        trunc=trunc,
        df=df,
        outliers_list=outliers_list,
        outliers_name=outliers_name,
        color=color,
        marker=marker)
    
    ax = axes[1]
    ax.axvline(q,color='crimson',ls='--')
    ax.set_title(f't{trunc} q{q_}',fontsize=30)
    
    ax = axes[2]
    ax.axvline(q,color='crimson',ls='--')

    if return_outliers:
        return(fig,axes,outliers_list)
    else:
        return(fig,axes)


def get_dict_testers_outliers_vs_each_condition(df,outliers_list,outliers_name):
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
        toutvscondition = create_and_fit_tester_for_two_sample_test_kfdat(df=dfoutvscond,
                                                    meta=metaoutvscond.copy(),
                                                    data_name=f'{outliers_name}_{condition}',
                                                    condition='population',
                                                    nystrom=False,
                                                    center_by=None,)
        dict_output[condition] = toutvscondition
    return(dict_output)

def get_dict_testers_condition_vs_each_other_condition(df,condition_of_interest='48HREV'):
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
            tout = create_and_fit_tester_for_two_sample_test_kfdat(df=dfout,
                                                        meta=metaout.copy(),
                                                        data_name=f'{condition_of_interest}_{condition}',
                                                        condition='population',
                                                        nystrom=False,
                                                        center_by=None,)
            dict_output[condition] = tout
    return(dict_output)


def get_tester_outliers_vs_all(df,outliers_list,outliers_name):

    dfout = df[df.index.isin(outliers_list)]
    metaout = get_meta_from_df(dfout)
    metaout['population'] = [f'out']*len(metaout)
    
    dfothers = df[~df.index.isin(outliers_list)]
    metaothers = get_meta_from_df(dfothers)
    metaothers['population'] = [f'others']*len(metaothers)
    
    dfoutvsothers = pd.concat([dfout,dfothers])
    metaoutvsothers = pd.concat([metaout,metaothers])
    toutvsothers = create_and_fit_tester_for_two_sample_test_kfdat(df=dfoutvsothers,
                                                    meta=metaoutvsothers.copy(),
                                                    data_name=f'{outliers_name}_all_other_cells',
                                                    condition='population',
                                                    nystrom=False,
                                                    center_by=None,)
    return(toutvsothers)

def figures_outliers_of_reversion(tester,trunc,df,outliers_list,outliers_name,color=None,marker=None):
    
    
    fig,axes = plt.subplots(ncols=4,figsize=(28,7)) 
                
    ax = axes[0]
    tester.plot_pval_and_errors(fig=fig,ax=ax,truncations_of_interest=[1,3,5],t=20)

    ax = axes[1]
    tester.hist_discriminant(t=trunc,fig=fig,ax=ax,)


    ax = axes[2]
    tester.plot_residuals(t=trunc,fig=fig,ax=ax,highlight=outliers_list,color=color,marker=marker)
    ax.set_title(outliers_name,fontsize=30)

    ax = axes[3]
    tester.fit_tester_with_ignored_outliers(outliers_list=outliers_list,
                                            outliers_name=outliers_name)
    tester.plot_pval_and_errors(fig=fig,ax=ax,truncations_of_interest=[1,3,5],t=20,outliers_in_obs=outliers_name)
    

    if color is not None:
        true_condition = tester.condition
        tester.condition = color
    effectifs_outliers = " ".join([f'{k}:{len(v[v.isin(outliers_list)])}' for k,v in tester.get_index().items()])
    ax.set_title(f'without\n{effectifs_outliers}', fontsize=30)
    if color is not None:
        tester.condition = true_condition

    if len(outliers_list)>0:

        dtest = get_dict_testers_outliers_vs_each_condition(df,outliers_list,outliers_name)
        fig_,axes_ = plt.subplots(ncols=4,figsize=(28,7))     
        for condition,ax in zip(['0H','24H','48HDIFF','48HREV'],axes_):

            dtest[condition].plot_pval_and_errors(
                fig=fig_,ax=ax,truncations_of_interest=[1,3,5],t=20)

            ax.set_title(f'out vs {condition}',fontsize=30)
    else:
        axes[3].set_title('no outliers',fontsize=30)
    fig.tight_layout()
    return(fig,axes)


