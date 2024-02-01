import pandas as pd
import matplotlib.pyplot as plt
from .utils_matplotlib import *


def compute_nreject_from_dfpvals(dfpvals,threshold=.05):
    return((dfpvals<threshold).sum(axis=1))

def compute_dict_of_genes_by_nrejects_from_dfpvals(dfpvals,threshold=.05,verbose=0):
    reject = compute_nreject_from_dfpvals(dfpvals,threshold=threshold)
    nc = len(dfpvals.columns)
    gs = {}
    for i in range(nc+1):
        genes = dfpvals[reject==i].index
        if verbose>0:
            print(f'{i}: {len(genes)}',end = ' |' )
        gs[i] = genes
    if verbose>0:
        print('')
    return(gs)

def boxplot_of_zp_per_nreject(self,dfpvals,threshold=.05,verbose=0):
    df = compute_df_zp_per_nreject(self,
                                   dfpvals=dfpvals,
                                   threshold=threshold,
                                  verbose=verbose)

    fig,ax = custom_boxplot(df)
    ax.set_yticks([i/10 for i in range(11)])
    ax.set_xlabel('n methods rejecting H0',fontsize=20)
    return(fig,ax)
    
def compute_df_zp_per_nreject(self,dfpvals,threshold=.05,verbose=0):
    var = self.var
    nc = len(dfpvals.columns)
    gs = compute_dict_of_genes_by_nrejects_from_dfpvals(dfpvals=dfpvals,
                                                       threshold=threshold,
                                                       verbose=verbose)
    zp = {}
    for i in range(nc+1):
        zp[i] = var[var.index.isin(gs[i])]['pz']
    return(pd.concat(zp,axis=1))

def bar_nreject_per_method(dfpvals,threshold=.05):
    de_reject = {}
    reject = compute_nreject_from_dfpvals(dfpvals,threshold=threshold)
    for de in dfpvals.columns:
        pvals = dfpvals[de]
        de_reject[de] = len(pvals[pvals<threshold])
    ax = pd.DataFrame(de_reject,index=['n']).plot.bar()
    ax.set_title('Number of rejected variables')
  
def compute_ndeg_per_nreject(self,dfpvals,threshold=.05,pz_threshold=0,pz_orientation='>',proportion=True,verbose=0):
    _,nc = dfpvals.shape
    de_scores = {}
    for de in dfpvals.columns:
        de_scores[de] = []
        for i in range(nc+1):
            
            genes = select_genes_per_nreject_and_pz(self,
                                                    dfpvals=dfpvals,
                                                    nreject=i,
                                                    threshold=threshold,
                                                    pz_threshold=pz_threshold,
                                                    pz_orientation=pz_orientation,
                                                    verbose=verbose)
            
            pvals = dfpvals[dfpvals.index.isin(genes)][de]
            pvals = pvals[pvals<threshold]
            
            ng = len(genes) if len(genes)>0 else 1
            nde = len(pvals)
            ratio = nde/ng if nde != ng else 1.2  
            if proportion:
                de_scores[de] += [ratio]
            else:
                de_scores[de] += [nde]
    return(pd.DataFrame(de_scores,index=range(nc+1)))

def bar_of_deg_per_nreject(self,dfpvals,threshold=.05,pz_threshold=0,pz_orientation='>',proportion=True):
    df = compute_ndeg_per_nreject(self,dfpvals,
                             threshold=threshold,
                             pz_threshold=pz_threshold,
                             pz_orientation=pz_orientation,
                             proportion=proportion)
    fig,ax = plt.subplots(figsize=(20,6))
    ax.axhline(1,color='grey',alpha=.5)
    df.plot.bar(ax=ax)

def group_deg_per_method_rejecting_them(dfpvals,goi,threshold):
    rejectors = {}
    for g in goi:
        rej = (dfpvals.loc[[g]]<threshold).T
        rejector = " | ".join(rej[rej[g]].index.to_list())
        if rejector not in rejectors:
            rejectors[rejector] = []
        rejectors[rejector] += [g]
    return(rejectors)

def print_characteristics_of_deg_per_method_rejecting_them(self,dfpvals,goi,threshold=.05):
    rejectors = group_deg_per_method_rejecting_them(dfpvals=dfpvals,
                                                    goi=goi,
                                                    threshold=threshold)
    var = self.var
    for i,de in enumerate(rejectors.keys()):
        genes= rejectors[de]
        varg = var[var.index.isin(genes)][['var','pz']]
        print(de,len(genes),' var = ',varg.mean()['var'],'pz = ',varg.mean()['pz'])

def select_genes_per_nreject_and_pz(self,dfpvals,nreject,threshold=.05,pz_threshold=.9,pz_orientation='<',verbose=0):
    dfz = compute_df_zp_per_nreject(self,
                                    dfpvals=dfpvals,
                                    threshold=threshold,
                                   verbose=verbose)[nreject]
    dfz = dfz[~dfz.isna()]
    nz = len(dfz)
    if verbose >0:
        print(f'{nz} genes rejected by {nreject} methods')
    if pz_orientation=='<':
        goi = dfz[dfz<pz_threshold].index
        if verbose>0:
            print(f'{len(goi)}/{nz} DE genes with pz < {pz_threshold} ')
    if pz_orientation=='>':
        goi = dfz[dfz>pz_threshold].index
        if verbose>0:
            print(f'{len(goi)}/{nz} DE genes with pz > {pz_threshold} ')
    if pz_orientation=='<>':
        dfz = dfz[pz_threshold[0]<dfz]
        goi = dfz[dfz<pz_threshold[1]].index
        if verbose>0:
            print(f'{len(goi)}/{nz} DE genes with  pz in {pz_threshold} ')
    return(goi)

def filter_genes_with_de_methods_and_pz(self,dfpvals,
                                        accepted_by=None,
                                        rejected_by=None,
                                        threshold=.05,
                                        pz_threshold=.9,
                                        pz_orientation='<',
                                       verbose=0):
    
    nreject = len(dfpvals.columns) if accepted_by is None else 0
    dfpvals_ = dfpvals if accepted_by is None else dfpvals[accepted_by]
    
    g_accepted = select_genes_per_nreject_and_pz(self,
                                                dfpvals=dfpvals_,
                                                nreject=nreject,
                                                threshold=threshold,
                                                pz_threshold=pz_threshold,
                                                pz_orientation=pz_orientation,
                                                verbose=verbose)
    
    
    
    nreject = 0 if rejected_by is None else len(rejected_by)
    dfpvals_ = dfpvals if rejected_by is None else dfpvals[rejected_by] 
    
    g_rejected = select_genes_per_nreject_and_pz(self,
                                                dfpvals=dfpvals_,
                                                nreject=nreject,
                                                threshold=threshold,
                                                pz_threshold=pz_threshold,
                                                pz_orientation=pz_orientation,
                                                verbose=verbose)

    goi = g_accepted[g_accepted.isin(g_rejected)]
    
    if verbose>0:
        print(f'accepted by {accepted_by}')
        print(f'rejected by {rejected_by}')
        print(f'{len(goi)} genes')
    return(goi)
    
def center_in_input_space(self,center_by=None,center_non_zero_only=False):
    if center_by == 'replicate':

        data_r = self.get_data(condition='replicate',dataframe=True,in_dict=True)
        all_data_ = self.get_data(condition='replicate',dataframe=True,in_dict=False)
        dfc = all_data_[all_data_!=0].mean() if center_non_zero_only else all_data_.mean()
        
        data_ ={}
        for k,v in data_r.items():
#             print(k)
            if center_non_zero_only:
                v[v!=0] = v[v!=0] - v[v!=0].mean() + dfc
                data_[k] = v
            else:
                data_[k] = v - v.mean() + dfc


    elif center_by == '#-replicate_+label':    
        data_r = self.get_data(condition='replicate',dataframe=True)
        data_l = self.get_data(condition='label',dataframe=True)
        if center_non_zero_only:
            mean_l = {k:v[v!=0].mean() for k,v in data_l.items()}
            mean_r = {k:v[v!=0].mean() for k,v in data_r.items()}
        else:
            mean_l = {k:v.mean() for k,v in data_l.items()}
            mean_r = {k:v.mean() for k,v in data_r.items()}

        metadata_l = self.get_metadata(condition='label',in_dict=True)
        r_in_l = {k:metadata_l[k]['replicate'].unique().to_list() for k in metadata_l.keys()}

        data_ = {}
        for l,ml in mean_l.items():
            for r,dr in data_r.items():
                if r in r_in_l[l]:
                    if center_non_zero_only:
                        dr[dr!=0] = dr[dr!=0] - mean_r[r] + ml
                        data_[r] = dr
                    else:
                        data_[r] =  dr - mean_r[r] + ml
                         
    
    new_df = pd.concat(data_.values())
    new_meta = self.get_metadata(condition='label',in_dict=False)
    new_meta = new_meta.iloc[new_meta.index.get_indexer(new_df.index)].copy()

    return(new_df,new_meta)

