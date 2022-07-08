import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as matplotlib
import numpy as np

'''
Toutes les fonction de ce script sont spécifique au single cell et à des données qui correspondent
a un certain format que j'ai utilisé. 
'''

def update_var_from_dataframe(self,df):

    for c in df.columns:
        print(c,end=' ')
        if c not in self.var:
            self.var[c] = df[c]
        else:
            print('update',end= '|')
            self.var[c].update(df[c])

def save_univariate_test_results_in_var(self,path):
    for cu in self.var.columns:
        if 'univariate' in cu:
            print(cu)
            dname = cu.split(sep='_univariate')[0]
            cols = [c for c in self.var.columns if f'{dname}_' in c]
            df = self.var[cols]
            df = df[df[cu]==True]
            print(df.shape)
            df.to_csv(f'{path}{cu}.csv')

def load_univariate_test_results_in_var(self,path,file):
    df = pd.read_csv(f'{path}{file}',index_col=0)
    print(file,len(df))
    self.update_var_from_dataframe(df)
    

def visualize_univariate_test_CRCL(self,variable,vtest,column,patient=True,data_name='data'):
    if patient: 
        fig,axes = plt.subplots(ncols=3,figsize=(22,7))
    else : 
        fig,axes = plt.subplots(ncols=2,figsize=(14,7))
    ax = axes[0]
    self.plot_density_of_variable(variable,data_name=data_name,fig=fig,ax=ax)
    if patient:
        ax = axes[1]
        self.plot_density_of_variable(variable,data_name=data_name,fig=fig,ax=ax,color='patient')
        ax = axes[2]
    else:
        ax = axes[1]    
    t1 = 1
    # t50 = vtest.select_trunc_by_between_reconstruction_ratio(ratio=.5)
    tr1 = vtest.select_trunc_by_between_reconstruction_ressaut(which_ressaut='first')
    tr2 = vtest.select_trunc_by_between_reconstruction_ressaut(which_ressaut='second')
    tr3 = vtest.select_trunc_by_between_reconstruction_ressaut(which_ressaut='third')
    trm = vtest.select_trunc_by_between_reconstruction_ressaut(which_ressaut='max')

    toi =[ t for t in [t1,tr1,tr2,tr3,trm] if t<100] 
    vtest.plot_pval_and_errors(column,truncations_of_interest=toi,
                         fig=fig,ax=ax)
    ax.set_xlim(0,20)
    for t in toi:
        ax.axvline(t,ls='--',alpha=.8)
    return(fig,axes)


def plot_density_of_variable(self,variable,fig=None,ax=None,data_name ='data',color=None,condition_mean=True,threshold=None):
    if fig is None:
        fig,ax =plt.subplots(figsize=(10,6))
        
    proj = self.init_df_proj(variable,name=data_name)
    cond = self.obs['sample']
    xv = proj[cond=='x'][variable]
    yv = proj[cond=='y'][variable]
    
    
    # je crois que j'ai déjà une fonction qui fait ça 
    outliers = None
    
    if threshold is not None: 
        print(f'{len(xv[xv>threshold])} and {len(yv[yv>threshold])} cells excluded')
        outliers = self.determine_outliers_from_condition(threshold = threshold,
                                           which=variable,
                                           column_in_dataframe=data_name,
                                           t=0,
                                           orientation='>')
    
        outliers_name=  f'{variable}>{threshold}'
        self.add_outliers_in_obs(outliers,outliers_name )
        
    self.density_proj(t=0,which=variable,name=data_name,fig=fig,ax=ax,color=color)
    
    
    # pas vraiment utile 
    if condition_mean:
        ax.axvline(xv.mean(),c='blue')
        ax.axvline(yv.mean(),c='orange')

        ax.axvline(xv[xv>0].mean(),c='blue',ls='--',alpha=.5)
        ax.axvline(yv[yv>0].mean(),c='orange',ls='--',alpha=.5)
    
    
    title = f'{variable} {data_name}\n'
    zero_proportions = self.get_zero_proportions_of_variable(variable,color)
    for c in zero_proportions.keys():
        if '_nz' in c:
            nz = zero_proportions[c]
            title += f' {c}:{nz}z'
    
    ax.set_title(title,fontsize=20)
    return(fig,ax)

def get_zero_proportions_of_variable(self,variable,condition=None):
    condition = 'sample' if condition is None else condition
    proj = self.init_df_proj(variable,name='counts')
    proj[condition] = self.obs[condition].astype('category')
    output = {}
    for c in proj[condition].cat.categories:
        df = proj[proj[condition]==c][variable]
        n = len(df)
        nz = len(df[df==0])
        pz = nz/n
        output[f'{c}_pz'] = pz
        output[f'{c}_nz'] = nz
        output[f'{c}_n'] = n
    return(output)

def add_zero_proportions_to_var(self,condition=None):
    '''
    The pd.dataframe attribute var is filled with columns corresponding to the number of zero and the 
    proportion of zero of the genes, with respect to the sample or a condition if informed. 
    '''
    zp = [self.get_zero_proportions_of_variable(v,condition) for v in self.variables]
    self.update_var_from_dataframe(pd.DataFrame(zp,index=self.variables))
    
def filter_genes_wrt_pval(pval,exceptions=[],focus=None,zero_pvals=False):
    
    pval = pval if focus is None else pval[pval.index.isin(focus)]
    pval = pval[~pval.index.isin(exceptions)]
    pval = pval[~pval.isna()]
    pval = pval[pval == 0] if zero_pvals else pval[pval>0]
    return(pval)


def volcano_plot(self,var_prefix,color=None,exceptions=[],focus=None,zero_pvals=False,fig=None,ax=None):
    # quand la stat est trop grande, la fonction chi2 de scipy.stat renvoie une pval nulle
    # on ne peut pas placer ces gènes dans le volcano plot alors ils ont leur propre graphe
    
    if fig is None:
        fig,ax = plt.subplots(figsize=(9,15))
    
    pval = filter_genes_wrt_pval(self.var[f'{var_prefix}_pval'],exceptions,focus,zero_pvals)
    kfda = self.var[f'{var_prefix}_kfda']
    errB = self.var[f'{var_prefix}_errB']
    kfda = kfda[kfda.index.isin(pval.index)]
    errB = errB[errB.index.isin(pval.index)]
    
    logkfda = np.log(kfda)
    
    xlim = (logkfda.min()-1,logkfda.max()+1)
    c = color_volcano_plot(self,var_prefix,pval.index,color=color)
    
    genes = []
    if zero_pvals:
#         print('zero')
        ax.set_title(f'{var_prefix} \ng enes strongly rejected',fontsize=30)
        ax.set_xlabel(f'log(kfda)',fontsize=20)
        ax.set_ylabel(f'errB',fontsize=20)
        
        for g in pval.index.tolist():
#             print(g,logkfda[g],errB[g],c[g])
            ax.text(logkfda[g],errB[g],g,color=c[g])
            ax.set_xlim(xlim)
            ax.set_ylim(0,1)
            genes += [g]
        

    else:
#         print('nz')
        ax.set_title(f'{var_prefix}\n non zero pvals',fontsize=30)
        ax.set_xlabel(f'log(kfda)',fontsize=20)
        ax.set_ylabel(f'-log(pval)',fontsize=20)
        logpval = -np.log(pval)
        
        
        for g in pval.index.tolist():
#             print(g,logkfda[g],logpval[g],c[g])
            ax.text(logkfda[g],logpval[g],g,color=c[g])
            ax.set_xlim(xlim)
            ax.set_ylim(0,1000)
            genes += [g]

            
    return(genes,fig,ax)

def volcano_plot_zero_pvals_and_non_zero_pvals(self,var_prefix,color_nz='errB',color_z='t',
                                               exceptions=[],focus=None):
    fig,axes = plt.subplots(ncols=2,figsize=(18,15))
    
    genes_nzpval,_,_ = volcano_plot(self,
                                var_prefix,
                                color=color_nz,
                                exceptions=exceptions,
                                focus=focus,
                                zero_pvals=False,
                                fig=fig,ax=axes[0])
    
    genes_zpval,_,_ = volcano_plot(self,
                               var_prefix,
                               color=color_z,
                               exceptions=exceptions,
                               focus=focus,
                                zero_pvals=True,
                               fig=fig,ax=axes[1])
    
    return(genes_nzpval,genes_zpval,fig,axes)
    

def color_map_color(value, cmap_name='viridis', vmin=0, vmax=1):
    # norm = plt.Normalize(vmin, vmax)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)  # PiYG
    rgb = cmap(norm(abs(value)))[:3]  # will return rgba, we take only first 3 so we get rgb
    color = matplotlib.colors.rgb2hex(rgb)
    return color

def color_volcano_plot(self,var_prefix,index,color=None):
    if color is None:
        
        return(pd.Series(['b']*len(index),index=index))
    
    var_ = self.var[self.var.index.isin(index)]
    colors = []
    if f'{var_prefix}_{color}' in var_:
        values = var_[f'{var_prefix}_{color}']

    
    for g in index:
        if color in ['errB','pval']:
            colors += [color_map_color(values[g])]
        elif color in ['t','kfda']:
            colors += [values[g]] 

        if color == 'mean_proportions':
            px = var_['x_pz'][g]
            py = var_['y_pz'][g]
            
            colors += [(px+py)/2]
        
        if color == 'diff_proportions':
            px = var_['x_pz'][g]
            py = var_['y_pz'][g]
            
            colors += [px-py]
#     print(colors)
    if color in ['t','kfda','mean_proportions','diff_proportions']:
        vmax = np.max(colors)
        vmin = np.min(colors)
        colors = [color_map_color(v,vmin=vmin,vmax=vmax) for v in colors]
    return(pd.Series(colors,index=index))



