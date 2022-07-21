
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
import scanpy as sc
from tester import Tester
from importlib import reload 


"""
Un vieux fourre tout de fonctions pour des applications spécifiques au single-cell. 
Plusieurs fonctions sont pensées pour être interfacées avec les objets AnnData de Scanpy, 
J'ai finalement décidé de m'en éloigner même si je m'en suis inspiré pour les attributs var et obs. 
Dans la finalité, ça peut être intéressant de penser la compatibilité avec Scanpy. 
"""

def split_H0(A: np.array,return_index=False):
    nA = len(A)
    permutation = np.random.permutation(np.arange(nA))
    p1,p2 = permutation[:nA//2],permutation[nA//2:]
    if return_index:
        return((A[p1],p1),(A[p2],p2))
    else:
        return(A[p1],A[p2])

def split_adata_H0(adata:sc.AnnData,layer='norm_data',return_index=False):
    A = np.nan_to_num(adata.layers[layer].toarray())
    splitted = split_H0(A,return_index=return_index)
    
    if return_index:
        A,p1 = splitted[0]
        B,p2 = splitted[1]
        index1,index2 = adata.obs.index[p1],adata.obs.index[p2]
        return((A,index1),(B,index2))

    else:
        return(splitted)

def get_ncells(adata,obs_col,obs_val):
    cells = adata.obs[obs_col]==obs_val
    return(len(adata[cells]))

def select_adata(adata,obs_col,obs_val):
    cells = adata.obs[obs_col]==obs_val
    return(adata[cells])

def tester_of_Controle_Traite(adata,obs_col='treatment',obs_values=('Control','BH3'),layer='norm_data',spec=None):
    """
    spec values in 'bool','H0CC','H0TT'
    """
    adata1,adata2 = select_adata(adata,obs_col,obs_values[0]),select_adata(adata,obs_col,obs_values[1])
    if spec in ['H0CC','H0TT']:
        xi,yi = split_adata_H0(adata1,return_index=True) if spec =='H0CC' else split_adata_H0(adata2,return_index=True)
        x,ix = xi
        y,iy = yi
    else:
        x,y = np.nan_to_num(adata1.layers[layer].toarray()),np.nan_to_num(adata2.layers[layer].toarray())
        ix,iy = adata1.obs.index,adata2.obs.index
        if spec=='bool':
            x,y = 1.*np.array(x,dtype=bool),1.*np.array(y,dtype=bool)
    return(Tester(x=x,y=y,x_index=ix,y_index=iy,variables=adata.var.index))

def get_ncells_from_proj(df_proj,xy):
    return(len(df_proj.loc[df_proj['sample']==xy]))

def scatter_proj_kfda(df_proj,ax,proj,gene_color = None):
    p1,p2 = proj
    for xy,l in zip('xy','CT'):
        dfxy = df_proj.loc[df_proj['sample']==xy]
            
        m = 'x' if xy =='x' else '+'
        
        if gene_color is None:
            c = 'xkcd:cerulean' if xy =='x' else 'xkcd:light orange'
        else:
            c = dfxy[gene_color]
        

        x_ = dfxy[f'p_{p1}']
        y_ = dfxy[f'p_{p2}']
        ax.scatter(x_,y_,c=c,s=30,label=l,alpha=.8,marker = m)

    for xy,l in zip('xy','CT'):

        dfxy = df_proj.loc[df_proj['sample']==xy]
        x_ = dfxy[f'p_{p1}']
        y_ = dfxy[f'p_{p2}']
        mx_ = x_.mean()
        my_ = y_.mean()
        ax.scatter(mx_,my_,edgecolor='black',linewidths=3,s=200)

    ax.set_xlabel(f't={p1}')                    
    ax.set_ylabel(f't={p2}')
    ax.legend()
    
def density_proj_kfda(df_proj,ax,proj):
    for xy,l in zip('xy','CT'):
        dfxy = df_proj.loc[df_proj['sample']==xy][f'p_{proj}']
        ax.hist(dfxy,density=True,label=l,alpha=.8,bins=int(np.floor(np.sqrt(len(dfxy)))))
        ax.axvline(dfxy.mean())
    ax.set_xlabel(f't={proj}')    
    ax.legend()

def initiate_kfda_plot(ax,trunc = range(1,60),asymp_ls='--',asymp_c = 'crimson',ylim=(-5,2000)):
    
    yas = [chi2.ppf(0.95,t) for t in trunc]    
    ax.plot(trunc,yas,ls=asymp_ls,c=asymp_c)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('t',fontsize= 20)
    
def correlationmatrix_kfda_genes(df_array,df_proj,csvfile,pathfile,trunc=range(1,60)):

    df_proj.index = df_array.index # inutile dans un futur proche 
    
    if csvfile in os.listdir(pathfile):
        acorr = pd.read_csv(pathfile+csvfile,header=0,index_col=0)
    else:
        for t in trunc:
            df_array[f'p_{t}'] = pd.Series(df_proj[f'p_{t}'],
                                   index = df_proj.index) # toujours utile dans ce futur proche ?
        acorr = df_array.corr()
        acorr.to_csv(pathfile+csvfile)
    return(acorr)

def add_genes_of_interest_in_df_proj(acorr,df_proj,df_array,ngenes = 4,trunc=range(1,60)):
    '''
    Utilise la matrice de corrélations "acorr" pour trouver les gènes corrélés à chaque direction de la KFDA. 
    Les niveau d'expressions des "ngenes" gènes les plus corrélés sont copiés du DataFrame "df_array"
    au DataFrame "df_proj" pour intégrer l'info des gènes plus facilement aux tracés des directions de la KFDA. 
    '''
    
    sorted_genes_dict = {}
    for t in trunc:
        
        sorted_genes = np.abs(acorr[f'p_{t}']).sort_values(ascending=False)
        sorted_genes_dict[f'p_{t}'] = sorted_genes
        
        for g in range(1,ngenes+1): # on commence à 1 car l'indice 0 correspond à la direction p_{t}
            gname = sorted_genes.index[g]
            i = 1
            
            while gname[:2] == 'p_':
                gname = sorted_genes.index[g+i]
                i += 1
            df_proj[gname] = df_array[gname]

def generate_which_dict(path_kfda,dataset_id,genes_selection='sg',spec=''):

    path_proj_kfda = path_kfda  + dataset_id + genes_selection + spec + 'projfdaxis' + '.csv'
    path_kfdat = path_kfda  + dataset_id + genes_selection + spec + 'kfda' + '.csv'
    path_proj_kpca = path_kfda  + dataset_id + genes_selection + spec + 'proj' + '.csv'
    path_corr_pv = path_kfda  + dataset_id + genes_selection + spec + 'corr_pv' + '.csv'
    
    return({
            'kfdat':path_kfdat,
            'proj_kfda':{'discriminant':path_proj_kfda},
            'proj_kpca':{'variant':path_proj_kpca},
            'correlations':{'genes':path_corr_pv}
            })



