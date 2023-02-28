from scipy.stats import chi2
import pandas as pd
from ktest.tester import Ktest
import numpy as np
from joblib import parallel_backend
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from time import time
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import chi2
from scipy.cluster.hierarchy import dendrogram

"""
Ce fichier est un fichier fourre tout où j'ai mis la plupart des fonctions dont j'ai régulièrement 
besoin et qui ne sont pas directement à intégrer au package, il y a notamment toute une 
série de méta-fonctions qui me permettent de gérer plus facilement une situation où j'ai plusieurs 
objets Ktest en parallèle correspondant à différents modèles ou jeux de données. 
"""

"""
Ces fonctions calculent la stat à partir de données et renvoie directement la stat 
L'objet Ktest reste en back et n'est pas accessible. 
"""

# def compute_standard_kfda(x,y,name,pval=True,kernel='gauss_median',params_model={}):
    # removed function that returned test.df_kfdat,test.df_pval

# def compute_standard_mmd(x,y,name,kernel='gauss_median',params_model={}):
    # removed function that returned test.dict_mmd


"""
Dans le cas d'un grid search pour tuner plusieurs paramètres, ces fonctions 
génèrent des tableaux de graphes pour visualiser l'influence de chaque paramètre sur les résultats. 
"""
        
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
"""
J'ai souvent l'occasion de faire des tests de deux échantillons sur plus de deux jeux de données. 
La solution simple et efficace que j'ai trouvée pour gérer les multiples instances de l'objet Ktest, 
une pour chaque couple d'échantillons comparés, consiste à nommer chaque instance et de toutes les 
stocker dans un dictionnaire que j'appelle toujours "dict_tests". 
Dans ce genre de situation, les données sont réutilisées plusieurs fois, je nomme chaque jeu de données 
et le stocke dans un dictionnaire que j'appelle "dict_data". 
En général, les noms des instances de Ktest dans "dict_tests" correspondent aux noms dans "dict_data" qu'ont 
les deux jeux de données comparés, séparés par un "_". 
Quand je veux comparer des concaténations de jeux de données présents dans dict_data, le nom dans "dict_tests" 
contient les noms des jeux de données séparés par une virgule ",". 

Tout ça est explicité plus en détail dans la description de la fonction "add_Ktest_to_dict_tests_from_name_and_dict_data"

Plusieurs fonctions ci-dessous servent uniquement à construire le dendrogramme d'un clustering hierarchique 
qui prend la statistique KFDA comme metrique. La fonctions correspondante est "plot_custom_dendrogram_from_cats"
"""


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
            t = test.t
            stat = test.df_kfdat['kfda'][t]
            dist[cat1][cat2] = stat
            dist[cat2][cat1] = stat
    return(pd.DataFrame(dist).fillna(0).to_numpy())

def get_data_in_dict_data_from_name(name,dict_data):
    cat1 = name.split(sep='_')[0]
    cat2 = name.split(sep='_')[1]

    df1 = pd.concat([dict_data[c] for c in [c for c in cat1.split(sep=',')]],axis=0) if ',' in cat1 else dict_data[cat1]
    df2 = pd.concat([dict_data[c] for c in [c for c in cat2.split(sep=',')]],axis=0) if ',' in cat2 else dict_data[cat2]
    return(df1,df2)


def add_Ktest_to_dict_tests_from_name_and_dict_data(name,dict_data,dict_tests,dict_meta=None,params_model={},center_by=None,free_memory=True):
    '''
    This function was specifically developped for the analysis of the CRCL data but it can be generalized. 
    This function takes the parameter name to determine which datasets stored in dict_data to compare and store 
    the resultant Ktest object at the key name of the dictionnary dict_tests. 
    
    The syntax for name is the following: 
    concatenated datasets are separated by a comma, 
    compared datasets are separated by an underscore '_'. 
    For example, if we have 'A','B' and 'C' in dict_data, 
    To compare 'A' and 'B' versus 'C', the parameter name should be equal to 'A,B_C'. 
        
    
    
    Parameters
    ----------
        name : str,
        Contains the information of the datasets on which we want to compute a statistic,
        It is also the key in which the resulting Ktest object will be stored in dict_tests. 
        The syntax for name is the following: 
        concatenated datasets are separated by a comma, 
        compared datasets are separated by an underscore '_'. 
        For example, if we have 'A','B' and 'C' in dict_data, 
        To compare 'A' and 'B' versus 'C', the parameter name should be equal to 'A,B_C'.
        
        dict_data : dict,
        Contains the pandas.DataFrames of every single dataset of interest, 
        each DataFrame is stored at a key caracterizing the dataset. 
        These keys are refered as categories or cat. 
        
        dict_tests : dict,
        The dictionnary to update by adding the Ktest object of the comparison defined by the parameter name. 
        
        dict_meta (optionnal) : dict,
        Contains the pandas.DataFrames of the metadata of every single dataset of interest, 
        each DataFrame is stored at a key caracterizing the dataset. 
        These keys are refered as categories or cat. 
        
        params_model (optionnal): dict, 
        Contains the information of which version of the statistic to compute,
        e.g. how to use the nystrom method if needed. 
        The keys to inform in params model are (should be simplified in the future):
            'approximation_cov' in 'standard','nystrom1', 'nystrom2','nystrom3'
            'approximation_mmd' in 'standard','nystrom1', 'nystrom2','nystrom3'
            if nystrom is used in one of the two approximations : 
                'n_landmarks' (int): the number of landmarks
                'n_anchors' (int): the number of anchors
                'landmark_method' in 'random', 'kmeans'
                'anchor_basis' in 'W','S','K'
        
        center_by (optionnal): str,
        A parameter to correct some effects corresponding to the metadata. 
        More information in the description of the function init_center_by of Ktest
        
        free_memory (default = True): boolean,
        If True, the data and the eigenvectors are not stored in the Ktest object. 
        This is usefull when many comparisons are done and stored in dict_tests to save place in the RAM. 
        
        
        
                
    Returns
    ------- 
        A new Ktest object is added in dict_tests at the key name. 
    '''

    if name not in dict_tests:
        cat1 = name.split(sep='_')[0]
        cat2 = name.split(sep='_')[1]
        df1,df2 = get_data_in_dict_data_from_name(name,dict_data)


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
        colpatient1 = []
        colpatient2 = []
        colcelltype1 = []
        colcelltype2 = []
        

        for cat,colcat,colsexe,colpatient,colcelltype in zip([cat1,cat2],[colcat1,colcat2],[colsexe1,colsexe2],[colpatient1,colpatient2],[colcelltype1,colcelltype2]):
            for c in cat.split(sep=','):
                colcat += [c]*len(dict_data[c])
                colsexe += [c[0]]*len(dict_data[c])
                colpatient += [c[:2]]*len(dict_data[c])
                colcelltype += [c[2:]]*len(dict_data[c])
                
        df1_meta['cat'] = colcat1
        df2_meta['cat'] = colcat2
        df1_meta['sexe'] = colsexe1
        df2_meta['sexe'] = colsexe2
        df1_meta['patient'] = colpatient1
        df2_meta['patient'] = colpatient2
        df1_meta['celltype'] = colcelltype1
        df2_meta['celltype'] = colcelltype2
        df1_meta['cat'] = df1_meta['cat'].astype('category')
        df2_meta['cat'] = df2_meta['cat'].astype('category')
        df1_meta['sexe'] = df1_meta['sexe'].astype('category')
        df2_meta['sexe'] = df2_meta['sexe'].astype('category')
        df1_meta['patient'] = df1_meta['patient'].astype('category')
        df2_meta['patient'] = df2_meta['patient'].astype('category')
        df1_meta['celltype'] = df1_meta['celltype'].astype('category')
        df2_meta['celltype'] = df2_meta['celltype'].astype('category')
        
#         print(len(df1),len(df2))
        if len(df1)>10 and len(df2)>10:
            t0=time()
            print(name,len(df1),len(df2),end=' ')
            test = Ktest()
#             center_by = 'cat' if center_by_cat else None
            test.init_data_from_dataframe(df1,df2,dfx_meta = df1_meta,dfy_meta=df2_meta,center_by=center_by)
            test.obs['cat']=test.obs['cat'].astype('category')
            test.obs['sexe']=test.obs['sexe'].astype('category')
            test.obs['patient']=test.obs['patient'].astype('category')
            test.init_model(**params_model)
            test.kfdat_statistic(name = 'kfda')
            t = test.t
            
                        
            test.compute_proj_kfda(t=20,name='kfda')
            
            kfda = test.df_kfdat['kfda'][t]
            test.compute_pvalue()
            pval = test.df_pval['kfda'][t]
            if free_memory:
                test.x = np.array(0)
                test.y = np.array(0) 
                del(test.spev['xy']['standard']['ev'])
#                 test.spev = {'x':{},'y':{},'xy':{},'residuals':{}}
            
            dict_tests[name] = test
            print(f'{time()-t0:.3f} t={t} kfda={kfda:.4f} pval={pval}')
        else: 
            print(f'{len(df1)} and {len(df2)} is not enough data')

    else:
        print(f'{name} in dict_tests')

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
    ax.legend(fontsize=30)
    # fig.suptitle(f'{cat1} vs {cat2} {infos}: pval={pval:.5f}',fontsize=30,y=1.04)
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

def get_kfda_from_name_and_dict_tests(name,dict_tests):
    if name in dict_tests:
        test = dict_tests[name]
        t = test.t
        kfda = test.df_kfdat['kfda'][t]
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
                add_Ktest_to_dict_tests_from_name_and_dict_data(name,dict_data,dict_tests,params_model=params_model)
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
    pval = test.df_pval['kfda'][t]
    kfda = test.df_kfdat['kfda'][t]
    c = 'green' if pval >.05 else 'red'
    yaccept = chi2.ppf(.95,t)
    ax.scatter(x,yaccept,s=500,c='red',alpha=.5,marker='_',)
    ax.scatter(x,y,s=300,c=c,edgecolor='black',alpha=1,linewidths=3,)

def get_name_MvsF(ct,data_type,cts,dict_data):
    mwi = [f'{mw}{i}' for i in '123' for mw in 'MW']
    name = ''
    for m in mwi:
        if 'M' in m:
            if ct in cts: 
                for celltype in cts[ct]:
                    cat = f'{m}{celltype}{data_type}'
                    if cat in dict_data:
                        name += f'{cat},'
            else:
                cat = f'{m}{ct}{data_type}'
                if cat in dict_data:
                    name += f'{cat},'

    name = name[:-1]
    name += '_'
    for m in mwi:
        if 'W' in m:
            if ct in cts: 
                for celltype in cts[ct]:
                    cat = f'{m}{celltype}{data_type}'
                    if cat in dict_data:
                        name += f'{cat},'
            else:
                cat = f'{m}{ct}{data_type}'
                if cat in dict_data:
                    name += f'{cat},'
    
    name = name[:-1]
    if len(name.split(sep='_'))==2 and name[0]!='_':
        return(name)
    else: 
        return('')


 