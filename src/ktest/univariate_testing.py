import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as matplotlib
import numpy as np
from time import time
from joblib import Parallel, delayed, parallel_backend
import os
'''
La question du test univarié sur des données univariées est difficile notamment par rapport à ce qu'on veut
retenir de chaque test univarié unique. 
Ces fonctions permettent de simplifier l'approche de test univariée à partir de données multivariées et 
la visualisation des résultats.  
'''


    
def parallel_univariate_kfda(self,params_model={},center_by=None,name='',n_jobs=1,lots=100,fromto=[0,-1],save_path=None):
    """
    parallel univariate testing.
    Variables are tested by groups of `lots` variables in order to save intermediate results if the 
    procedure is interupted before the end. 
    """
    dname = name
    cb = '' if center_by is None else center_by

    if 'approximation_cov' in params_model:
        dname += params_model['approximation_cov']+cb
    else:
        dname += f'standard{cb}'
    print(dname)
        
    fromto[1] = len(voi) if fromto[1]==-1 else fromto[1]
    voi = self.variables[fromto[0]:fromto[1]]
    
    
    ntot = len(self.variables)
    ntested = 0
    nnot_tested = len(voi)
    if f'{dname}_univariate' in self.var:
        tested = self.var[self.var[f'{dname}_univariate']==1].index
        ntested = len(tested)
        nnot_tested = ntot-ntested

        voi = voi[~voi.isin(tested)]
        print(f'filter {ntested} variables already tested')

    ntotest = len(voi)
    print(f'variables : total:{ntot}  t:{ntested} nt:{nnot_tested} ->{ntotest}')
        

    
    if n_jobs==1:
        a=[]
        for v in voi:
            a+=[self.univariate_kfda(v,
                                     params_model=params_model,
                                     center_by=center_by,
                                     name=name,
                                    visualize_test=False,
                                     return_visualize=False,
                                     parallel=True)]
    elif n_jobs >1:
        # with parallel_backend('loky'):
        
        vss = [voi[h*lots:(h+1)*lots] for h in range(len(voi)//lots)]
        vss += [voi[len(voi)//lots:]]
        
        for vs in vss:  
            print(f'testing {len(vs)}/{ntot}')
            t0 = time()  
            with parallel_backend('loky'):
                a = Parallel(n_jobs=n_jobs)(delayed(self.univariate_kfda)(
                    v,
                    params_model=params_model,
                    center_by=center_by,
                    name=name,
                    visualize_test=False,
                    return_visualize=False,
                    parallel=True)
                    for v  in  vs)
            
            print(f'Tested in {time() - t0} \n\n')
            for a_,v in zip(a,vs):
                if a_ is None: 
                    print(f'{v} was not tested')
                if a_ is not None:
    #                 print(a_)
                    if a_['v'] != v:
                        print(f'pb with {v} different from ', a_['v'])
                        v = a_['v']                    
                    for key,value in a_.items():
                        if key != 'v':
                            self.vard[v][key] = value

            if save_path is not None:
                print('saving')
                
                df = pd.DataFrame(self.vard).T
                self.update_var_from_dataframe(df)
                file = f'{dname}_univariate.csv'
                if file in os.listdir(save_path):
                    self.load_univariate_test_results_in_var(save_path,file,)
                self.save_univariate_test_results_in_var(save_path)

            


def univariate_kfda(self,variable,params_model={},visualize_test=True,return_visualize=False,center_by=None,name='',parallel=False,patient=False):
    from .tester import Tester
    zp = self.get_zero_proportions_of_variable(variable)
    # print(zp)
    if zp['x_pz'] == 1 and zp['y_pz'] == 1:
        print("gene not expressed in the dataset")
    else:
        dname = name
        cb = '' if center_by is None else center_by

        if 'approximation_cov' in params_model:
            dname += params_model['approximation_cov']+cb
            print(dname)
        else:
            dname += f'standard{cb}'
            print(dname)

        if f'{dname}_univariate' in self.var and self.var[f'{dname}_univariate'][variable] == True:
            print(f'{dname} already computed for {variable}')

        else : 

            proj = self.init_df_proj(variable)
            cond = self.obs['sample']
            xv = proj[cond=='x'][variable]
            yv = proj[cond=='y'][variable]
        #     print(xv,yv)
            vtest = Tester()
            vtest.init_data(xv,yv,center_by=center_by)
            vtest.init_model(**params_model)
            vtest.obs = self.obs
            vtest.initialize_kfdat()
    #         sp = vtest.spev['xy']['anchors']['w']['sp']
            column = vtest.kfdat(name='kfda')

            t1 = 1
    #         t50 = vtest.select_trunc_by_between_reconstruction_ratio(ratio=.5)
            tr1 = vtest.select_trunc_by_between_reconstruction_ressaut(which_ressaut='first')
            tr2 = vtest.select_trunc_by_between_reconstruction_ressaut(which_ressaut='second')
            tr3 = vtest.select_trunc_by_between_reconstruction_ressaut(which_ressaut='third')
            trm = vtest.select_trunc_by_between_reconstruction_ressaut(which_ressaut='max')

            ts = [t1,tr1,tr2,tr3,trm]
            tnames = ['1','r1','r2','r3','rmax']
            for t,tname in zip(ts,tnames):
                if t<100:

                    col = f'{dname}_t{tname}'
                    pval = vtest.df_pval[column][t]
                    kfda = vtest.df_kfdat[column][t]
                    errB = 1-vtest.get_between_covariance_projection_error_associated_to_t(t)

                    self.vard[variable][f'{col}_pval'] = pval
                    self.vard[variable][f'{col}_kfda'] = kfda 
                    self.vard[variable][f'{col}_errB'] = errB
                    if tname != '1':
                        self.vard[variable][f'{col}_t'] = t

            self.vard[variable][f'{dname}_univariate'] = True
            

            tab = '\t' if len(variable)>6 else '\t\t'
            x_pz = zp['x_pz']
            y_pz = zp['y_pz']
            tr1 = self.vard[variable][f'{dname}_tr1_t']
            tr2 = self.vard[variable][f'{dname}_tr2_t']
            errB1 = self.vard[variable][f'{dname}_tr1_errB']
            errB2 = self.vard[variable][f'{dname}_tr2_errB']
            pval1 = self.vard[variable][f'{dname}_tr1_pval']
            pval2 = self.vard[variable][f'{dname}_tr2_pval']

            string = f'{variable} {tab} pz: x{x_pz:.2f} y{y_pz:.2f} \t pval{tr1:1.0f}:{pval1:1.0e}  r{tr1:1.0f}:{errB1:.2f} \t pval{tr2:1.0f}:{pval2:1.0e}  r{tr2:1.0f}:{errB2:.2f}'
            print(string)
            


            if parallel:
                return({'v':variable,**self.vard[variable]})
            
            if visualize_test:
                fig,axes = self.visualize_univariate_test_CRCL(variable,vtest,column=column,patient=patient)
                if return_visualize:
                    return(vtest,fig,axes)

            return(vtest) 
       


def update_var_from_dataframe(self,df):

    for c in df.columns:
        print(c,end=' ')
        token = False
        if 'univariate' in c and c in self.var:
            token = True
            nbef = sum(self.var[c]==1)
        if c not in self.var:
            self.var[c] = df[c].astype('float64')
        else:
            print('update',end= '|')
            self.var[c].update(df[c].astype('float64'))
        if token:
            naft = sum(self.var[c]==1)
            print(f'\n tested from {nbef} to {naft}')

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

def load_univariate_test_results_in_var(self,path,file,verbose=0):
    df = pd.read_csv(f'{path}{file}',index_col=0)
    if verbose >0:
        print(file,len(df))
    self.update_var_from_dataframe(df)
    

def visualize_univariate_test_CRCL(self,variable,vtest,column,patient=True,data_name='data',):

    fig,axes = plt.subplots(ncols=3,figsize=(22,7))
    
    ax = axes[0]
    self.plot_density_of_variable(variable,data_name=data_name,fig=fig,ax=ax)
    
    if patient:
        ax = axes[1]
        self.plot_density_of_variable(variable,data_name=data_name,fig=fig,ax=ax,color='patient')
        ax = axes[2]
    else:
        ax = axes[1]    
        self.plot_density_of_variable(variable,data_name='counts',fig=fig,ax=ax)
        ax = axes[2]    

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


def plot_density_of_variable(self,variable,fig=None,ax=None,data_name ='data',color=None,condition_mean=True,threshold=None,labels='MF'):
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
        
    self.density_proj(t=0,which=variable,name=data_name,fig=fig,ax=ax,color=color,labels=labels)
    
    
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
    
def filter_genes_wrt_pval(pval,exceptions=[],focus=None,zero_pvals=False,threshold=1):
    
    pval = pval if focus is None else pval[pval.index.isin(focus)]
    pval = pval[~pval.index.isin(exceptions)]
    pval = pval[pval<threshold]
    pval = pval[~pval.isna()]
    pval = pval[pval == 0] if zero_pvals else pval[pval>0]
    return(pval)



def volcano_plot(self,var_prefix,color=None,exceptions=[],focus=None,zero_pvals=False,fig=None,ax=None,BH=False,threshold=1):
    # quand la stat est trop grande, la fonction chi2 de scipy.stat renvoie une pval nulle
    # on ne peut pas placer ces gènes dans le volcano plot alors ils ont leur propre graphe
    
    if fig is None:
        fig,ax = plt.subplots(figsize=(9,15))
    BH_str = 'BHc' if BH else ''
    zpval_str = '= 0' if zero_pvals else '>0'
    
    pval_name = f'{var_prefix}_pval{BH_str}' 
    pval = filter_genes_wrt_pval(self.var[pval_name],exceptions,focus,zero_pvals,threshold)
    print(f'{var_prefix} ngenes with pvals {BH_str} {zpval_str}: {len(pval)}')
    genes = []
    if len(pval) != 0:
        kfda = self.var[f'{var_prefix}_kfda']
        errB = self.var[f'{var_prefix}_errB']
        kfda = kfda[kfda.index.isin(pval.index)]
        errB = errB[errB.index.isin(pval.index)]

        logkfda = np.log(kfda)

        xlim = (logkfda.min()-1,logkfda.max()+1)
        c = color_volcano_plot(self,var_prefix,pval.index,color=color)

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
                ax.set_ylim(0,logpval.max()*1.1)
                genes += [g]


    return(genes,fig,ax)

def volcano_plot_zero_pvals_and_non_zero_pvals(self,var_prefix,color_nz='errB',color_z='t',
                                               exceptions=[],focus=None,BH=False,threshold=1):
    fig,axes = plt.subplots(ncols=2,figsize=(18,15))
    
    genes_nzpval,_,_ = volcano_plot(self,
                                var_prefix,
                                color=color_nz,
                                exceptions=exceptions,
                                focus=focus,
                                zero_pvals=False,
                                fig=fig,ax=axes[0],
                                   BH=BH,
                                   threshold=threshold)
    
    genes_zpval,_,_ = volcano_plot(self,
                               var_prefix,
                               color=color_z,
                               exceptions=exceptions,
                               focus=focus,
                               zero_pvals=True,
                               fig=fig,ax=axes[1],
                               BH=BH,
                               threshold=threshold)
    
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


"""
Les tests univariés sont couteux, ils nécessitent de faire un test par variable d'intérêt. 
Ces fonctions permettent de paralleliser ces calculs. 
Ce sont d'anciennes fonctions que j'ai codées dans un autre contexte et que je n'ai pas encore pris le 
temps de rendre compatible avec ma nouvelle façon de faire des tests univariés. 
Ce qui change, c'est les infos que je veux garder à la fin, et le fait que je veux que le calcul univarié
soit une fonction liée à la classe Tester.  
"""
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
