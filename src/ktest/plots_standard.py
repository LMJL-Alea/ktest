from os import POSIX_FADV_SEQUENTIAL
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

from .kernel_statistics import Statistics
from .utils_plot import init_plot_kfdat,init_plot_pvalue,text_truncations_of_interest

from .kernel_statistics import Statistics


# from functions import get_between_covariance_projection_error

from adjustText import adjust_text


from scipy.stats import chi2
import numpy as np
import torch
from torch import mv,dot,sum,cat,tensor,float64




class Plot_Standard(Statistics):

    def __init__(self):
        super(Plot_Standard,self).__init__()



    def plot_several_kfdat(self,fig=None,ax=None,ylim=None,t=None,columns=None,title=None,title_fontsize=40,mean=False,mean_label='mean',mean_color = 'xkcd: indigo',contrib=False,label_asymp=False,asymp_arg=None,legend=True):
        # try:
            if columns is None:
                columns = self.df_kfdat.columns
            kfdat = self.df_kfdat[columns].copy()

            t = max([(~kfdat[c].isna()).sum() for c in columns]) if t is None and len(columns)==0 else \
                100 if t is None else t 
            trunc = range(1,t)  
            no_data_to_plot = len(columns)==0

            fig,ax = init_plot_kfdat(fig=fig,ax=ax,ylim=ylim,t=t,label=label_asymp,title=title,title_fontsize=title_fontsize,asymp_arg=asymp_arg)
            
            if mean:
                ax.plot(kfdat.mean(axis=1),label=mean_label,c=mean_color)
                ax.plot(kfdat.mean(axis=1)- 2* kfdat.std(axis=1)/(~kfdat[columns[0]].isna()).sum(),c=mean_color,ls = '--',alpha=.5)
                ax.plot(kfdat.mean(axis=1)+ 2* kfdat.std(axis=1)/(~kfdat[columns[0]].isna()).sum(),c=mean_color,ls = '--',alpha=.5)
            if len(self.df_kfdat.columns)>0:
                kfdat.plot(ax=ax)
            if ylim is None and not no_data_to_plot:
                # probleme je voulais pas yas dans cette fonction 
                yas = [chi2.ppf(0.95,t) for t in trunc] 
                # ymax = np.max([yas[-1], np.nanmax(np.isfinite(kfdat[kfdat.index == trunc[-1]]))]) # .max(numeric_only=True)
                ymax = np.max([yas[-1]*5, np.nanmax(np.isfinite(kfdat))]) # .max(numeric_only=True)
                ylim = (-5,ymax)
                ax.set_ylim(ylim)
            if contrib and len(columns)==1:
                self.plot_kfdat_contrib(fig,ax,t,columns[0])
            if legend:
                ax.legend()  
            return(fig,ax)

    def plot_kfdat_contrib(self,fig=None,ax=None,t=None,name=None):
        xp,yspnorm,ypnorm = self.compute_kfdat_contrib(t)
        ax1 = ax.twinx()
        
        ax1.bar(xp,yspnorm,color='orange',alpha=.4)
        ax1.bar(xp,ypnorm,color='red',alpha=.2)

        label_ysp = r'$\Vert \widehat{\Sigma}_W^{1/2} f_{X,t}^W \Vert_{\mathcal{H}}^2/tr(\widehat{\Sigma}_W) =  \lambda_{X,t}^w/tr(\widehat{\Sigma}_W)$' 
        label_yp = r'$\Vert \widehat{\Sigma}_B^{1/2} f_{X,t}^W \Vert_{\mathcal{H}}^2 \propto \frac{n_1 n_2}{n} \langle f_{X,t}^w, \widehat{\mu}_2 - \widehat{\mu}_1 \rangle_{\mathcal{H}}$'
        
        spectrum = mpatches.Patch(color='orange',alpha=.2,label = label_ysp)
        mmdproj = mpatches.Patch(color='red',alpha=.2,label = label_yp)
        kfda = mlines.Line2D([], [], color='#1f77b4', label=name)
        ax.legend(handles=[kfda,spectrum,mmdproj],fontsize=20)    

        return(fig,ax)



    def plot_kfdat(self,t=20,fig=None,ax=None,title=None,legend=True,
                    ylim=None,
                    log=False,contrib=False,aggregated=True,
                    truncations_of_interest=None,adjust=True,
                    color_agg='black',color_uni='xkcd:blue',ls_agg='-',ls_uni='-',
                    label_agg='kfda',label_uni='axis kfda',
                    asymp_arg=None):
        fig,ax = init_plot_kfdat(fig=fig,ax=ax,ylim=ylim,t=t,title=title,asymp_arg=asymp_arg)
            
        if aggregated:
            kfda = self.get_kfda(log=log).loc[:t]
            ax.plot(kfda,label=label_agg,ls=ls_agg,c=color_agg) 

            if truncations_of_interest is not None:
                values = [0]+kfda 
                text_truncations_of_interest(truncations_of_interest,ax,values,adjust,log=log)

        if contrib:
            kfda = self.get_kfda(contrib=contrib,log=log).loc[:t]
            ax.plot(kfda,label=label_uni,alpha=.5,ls=ls_uni,color=color_uni) 

            if truncations_of_interest is not None and not aggregated:
                values = [0]+kfda 
                text_truncations_of_interest(truncations_of_interest,ax,values,adjust,log=log)
        # ax.set_xlim(0,t)
        if legend:
            ax.legend(fontsize=20)

        return(fig,ax)


    def plot_pvalue(self,t=20,fig=None,ax=None,title=None,legend=True,log=False,contrib=False,aggregated=True,truncations_of_interest=None,adjust=True,
                color_agg='black',color_uni='xkcd:blue',ls_agg='dashdot',ls_uni='--',label_agg='p-value',label_uni='axis p-value'):
        fig,ax = init_plot_pvalue(fig=fig,ax=ax,ylim=(-.05,1.05),t=t,title=title,log=log)

        if aggregated:
            pval = self.get_pvalue(log=log).loc[:t]
            ax.plot(pval,label=label_agg,ls=ls_agg,c=color_agg) 

            if truncations_of_interest is not None:
                values = [0]+pval 
                text_truncations_of_interest(truncations_of_interest,ax,values,adjust,log=log)

        if contrib:
            pval = self.get_pvalue(contrib=contrib,log=log).loc[:t]
            ax.plot(pval,label=label_uni,alpha=.5,ls=ls_uni,color=color_uni) 

            if truncations_of_interest is not None and not aggregated:
                values = [0]+pval 
                text_truncations_of_interest(truncations_of_interest,ax,values,adjust,log=log)
               
        # ax.set_xlim(0,t)
        if legend:
            ax.legend(fontsize=20)

        return(fig,ax)


    def plot_several_pvalues(self,t=20,fig=None,ax=None,columns=None,legend=True,log=False,contrib=False,ylim=None,title=None,title_fontsize=40,label_asymp=False):

        fig,ax = init_plot_pvalue(fig=fig,ax=ax,ylim=ylim,t=t,label=label_asymp,
                                title=title,title_fontsize=title_fontsize,log=log)
    
        if columns is None:
            columns = self.df_kfdat.columns
        for c in columns:
            pval = self.get_pvalue(contrib=contrib,log=log,name=c).loc[:t]
            ax.plot(pval,label=c)
        # pvals = self.df_pval[columns].copy()
        # if log :
        #     pvals = -np.log(pvals+1)
        # # t_ = (~pvals.isna()).sum().min()
        # pvals.plot(ax=ax)
        # ax.set_xlim(0,t)
        
        if legend:
            ax.legend()
        
        return(fig,ax)

    def plot_spectrum(self,fig=None,ax=None,t=None,anchors=False,
                        cumul=False,part_of_inertia=False,log=False,
                        label=None,truncations_of_interest = None):
        if ax is None:
            fig,ax = plt.subplots(figsize=(10,10))

        sp = self.get_spectrum(anchors=anchors,cumul=cumul,part_of_inertia=part_of_inertia,log=log)

        if truncations_of_interest is not None:
            values = cat(tensor(0),sp) 
            text_truncations_of_interest(truncations_of_interest,ax,values)
       
        trunc = range(1,len(sp) if t is None else min(t,len(sp)))
        ax.plot(trunc,sp[:trunc[-1]],label=label)
        ax.set_xlabel('t',fontsize= 20)
        return(fig,ax)

    def get_axis_label(self,proj,t):
        label = proj if proj in self.get_variables() else \
                 t if proj=='obs' else \
                 f'DA{t}' if proj == 'proj_kfda' else \
                 f'PC{t}' if proj == 'proj_kpca' else \
                 f'R{t}'

        if proj == 'proj_kfda':
            pval = self.df_pval[self.get_kfdat_name()].loc[t]
            label += f' pval={pval:.1e}' if pval<.01 else f' pval={pval:1.2f}'
                    
        sp,ev = self.get_spev('covw')
        if proj in ['proj_kfda','proj_kpca']:
            lmbda = sp[t-1]
            label += r' $\lambda$'
            label += f'={lmbda:.1e}' if lmbda<.01 else f'={lmbda:.2f}' 
        return(label)

    def density_proj(self,
                    t,
                    proj='proj_kfda',
                    name=None,
                    orientation='vertical',
                    color=None,
                    fig=None,ax=None,
                    show_conditions=True,
                    legend_fontsize=15,
                    condition=None,
                    samples=None,
                    samples_colors=None,
                    marked_obs_to_ignore=None):


        labels = list(self.get_index(condition=condition,samples=samples,marked_obs_to_ignore=marked_obs_to_ignore).keys())

        if fig is None:
            fig,ax = plt.subplots(ncols=1,figsize=(12,6))

        properties = self.get_plot_properties(
                        color=color,
                        labels=labels,
                        show_conditions=show_conditions,
                        condition=condition,
                        samples=samples,
                        color_list=samples_colors,
                        marked_obs_to_ignore=marked_obs_to_ignore)

        df_proj= self.init_df_proj(proj,name)
        
        # quand beaucoup de plot se chevauchent, ça serait sympa de les afficher en 3D pour mieux les voir 
        
        for kprop,vprop in properties.items():
    #         print(kprop,vprop['mean_plot_args'].keys())
            if len(vprop['index'])>0:
                dfxy = df_proj.loc[df_proj.index.isin(vprop['index'])]
                dfxy = dfxy[proj] if proj in dfxy else dfxy[str(t)]                        
                
                ax.hist(dfxy,density=True,histtype='bar',alpha=.5,orientation=orientation,**vprop['hist_args'])
                if 'label' in vprop['hist_args']:
                    del(vprop['hist_args']['label'])
                if 'edgecolor' not in vprop['hist_args']:
    #                 print(ax._children[-1].__dict__)
                    vprop['hist_args']['edgecolor'] = ax._children[-1]._facecolor
                if 'color' not in vprop['hist_args']:
                    vprop['hist_args']['color'] = ax._children[-1]._facecolor
                
                ax.hist(dfxy,density=True,histtype='step',lw=3,orientation=orientation,**vprop['hist_args'])
                
                # si je voulais faire des mean qui correspondent à d'autre pop que celles des histogrammes,
                # la solution la plus simple serait de faire une fonction spécifique 'plot_mean_hist' par ex
                # dédiée au tracé des lignes verticales correspondant aux means que j'appelerais séparément. 
                if orientation =='vertical':
                    ax.axvline(dfxy.mean(),c=vprop['hist_args']['color'],lw=1.5)
                else:
                    ax.axhline(dfxy.mean(),c=vprop['hist_args']['color'],lw=1.5)

        if orientation == 'vertical':
            ax.set_xlabel(self.get_axis_label(proj,t),fontsize=25)
        else:
            ax.set_ylabel(self.get_axis_label(proj,t),fontsize=25)

        ax.legend(fontsize=legend_fontsize)

        # fig.tight_layout()
        return(fig,ax)
        
    def get_plot_properties(self,
                    marker=None,
                    color=None,
                    show_conditions=True,
                    labels='CT',
                    legend=True,
                    condition=None,
                    samples=None,
                    marked_obs_to_ignore=None,
                    marker_list = ['.','x','+','d','1','*',(4,1,0),(4,1,45),(7,1,0),(20,1,0),'s'],
                    big_marker_list = ['o','X','P','D','v','*',(4,1,0),(4,1,45),(7,1,0),(20,1,0),'s'],
                    color_list = ['xkcd:cerulean','xkcd:light orange','xkcd:grass green']
                        #color_list,marker_list,big_marker_list,show_conditions
                        ):
        if color_list is None:
            color_list = ['xkcd:cerulean','xkcd:light orange','xkcd:grass green']
        properties = {}
        # cx_ = color_list[0] # 'xkcd:cerulean'
        # cy_ = color_list[1] #'xkcd:light orange'
        mx_ = 'o'
        my_ = 's'
        variables = self.data[self.data_name]['variables']
        marked_obs_to_ignore = self.marked_obs_to_ignore if marked_obs_to_ignore is None else marked_obs_to_ignore
        
        dict_index = self.get_index(condition=condition,samples=samples,marked_obs_to_ignore=marked_obs_to_ignore)
        dict_data = self.get_dataframes_of_data(condition=condition,samples=samples,marked_obs_to_ignore=marked_obs_to_ignore)
        samples_list = self.get_samples_list(condition,samples) 
        
        color_list = color_list if len(color_list)>=len(samples_list) else [None]*len(samples_list)   
        if not isinstance(color_list,dict): 
            color_list = {k:c for k,c in zip(samples_list,color_list)}
        coef_bins = 3
        if marker is None and color is None : 
            # print("plot properties marker is None and color is None")
            for i,k in enumerate(samples_list):

                ipop = dict_index[k]
                lab = f'{k}({len(ipop)})' if legend else None
                bins = coef_bins*int(np.floor(np.sqrt(len(ipop))))
                m = marker_list[i]
                c = color_list[k]
                bm = big_marker_list[i]

                properties[k] = {'index':ipop,
                                 'plot_args':{'marker':m,'c':c},
                                 'mean_plot_args':{'marker':bm,'label':lab,'color':c},
                                 'hist_args':{'bins':bins,'label':lab}
                }
                if c is not None:
                    properties[k]['hist_args']['color']=c
        
        elif isinstance(color,str) and marker is None:
            # print("plot properties isinstance(color,str) and marker is None")
            if color in list(variables):
                # print("color in list(variables)")

                for i,k in enumerate(samples_list):
                    
                    ipop = dict_index[k]
                    n = len(ipop)
                    df = dict_data[k]
                    c = df[color]
                    cm = color_list[k]
                    lab = f'{k}({n})' if legend else None
                    m = marker_list[i]
                    bm = big_marker_list[i]
                    properties[k] = {'index':ipop,
                                    'plot_args':{'marker':m,'c':c},
                                    'mean_plot_args':{'marker':bm,'label':lab,'color':cm},
                                    }

            elif color in list(self.obs.columns):
                # print("color in list(variables)")
                if self.obs[color].dtype == 'category':
                    for i,pop in enumerate(self.obs[color].cat.categories):
                        if show_conditions: 
                            for j,k in enumerate(samples_list):

                                ipop = self.obs.loc[self.obs[color]==pop].index
                                n = len(ipop)
                                ipop = ipop[ipop.isin(dict_index[k])]
                                
                                m = marker_list[j%len(marker_list)]
                                mean_m = big_marker_list[j%len(big_marker_list)]

                                if marked_obs_to_ignore is not None:
                                    obs_to_ignore = self.obs[self.obs[marked_obs_to_ignore]].index
                                    ipop = ipop[~ipop.isin(obs_to_ignore)]
                                lab = f'{k} {pop} ({n})' if legend else None
                                bins = coef_bins*int(np.floor(np.sqrt(n)))
                                properties[f'{pop}{k}'] = {'index':ipop,
                                                    'plot_args':{'marker':m},
                                                    'mean_plot_args':{'marker':mean_m,'label':lab},
                                                    'hist_args':{'bins':bins,'label':lab}}

        
                        else:
                            # print('not show conditions')

                            ipop = self.obs.loc[self.obs[color]==pop].index
                            if marked_obs_to_ignore is not None:
                                obs_to_ignore = self.obs[self.obs[marked_obs_to_ignore]].index
                                ipop = ipop[~ipop.isin(obs_to_ignore)]

                            lab = f'{pop} ({len(ipop)})' if legend else None
                            bins=coef_bins*int(np.floor(np.sqrt(len(ipop))))
                            c = color_list[k]
                            
                            properties[pop] = {'index':ipop,
                                            'plot_args':{'marker':'.','c':c},
                                            'mean_plot_args':{'marker':'o','label':lab,'color':c},
                                            'hist_args':{'bins':bins,'label':lab}}
                            if c is not None:
                                properties[k]['hist_args']['color']=c
                            
                else: # pour une info numérique 
                    # print(f'{color} is not categorical in obs')
                    for i,k in enumerate(samples_list):

                        m = marker_list[i%len(marker_list)]
                        mean_m = big_marker_list[i%len(big_marker_list)]
                        

                        ipop = dict_index[k]
                        n = len(ipop)
                        if marked_obs_to_ignore is not None:
                            obs_to_ignore = self.obs[self.obs[marked_obs_to_ignore]].index
                            ipop = ipop[~ipop.isin(obs_to_ignore)]
                        c = self.obs.loc[self.obs.index.isin(ipop)][color]
                        lab = f'{k}({n})' if legend else None
                        properties[k] = {'index':ipop,
                                        'plot_args':{'c':c,'marker':m},
                                        'mean_plot_args':{'marker':mean_m,'label':lab}}


                        
        elif color is None and isinstance(marker,str):
            print('color is None and marker is specified : this case is not treated yet')
        
        elif isinstance(color,str) and isinstance(marker,str):
            if marker in list(self.obs.columns) and self.obs[marker].dtype == 'category':
                if color in list(variables):
                    print('variable')
                    x,y = self.get_xy()

                    for im,popm in enumerate(self.obs[marker].cat.categories):
                        
                        m = marker_list[im%len(marker_list)]
                        mean_m = big_marker_list[im%len(big_marker_list)]
                        
                        ipopx = self.obs.loc[self.obs[marker]==popm].loc[self.obs['sample']=='x'].index
                        ipopy = self.obs.loc[self.obs[marker]==popm].loc[self.obs['sample']=='y'].index
                        
                        if marked_obs_to_ignore is not None:
                            obs_to_ignore = self.obs[self.obs[marked_obs_to_ignore]].index
                            ipopx = ipopx[~ipopx.isin(obs_to_ignore)]
                            ipopy = ipopy[~ipopy.isin(obs_to_ignore)]


                        maskx = self.get_xy_index(sample='x').isin(ipopx)
                        masky = self.get_xy_index(sample='y').isin(ipopy)
                        
                        cx = x[maskx,variables.get_loc(color)]
                        cy = y[masky,variables.get_loc(color)]
                        
                        labx = f'{labels[0]} {popm} ({len(ipopx)})' if legend else None
                        laby = f'{labels[1]} {popm} ({len(ipopy)})' if legend else None

                    properties['x']  = {'index':ipopx,
                                        'plot_args':{'marker':m,'c':cx},
                                        'mean_plot_args':{'marker':mean_m,'color':color_list[popm],'label':labx}}
                    properties['y']  = {'index':ipopy,
                                        'plot_args':{'marker':m,'c':cy},
                                        'mean_plot_args':{'marker':mean_m,'color':color_list[popm],'label':laby}}

                elif color in list(self.obs.columns):
                    print('in obs')
                    if self.obs[color].dtype == 'category':
                        for popc in self.obs[color].cat.categories:
                            for im,popm in enumerate(self.obs[marker].cat.categories):
                                
                                m = marker_list[im%len(marker_list)]
                                mean_m = big_marker_list[im%len(big_marker_list)]
                                
                                if show_conditions: 
                                    ipopx = self.obs.loc[self.obs[color]==popc]\
                                                    .loc[self.obs['sample']=='x']\
                                                    .loc[self.obs[marker]==popm].index
                                    
                                    ipopy = self.obs.loc[self.obs[color]==popc]\
                                                    .loc[self.obs['sample']=='y']\
                                                    .loc[self.obs[marker]==popm].index
                                    if marked_obs_to_ignore is not None:
                                        obs_to_ignore = self.obs[self.obs[marked_obs_to_ignore]].index
                                        ipopx = ipopx[~ipopx.isin(obs_to_ignore)]
                                        ipopy = ipopy[~ipopy.isin(obs_to_ignore)]
                                    labx = f'{labels[0]} {popc} {popm} ({len(ipopx)})' if legend else None
                                    laby = f'{labels[1]} {popc} {popm} ({len(ipopy)})' if legend else None

                                    properties[f'{popc}{popm}x'] = {
                                        'index':ipopx,
                                        'plot_args':{'marker':m},
                                        'mean_plot_args':{'marker':mean_m,'label':labx}}
                                    properties[f'{popc}{popm}y'] = {
                                        'index':ipopy,
                                        'plot_args':{'marker':m},
                                        'mean_plot_args':{'marker':mean_m,'label':laby}}
                                
                                else:
                                    ipop = self.obs.loc[self.obs[color]==popc].loc[self.obs[marker]==popm].index
                                    
                                    if marked_obs_to_ignore is not None:
                                        obs_to_ignore = self.obs[self.obs[marked_obs_to_ignore]].index
                                        ipop = ipop[~ipop.isin(obs_to_ignore)]
                                    
                                    lab = f'{popc} {popm} ({len(ipop)})' if legend else None
                                    properties[f'{popc}{popm}'] = {'index':ipop,
                                                                'plot_args':{'marker':m},
                                                                'mean_plot_args':{'marker':mean_m,'label':lab}}


                    
                    else: # pour une info numérique 
                        for im,popm in enumerate(self.obs[marker].cat.categories):

                            m = marker_list[im%len(marker_list)]
                            mean_m = big_marker_list[im%len(big_marker_list)]

                            if show_conditions:

                                ipopx = self.obs.loc[self.obs['sample']=='x']\
                                                .loc[self.obs[marker]==popm].index

                                ipopy = self.obs.loc[self.obs['sample']=='y']\
                                                .loc[self.obs[marker]==popm].index
                                            
                                if marked_obs_to_ignore is not None:
                                    obs_to_ignore = self.obs[self.obs[marked_obs_to_ignore]].index
                                    ipopx = ipopx[~ipopx.isin(obs_to_ignore)]
                                    ipopy = ipopy[~ipopy.isin(obs_to_ignore)]

                                cx = self.obs.loc[self.obs.index.isin(ipopx)][color]
                                cy = self.obs.loc[self.obs.index.isin(ipopy)][color]

                                labx = f'{labels[0]} {popm} ({len(ipopx)})' if legend else None
                                laby = f'{labels[1]} {popm} ({len(ipopy)})' if legend else None
                                            
                                properties[f'{popm}x'] = {
                                    'index':ipopx,
                                    'plot_args':{'c':cx,'marker':m,},
                                    'mean_plot_args':{'marker':mean_m,'color':cx_,'label':labx}}
                                properties[f'{popm}y'] = {
                                    'index':ipopy,
                                    'plot_args':{'c':cy,'marker':m,},
                                    'mean_plot_args':{'marker':mean_m,'color':cy_,'label':laby}}

                            else:
                                ipop = self.obs.loc[self.obs[marker]==popm].index
                                
                                if marked_obs_to_ignore is not None:
                                    obs_to_ignore = self.obs[self.obs[marked_obs_to_ignore]].index
                                    ipop = ipop[~ipop.isin(obs_to_ignore)]

                                c = self.obs.loc[self.obs.index.isin(ipop)][color]
                                
                                lab = f'{popm} ({len(ipop)})' if legend else None

                                properties[f'{popm}'] = {'index':ipop,
                                                        'plot_args':{'c':c,'marker':m,},
                                                        'mean_plot_args':{'marker':mean_m,'label':lab}}
                                
                                
            else:
                print(f"{marker} is not in self.obs or is not categorical, \
                use self.obs['{marker}'] = self.obs['{marker}'].astype('category')")
            
        else:

                print(f'{color} and {marker} not found in obs and variables')

        return(properties)

    def scatter_proj(self,projection,xproj='proj_kpca',yproj=None,xname=None,yname=None,
                    highlight=None,color=None,marker=None,show_conditions=True,text=False,fig=None,ax=None,
                    alpha=.8,legend=True,legend_fontsize=15):
        labels = list(self.get_index().keys())
        
        if fig is None:
            fig,ax = plt.subplots(ncols=1,figsize=(12,6))

        p1,p2 = projection
        yproj = xproj if yproj is None else yproj
        if xproj == yproj and yname is None:
            yname = xname
        # print(f'xproj={xproj} xname={xname}  yproj={yproj} yname={yname}')
        df_abscisse = self.init_df_proj(xproj,xname,)
        df_ordonnee = self.init_df_proj(yproj,yname,)
        properties = self.get_plot_properties(marker=marker,color=color,labels=labels,show_conditions=show_conditions,legend=legend)
        texts = []
        
        for kprop,vprop in properties.items():
    #         print(kprop,vprop['mean_plot_args'].keys())
            if len(vprop['index'])>0:
                x_ = df_abscisse.loc[df_abscisse.index.isin(vprop['index'])]
                y_ = df_ordonnee.loc[df_ordonnee.index.isin(vprop['index'])]

                x_ = x_[p1] if p1 in x_ else x_[str(p1)]                        
                y_ = y_[p2] if p2 in y_ else y_[str(p2)]                        
                


                # alpha = .2 if text else .8
                # print(len(x_),len(y_)) 
                ax.scatter(x_,y_,s=30,alpha=alpha,**vprop['plot_args'])
                if 'mean_plot_args' in vprop and 'color' not in vprop['mean_plot_args']:
                    vprop['mean_plot_args']['color'] = ax._children[-1]._facecolors[0]
                    
        for kprop,vprop in properties.items():
            if len(vprop['index'])>0:
                x_ = df_abscisse.loc[df_abscisse.index.isin(vprop['index'])][f'{p1}']
                y_ = df_ordonnee.loc[df_ordonnee.index.isin(vprop['index'])][f'{p2}']
                
                if highlight is not None and type(highlight)==str:
                    if highlight in self.obs and self.obs[highlight].dtype == bool:
                        highlight = self.obs[self.obs[highlight]].index

                if highlight is not None and vprop['index'].isin(highlight).sum()>0:
                    
                    ihighlight = vprop['index'][vprop['index'].isin(highlight)]
                    
                    xhighlight_ = df_abscisse.loc[df_abscisse.index.isin(ihighlight)][f'{p1}']
                    yhighlight_ = df_ordonnee.loc[df_ordonnee.index.isin(ihighlight)][f'{p2}']
                    c = vprop['plot_args']['color'] if 'color' in vprop['plot_args'] else \
                        vprop['mean_plot_args']['color'] if 'mean_plot_args' in vprop and 'color' in vprop['mean_plot_args'] \
                        else 'xkcd:neon purple'
                    ax.scatter(xhighlight_,yhighlight_,color=c,s=100,marker='*',edgecolor='black',linewidths=1)    
                    
                if 'mean_plot_args' in vprop:
                    mx_ = x_.mean()
                    my_ = y_.mean()
                    ax.scatter(mx_,my_,edgecolor='black',linewidths=1.5,s=200,**vprop['mean_plot_args'],alpha=1)

                    if text :
                        texts += [ax.text(mx_,my_,kprop,fontsize=20)]
                        
        if text:

            adjust_text(texts,only_move={'points': 'y', 'text': 'y', 'objects': 'y'})
            
        if 'title' in properties :
            ax.set_title(properties['title'],fontsize=20)

        ax.set_xlabel(self.get_axis_label(xproj,p1),fontsize=25)                    
        ax.set_ylabel(self.get_axis_label(yproj,p2),fontsize=25)
        
        ax.legend(fontsize=legend_fontsize)

        return(fig,ax)      




    def init_axes_projs(self,fig,axes,projections,suptitle,kfda,kfda_ylim,t,kfda_title,spectrum,spectrum_label):
        if axes is None:
            rows=1;cols=len(projections) + kfda + spectrum
            fig,axes = plt.subplots(nrows=rows,ncols=cols,figsize=(6*cols,6*rows))
        if suptitle is not None:
            fig.suptitle(suptitle,fontsize=50,y=1.04)
        if kfda:
            
            #params a ajouter si besoin ? 
            # columns=None,asymp_ls='--',asymp_c = 'crimson',title=None,title_fontsize=40,mean=False,mean_label='mean',mean_color = 'xkcd: indigo')
            self.plot_kfdat(fig=fig,ax = axes[0],ylim=kfda_ylim,t = t,title=kfda_title)
            axes = axes[1:]
        if spectrum:
            self.plot_spectrum(fig=fig,ax=axes[0],t=t,title='spectrum',label=spectrum_label)
            axes = axes[1:]
        return(fig,axes)


    def density_projs(self,fig=None,axes=None,proj='proj_kpca',name=None,projections=range(1,10),suptitle=None,kfda=False,kfda_ylim=None,t=None,kfda_title=None,spectrum=False,spectrum_label=None,show_conditions=True):
        fig,axes = self.init_axes_projs(fig=fig,axes=axes,projections=projections,suptitle=suptitle,kfda=kfda,
                                        kfda_ylim=kfda_ylim,t=t,kfda_title=kfda_title,spectrum=spectrum,spectrum_label=spectrum_label)
        if not isinstance(axes,np.ndarray):
            axes = [axes]
        for ax,t in zip(axes,projections):
            self.density_proj(t=t,proj=proj,name=name,fig=fig,ax=ax,show_conditions=show_conditions)
        fig.tight_layout()
        return(fig,axes)

    def scatter_projs(self,fig=None,axes=None,xproj='proj_kpca',yproj=None,xname=None,yname=None,projections=[(i*2,i*2+1) for i in range(4)],suptitle=None,
                        highlight=None,color=None,marker=None,kfda=False,kfda_ylim=None,t=None,kfda_title=None,spectrum=False,spectrum_label=None,iterate_over='projections',show_conditions=True):
        to_iterate = projections if iterate_over == 'projections' else color
        fig,axes = self.init_axes_projs(fig=fig,axes=axes,projections=to_iterate,suptitle=suptitle,kfda=kfda,
                                        kfda_ylim=kfda_ylim,t=t,kfda_title=kfda_title,spectrum=spectrum,spectrum_label=spectrum_label)
        if not isinstance(axes,np.ndarray):
            axes = [axes]
        for ax,obj in zip(axes,to_iterate):
            if iterate_over == 'projections':
                self.scatter_proj(obj,xproj=xproj,yproj=yproj,xname=xname,yname=yname,highlight=highlight,color=color,marker=marker,fig=fig,ax=ax,show_conditions=show_conditions)
            elif iterate_over == 'color':
                self.scatter_proj(projections,xproj=xproj,yproj=yproj,xname=xname,yname=yname,highlight=highlight,color=obj,marker=marker,fig=fig,ax=ax,show_conditions=show_conditions)
        fig.tight_layout()
        return(fig,axes)
    
    def plot_correlation_proj_var(self,fig=None,ax=None,name=None,nvar=30,projections=range(1,10),title=None,prefix_col=''):
        if name is None:
            name = self.get_names()['correlations'][0]
        if ax is None:
            fig,ax = plt.subplots(figsize=(10,10))
        if title is not None:
            ax.set_title(title,fontsize=40)
        for proj in projections:
            col = f'{prefix_col}{proj}'
            val  = list(np.abs(self.corr[name][col]).sort_values(ascending=False).values)[:nvar]
            print(val)
            ax.plot(val,label=col)
        ax.legend()
        return(fig,ax)

    def hist_discriminant(self,
                    t,
                    color=None,
                    fig=None,
                    ax=None,
                    show_conditions=True,
                    orientation='vertical',
                    legend_fontsize=15,
                    condition=None,
                    samples=None,
                    samples_colors=None,
                    marked_obs_to_ignore=None,
                    verbose=0):

        self.projections(t=t,condition=condition,samples=samples,
                        marked_obs_to_ignore=marked_obs_to_ignore,verbose=verbose)

        kfdat_name = self.get_kfdat_name(
                        condition=condition,
                        samples=samples,
                        marked_obs_to_ignore=marked_obs_to_ignore)

        fig,ax = self.density_proj(
                        t,
                        proj='proj_kfda',
                        name=kfdat_name,
                        orientation=orientation,
                        color=color,
                        fig=fig,ax=ax,
                        show_conditions=show_conditions,
                        legend_fontsize=legend_fontsize,
                        condition=condition,
                        samples=samples,
                        samples_colors=samples_colors,
                        marked_obs_to_ignore=marked_obs_to_ignore)
        return(fig,ax)


    def hist_mmd_discriminant(self,color=None,fig=None,ax=None,show_conditions=True,orientation='vertical',legend_fontsize=15):
        mmd_name = self.get_mmd_name()
        fig,ax = self.density_proj(t='mmd',proj='proj_mmd',name=mmd_name,orientation=orientation,color=color,
                        fig=fig,ax=ax,show_conditions=show_conditions,legend_fontsize=legend_fontsize)
        return(fig,ax)
        
    def hist_tmmd_discriminant(self,t,color=None,fig=None,ax=None,show_conditions=True,orientation='vertical',legend_fontsize=15):
        mmd_name = self.get_mmd_name()
        fig,ax = self.density_proj(t=t,proj='proj_tmmd',name=mmd_name,orientation=orientation,color=color,
                        fig=fig,ax=ax,show_conditions=show_conditions,legend_fontsize=legend_fontsize)
        return(fig,ax)

    def hist_pc(self,t,color=None,fig=None,ax=None,show_conditions=True,orientation='vertical',legend_fontsize=15):
        kfdat_name = self.get_kfdat_name()
        fig,ax = self.density_proj(t,proj='proj_kpca',name=kfdat_name,orientation=orientation,color=color,
                        fig=fig,ax=ax,show_conditions=show_conditions,legend_fontsize=legend_fontsize)
        return(fig,ax)

    def plot_nextPC(self,t,fig=None,ax=None,color=None,marker=None,highlight=None,show_conditions=True,legend_fontsize=15):
        fig,ax = self.scatter_proj(projection=[t,t+1],
                               xproj='proj_kfda',
                               yproj='proj_kpca',
                               xname=self.get_kfdat_name(),
                               yname=self.get_kfdat_name(),
                               color=color,
                               marker=marker,
                               highlight=highlight,
                               show_conditions=show_conditions,
                               fig=fig,
                               ax=ax,
                               legend_fontsize=legend_fontsize)
        ax.set_title(f'D={t} PC={t+1}',fontsize=30)
        return(fig,ax)

    def plot_nextDA(self,t,fig=None,ax=None,color=None,marker=None,highlight=None,show_conditions=True,legend_fontsize=15):
        fig,ax = self.scatter_proj(projection=[t,t+1],
                               xproj='proj_kfda',
                               yproj='proj_kfda',
                               xname=self.get_kfdat_name(),
                               yname=self.get_kfdat_name(),
                               color=color,
                               marker=marker,
                               highlight=highlight,
                               show_conditions=show_conditions,
                               fig=fig,
                               ax=ax,
                               legend_fontsize=legend_fontsize)
        ax.set_title(f'D={t} D={t+1}',fontsize=30)
        return(fig,ax)


    def plot_residuals(self,t=1,center='w',fig=None,ax=None,color=None,marker=None,highlight=None,legend_fontsize=15):
        self.residuals(t=t,center=center)
        residuals_name = self.get_residuals_name(t=t,center=center)
        kfdat_name = self.get_kfdat_name()
        fig,ax=self.scatter_proj([t,1],xproj='proj_kfda',yproj='proj_residuals',
                                xname=kfdat_name,yname=residuals_name,
                          fig=fig,ax=ax,color=color,marker=marker,highlight=highlight,legend_fontsize=legend_fontsize)
        ax.set_title(f'discriminant and orthogonal axis t={t}',fontsize=30)
        return(fig,ax)

    def plot_kpca(self,t=1,fig=None,ax=None,color=None,marker=None,highlight=None,legend_fontsize=15):
        kfdat_name = self.get_kfdat_name()
        fig,ax=self.scatter_proj([t,t+1],xproj='proj_kpca',xname=kfdat_name,
                            fig=fig,ax=ax,color=color,marker=marker,highlight=highlight,legend_fontsize=legend_fontsize)
        ax.set_title('KPCA',fontsize=30)      
        return(fig,ax)



    def plot_nreject(self,t=20,fig=None,ax=None,label=None,legend=True):

        if fig is None:
            fig,ax = plt.subplots(ncols=1,figsize=(12,6))

        pval =  self.df_pval_contributions[self.get_kfdat_name()]
    #     print(pval)
        pval = pval[[i for i in range(1,t)]]
        responses=(pval<.05).cumsum()
        ax.set_xlim(0,t)
        ax.plot(responses,label=label)
        if legend:
            ax.legend()
        
        return(fig,ax)

    def plot_nreject_weighted(self,t=20,fig=None,ax=None,label=None,legend=True,cumul=True):

        if fig is None:
            fig,ax = plt.subplots(ncols=1,figsize=(12,6))

        kfdat_contrib = self.df_kfdat_contributions[self.get_kfdat_name()]
        responses = []
        for v in kfdat_contrib.values:
            pval = chi2.sf(v,1)
            if pval>.05:
                responses+=[0]
            else:
                if v > chi2.ppf(0.95,300):
                    responses += [300]
                else:
                    score = np.where(np.array([v<chi2.ppf(.95,t_) for t_ in range(1,300)]) == True)[0][0]
    #                 print('score',score)
                    responses += [score]
        
    #     print(responses)
        
        ax.set_xlim(0,t)
        if cumul:
            ax.plot(np.array(responses)[:t].cumsum(),label=label)
        else:
            ax.plot(np.array(responses)[:t],label=label)
        if legend:
            ax.legend()
        
        return(fig,ax)