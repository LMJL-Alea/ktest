from os import POSIX_FADV_SEQUENTIAL
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

from .statistics import Statistics
from .utils_plot import init_plot_kfdat,init_plot_pvalue,text_truncations_of_interest

from .statistics import Statistics


# from functions import get_between_covariance_projection_error

from adjustText import adjust_text


from scipy.stats import chi2
import numpy as np
import torch
from torch import mv,dot,sum,cat,tensor,float64




class Plot_Standard(Statistics):

    def __init__(self):
        super(Plot_Standard,self).__init__()

    def plot_kfdat(self,fig=None,ax=None,ylim=None,t=None,columns=None,title=None,title_fontsize=40,mean=False,mean_label='mean',mean_color = 'xkcd: indigo',contrib=False,label_asymp=False,asymp_arg=None,legend=True):
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
                
                ymax = np.max([yas[-1], np.nanmax(np.isfinite(kfdat[kfdat.index == trunc[-1]]))]) # .max(numeric_only=True)
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

    def plot_pvalue(self,column,t=20,fig=None,ax=None,legend=True,label=None,log=False,ylim=None,title=None,title_fontsize=40,label_asymp=False):

        fig,ax = init_plot_pvalue(fig=fig,ax=ax,ylim=ylim,t=t,label=label_asymp,
                                title=title,title_fontsize=title_fontsize,log=log)
    
        pvals = self.df_pval[column].copy()
        if log :
            pvals = -np.log(pvals)
        t_ = (~pvals.isna()).sum()
        ax.plot(pvals[:t_],label=column if label is None else label)
        ax.set_xlim(0,t)
        
        if legend:
            ax.legend()
        
        return(fig,ax)

    def plot_spectrum_new(self,fig=None,ax=None,t=None,title=None,anchors=False,label=None,truncations_of_interest = None ):
        if ax is None:
            fig,ax = plt.subplots(figsize=(10,10))
        if title is not None:
            ax.set_title(title,fontsize=40)
        
        
        cov = self.approximation_cov

        if anchors:
            if 'nystrom' not in cov:
                print('impossible to plot anchor spectrum')
            else:

                sp,_ = self.get_spev(slot='anchors')
                
        else:
            sp,_ = self.get_spev(slot='covw')

        if truncations_of_interest is not None:
            values = cat(tensor(0),sp)
            text_truncations_of_interest(truncations_of_interest,ax,values)
            # texts = []
            # for t in set(truncations_of_interest):
            #     if len(sp)>t-1:
            #         spt = sp[t-1]
            #         text = f'{spt:.2f}' if spt >=.01 else f'{spt:1.0e}'
            #         ax.scatter(x=t,y=spt,s=20)
            #         texts += [ax.text(t,spt,text,fontsize=20)]
            # adjust_text(texts,only_move={'points': 'y', 'text': 'y', 'objects': 'y'})
        
        t = len(sp) if t is None else min(t,len(sp))
        trunc = range(1,t)
        ax.plot(trunc,sp[:trunc[-1]],label=label)
        ax.set_xlabel('t',fontsize= 20)

        return(fig,ax)

    def density_proj(self,t,which='proj_kfda',name=None,orientation='vertical',labels='MW',color=None,fig=None,ax=None,show_conditions=True):
        if fig is None:
            fig,ax = plt.subplots(ncols=1,figsize=(12,6))

        properties = self.get_plot_properties(color=color,labels=labels,show_conditions=show_conditions)
        df_proj= self.init_df_proj(which,name)
        
        # quand beaucoup de plot se chevauchent, ça serait sympa de les afficher en 3D pour mieux les voir 
        
        for kprop,vprop in properties.items():
    #         print(kprop,vprop['mean_plot_args'].keys())
            if len(vprop['index'])>0:
                dfxy = df_proj.loc[df_proj.index.isin(vprop['index'])]
                dfxy = dfxy[which] if which in dfxy else dfxy[str(t)]                        
                
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
                    

        name_pval = self.get_kfdat_name()
        xlabel = which if which in self.data[self.data_name]['variables'] else which.split(sep='_')[1]+f': t={t}'
        xlabel += f'  pval={self.df_pval[name_pval].loc[t]:.3e}' if which == 'proj_kfda' else \
                f'  pval={self.df_pval_contributions[name_pval].loc[t]:.3e}' if which == 'proj_kfda' else \
                ''
        if orientation == 'vertical':
            ax.set_xlabel(xlabel,fontsize=25)
        else:
            ax.set_ylabel(xlabel,fontsize=25)

        ax.legend(fontsize=30)

        fig.tight_layout()
        return(fig,ax)
        
    def get_plot_properties(self,marker=None,color=None,show_conditions=True,labels='CT',
                            marker_list = ['.','x','+','d','1','*',(4,1,0),(4,1,45),(7,1,0),(20,1,0),'s'],
                            big_marker_list = ['o','X','P','D','v','*',(4,1,0),(4,1,45),(7,1,0),(20,1,0),'s'],
                            
                        #color_list,marker_list,big_marker_list,show_conditions
                        ):

        properties = {}
        cx_ = 'xkcd:cerulean'
        cy_ = 'xkcd:light orange'
        mx_ = 'o'
        my_ = 's'
        variables = self.data[self.data_name]['variables']
        outliers_in_obs = self.outliers_in_obs

        if marker is None and color is None : 
            ipopx = self.get_xy_index(sample='x')
            ipopy = self.get_xy_index(sample='y')

            labx = f'{labels[0]}({len(ipopx)})'
            laby = f'{labels[1]}({len(ipopy)})'
            
            binsx=int(np.floor(np.sqrt(len(ipopx))))
            binsy=int(np.floor(np.sqrt(len(ipopy))))
                            
            properties['x'] = {'index':ipopx,
                            'plot_args':{'marker':'x','color':cx_},
                            'mean_plot_args':{'marker':mx_,'color':cx_,'label':labx},
                            'hist_args':{'bins':binsx,'label':labx,'color':cx_}}
                            
            properties['y'] = {'index':ipopy,
                            'plot_args':{'marker':'+','color':cy_},
                            'mean_plot_args':{'marker':my_,'color':cy_,'label':laby},
                            'hist_args':{'bins':binsy,'label':laby,'color':cy_}}
                            
        
        elif isinstance(color,str) and marker is None:
            if color in list(variables):
                x,y = self.get_xy()
                
                ipopx = self.get_xy_index(sample='x')
                ipopy = self.get_xy_index(sample='y')

                cx = x[:,variables.get_loc(color)]
                cy = y[:,variables.get_loc(color)]
                
                labx = f'{labels[0]}({len(ipopx)})'
                laby = f'{labels[1]}({len(ipopy)})'
            
                properties['x']  = {'index':ipopx,
                                    'plot_args':{'marker':'x','c':cx},
                                    'mean_plot_args':{'marker':mx_,'color':cx_,'label':labx},
                                    }
                properties['y']  = {'index':ipopy,
                                    'plot_args':{'marker':'+','c':cy},
                                    'mean_plot_args':{'marker':my_,'color':cy_,'label':laby}}
                
                
            elif color in list(self.obs.columns):
                if self.obs[color].dtype == 'category':
                    for pop in self.obs[color].cat.categories:
                        if show_conditions: 
                    
                            ipopx = self.obs.loc[self.obs[color]==pop]\
                                            .loc[self.obs['sample']=='x'].index
                            ipopy = self.obs.loc[self.obs[color]==pop]\
                                            .loc[self.obs['sample']=='y'].index
                            if outliers_in_obs is not None:
                                outliers = self.obs[self.obs[outliers_in_obs]].index
                                ipopx = ipopx[~ipopx.isin(outliers)]
                                ipopy = ipopy[~ipopy.isin(outliers)]

                            labx = f'{labels[0]} {pop} ({len(ipopx)})'
                            laby = f'{labels[1]} {pop} ({len(ipopy)})'
                            
                            binsx=int(np.floor(np.sqrt(len(ipopx))))
                            binsy=int(np.floor(np.sqrt(len(ipopy))))
                            
                            properties[f'{pop}x'] = {'index':ipopx,
                                                    'plot_args':{'marker':'x'},
                                                    'mean_plot_args':{'marker':mx_,'label':labx},
                                                    'hist_args':{'bins':binsx,'label':labx}}
                            properties[f'{pop}y'] = {'index':ipopy,
                                                    'plot_args':{'marker':'+'},
                                                    'mean_plot_args':{'marker':my_,'label':laby},
                                                    'hist_args':{'bins':binsy,'label':laby}}

                            
                        else:
                            ipop = self.obs.loc[self.obs[color]==pop].index
                            if outliers_in_obs is not None:
                                outliers = self.obs[self.obs[outliers_in_obs]].index
                                ipop = ipop[~ipop.isin(outliers)]

                            lab = f'{pop} ({len(ipop)})'
                            bins=int(np.floor(np.sqrt(len(ipop))))
                            
                            
                            properties[pop] = {'index':ipop,
                                            'plot_args':{'marker':'.'},
                                            'mean_plot_args':{'marker':'o','label':lab},
                                            'hist_args':{'bins':bins,'label':lab}}
                            
                            
                else: # pour une info numérique 

                    ipopx = self.obs.loc[self.obs['sample']=='x'].index                                
                    ipopy = self.obs.loc[self.obs['sample']=='y'].index
                    
                    if outliers_in_obs is not None:
                        outliers = self.obs[self.obs[outliers_in_obs]].index
                        ipopx = ipopx[~ipopx.isin(outliers)]
                        ipopy = ipopy[~ipopy.isin(outliers)]

                    cx = self.obs.loc[self.obs.index.isin(ipopx)][color]
                    cy = self.obs.loc[self.obs.index.isin(ipopy)][color]


                    labx = f'{labels[0]}({len(ipopx)})'
                    laby = f'{labels[1]}({len(ipopy)})'

                    properties['x'] = {'index':ipopx,
                                    'plot_args':{'c':cx,'marker':'x'},
                                    'mean_plot_args':{'marker':mx_,'color':cx_,'label':labx}}
                    properties['y'] = {'index':ipopy,
                                    'plot_args':{'c':cy,'marker':'+'},
                                    'mean_plot_args':{'marker':my_,'color':cy_,'label':laby}}
                        

                        
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
                        
                        if outliers_in_obs is not None:
                            outliers = self.obs[self.obs[outliers_in_obs]].index
                            ipopx = ipopx[~ipopx.isin(outliers)]
                            ipopy = ipopy[~ipopy.isin(outliers)]


                        maskx = self.get_xy_index(sample='x').isin(ipopx)
                        masky = self.get_xy_index(sample='y').isin(ipopy)
                        
                        cx = x[maskx,variables.get_loc(color)]
                        cy = y[masky,variables.get_loc(color)]
                        
                        labx = f'{labels[0]} {popm} ({len(ipopx)})'
                        laby = f'{labels[1]} {popm} ({len(ipopy)})'

                    properties['x']  = {'index':ipopx,
                                        'plot_args':{'marker':m,'c':cx},
                                        'mean_plot_args':{'marker':mean_m,'c':cx_,'label':labx}}
                    properties['y']  = {'index':ipopy,
                                        'plot_args':{'marker':m,'c':cy},
                                        'mean_plot_args':{'marker':mean_m,'c':cy_,'label':laby}}

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
                                    if outliers_in_obs is not None:
                                        outliers = self.obs[self.obs[outliers_in_obs]].index
                                        ipopx = ipopx[~ipopx.isin(outliers)]
                                        ipopy = ipopy[~ipopy.isin(outliers)]
                                    labx = f'{labels[0]} {popc} {popm} ({len(ipopx)})'
                                    laby = f'{labels[1]} {popc} {popm} ({len(ipopy)})'

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
                                    
                                    if outliers_in_obs is not None:
                                        outliers = self.obs[self.obs[outliers_in_obs]].index
                                        ipop = ipop[~ipop.isin(outliers)]
                                    
                                    lab = f'{popc} {popm} ({len(ipop)})'
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
                                            
                                if outliers_in_obs is not None:
                                    outliers = self.obs[self.obs[outliers_in_obs]].index
                                    ipopx = ipopx[~ipopx.isin(outliers)]
                                    ipopy = ipopy[~ipopy.isin(outliers)]

                                cx = self.obs.loc[self.obs.index.isin(ipopx)][color]
                                cy = self.obs.loc[self.obs.index.isin(ipopy)][color]

                                labx = f'{labels[0]} {popm} ({len(ipopx)})'
                                laby = f'{labels[1]} {popm} ({len(ipopy)})'
                                            
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
                                
                                if outliers_in_obs is not None:
                                    outliers = self.obs[self.obs[outliers_in_obs]].index
                                    ipop = ipop[~ipop.isin(outliers)]

                                c = self.obs.loc[self.obs.index.isin(ipop)][color]
                                
                                lab = f'{popm} ({len(ipop)})'

                                properties[f'{popm}'] = {'index':ipop,
                                                        'plot_args':{'c':c,'marker':m,},
                                                        'mean_plot_args':{'marker':mean_m,'label':lab}}
                                
                                
            else:
                print(f"{marker} is not in self.obs or is not categorical, \
                use self.obs['{marker}'] = self.obs['{marker}'].astype('category')")
            
        else:

                print(f'{color} and {marker} not found in obs and variables')

        return(properties)

    def scatter_proj(self,projection,xproj='proj_kfda',yproj=None,xname=None,yname=None,
                    highlight=None,color=None,marker=None,show_conditions=True,labels='CT',text=False,fig=None,ax=None,):
        if fig is None:
            fig,ax = plt.subplots(ncols=1,figsize=(12,6))

        p1,p2 = projection
        yproj = xproj if yproj is None else yproj
        if xproj == yproj and yname is None:
            yname = xname
        df_abscisse = self.init_df_proj(xproj,xname,)
        df_ordonnee = self.init_df_proj(yproj,yname,)
        properties = self.get_plot_properties(marker=marker,color=color,labels=labels,show_conditions=show_conditions,)
        texts = []
        
        for kprop,vprop in properties.items():
    #         print(kprop,vprop['mean_plot_args'].keys())
            if len(vprop['index'])>0:
                x_ = df_abscisse.loc[df_abscisse.index.isin(vprop['index'])][f'{p1}']
                y_ = df_ordonnee.loc[df_ordonnee.index.isin(vprop['index'])][f'{p2}']
                            
                alpha = .2 if text else .8 
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

        name_pval = self.get_kfdat_name()
        xlabel = xproj if xproj in self.data[self.data_name]['variables'] else xproj.split(sep='_')[1]+f': t={p1}'
        # xlabel += f'  pval={self.df_pval[xname].loc[p1]:.3e}' if xproj == 'proj_kfda' else \
        #         f'  pval={self.df_pval_contributions[xname].loc[p1]:.3e}' if xproj == 'proj_kfda' else \
        #         ''
        xlabel += f'  pval={self.df_pval[name_pval].loc[p1]:.3e}' if xproj == 'proj_kfda' else \
                f'  pval={self.df_pval_contributions[name_pval].loc[p1]:.3e}' if xproj == 'proj_kfda' else \
                ''

        ylabel = yproj if yproj in self.data[self.data_name]['variables'] else yproj.split(sep='_')[1]+f': t={p2}'
        # ylabel += f'  pval={self.df_pval[yname].loc[p2]:.3e}' if yproj == 'proj_kfda' else \
        #         f'  pval={self.df_pval_contributions[yname].loc[p2]:.3e}' if yproj == 'proj_kfda' else \
        #         ''
        ylabel += f'  pval={self.df_pval[name_pval].loc[p2]:.3e}' if yproj == 'proj_kfda' else \
                f'  pval={self.df_pval_contributions[name_pval].loc[p2]:.3e}' if yproj == 'proj_kfda' else \
                ''

        ax.set_xlabel(xlabel,fontsize=25)                    
        ax.set_ylabel(ylabel,fontsize=25)
        
        ax.legend()

        return(fig,ax)      



    # désuet ? 
    def init_axes_projs(self,fig,axes,projections,sample,suptitle,kfda,kfda_ylim,t,kfda_title,spectrum,spectrum_label):
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
            self.plot_spectrum(fig=fig,ax=axes[0],t=t,title='spectrum',sample=sample,label=spectrum_label)
            axes = axes[1:]
        return(fig,axes)

    def density_projs(self,fig=None,axes=None,which='proj_kfda',sample='xy',name=None,projections=range(1,10),suptitle=None,kfda=False,kfda_ylim=None,t=None,kfda_title=None,spectrum=False,spectrum_label=None,labels='CT'):
        fig,axes = self.init_axes_projs(fig=fig,axes=axes,projections=projections,sample=sample,suptitle=suptitle,kfda=kfda,
                                        kfda_ylim=kfda_ylim,t=t,kfda_title=kfda_title,spectrum=spectrum,spectrum_label=spectrum_label)
        if not isinstance(axes,np.ndarray):
            axes = [axes]
        for ax,proj in zip(axes,projections):
            self.density_proj(t=proj,which=which,name=name,labels=labels,sample=sample,fig=fig,ax=ax)
        fig.tight_layout()
        return(fig,axes)

    def scatter_projs(self,fig=None,axes=None,xproj='proj_kfda',sample='xy',yproj=None,xname=None,yname=None,projections=[(1,i+2) for i in range(10)],suptitle=None,
                        highlight=None,color=None,kfda=False,kfda_ylim=None,t=None,kfda_title=None,spectrum=False,spectrum_label=None,iterate_over='projections',labels='CT'):
        to_iterate = projections if iterate_over == 'projections' else color
        fig,axes = self.init_axes_projs(fig=fig,axes=axes,projections=to_iterate,sample=sample,suptitle=suptitle,kfda=kfda,
                                        kfda_ylim=kfda_ylim,t=t,kfda_title=kfda_title,spectrum=spectrum,spectrum_label=spectrum_label)
        if not isinstance(axes,np.ndarray):
            axes = [axes]
        for ax,obj in zip(axes,to_iterate):
            if iterate_over == 'projections':
                self.scatter_proj(ax,obj,xproj=xproj,yproj=yproj,xname=xname,yname=yname,highlight=highlight,color=color,labels=labels,sample=sample)
            elif iterate_over == 'color':
                self.scatter_proj(ax,projections,xproj=xproj,yproj=yproj,xname=xname,yname=yname,highlight=highlight,color=obj,labels=labels,sample=sample)
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


