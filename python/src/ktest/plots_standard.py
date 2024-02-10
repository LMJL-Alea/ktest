import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
from .kernel_statistics import Statistics
from .utils_plot import init_plot_kfdat,init_plot_pvalue,text_truncations_of_interest
from .utils_matplotlib import custom_histogram,highlight_on_histogram
from .kernel_statistics import Statistics


# from functions import get_between_covariance_projection_error

from adjustText import adjust_text
from scipy.stats import chi2
import numpy as np
import torch
from torch import mv,dot,sum,cat,tensor,float64




class Plot_Standard(Statistics):

    # def __init__(self,data,obs=None,var=None,):
    #     super(Plot_Standard,self).__init__(data,obs=obs,var=var,)



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
            columns = self.df_pval.columns
        for c in columns:
            pval = self.get_pvalue(name=c,contrib=contrib,log=log).loc[:t]
            ax.plot(pval,label=c)
        
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
        label = proj if proj in self.variables else \
                 t if proj=='obs' else \
                 f'DA{t}' if proj == 'proj_kfda' else \
                 f'PC{t}' if proj == 'proj_kpca' else \
                 f'R{t}'

        if proj == 'proj_kfda':
            pval = self.df_pval[self.get_kfdat_name()].loc[t]
            label += f' pval={pval:.1e}' if pval<.01 else f' pval={pval:1.2f}'
                    
        if proj in ['proj_kfda','proj_kpca']:
            sp,ev = self.get_spev('covw')
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
                    alpha=.5,
                    fig=None,ax=None,
                    show_conditions=True,
                    legend=True,
                    legend_fontsize=15,
                    lw=2,
                    condition=None,
                    samples=None,
                    highlight=None,
                    highlight_color=None,
                    highlight_linewidth=3,
                    highlight_label=None,
                    highlight_marker='*',
                    samples_colors=None,
                    marked_obs_to_ignore=None,
                    hist_type='kde',
                    kde_bw=.2,
                    xshift=0,
                    yshift=0,
                    means=True,
                    normalize=False,
                    non_zero_only=False
                    ):

        # labels = self.get_samples_list(condition=condition,samples=samples,marked_obs_to_ignore=marked_obs_to_ignore)
        if fig is None:
            fig,ax = plt.subplots(ncols=1,figsize=(12,6))

        properties = self.get_plot_properties(
                        color=color,
                        # labels=labels,
                        show_conditions=show_conditions,
                        condition=condition,
                        samples=samples,
                        color_list=samples_colors,
                        marked_obs_to_ignore=marked_obs_to_ignore)

        df_proj= self.init_df_proj(proj,name)
        df_proj = df_proj[proj] if proj in df_proj else df_proj[str(t)]
        if non_zero_only:
            df_proj = df_proj[df_proj!=0]
        min,max = df_proj.min(),df_proj.max()
        # quand beaucoup de plot se chevauchent, ça serait sympa de les afficher en 3D pour mieux les voir 
        
        for kprop,vprop in properties.items():
            # print(kprop,vprop)
            if len(vprop['index'])>0:
                dfxy = df_proj.loc[df_proj.index.isin(vprop['index'])]
                # dfxy = dfxy[proj] if proj in dfxy else dfxy[str(t)]                        
                
                custom_histogram(dfxy,
                                fig=fig,
                                ax=ax,
                                orientation=orientation,
                                alpha=alpha,
                                label=vprop['hist_args']['label'] if legend else None,
                                color=vprop['hist_args']['color'],
                                hist_type=hist_type,
                                lw=lw,
                                kde_bw=kde_bw,
                                minmax = [min,max],
                                xshift=xshift,
                                yshift=yshift,
                                means=means,
                                normalize=normalize
                                )
                

                # si je voulais faire des mean qui correspondent à d'autre pop que celles des histogrammes,
                # la solution la plus simple serait de faire une fonction spécifique 'plot_mean_hist' par ex
                # dédiée au tracé des lignes verticales correspondant aux means que j'appelerais séparément. 

        
        for kprop,vprop in properties.items():
            if len(vprop['index'])>0:
                if highlight is not None and type(highlight)==str:
                    if highlight in self.obs and self.obs[highlight].dtype == bool:
                        highlight = self.obs[self.obs[highlight]].index

                if highlight is not None and vprop['index'].isin(highlight).sum()>0:
                    
                    c = highlight_color if highlight_color is not None else \
                            vprop['plot_args']['color'] if 'color' in vprop['plot_args'] else \
                            vprop['mean_plot_args']['color'] if 'mean_plot_args' in vprop and 'color' in vprop['mean_plot_args'] \
                            else 'xkcd:cyan'
                    dfxy = df_proj.loc[df_proj.index.isin(vprop['index'])]
                    # dfxy = dfxy[proj] if proj in dfxy else dfxy[str(t)]                        
                    ihighlight = vprop['index'][vprop['index'].isin(highlight)]
                    data_highlight = dfxy.loc[dfxy.index.isin(ihighlight)] 
                    
                    highlight_on_histogram(data=data_highlight,
                                            fig=fig,
                                            ax=ax,
                                            orientation=orientation,
                                            label=highlight_label,
                                            color=c,
                                            marker=highlight_marker,
                                            coef_bins=3,
                                            linewidths=highlight_linewidth,
                                            means=True)


        if orientation == 'vertical':
            ax.set_xlabel(self.get_axis_label(proj,t),fontsize=25)
        else:
            ax.set_ylabel(self.get_axis_label(proj,t),fontsize=25)

        if legend:
            ax.legend(fontsize=legend_fontsize)

        # fig.tight_layout()
        return(fig,ax)
        

    def scatter_proj(self,
                     projection,
                     xproj='proj_kpca',
                     yproj=None,
                     xname=None,
                     yname=None,
                    highlight=None,
                    color=None,
                    marker=None,
                    show_conditions=True,
                    text=False,
                    fig=None,
                    ax=None,
                    alpha=.8,
                    legend=True,
                    legend_fontsize=15,
                    condition=None,
                    samples=None,
                    samples_colors=None,
                    samples_markers=None,
                    marked_obs_to_ignore=None,):


        # labels = self.get_samples_list(condition=condition,samples=samples,marked_obs_to_ignore=marked_obs_to_ignore)
        if fig is None:
            fig,ax = plt.subplots(ncols=1,figsize=(12,6))
        
        self.projections(t=np.max([p for p in projection if isinstance(p,int)]),condition=condition,samples=samples,
                        marked_obs_to_ignore=marked_obs_to_ignore)

        p1,p2 = projection
        yproj = xproj if yproj is None else yproj
        if xproj == yproj and yname is None:
            yname = xname
        # print(f'xproj={xproj} xname={xname}  yproj={yproj} yname={yname}')
        df_abscisse = self.init_df_proj(xproj,xname,)
        df_ordonnee = self.init_df_proj(yproj,yname,)
        properties = self.get_plot_properties(
                marker=marker,
                color=color,
                # labels=labels,
                show_conditions=show_conditions,
                condition=condition,
                samples=samples,
                color_list=samples_colors,
                marker_list=samples_markers,
                marked_obs_to_ignore=marked_obs_to_ignore,
                legend=legend)
        
        texts = []
        
        for kprop,vprop in properties.items():
    #         print(kprop,vprop['mean_plot_args'].keys())
            if len(vprop['index'])>0:
                x_ = df_abscisse.loc[df_abscisse.index.isin(vprop['index'])]
                y_ = df_ordonnee.loc[df_ordonnee.index.isin(vprop['index'])]

                x_ = x_[p1] if p1 in x_ else x_[str(p1)]                        
                y_ = y_[p2] if p2 in y_ else y_[str(p2)]                        
                
                ax.scatter(x_,y_,s=30,alpha=alpha,**vprop['plot_args'])
                if 'mean_plot_args' in vprop and 'color' not in vprop['mean_plot_args']:
                    vprop['mean_plot_args']['color'] = ax._children[-1]._facecolors[0]
                    
        for kprop,vprop in properties.items():
            if len(vprop['index'])>0:
                x_ = df_abscisse.loc[df_abscisse.index.isin(vprop['index'])]
                y_ = df_ordonnee.loc[df_ordonnee.index.isin(vprop['index'])]
                
                x_ = x_[p1] if p1 in x_ else x_[str(p1)]                        
                y_ = y_[p2] if p2 in y_ else y_[str(p2)]                        
                
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
                    ax.scatter(xhighlight_.mean(),yhighlight_.mean(),color=c,s=200,marker='*',edgecolor='black',linewidths=3,label=f'({len(ihighlight)})')    
                    
                if 'mean_plot_args' in vprop:
                    mx_ = x_.mean()
                    my_ = y_.mean()
                    ax.scatter(mx_,my_,edgecolor='black',linewidths=1.5,s=300,**vprop['mean_plot_args'],alpha=1)

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
                    alpha=.5,
                    legend_fontsize=15,
                    condition=None,
                    samples=None,
                    highlight=None,
                    highlight_color=None,
                    highlight_linewidth=3,
                    highlight_label=None,
                    highlight_marker='*',
                    samples_colors=None,
                    marked_obs_to_ignore=None,
                    hist_type='kde',
                    kde_bw=.2,
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
                        alpha=alpha,
                        fig=fig,ax=ax,
                        show_conditions=show_conditions,
                        legend_fontsize=legend_fontsize,
                        highlight=highlight,
                        highlight_color=highlight_color,
                        highlight_linewidth=highlight_linewidth,
                        highlight_label=highlight_label,
                        highlight_marker=highlight_marker,
                        condition=condition,
                        samples=samples,
                        samples_colors=samples_colors,
                        marked_obs_to_ignore=marked_obs_to_ignore,
                        hist_type=hist_type,
                        kde_bw=kde_bw,
                        )
        return(fig,ax)


    def hist_mmd_discriminant(self,
                              color=None,
                              fig=None,
                              ax=None,
                              show_conditions=True,
                              orientation='vertical',
                              highlight=None,
                    highlight_label=None,
                    alpha=.5,
                              legend_fontsize=15):
        mmd_name = self.get_mmd_name()
        fig,ax = self.density_proj(t='mmd',
                                   proj='proj_mmd',
                                   name=mmd_name,
                                   orientation=orientation,
                                   color=color,
                                   alpha=alpha,
                                   fig=fig,
                                   ax=ax,
                                   show_conditions=show_conditions,
                                   legend_fontsize=legend_fontsize,
                                   highlight=highlight,
                                    highlight_label=highlight_label,
                                   )
        return(fig,ax)

    def hist_pc(self,
                t,
                color=None,
                fig=None,
                ax=None,
                show_conditions=True,
                orientation='vertical',
                alpha=.5,
                legend_fontsize=15,
                condition=None,
                samples=None,
                samples_colors=None,
                highlight=None,
                highlight_color=None,
                highlight_marker='*',
                highlight_linewidth=3,
                highlight_label=None,
                marked_obs_to_ignore=None,
                hist_type='kde',
                kde_bw=.2,
                verbose=0):
        
        self.projections(t=t,condition=condition,samples=samples,
                        marked_obs_to_ignore=marked_obs_to_ignore,verbose=verbose)

        kfdat_name = self.get_kfdat_name(
                        condition=condition,
                        samples=samples,
                        marked_obs_to_ignore=marked_obs_to_ignore)
        
        fig,ax = self.density_proj(
                        t,
                        proj='proj_kpca',
                        name=kfdat_name,
                        orientation=orientation,
                        color=color,
                        alpha=alpha,
                        fig=fig,ax=ax,
                        show_conditions=show_conditions,
                        legend_fontsize=legend_fontsize,
                        highlight=highlight,
                        highlight_color=highlight_color,
                        highlight_linewidth=highlight_linewidth,
                        highlight_label=highlight_label,
                        highlight_marker=highlight_marker,
                        condition=condition,
                        samples=samples,
                        samples_colors=samples_colors,
                        marked_obs_to_ignore=marked_obs_to_ignore,
                        hist_type=hist_type,
                        kde_bw=kde_bw,
                        )
        return(fig,ax)



    def plot_nextPC(self,
                    t,
                    fig=None,
                    ax=None,
                    color=None,
                    marker=None,
                    highlight=None,
                    show_conditions=True,
                    legend_fontsize=15,
                    condition=None,
                    samples=None,
                    samples_colors=None,
                    samples_markers=None,
                    marked_obs_to_ignore=None,
                    verbose=0):

        self.projections(t=t+1,condition=condition,samples=samples,
                        marked_obs_to_ignore=marked_obs_to_ignore,verbose=verbose)

        kfdat_name = self.get_kfdat_name(
                        condition=condition,
                        samples=samples,
                        marked_obs_to_ignore=marked_obs_to_ignore)

        fig,ax = self.scatter_proj(projection=[t,t+1],
                               xproj='proj_kfda',
                               yproj='proj_kpca',
                               xname=kfdat_name,
                               yname=kfdat_name,
                               color=color,
                               marker=marker,
                               highlight=highlight,
                               show_conditions=show_conditions,
                               fig=fig,
                               ax=ax,
                               legend_fontsize=legend_fontsize,
                               condition=condition,
                               samples=samples,
                               samples_colors=samples_colors,
                               samples_markers=samples_markers,
                               marked_obs_to_ignore=marked_obs_to_ignore)
        
        ax.set_title(f'D={t} PC={t+1}',fontsize=30)
        return(fig,ax)

    def plot_nextDA(self,
                    t,
                    fig=None,
                    ax=None,
                    color=None,
                    marker=None,
                    highlight=None,
                    show_conditions=True,
                    legend_fontsize=15,
                    condition=None,
                    samples=None,
                    samples_colors=None,
                    samples_markers=None,
                    marked_obs_to_ignore=None,
                    verbose=0):


        self.projections(t=t+1,condition=condition,samples=samples,
                        marked_obs_to_ignore=marked_obs_to_ignore,verbose=verbose)

        kfdat_name = self.get_kfdat_name(
                        condition=condition,
                        samples=samples,
                        marked_obs_to_ignore=marked_obs_to_ignore)

        fig,ax = self.scatter_proj(projection=[t,t+1],
                               xproj='proj_kfda',
                               yproj='proj_kfda',
                               xname=kfdat_name,
                               yname=kfdat_name,
                               color=color,
                               marker=marker,
                               highlight=highlight,
                               show_conditions=show_conditions,
                               fig=fig,
                               ax=ax,
                               legend_fontsize=legend_fontsize,
                               condition=condition,
                               samples=samples,
                               samples_colors=samples_colors,
                               samples_markers=samples_markers,
                               marked_obs_to_ignore=marked_obs_to_ignore)

        ax.set_title(f'D={t} D={t+1}',fontsize=30)
        return(fig,ax)


    def plot_orthogonal(self,
                        t=1,
                        center='w',
                        fig=None,
                        ax=None,
                        color=None,
                        marker=None,
                        highlight=None,
                        show_conditions=True,
                        legend_fontsize=15,
                        condition=None,
                        samples=None,
                        samples_colors=None,
                        samples_markers=None,
                        marked_obs_to_ignore=None,
                        verbose=0):

        self.projections(t=t,condition=condition,samples=samples,
                        marked_obs_to_ignore=marked_obs_to_ignore,verbose=verbose)
        self.orthogonal(t=t,center=center)

        kfdat_name = self.get_kfdat_name(
                        condition=condition,
                        samples=samples,
                        marked_obs_to_ignore=marked_obs_to_ignore)
        orthogonal_name = self.get_orthogonal_name(t=t,center=center)
        

        fig,ax = self.scatter_proj(projection=[t,1],
                        xproj='proj_kfda',
                        yproj='proj_orthogonal',
                        xname=kfdat_name,
                        yname=orthogonal_name,
                        color=color,
                        marker=marker,
                        highlight=highlight,
                        show_conditions=show_conditions,
                        fig=fig,
                        ax=ax,
                        legend_fontsize=legend_fontsize,
                        condition=condition,
                        samples=samples,
                        samples_colors=samples_colors,
                        samples_markers=samples_markers,
                        marked_obs_to_ignore=marked_obs_to_ignore)
        
        ax.set_title(f'discriminant and orthogonal axis t={t}',fontsize=30)
        return(fig,ax)

    def plot_kpca(self,
                    t=1,
                    fig=None,
                    ax=None,
                    color=None,
                    marker=None,
                    highlight=None,
                    show_conditions=True,
                    legend_fontsize=15,
                    condition=None,
                    samples=None,
                    samples_colors=None,
                    samples_markers=None,
                    marked_obs_to_ignore=None,
                    verbose=0):
        


        self.projections(t=t+1,condition=condition,samples=samples,
                        marked_obs_to_ignore=marked_obs_to_ignore,verbose=verbose)
        
        kfdat_name = self.get_kfdat_name(
                        condition=condition,
                        samples=samples,
                        marked_obs_to_ignore=marked_obs_to_ignore)
        

        fig,ax = self.scatter_proj(projection=[t,t+1],
                        xproj='proj_kpca',
                        yproj='proj_kpca',
                        xname=kfdat_name,
                        yname=kfdat_name,
                        color=color,
                        marker=marker,
                        highlight=highlight,
                        show_conditions=show_conditions,
                        fig=fig,
                        ax=ax,
                        legend_fontsize=legend_fontsize,
                        condition=condition,
                        samples=samples,
                        samples_colors=samples_colors,
                        samples_markers=samples_markers,
                        marked_obs_to_ignore=marked_obs_to_ignore)
        
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
    

      
    def get_plot_properties(self,
                    marker=None,
                    color=None,
                    show_conditions=True,
                    legend=True,
                    condition=None,
                    samples=None,
                    marked_obs_to_ignore=None,
                    marker_list = None,
                    big_marker_list = None,
                    color_list = None,
                    coef_bins = 3,
                    verbose=0):            
            

        properties = {}
        
        index_to_plot = self.get_indexes_to_plot(marker=marker,color=color,show_conditions=show_conditions,
                        condition=condition,samples=samples,marked_obs_to_ignore=marked_obs_to_ignore)

        marker_dict,big_marker_dict = self.get_marker_to_plot(marker=marker,
                    show_conditions=show_conditions,
                    condition=condition,samples=samples,
                    marker_list = marker_list,big_marker_list = big_marker_list,)
        color_dict,mean_color_dict = self.get_color_to_plot(color=color,
                    show_conditions=show_conditions,
                    condition=condition,
                    samples=samples,
                    marked_obs_to_ignore=marked_obs_to_ignore,
                    color_list = color_list)
        
        
        # print('marker:',marker_dict.keys())
        # print('color:',color_dict.keys())
        # print('index:',index_to_plot.keys())
        for k in index_to_plot.keys():
            pop_index = index_to_plot[k]['index']
            popc = index_to_plot[k]['popc']
            popm = index_to_plot[k]['popm']
            
            # print(f'k:{k} popc:{popc} popm:{popm}')
            
            c = color_dict[popc]
            if isinstance(c,pd.DataFrame):
                c = c.loc[pop_index,color]
            cm = mean_color_dict[popc]
            m = marker_dict[popm]
            bm = big_marker_dict[popm]

            n = len(pop_index)
            lab = f'{k} ({n})' if legend else None
            bins = coef_bins*int(np.floor(np.sqrt(len(pop_index))))

            properties[k] = {'index':pop_index,
                                'plot_args':{'marker':m,'c':c},
                                'mean_plot_args':{'marker':bm,'label':lab,'color':cm},
                                'hist_args':{'bins':bins,'label':lab,'color':c}}
        return(properties)

    def get_indexes_to_plot(self,
                        marker=None,
                        color=None,
                        show_conditions=True,
                        condition=None,
                        samples=None,
                        marked_obs_to_ignore=None):

                
        index = self.get_index(condition=condition,samples=samples,marked_obs_to_ignore=marked_obs_to_ignore)
        all_index = self.get_index(condition=condition,samples=samples,marked_obs_to_ignore=marked_obs_to_ignore,in_dict=False)
        index_to_plot = {}
        
        if marker is None and color is None : 
            if show_conditions:
                for pop,pop_index in index.items():
                    index_to_plot[pop] = {'index':pop_index,'popm':pop,'popc':pop}
            else:
                index_to_plot['pop'] = {'index':all_index,'popm':'popm','popc':'popc'}

        elif isinstance(color,str) and marker is None:
            if color in list(self.variables):
                if show_conditions:
                    for pop,pop_index in index.items():
                        index_to_plot[pop] = {'index':pop_index,'popm':pop,'popc':pop}
                else:
                    index_to_plot['pop'] = {'index':all_index,'popm':'popm','popc':'popc'}

            elif color in list(self.obs.columns):
                if self.obs[color].dtype == 'category':
                    for popc in self.obs[color].cat.categories:
                        popc_index = self.obs.loc[self.obs[color]==popc].index
                        
                        if show_conditions: 
                            for popm,popm_index in index.items():
                                pop_index = popm_index[popm_index.isin(popc_index)]
                                index_to_plot[f'{popc} {popm}'] = {'index':pop_index,
                                                               'popm':popm,
                                                               'popc':popc}
                        else:
                            pop_index = all_index[all_index.isin(popc_index)]                            
                            index_to_plot[popc] = {'index':pop_index,
                                            'popm':'popm',
                                            'popc':popc}
                            

                else: # pour une info numérique 
                    if show_conditions:
                        for pop,pop_index in index.items():
                            index_to_plot[pop] = {'index':pop_index,'popm':pop,'popc':pop}
                    else:
                        index_to_plot['pop'] = {'index':all_index,'popm':'popm','popc':'popc'}


        elif color is None and isinstance(marker,str):
            print('color is None and marker is specified : this case is not treated yet')

        elif isinstance(color,str) and isinstance(marker,str):
            if marker in list(self.obs.columns) and self.obs[marker].dtype == 'category':
                if color in list(self.variables):                    
                    for popm in self.obs[marker].cat.categories:  
                        popm_index = self.obs.loc[self.obs[marker]==popm].index
                        if show_conditions: 
                            for popc in index.keys():
                                pop_index = popm_index[popm_index.isin(index[popc])]
                                index_to_plot[f'{popc} {popm}'] = {'index':pop_index,
                                                               'popm':popm,
                                                               'popc':popc}
                        else:
                            pop_index = all_index[all_index.isin(popm_index)]   
                            index_to_plot[popm] = {'index':pop_index,
                                                    'popm':popm,
                                                    'popc':'popc'}

                elif color in list(self.obs.columns):
                    if self.obs[color].dtype == 'category':
                        for popc in self.obs[color].cat.categories:
                            popc_index = self.obs[self.obs[color]==popc].index
                            for popm in self.obs[marker].cat.categories:
                                popm_index = self.obs[self.obs[marker]==popm].index
                                pop_index = popm_index[popm_index.isin(popc_index)]

                                if show_conditions: 
                                    for k in index.keys():
                                        pop_index = pop_index[pop_index.isin(index[k])]
                                        index_to_plot[f'{k} {popc} {popm}'] = {'index':pop_index,
                                                    'popm':popm,
                                                    'popc':popc}
                                else:
                                    pop_index = all_index[all_index.isin(pop_index)]   
                                    index_to_plot[f'{popc} {popm}'] = {'index':pop_index,
                                                    'popm':popm,
                                                    'popc':popc}
                                    
                    else: # pour une info numérique 
                        for popm in self.obs[marker].cat.categories:
                            popm_index = self.obs.loc[self.obs[marker]==popm].index
                            if show_conditions: 
                                for popc in index.keys():
                                    pop_index = popm_index[popm_index.isin(index[popc])]
                                    index_to_plot[f'{popc} {popm}'] = {'index':pop_index,
                                                    'popm':popm,
                                                    'popc':popc}
                            else:
                                pop_index = all_index[all_index.isin(popm_index)]
                                index_to_plot[popm] = {'index':pop_index,
                                                    'popm':popm,
                                                    'popc':'popc'}

            else:
                print(f"{marker} is not in self.obs or is not categorical, \
                use self.obs['{marker}'] = self.obs['{marker}'].astype('category')")
        else:
                print(f'{color} and {marker} not found in obs and variables')
        return(index_to_plot)
    
    def get_marker_to_plot(self,
                    marker=None,
                    show_conditions=True,
                    condition=None,
                    samples=None,
                    marker_list = None,
                    big_marker_list = None,
                        #color_list,marker_list,big_marker_list,show_conditions
                        ):

        if marker_list is None:
            marker_list = ['.','x','+','d','1','*',(4,1,0),(4,1,45),(7,1,0),(20,1,0),'s']
        if big_marker_list is None:
            big_marker_list = ['o','X','P','D','v','*',(4,1,0),(4,1,45),(7,1,0),(20,1,0),'s']
        
        nm = len(marker_list)
        samples_list = self.get_samples_list(condition,samples) 
        
        marker_dict = {}
        big_marker_dict = {}

        if marker is None and not show_conditions:
            marker_dict['popm'] = '.'
            big_marker_dict['popm'] = 'o'
        else:
            if marker is None:
                popm_list = samples_list
            elif marker in list(self.obs.columns) and self.obs[marker].dtype == 'category':
                popm_list = self.obs[marker].cat.categories
            else: 
                print(f'marker {marker} not recognized')
            
            for i,popm in enumerate(popm_list):

                    m = marker_list[i%nm]
                    bm = big_marker_list[i%nm]
                    
                    marker_dict[popm] = m
                    big_marker_dict[popm] = bm

        return(marker_dict,big_marker_dict)

    def get_color_to_plot(self,
                    color=None,
                    show_conditions=True,
                    condition=None,
                    samples=None,
                    marked_obs_to_ignore=None,
                    color_list = None):
        if isinstance(color_list,dict):
            color_dict = color_list
            mean_color_dict=color_list

        else:
            if color_list is None:
                color_list = ['xkcd:cerulean','xkcd:light orange',
            'xkcd:grass green','xkcd:cerise','xkcd:mocha','xkcd:greeny blue',
            'xkcd:vibrant blue','xkcd:candy pink','xkcd:lavender','xkcd:pale green',
            'xkcd:peach','xkcd:goldenrod','xkcd:mahogany','xkcd:terra cotta',
            'xkcd:acid green','xkcd:teal blue','xkcd:dusty pink','xkcd:pinky red']
                
            nc = len(color_list)
            color_dict={}
            mean_color_dict={}
            variables = self.variables
            
            samples_list = self.get_samples_list(condition,samples) 
            
            
            
            if color is None : 
                if show_conditions:
                    for i,popc in enumerate(samples_list):
                        color_dict[popc] = color_list[i%nc]
                        mean_color_dict[popc] = color_list[i%nc]
                else:
                    color_dict['popc'] = color_list[0]
                    mean_color_dict['popc'] = color_list[0]

            elif isinstance(color,str):
                if color in list(variables):
                    data = self.get_data(condition=condition,
                                    samples=samples,
                                    marked_obs_to_ignore=marked_obs_to_ignore,
                                    in_dict=False,
                                    dataframe=True)

                    if show_conditions:
                        for i,popc in enumerate(samples_list):
                            color_dict[popc] = data[color].to_frame()
                            mean_color_dict[popc] = color_list[i%nc]
                    else:                    
                        color_dict['popc'] = data[color].to_frame()
                        mean_color_dict['popc'] = color_list[0]

                elif color in list(self.obs.columns):
                    # print("color in list(variables)")
                    if self.obs[color].dtype == 'category':
                        for i,popc in enumerate(self.obs[color].cat.categories):
                            color_dict[popc] = color_list[i%nc]
                            mean_color_dict[popc] = color_list[i%nc]                        

                    else: # pour une info numérique 
                        # print(f'{color} is not categorical in obs')
                        if show_conditions:
                            for i,popc in enumerate(samples_list):
                                color_dict[popc] = self.obs[color].to_frame()
                                mean_color_dict[popc] = color_list[i%nc]  
                        else:
                            color_dict['popc'] = self.obs[color].to_frame()
                            mean_color_dict['popc'] = color_list[i%nc]  
            else:
                    print(f'{color} not found in obs and variables')
        return(color_dict,mean_color_dict)
