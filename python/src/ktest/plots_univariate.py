import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plots_summarized import Plot_Summarized
from .univariate_testing import Univariate

from .utils_plot import init_plot_pvalue,text_truncations_of_interest
from .utils_univariate import filter_genes_wrt_pval


class Plot_Univariate(Plot_Summarized,Univariate):

    def visualize_univariate_test_CRCL(self,variable,vtest,column,patient=True,data_name='data',):

        fig,axes = plt.subplots(ncols=3,figsize=(22,7))
        
        ax = axes[0]
        self. _of_variable(variable,data_name=data_name,fig=fig,ax=ax)
        
        if patient:
            ax = axes[1]
            self.plot_density_of_variable(variable,data_name=data_name,fig=fig,ax=ax,color='patient')
            ax = axes[2]
        else:
            ax = axes[1]    
            self.plot_density_of_variable(variable,data_name='counts',fig=fig,ax=ax)
            ax = axes[2]    


        vtest.plot_pval_and_errors(column,fig=fig,ax=ax)
        ax.set_xlim(0,20)
        return(fig,axes)
                                 
    def plot_density_of_variable(self,
                                 variable,
                                 fig=None,
                                 ax=None,
                                 color=None,
                                 highlight=None,
                                 highlight_color=None,
                                 highlight_linewidth=3,
                                 highlight_label=None,
                                 samples_colors=None,
                                 alpha=.5,
                                 hist_type='kde',
                                 kde_bw=.2,
                                 show_conditions=True,
                                 legend=True,
                                 legend_fontsize=15,
                                 lw=2,
                                 condition=None,
                                 samples=None,
                                 marked_obs_to_ignore=None,
                                 xshift=0,
                                 yshift=0,
                                 orientation='vertical',
                                 normalize=False,
                                 means=True,
                                 non_zero_only=False):
        if fig is None:
            fig,ax =plt.subplots(figsize=(10,6))
            
        self.density_proj(t=0,
                          proj=variable,
                          fig=fig,
                          ax=ax,
                          color=color,
                          alpha=alpha,
                          highlight=highlight,
                          highlight_label=highlight_label,
                          highlight_color=highlight_color,
                          highlight_linewidth=highlight_linewidth,
                          show_conditions=show_conditions,
                          legend=legend,
                          legend_fontsize=legend_fontsize,
                          lw=lw,
                          hist_type=hist_type,kde_bw=kde_bw,
                            condition=condition,
                            samples=samples,
                            marked_obs_to_ignore=marked_obs_to_ignore,
                            samples_colors=samples_colors,
                            orientation=orientation,
                            xshift=xshift,
                            yshift=yshift,
                            normalize=normalize,
                            means=means,
                            non_zero_only=non_zero_only
                            )
        
        title = f'{variable}\n'
        zero_proportions = self.compute_zero_proportions_of_variable(variable)
        for c in zero_proportions.keys():
            if '_nz' in c:
                nz = zero_proportions[c]
                c_ = c.split(sep='_nz')[0]
                title += f' {c_}: {nz}z '
        
        ax.set_title(title,fontsize=20)
        return(fig,ax)

    def plot_density_of_variables_horizon(self,
                                        genes,
                                        fig=None,
                                        ax=None,
                                        kde_bw=.2,
                                        xshift_max=8,
                                        yshift_max=3,
                                        legend=True,
                                        means=True,
                                        lw=3,    
                                        color=None,
                                        highlight=None,
                                        highlight_color=None,
                                        highlight_linewidth=.5,
                                        highlight_label=None,
                                        samples_colors=None,
                                        alpha=.5,
                                        show_conditions=True,
                                        legend_fontsize=15,
                                        condition=None,
                                        samples=None,
                                        marked_obs_to_ignore=None,
                                        orientation='vertical',
                                        normalize=True,
                                        non_zero_only=False,
                                        ):
        if fig is None:
            fig,ax=plt.subplots(figsize=(7,7))

        ng = len(genes)
        
        yshifts = [i/ng*yshift_max for i in range(ng)]
        xshifts = [i/ng*xshift_max for i in range(ng)]
        ax.plot(xshifts,yshifts,alpha=.5,c='grey')
        ax.scatter(xshifts,yshifts,marker='|',s=100,color='grey')
        
        for i in range(ng)[::-1]:
            if legend :
                legend_=True if i==0 else False
            else:
                legend_=legend
            x= xshifts[i]
            y=yshifts[i]
            g=genes[i]
            
            ax.axhline(y,alpha=.5,c='grey')
                        
            self.plot_density_of_variable(variable=g,
                                        fig=fig,
                                        ax=ax,
                                        color=color,
                                        highlight=highlight,
                                        highlight_color=highlight_color,
                                        highlight_linewidth=highlight_linewidth,
                                        highlight_label=highlight_label,
                                        samples_colors=samples_colors,
                                        alpha=alpha,
                                        hist_type='kde',
                                        kde_bw=kde_bw,
                                        show_conditions=show_conditions,
                                        legend=legend_,
                                        legend_fontsize=legend_fontsize,
                                        lw=lw,
                                        condition=condition,
                                        samples=samples,
                                        marked_obs_to_ignore=marked_obs_to_ignore,
                                        xshift=x,
                                        yshift=y,
                                        orientation=orientation,
                                        means=means,
                                        normalize=normalize,
                                        non_zero_only=non_zero_only)  

        ax.set_yticks(yshifts)
        ax.set_yticklabels(genes,fontsize=10)

        
        return(fig,ax)
    

    def plot_density_of_variables_violin(self,
                                    variables,
                                    fig=None,
                                    ax=None,
                                    color=None,
                                    highlight=None,
                                    highlight_color=None,
                                    highlight_linewidth=3,
                                    highlight_label=None,
                                    samples_colors=None,
                                    alpha=.5,
                                    kde_bw=.2,
                                    show_conditions=True,
                                    legend=True,
                                    legend_fontsize=15,
                                    lw=2,
                                    condition=None,
                                    samples=None,
                                    marked_obs_to_ignore=None,
                                    orientation='horizontal',
                                    means=True,
                                    separation=2.2,
                                    non_zero_only=False):
        if fig is None:
            fig,ax = plt.subplots(figsize=(12,6))
        
        yshifts = [i*separation for i in range(len(variables))]
        for i,v in enumerate(variables):
            if legend and i != 0:
                legend=False
            self.plot_density_of_variable(variable=v,
                                        fig=fig,
                                    ax=ax,
                                    color=color,
                                    highlight=highlight,
                                    highlight_color=highlight_color,
                                    highlight_linewidth=highlight_linewidth,
                                    highlight_label=highlight_label,
                                    samples_colors=samples_colors,
                                    alpha=alpha,
                                    hist_type='violin',
                                    kde_bw=kde_bw,
                                    show_conditions=show_conditions,
                                    legend=legend,
                                    legend_fontsize=legend_fontsize,
                                    lw=lw,
                                    condition=condition,
                                    samples=samples,
                                    marked_obs_to_ignore=marked_obs_to_ignore,
                                    xshift=0,
                                    yshift=yshifts[i],
                                    orientation=orientation,
                                    means=means,
                                    non_zero_only=non_zero_only)  
        if orientation == 'vertical':
            ax.set_yticks(yshifts)
            ax.set_yticklabels(variables,rotation=0,fontsize=10)
        else:
            ax.set_xticks(yshifts)
            ax.set_xticklabels(variables,rotation=90,fontsize=10)
        return(fig,ax)


    def plot_density_of_variables(self,variables,fig=None,axes=None,color=None):
        if fig is None:
            ncols = len(variables)
            fig,axes = plt.subplots(ncols=ncols, figsize=(ncols*10,6))

        for variable,ax in zip(variables,axes):
            self.plot_density_of_variable(variable=variable,fig=fig,ax=ax,color=color)
        return(fig,axes)

    def scatter_2variables(self,v1,v2,fig=None,ax=None,color=None,marker=None,highlight=None,show_conditions=True,text=False,alpha=.8,legend=True,legend_fontsize=15):
        if fig is None:
            fig,ax = plt.subplots(figsize=(7,7))
        if v1 in self.get_variables() and v2 in self.get_variables():
            self.scatter_proj(projection=[v1,v2],xproj=v1,yproj=v2,fig=fig,ax=ax,
                        color=color,marker=marker,highlight=highlight,
                        show_conditions=show_conditions,text=text,alpha=alpha,
                        legend=legend,legend_fontsize=legend_fontsize)
        return(fig,ax)

    def scatter_variables(self,variables,fig=None,axes=None,color=None,marker=None,highlight=None,show_conditions=True,text=False,alpha=.8,legend=True,legend_fontsize=15):
        nv = len(variables)
        if nv == 2:
            v1,v2 = variables
            fig,axes = self.scatter_2variables(v1,v2,fig=fig,ax=ax,
                        color=color,marker=marker,highlight=highlight,
                        show_conditions=show_conditions,text=text,alpha=alpha,
                        legend=legend,legend_fontsize=legend_fontsize)        

        elif nv>2:
            if fig is None:
                fig,axes = plt.subplots(ncols=nv,nrows=nv,figsize=(7*nv,7*nv))
            for i,vi in enumerate(variables):
                for j,vj in enumerate(variables):
                    ax = axes[i,j]
                    if vi != vj and vi in self.get_variables() and vj in self.get_variables():
                        self.scatter_2variables(vi,vj,fig=fig,ax=ax,
                        color=color,marker=marker,highlight=highlight,
                        show_conditions=show_conditions,text=text,alpha=alpha,
                        legend=legend,legend_fontsize=legend_fontsize)        

        return(fig,axes)


    def plot_pval_and_errors_of_variable(self,variable,t=30,name=None,fig=None,ax=None,truncations_of_interest=[1,3,6],adjust=True,
                    pval=True,var=True,diff=True):
        if fig is None:
            fig,ax = plt.subplots(figsize=(12,6))
        
        fig,ax = init_plot_pvalue(fig=fig,ax=ax,t=t)
        
        if pval:
            self.plot_pvalue_of_variable(variable=variable,name=name,t=t,fig=fig,ax=ax,truncations_of_interest=truncations_of_interest,adjust=adjust,)
        
        if var:
            errW = [1]+[self.get_var()[self.get_column_name_in_var(t=trunc,name=name,output='errW')][variable] \
                for trunc in range(1,t)]
            ax.plot(range(t),errW,label='w-variability')
        
        if diff:
            errB = [1]+[self.get_var()[self.get_column_name_in_var(t=trunc,name=name,output='errB')][variable] \
                for trunc in range(1,t)]
            ax.plot(range(t),errB,label='difference')
        
        
        ax.legend(fontsize=20)
        

        ax.set_xlabel('Truncation',fontsize=30)
        ax.set_ylabel('Errors or pval',fontsize=30)
        ax.set_title(variable,fontsize=30)
        ax.set_xlim(-1,t+1)
        return(fig,ax)

    def plot_pvalue_of_variable(self,variable,name=None,t=30,fig=None,ax=None,truncations_of_interest=[1,3,5],adjust=True,color=None,ls=None,label=None,corrected=True):
        if fig is None:
            fig,ax = plt.subplots(figsize=(12,6))
        fig,ax = init_plot_pvalue(fig=fig,ax=ax,t=t)

        pval = [self.get_var()[self.get_column_name_in_var(t=trunc,name=name,output='pval',corrected=corrected)][variable] \
            for trunc in range(1,t)]
        label = f'{variable} p-value' if label is None else label
        ax.plot(range(1,t),pval,label=label,color=color,ls=ls)
        if truncations_of_interest is not None:
            text_truncations_of_interest(truncations_of_interest,ax,[0]+pval,adjust=adjust)
        ax.set_ylim(-.05,1.05)  
        ax.legend(fontsize=20)

    def plot_discriminant_of_expression_univariate_with_surroundings(self,variable,t,
                                                color=None,marker=None,highlight=None,
                                                samples_colors=None,
                                                previous_discriminant=False,
                                                hist_type='hist',
                                                kde_bw =.2,
                                                condition=None,
                                                samples=None,
                                                pval_t=None,
                                                figsize=(15,7.5),
                                                height_ratios=[1,2],
                                                width_ratios=[3,2]
                                                ):
        
        fig = plt.figure(figsize=figsize,constrained_layout=True)
        axd = fig.subplot_mosaic("AB\nCD",
                                 gridspec_kw=dict(height_ratios=height_ratios,
                                                  width_ratios=width_ratios),)
        
        
        # Pval and errors 
        if pval_t is not None: 
            self.plot_pval_and_errors(t=pval_t,
                                    fig=fig,
                                    ax=axd['B'],
                                    truncations_of_interest=[t],
                                    pval_aggregated=True,pval_contrib=False,
                                    var_within=False,var_conditions=False,
                                    kfdr=True,
                                    diff=False,grid=True,
                                    alpha=.8)
            axd['B'].legend(fontsize=15)

        else:
            axd['B'].set_axis_off()

        self.plot_density_of_variable(variable=variable,
                                      fig=fig,
                                      ax=axd['A'],
                                      color=color,
                                      highlight=highlight,
                                      hist_type=hist_type,
                                      kde_bw=kde_bw,
                                      samples_colors=samples_colors,
                                      condition=condition,
                                      samples=samples)

        self.plot_discriminant_of_expression_univariate(variable=variable,
                                                        t=t,
                                                        color=color,
                                                        marker=marker,
                                                        highlight=highlight,
                                                        previous_discriminant=previous_discriminant,
                                                        fig=fig,
                                                        ax=axd['C']
                                                        )
        self.hist_discriminant(t=t,fig=fig,ax=axd['D'],orientation='horizontal',
                                hist_type=hist_type,
                               kde_bw=kde_bw)
        
        axd['A'].legend([])
        axd['A'].set_xlabel('')
        axd['B'].set_xlabel('')
        axd['B'].set_ylabel('')
        axd['C'].set_ylabel('')
        axd['C'].set_title('')
        axd['D'].legend([])
        axd['D'].set_title(f'Discriminant t={t}',fontsize=30)
        axd['D'].set_ylabel('')
        axd['D'].sharey(axd['C'])
        #     axd['A'].set_title(f'{g} expression',fontsize=30)
        #     axd['B'].axhline(0,c='crimson',ls='--',alpha=.5,lw='2')
        #     axd['A'].legend(bbox_to_anchor=(.98,1.02),fontsize=30)
        # title = f'{variable} t={t}'
        # fig.suptitle(title,fontsize=30,y=1.02)
        fig.tight_layout()
        
        return(fig,axd)

    def plot_discriminant_of_expression_univariate(self,variable,t,
                                                color=None,marker=None,highlight=None,
                                                previous_discriminant=False,
                                                fig=None,ax=None,
                                                ):
        if fig is None:
            fig,ax = plt.subplots(figsize=(12,6))        

        self.scatter_proj(projection=[variable,t],
                          xproj=variable,
                          yproj='proj_kfda',
                          yname=self.get_kfdat_name(),
                          color=color,
                          marker=marker,
                          highlight=highlight,
                          fig=fig,ax=ax)
        if previous_discriminant and t>1:
            self.scatter_proj(projection=[variable,t-1],
                              xproj=variable,
                              yproj='proj_kfda',
                              yname=self.get_kfdat_name(),
                              color=color,
                              marker=marker,
                              highlight=highlight,
                              fig=fig,ax=ax,
                              alpha=.2,
                              legend=False)
        return(fig,ax)

    def plot_mmd_discriminant_of_expression_univariate(self,
                                                       variable,
                                                       color=None,
                                                       marker=None,
                                                       highlight=None,
                                                        highlight_label=None,
                                                ):

        fig = plt.figure(figsize=(15,7.5),constrained_layout=True)
        axd = fig.subplot_mosaic("AB\nCD",
                                 gridspec_kw=dict(height_ratios=[1, 2],
                                                  width_ratios=[3,2]),)
        yproj = 'proj_mmd' 

        # Pval and errors 
    #     self.plot_pval_and_errors(fig=fig,ax=axd['B'],truncations_of_interest=[trunc],adjust=False)

        self.density_proj(t=variable,
                          proj=variable,
                          fig=fig,
                          ax=axd['A'],
                          highlight=highlight,
                    highlight_label=highlight_label)
        
        self.scatter_proj(projection=[variable,'mmd'],xproj=variable,yproj=yproj,yname=self.get_mmd_name(),
                        color=color,marker=marker,highlight=highlight,fig=fig,ax=axd['C'])
        
        self.hist_mmd_discriminant(fig=fig,ax=axd['D'],orientation='horizontal')

        axd['A'].legend([])
        axd['A'].set_xlabel('')
        axd['B'].set_xlabel('')
        axd['B'].set_ylabel('')
    #     axd['B'].legend(fontsize=15)
        axd['C'].set_title('')
        axd['D'].legend([])
        axd['D'].set_title(f'MMD axis',fontsize=30)
        axd['D'].set_ylabel('')
        axd['D'].sharey(axd['C'])
        #     axd['A'].set_title(f'{g} expression',fontsize=30)
        #     axd['B'].axhline(0,c='crimson',ls='--',alpha=.5,lw='2')
        #     axd['A'].legend(bbox_to_anchor=(.98,1.02),fontsize=30)
    #     pval = self.df_pval[self.get_kfdat_name()].loc[trunc]
        title = f'{variable} MMD'
    #     title += f'{pval:.1e}' if pval<0.01 else f'{pval:.2f}'
        fig.suptitle(title,fontsize=30,y=1.02)
        fig.tight_layout()

        return(fig,axd)

    def plot_pc_of_expression_univariate(self,
                                         variable,
                                         t,
                                         color=None,
                                         marker=None,
                                         highlight=None,
                                    highlight_label=None,
                                                ):
        
        fig = plt.figure(figsize=(15,7.5),constrained_layout=True)
        axd = fig.subplot_mosaic("AB\nCD",gridspec_kw=dict(height_ratios=[1, 2],width_ratios=[3,2]),)
        
        # Pval and errors 
        toi = [t] if t <=30 else []
        self.plot_pval_and_errors(t=30,fig=fig,ax=axd['B'],truncations_of_interest=toi,adjust=False)
        
        self.density_proj(t=variable,
                          proj=variable,
                          fig=fig,
                          ax=axd['A'],
                          highlight=highlight,
                    highlight_label=highlight_label,
                          )
        self.scatter_proj(projection=[variable,t],xproj=variable,yproj='proj_kpca',yname=self.get_kfdat_name(),
                        color=color,marker=marker,highlight=highlight,fig=fig,ax=axd['C'])
        self.hist_pc(t=t,fig=fig,ax=axd['D'],orientation='horizontal')
        axd['A'].legend([])
        axd['A'].set_xlabel('')
        axd['B'].legend(fontsize=15)
        axd['B'].set_ylabel('')
        axd['B'].set_xlabel('')
        axd['C'].set_title('')
        axd['D'].set_title(f'PC{t}',fontsize=30)
        axd['D'].legend([])
        axd['D'].set_ylabel('')
        axd['D'].sharey(axd['C'])
        #     axd['B'].axhline(0,c='crimson',ls='--',alpha=.5,lw='2')
        
        pval = self.df_pval[self.get_kfdat_name()].loc[t]
        title = f'{variable} PC{t} pval='
        title += f'{pval:.1e}' if pval<0.01 else f'{pval:.2f}'
        fig.suptitle(title,fontsize=30,y=1.02)
        fig.tight_layout()
        
        return(fig,axd)

    def plot_pc_and_discriminant_of_expression_univariate(self,  
                                                        variable,
                                                        t,
                                                        color=None,
                                                        marker=None,
                                                        highlight=None,
                                                        highlight_label=None,
                                                        previous_discriminant=False):
        
        fig = plt.figure(figsize=(15,15),constrained_layout=True)
        axd = fig.subplot_mosaic("AB\nCD\nEF",gridspec_kw=dict(height_ratios=[1, 2,2],width_ratios=[3,2]),)
        # Pval and errors 
        self.plot_pval_and_errors(fig=fig,ax=axd['B'],truncations_of_interest=[t],adjust=False)
        
        # expression
        self.density_proj(t=variable,
                          proj=variable,
                          fig=fig,
                          ax=axd['A'],
                          highlight=highlight,
                        highlight_label=highlight_label,
                          )
        
        # PC
        self.scatter_proj(projection=[variable,t],xproj=variable,yproj='proj_kpca',yname=self.get_kfdat_name(),
                        color=color,marker=marker,highlight=highlight,fig=fig,ax=axd['C'])
        self.hist_pc(t=t,fig=fig,ax=axd['D'],orientation='horizontal')
        # Discriminant 
        self.scatter_proj(projection=[variable,t],xproj=variable,yproj='proj_kfda',yname=self.get_kfdat_name(),
                        color=color,marker=marker,highlight=highlight,fig=fig,ax=axd['E'])
        if previous_discriminant and t>1:
            self.scatter_proj(projection=[variable,t-1],xproj=variable,yproj='proj_kfda',yname=self.get_kfdat_name(),
                        color=color,marker=marker,highlight=highlight,fig=fig,ax=axd['E'],
                        alpha=.2,legend=False)
        self.hist_discriminant(t=t,fig=fig,ax=axd['F'],orientation='horizontal')
        
        
        axd['A'].legend([])
        axd['A'].set_xlabel('')
        axd['B'].set_ylabel('')
        axd['B'].legend(fontsize=15)
        axd['C'].set_title(f'PC{t}',fontsize=30)
        axd['D'].legend([])
        axd['D'].set_ylabel('')
        axd['D'].set_title(f'PC{t}',fontsize=30)
        axd['D'].sharey(axd['C'])
        axd['E'].set_title(f'DA{t}',fontsize=30)
        axd['F'].set_title(f'DA{t}',fontsize=30)
        axd['F'].legend([])
        axd['F'].set_ylabel('')

        #     axd['B'].axhline(0,c='crimson',ls='--',alpha=.5,lw='2')
        #     axd['D'].axhline(0,c='crimson',ls='--',alpha=.5,lw='2')
        
        pval = self.df_pval[self.get_kfdat_name()].loc[t]
        title = f'{variable} PC{t} pval='
        title += f'{pval:.1e}' if pval<0.01 else f'{pval:.2f}'
        fig.suptitle(title,fontsize=30,y=1.02)
        fig.tight_layout()
        
        return(fig,axd)
   
    def volcano_plot(self,t,name='',color=None,exceptions=[],
                    focus=None,zero_pvals=False,fig=None,ax=None,corrected=False,threshold=1,plot_others=False):
        # quand la stat est trop grande, la fonction chi2 de scipy.stat renvoie une pval nulle
        # on ne peut pas placer ces gènes dans le volcano plot alors ils ont leur propre graphe

        if fig is None:
            fig,ax = plt.subplots(figsize=(4,10))

        col = self.get_column_name_in_var(t=t,corrected=corrected,name=name,output='pval') 
        
        zpval_str = '= 0' if zero_pvals else '>0'
        var = self.get_var()
        BH_str = 'after correction' if corrected else ''

        if corrected and col not in var:
            self.correct_BenjaminiHochberg_pval_univariate(t=t,name=name)
        pval = var[col]
        pval = filter_genes_wrt_pval(pval,exceptions,focus,zero_pvals,threshold)
        
        print(f'{col} ngenes with pvals {BH_str} {zpval_str}: {len(pval)}')
        
        genes = []
        if len(pval) != 0:
            col_kfda = self.get_column_name_in_var(t=t,corrected=corrected,name=name,output='kfda') 
            col_errB = self.get_column_name_in_var(t=t,corrected=corrected,name=name,output='errB') 
        
            kfda = var[col_kfda]
            errB = var[col_errB]
            logkfda = np.log(kfda[kfda.index.isin(pval.index)])
            errB = errB[errB.index.isin(pval.index)]

            zpval_logkfda = np.log(kfda[~kfda.index.isin(pval.index)])
    #         print(len(logkfda),len(zpval_logkfda))

            xlim = (logkfda.min()-1,logkfda.max()+1)
    #         xlim = (logkfda.min()-1,zpval_logkfda.max()+1)
    #             c = self.color_volcano_plot(var_prefix,pval.index,color=color)

            if zero_pvals:
        #         print('zero')
                ax.set_title(f'col \ngenes strongly rejected',fontsize=30)
                ax.set_xlabel(f'log(kfda)',fontsize=20)
                ax.set_ylabel(f'errB',fontsize=20)

                for g in pval.index.tolist():
        #             print(g,logkfda[g],errB[g],c[g])
                    ax.text(logkfda[g],errB[g],g)#,color=c[g])
                    genes += [g]
                
                ax.set_xlim(xlim)
                ax.set_ylim(0,1)


            else:
        #         print('nz')
                ax.set_title(f'{col}\n non zero pvals',fontsize=30)
                ax.set_xlabel(f'log(kfda)',fontsize=20)
                ax.set_ylabel(f'-log(pval)',fontsize=20)
                logpval = -np.log(pval)


                for g in pval.sort_values().index.tolist():
        #             print(g,logkfda[g],logpval[g],c[g])
                    ax.text(logkfda[g],logpval[g],g)#,color=c[g])
                    ax.set_xlim(xlim)
                    ax.set_ylim(0,logpval.max()*1.1)
                    genes += [g]
                if plot_others:
                    for g in zpval_logkfda.index.tolist():
        #                 print(g)
                        ax.text(zpval_logkfda[g],logpval.max(),g)

        return(genes,fig,ax)
    
    def plot_pval_of_correlated_genes(self,tcorrmax,tpvalmax,proj,t,name,fig=None,axes=None):
        if fig is None:
            fig,axes = plt.subplots(ncols=tcorrmax,figsize=(8*tcorrmax,10))

        axes = [axes] if tcorrmax == 1 else axes



        for tcorr,ax in zip(range(1,tcorrmax+1),axes):
            corr = self.find_correlated_variables(proj=proj,nvar=0,t=tcorr)

            out = []
            if proj == 'proj_kfda':
                print(f'top correlated with DA{tcorr}:',corr.index[:10])
            if proj == 'proj_kpca':
                print(f'top correlated with PC{tcorr}:',corr.index[:10])
                
            for g in corr.index[::-1]:
                dout = {}
                for tpval in range(1,tpvalmax+1):
                    col = self.get_column_name_in_var(t=t,corrected=True,name=name,output='pval') 
                    # pval_name = f'{prefix}_{self.get_kfdat_name()}_t{tpval}_pvalBHc'
                    if col not in self.get_var().columns:
                        self.correct_BenjaminiHochberg_pval_univariate(t=t,name=name)
                            # var_prefix=pval_name[:-8])
                    dout[f't{tpval}'] = -np.log(self.get_var()[col][g]+1)/np.log(10)
                out += [dout]

            dfout = pd.DataFrame(out,index=corr.index)
    #         print('min',dfout.min().min())
    #         print('max',dfout.max().max())
            ax.set_title(f'tcorr={tcorr}',fontsize=30)
            pcm = ax.pcolor(dfout,vmin=-.3,vmax=0)
            ax.set_yticks(range(1,len(corr)+1))
            ax.set_yticklabels(corr.index[::-1])
            fig.colorbar(pcm, ax=ax, extend='max')
        fig.tight_layout()
        return(fig,axes)

    def plot_correlation_of_DEgenes(self,tcorrmax,tpvalmax,proj,t,name,fig=None,axes=None,highlightDE=False):
        if fig is None:
            fig,axes = plt.subplots(ncols=tpvalmax,figsize=(8*tpvalmax,10))

        axes = [axes] if tpvalmax == 1 else axes

        for tpval,ax in zip(range(1,tpvalmax+1),axes):
            
            col = self.get_column_name_in_var(t=t,corrected=True,name=name,output='pval') 
            if col not in self.get_var().columns:
                self.correct_BenjaminiHochberg_pval_univariate(t,name)
            pval = self.get_var()[col]
            pval = pval.sort_values()

            out = []
            print(f'top DE genes for {tpval}:',pval.index[:10])
                
            for g in pval.index[::-1]:
                out += [{f't{tcorr}':self.get_corr_of_variable(proj,g,tcorr) for tcorr in range(1,tcorrmax+1)}]

            dfout = pd.DataFrame(out,index=pval.index)
    #         print('min',dfout.min().min())
    #         print('max',dfout.max().max())

            ax.set_title(f'tpval={tpval}',fontsize=30)
            pcm=ax.pcolor(dfout,vmin=0,vmax=1)
            if highlightDE:
                ax.set_yticks(range(len(pval)+1-len(pval[pval<.05]),len(pval)+1))
                ax.set_yticklabels(pval[pval<.05].index[::-1])
            else:
                ax.set_yticks(range(1,len(pval)+1))
                ax.set_yticklabels(pval.index[::-1])
            fig.colorbar(pcm, ax=ax, extend='max')
        fig.tight_layout()
        plt.show()
        return(fig,axes)
    # def volcano_plot(self,var_prefix,color=None,exceptions=[],focus=None,zero_pvals=False,fig=None,ax=None,BH=False,threshold=1):
    #     # quand la stat est trop grande, la fonction chi2 de scipy.stat renvoie une pval nulle
    #     # on ne peut pas placer ces gènes dans le volcano plot alors ils ont leur propre graphe

    #     if fig is None:
    #         fig,ax = plt.subplots(figsize=(9,15))

    #     BH_str = 'BHc' if BH else ''
    #     zpval_str = '= 0' if zero_pvals else '>0'
    #     dn = self.data_name

    #     pval_name = f'{var_prefix}_pval{BH_str}' 
    #     pval = filter_genes_wrt_pval(self.var[dn][pval_name],exceptions,focus,zero_pvals,threshold)
    #     print(f'{var_prefix} ngenes with pvals {BH_str} {zpval_str}: {len(pval)}')
    #     genes = []
    #     if len(pval) != 0:
    #         kfda = self.var[dn][f'{var_prefix}_kfda']
    #         errB = self.var[dn][f'{var_prefix}_errB']
    #         kfda = kfda[kfda.index.isin(pval.index)]
    #         errB = errB[errB.index.isin(pval.index)]

    #         logkfda = np.log(kfda)

    #         xlim = (logkfda.min()-1,logkfda.max()+1)
    # #             c = self.color_volcano_plot(var_prefix,pval.index,color=color)

    #         if zero_pvals:
    #     #         print('zero')
    #             ax.set_title(f'{var_prefix} \ng enes strongly rejected',fontsize=30)
    #             ax.set_xlabel(f'log(kfda)',fontsize=20)
    #             ax.set_ylabel(f'errB',fontsize=20)

    #             for g in pval.index.tolist():
    #     #             print(g,logkfda[g],errB[g],c[g])
    #                 ax.text(logkfda[g],errB[g],g)#,color=c[g])
    #                 ax.set_xlim(xlim)
    #                 ax.set_ylim(0,1)
    #                 genes += [g]


    #         else:
    #     #         print('nz')
    #             ax.set_title(f'{var_prefix}\n non zero pvals',fontsize=30)
    #             ax.set_xlabel(f'log(kfda)',fontsize=20)
    #             ax.set_ylabel(f'-log(pval)',fontsize=20)
    #             logpval = -np.log(pval)


    #             for g in pval.index.tolist():
    #     #             print(g,logkfda[g],logpval[g],c[g])
    #                 ax.text(logkfda[g],logpval[g],g)#,color=c[g])
    #                 ax.set_xlim(xlim)
    #                 ax.set_ylim(0,logpval.max()*1.1)
    #                 genes += [g]


    #     return(genes,fig,ax)


    def volcano_plot_zero_pvals_and_non_zero_pvals(self,var_prefix,color_nz='errB',color_z='t',
                                                exceptions=[],focus=None,BH=False,threshold=1):
        fig,axes = plt.subplots(ncols=2,figsize=(18,15))
        
        genes_nzpval,_,_ = self.olcano_plot(self,
                                    var_prefix,
                                    color=color_nz,
                                    exceptions=exceptions,
                                    focus=focus,
                                    zero_pvals=False,
                                    fig=fig,ax=axes[0],
                                    BH=BH,
                                    threshold=threshold)
        
        genes_zpval,_,_ = self.volcano_plot(self,
                                var_prefix,
                                color=color_z,
                                exceptions=exceptions,
                                focus=focus,
                                zero_pvals=True,
                                fig=fig,ax=axes[1],
                                BH=BH,
                                threshold=threshold)
        
        return(genes_nzpval,genes_zpval,fig,axes)

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

