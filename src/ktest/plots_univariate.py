import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plots_summarized import Plot_Summarized
from .truncation_selection import TruncationSelection
from .univariate_testing import Univariate

from .utils_plot import init_plot_pvalue,text_truncations_of_interest
from .utils_univariate import filter_genes_wrt_pval


class Plot_Univariate(TruncationSelection,Plot_Summarized,Univariate):

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

    def plot_density_of_variable(self,variable,fig=None,ax=None,color=None):
        if fig is None:
            fig,ax =plt.subplots(figsize=(10,6))
            
        self.density_proj(t=0,proj=variable,fig=fig,ax=ax,color=color)
        
        title = f'{variable}\n'
        zero_proportions = self.compute_zero_proportions_of_variable(variable)
        for c in zero_proportions.keys():
            if '_nz' in c:
                nz = zero_proportions[c]
                c_ = c.split(sep='_nz')[0]
                title += f' {c_}: {nz}z '
        
        ax.set_title(title,fontsize=20)
        return(fig,ax)

    def plot_pval_and_errors_of_variable(self,variable,t=30,fig=None,ax=None,truncations_of_interest=[1,3,6],adjust=True):
        if fig is None:
            fig,ax = plt.subplots(figsize=(7,7))
        
        fig,ax = init_plot_pvalue(fig=fig,ax=ax,t=t)
        
        pval = [self.get_var()[f'_{self.get_kfdat_name()}_t{trunc}_pval'][variable] for trunc in range(1,t)]
        errB = [1]+[self.get_var()[f'_{self.get_kfdat_name()}_t{trunc}_errB'][variable] for trunc in range(1,t)]
        errW = [1]+[self.get_var()[f'_{self.get_kfdat_name()}_t{trunc}_errW'][variable] for trunc in range(1,t)]
        
        
        ax.plot(range(1,t),pval,label='p-value')
        ax.plot(range(t),errB,label='explained difference')
        ax.plot(range(t),errW,label='explained variance')
        ax.legend(fontsize=20)
        
        if truncations_of_interest is not None:
            text_truncations_of_interest(truncations_of_interest,ax,[0]+pval,adjust=adjust)
            
        ax.set_xlabel('Truncation',fontsize=30)
        ax.set_ylabel('Errors or pval',fontsize=30)
        ax.set_title(variable,fontsize=30)
        ax.set_xlim(-1,t+1)
        return(fig,ax)

    def plot_discriminant_of_expression_univariate(self,variable,trunc,
                                                color=None,marker=None,highlight=None,
                                                ):
        
        fig = plt.figure(figsize=(15,7.5),constrained_layout=True)
        axd = fig.subplot_mosaic("AB\nCD",gridspec_kw=dict(height_ratios=[1, 2],width_ratios=[3,2]),)
        
        
        # Pval and errors 
        self.plot_pval_and_errors(fig=fig,ax=axd['B'],truncations_of_interest=[trunc],adjust=False)
        
        self.density_proj(t=variable,proj=variable,fig=fig,ax=axd['A'])
        self.scatter_proj(projection=[variable,trunc],xproj=variable,yproj='proj_kfda',yname=self.get_kfdat_name(),
                        color=color,marker=marker,highlight=highlight,fig=fig,ax=axd['C'])
        self.hist_discriminant(t=trunc,fig=fig,ax=axd['D'],orientation='horizontal')
        axd['A'].legend([])
        axd['A'].set_xlabel('')
        axd['B'].set_xlabel('')
        axd['B'].set_ylabel('')
        axd['B'].legend(fontsize=15)
        axd['C'].set_title('')
        axd['D'].legend([])
        axd['D'].set_title(f'Discriminant t={trunc}',fontsize=30)
        axd['D'].set_ylabel('')
        axd['D'].sharey(axd['C'])
        #     axd['A'].set_title(f'{g} expression',fontsize=30)
        #     axd['B'].axhline(0,c='crimson',ls='--',alpha=.5,lw='2')
        #     axd['A'].legend(bbox_to_anchor=(.98,1.02),fontsize=30)
        pval = self.df_pval[self.get_kfdat_name()].loc[trunc]
        title = f'{variable} DA{trunc} pval='
        title += f'{pval:.1e}' if pval<0.01 else f'{pval:.2f}'
        fig.suptitle(title,fontsize=30,y=1.02)
        fig.tight_layout()
        
        return(fig,axd)


    def plot_pc_of_expression_univariate(self,variable,trunc,
                                                color=None,marker=None,highlight=None,
                                                ):
        
        fig = plt.figure(figsize=(15,7.5),constrained_layout=True)
        axd = fig.subplot_mosaic("AB\nCD",gridspec_kw=dict(height_ratios=[1, 2],width_ratios=[3,2]),)
        
        # Pval and errors 
        self.plot_pval_and_errors(fig=fig,ax=axd['B'],truncations_of_interest=[trunc],adjust=False)
        
        self.density_proj(t=variable,proj=variable,fig=fig,ax=axd['A'])
        self.scatter_proj(projection=[variable,trunc],xproj=variable,yproj='proj_kpca',yname=self.get_kfdat_name(),
                        color=color,marker=marker,highlight=highlight,fig=fig,ax=axd['C'])
        self.hist_pc(t=trunc,fig=fig,ax=axd['D'],orientation='horizontal')
        axd['A'].legend([])
        axd['A'].set_xlabel('')
        axd['B'].legend(fontsize=15)
        axd['B'].set_ylabel('')
        axd['B'].set_xlabel('')
        axd['C'].set_title('')
        axd['D'].set_title(f'PC{trunc}',fontsize=30)
        axd['D'].legend([])
        axd['D'].set_ylabel('')
        axd['D'].sharey(axd['C'])
        #     axd['B'].axhline(0,c='crimson',ls='--',alpha=.5,lw='2')
        
        pval = self.df_pval[self.get_kfdat_name()].loc[trunc]
        title = f'{variable} PC{trunc} pval='
        title += f'{pval:.1e}' if pval<0.01 else f'{pval:.2f}'
        fig.suptitle(title,fontsize=30,y=1.02)
        fig.tight_layout()
        
        return(fig,axd)


    def plot_pc_and_discriminant_of_expression_univariate(self,variable,trunc,
                                                color=None,marker=None,highlight=None,
                                                ):
        
        fig = plt.figure(figsize=(15,15),constrained_layout=True)
        axd = fig.subplot_mosaic("AB\nCD\nEF",gridspec_kw=dict(height_ratios=[1, 2,2],width_ratios=[3,2]),)
        # Pval and errors 
        self.plot_pval_and_errors(fig=fig,ax=axd['B'],truncations_of_interest=[trunc],adjust=False)
        
        # expression
        self.density_proj(t=variable,proj=variable,fig=fig,ax=axd['A'])
        
        # PC
        self.scatter_proj(projection=[variable,trunc],xproj=variable,yproj='proj_kpca',yname=self.get_kfdat_name(),
                        color=color,marker=marker,highlight=highlight,fig=fig,ax=axd['C'])
        self.hist_pc(t=trunc,fig=fig,ax=axd['D'],orientation='horizontal')
        # Discriminant 
        self.scatter_proj(projection=[variable,trunc],xproj=variable,yproj='proj_kfda',yname=self.get_kfdat_name(),
                        color=color,marker=marker,highlight=highlight,fig=fig,ax=axd['E'])
        self.hist_discriminant(t=trunc,fig=fig,ax=axd['F'],orientation='horizontal')
        
        
        axd['A'].legend([])
        axd['A'].set_xlabel('')
        axd['B'].set_ylabel('')
        axd['B'].legend(fontsize=15)
        axd['C'].set_title(f'PC{trunc}',fontsize=30)
        axd['D'].legend([])
        axd['D'].set_ylabel('')
        axd['D'].set_title(f'PC{trunc}',fontsize=30)
        axd['D'].sharey(axd['C'])
        axd['E'].set_title(f'DA{trunc}',fontsize=30)
        axd['F'].set_title(f'DA{trunc}',fontsize=30)
        axd['F'].legend([])
        axd['F'].set_ylabel('')

        #     axd['B'].axhline(0,c='crimson',ls='--',alpha=.5,lw='2')
        #     axd['D'].axhline(0,c='crimson',ls='--',alpha=.5,lw='2')
        
        pval = self.df_pval[self.get_kfdat_name()].loc[trunc]
        title = f'{variable} PC{trunc} pval='
        title += f'{pval:.1e}' if pval<0.01 else f'{pval:.2f}'
        fig.suptitle(title,fontsize=30,y=1.02)
        fig.tight_layout()
        
        return(fig,axd)
   
   
   
    # def volcano_plot(self,var_prefix,color=None,exceptions=[],focus=None,zero_pvals=False,fig=None,ax=None,BH=False,threshold=1):
    #     # quand la stat est trop grande, la fonction chi2 de scipy.stat renvoie une pval nulle
    #     # on ne peut pas placer ces gènes dans le volcano plot alors ils ont leur propre graphe
        
    #     if fig is None:
    #         fig,ax = plt.subplots(figsize=(9,15))
    #     BH_str = 'BHc' if BH else ''
    #     zpval_str = '= 0' if zero_pvals else '>0'
        
    #     pval_name = f'{var_prefix}_pval{BH_str}' 
    #     pval = filter_genes_wrt_pval(self.var[pval_name],exceptions,focus,zero_pvals,threshold)
    #     print(f'{var_prefix} ngenes with pvals {BH_str} {zpval_str}: {len(pval)}')
    #     genes = []
    #     if len(pval) != 0:
    #         kfda = self.var[f'{var_prefix}_kfda']
    #         errB = self.var[f'{var_prefix}_errB']
    #         kfda = kfda[kfda.index.isin(pval.index)]
    #         errB = errB[errB.index.isin(pval.index)]

    #         logkfda = np.log(kfda)

    #         xlim = (logkfda.min()-1,logkfda.max()+1)
    #         c = self.color_volcano_plot(var_prefix,pval.index,color=color)

    #         if zero_pvals:
    #     #         print('zero')
    #             ax.set_title(f'{var_prefix} \ng enes strongly rejected',fontsize=30)
    #             ax.set_xlabel(f'log(kfda)',fontsize=20)
    #             ax.set_ylabel(f'errB',fontsize=20)

    #             for g in pval.index.tolist():
    #     #             print(g,logkfda[g],errB[g],c[g])
    #                 ax.text(logkfda[g],errB[g],g,color=c[g])
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
    #                 ax.text(logkfda[g],logpval[g],g,color=c[g])
    #                 ax.set_xlim(xlim)
    #                 ax.set_ylim(0,logpval.max()*1.1)
    #                 genes += [g]


    #     return(genes,fig,ax)

    def volcano_plot(self,var_prefix,color=None,exceptions=[],focus=None,zero_pvals=False,fig=None,ax=None,BH=False,threshold=1):
        # quand la stat est trop grande, la fonction chi2 de scipy.stat renvoie une pval nulle
        # on ne peut pas placer ces gènes dans le volcano plot alors ils ont leur propre graphe

        if fig is None:
            fig,ax = plt.subplots(figsize=(9,15))

        BH_str = 'BHc' if BH else ''
        zpval_str = '= 0' if zero_pvals else '>0'
        dn = self.data_name

        pval_name = f'{var_prefix}_pval{BH_str}' 
        pval = filter_genes_wrt_pval(self.var[dn][pval_name],exceptions,focus,zero_pvals,threshold)
        print(f'{var_prefix} ngenes with pvals {BH_str} {zpval_str}: {len(pval)}')
        genes = []
        if len(pval) != 0:
            kfda = self.var[dn][f'{var_prefix}_kfda']
            errB = self.var[dn][f'{var_prefix}_errB']
            kfda = kfda[kfda.index.isin(pval.index)]
            errB = errB[errB.index.isin(pval.index)]

            logkfda = np.log(kfda)

            xlim = (logkfda.min()-1,logkfda.max()+1)
    #             c = self.color_volcano_plot(var_prefix,pval.index,color=color)

            if zero_pvals:
        #         print('zero')
                ax.set_title(f'{var_prefix} \ng enes strongly rejected',fontsize=30)
                ax.set_xlabel(f'log(kfda)',fontsize=20)
                ax.set_ylabel(f'errB',fontsize=20)

                for g in pval.index.tolist():
        #             print(g,logkfda[g],errB[g],c[g])
                    ax.text(logkfda[g],errB[g],g)#,color=c[g])
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
                    ax.text(logkfda[g],logpval[g],g)#,color=c[g])
                    ax.set_xlim(xlim)
                    ax.set_ylim(0,logpval.max()*1.1)
                    genes += [g]


        return(genes,fig,ax)


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

