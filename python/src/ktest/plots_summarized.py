
import matplotlib.pyplot as plt
import numpy as np

from .plots_wb_error import Plot_WBerrors
from .plots_standard import Plot_Standard
from .utils_plot import text_truncations_of_interest,replace_label,adjusted_xticks
from .dendrogram import Dendrogram

class Plot_Summarized(Plot_Standard,Plot_WBerrors,Dendrogram):
        
    def __init__(self):        
        super(Plot_Summarized, self).__init__()


    # reconstructions error 
    def plot_pval_and_errors(self,t=20,fig=None,ax=None,truncations_of_interest=[1,3,5],marked_obs_to_ignore=None,
                            log=False,cumul=False,adjust=True,decreasing=False,
                            log_spectrum=False,
                            pval_aggregated=True,pval_contrib=False,
                            var_within=False,var_conditions=True,
                            kfdr=False,
                            diff=True,grid=True,
                            alpha=.8):

        if fig is None:
            fig,ax = plt.subplots(ncols=1,figsize=(12,6))

        if marked_obs_to_ignore is not None:
            self.set_marked_obs_to_ignore(marked_obs_to_ignore=marked_obs_to_ignore)

        cumul = True if decreasing else cumul

        if any([pval_aggregated,pval_contrib]):
            self.plot_pvalue(fig=fig,ax=ax,t=t,aggregated=pval_aggregated,contrib=pval_contrib,truncations_of_interest=truncations_of_interest,adjust=adjust,log=log)

        if log_spectrum:
            self.plot_part_log_spectrum(t=t,fig=fig,ax=ax)
        
        df = self.get_diagnostics(t=t,
                                diff=diff,
                            var_within=var_within,
                            var_samples=var_conditions,
                            kfdr=kfdr,
                            cumul=cumul,log=log,decreasing=decreasing
                            ) 
        df.plot.bar(alpha=alpha,ax=ax)

        ax.legend(fontsize=20)
        ax.set_xlabel('t',fontsize=30)
        ax.set_xticks(adjusted_xticks(t))
        if grid:
            ax.grid(alpha=.2)

        n1,n2,n = self.get_n1n2n()
        samples = self.get_samples_list()
        ax.set_title(f'n{samples[0]}={n1} vs n{samples[1]}={n2}',fontsize=30)

        if marked_obs_to_ignore is not None:
            self.set_marked_obs_to_ignore()

        return(fig,ax)


    def what_if_we_ignored_cells_by_condition(self,threshold,orientation,t='1',column_in_dataframe='kfda',proj='proj_kfda',marked_obs_already_ignored=None):
        oname = f"{proj}[{column_in_dataframe}][{t}]{orientation}{threshold}"
    #     print(oname_)
    #     oname = f'outliers_kfdat1_{threshold}'
        
        observations = self.select_observations_from_condition(threshold = threshold,
                                    orientation =orientation, 
                                    t=t,
                                    column_in_dataframe=column_in_dataframe,
                                    proj=proj,
                                    already_marked_obs_to_consider=marked_obs_already_ignored)
        
        print(f'{oname} : {len(observations)} observations')

        self.mark_observations(observations_to_mark=observations,marking_name=oname)
        self.set_marked_obs_to_ignore(marked_obs_to_ignore=oname)
        self.multivariate_test()    
        self.projections(t=20)
        self.set_marked_obs_to_ignore()

        # a remplacer par self.compare_two_stats
        fig,axes = plt.subplots(ncols=4,figsize=(48,8))
        ax = axes[0]
        self.density_proj(t=int(t),
                          labels='MF',
                          proj=proj,
                          name=column_in_dataframe,
                          fig=fig,ax=ax)
        ax.axvline(threshold,ls='--',c='crimson')
        ax.set_title(column_in_dataframe,fontsize=20)

        ax = axes[1]
        self.plot_kfdat(fig,ax,t=20,columns = [column_in_dataframe,oname])
        
        ax = axes[2]
        self.plot_pval_and_errors(fig=fig,ax=ax,t=30)
        ax.legend()
        ax.set_xlabel('Truncation',fontsize=30)
        ax.set_ylabel('Errors or pval',fontsize=30)
        replace_label(ax,0,'p-value')
        ax.set_title('Before',fontsize=30)
        
        ax = axes[3]
        self.set_marked_obs_to_ignore(marked_obs_to_ignore=oname)
        self.plot_pval_and_errors(fig=fig,ax=ax,t=30)
        ax.legend()
        ax.set_xlabel('Truncation',fontsize=30)
        ax.set_ylabel('Errors or pval',fontsize=30)
        replace_label(ax,0,'p-value')
        ax.set_title(f'After ({oname})',fontsize=30)
        fig.tight_layout()
        self.set_marked_obs_to_ignore()
        return(oname)
    
    def fit_Ktest_with_ignored_observations(self,list_of_observations_to_ignore,list_name):
        print(f'{list_name} : ignoring {len(list_of_observations_to_ignore)} observations ')
        if list_name in self.obs:
            print(f'list_name {list_name} already in obs')
            self.set_marked_obs_to_ignore(marked_obs_to_ignore=list_name)
        else:
            self.mark_observations(observations_to_mark=list_of_observations_to_ignore,marking_name=list_name)
        self.set_marked_obs_to_ignore(marked_obs_to_ignore=list_name)
        self.multivariate_test() 
        self.projections(t=20)
        self.set_marked_obs_to_ignore()



    def what_if_we_ignored_list_of_obs(self,list_of_observations,list_name,t_errors=20,t_discriminant=None,t_orthogonal=None,nkpca=0):
        

        self.set_marked_obs_to_ignore(marked_obs_to_ignore=list_name)
        fig,axes = self.summary_plots_of_Ktest(title=list_name,
                                    t_errors=t_errors,
                                    t_discriminant=t_discriminant,
                                    t_orthogonal=t_orthogonal,
                                    nkpca=nkpca,
                                    highlight=list_of_observations)
        self.set_marked_obs_to_ignore()

        return(list_name,fig,axes)

    def summary_plots_of_Ktest(self,title,t_errors=None,t_discriminant=None,t_orthogonal=None,t_kpca=None,t_nextPC=None,color=None,marker=None,highlight=None):

        
        dict_t = {}
        dict_nt = {}
        
        errors = False if t_errors is None else True 
        ncols = errors
        for t,tname in zip([t_discriminant,t_orthogonal,t_kpca,t_nextPC],
                           ['discriminant','orthogonal','kpca','nextPC']):
            dict_t[tname] = [t] if isinstance(t,int) else t
            dict_nt[tname] = False if t is None else len(t)
            ncols += False if t is None else len(t)
        
        fig,axes = plt.subplots(ncols=ncols,figsize=(9*ncols,8))
        if ncols==1:
            axes = [axes]

        fig.suptitle(title,fontsize=50,y=1.04)
        i=0
        if errors: 
            ax = axes[i]
            i+=1
            self.plot_pval_and_errors(truncations_of_interest=[1,3,6],t=t_errors,fig=fig,ax=ax)
            
        if dict_nt['discriminant']:
            for t in dict_t['discriminant']:
                ax = axes[i]
                i+=1
                self.hist_discriminant(t=t,fig=fig,ax=ax)
                # self.density_proj(t = t,fig=fig,ax=ax)
                ax.set_title(f'discriminant axis t={t}',fontsize=30)
        if dict_nt['orthogonal']: 
            for t in dict_t['orthogonal']:
                ax = axes[i]
                i+=1
                self.orthogonal(t)
                self.plot_orthogonal(t,fig=fig,ax=ax,color=color,marker=marker,highlight=highlight)

        if dict_nt['kpca']:
            for t in dict_t['kpca']:
                ax = axes[i]
                i+=1
                self.plot_kpca(t=t,fig=fig,ax=ax,color=color,marker=marker,highlight=highlight)
        
        if dict_nt['nextPC']:
            for t in dict_t['nextPC']:
                ax = axes[i]
                i+=1
                self.plot_nextPC(t,fig=fig,ax=ax,color=color,marker=marker,highlight=highlight)
            # self.scatter_projs(projections=[[i*2+1,i*2+2] for i in range(nkpca)],xproj='proj_kpca',fig=fig,axes=axes[i:],color=color)
        fig.tight_layout()
        return(fig,axes)
                