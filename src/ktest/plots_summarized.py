
import matplotlib.pyplot as plt


from .plots_wb_error import Plot_WBerrors
from .plots_standard import Plot_Standard
from .utils_plot import text_truncations_of_interest,replace_label


class Plot_Summarized(Plot_Standard,Plot_WBerrors):
        
    def __init__(self):        
        super(Plot_Summarized, self).__init__()


    # reconstructions error 
    def plot_pval_and_errors(self,truncations_of_interest=[1,3,5],t=20,fig=None,ax=None,outliers_in_obs=None,adjust=True):
        if outliers_in_obs is not None:
            self.set_outliers_in_obs(outliers_in_obs=outliers_in_obs)
        column = self.get_kfdat_name()
        if fig is None:
            fig,ax = plt.subplots(ncols=1,figsize=(12,8))
        self.plot_pvalue(fig=fig,ax=ax,t=t)
        self.plot_between_covariance_reconstruction_error_with_respect_to_t(r'explained difference',
                                                                            fig,ax,t=t,scatter=False)
        self.plot_within_covariance_reconstruction_error_with_respect_to_t(r'explained variance',
                                                                        fig,ax,t=t,scatter=False)
        if truncations_of_interest is not None:
            values = self.df_pval[column]
            text_truncations_of_interest(truncations_of_interest,ax,values,adjust=adjust)
            # texts = []
            # for t in set(truncations_of_interest):
            #     pvalt = self.df_pval[column][t]
            #     text = f'{pvalt:.2f}' if pvalt >=.01 else f'{pvalt:1.0e}'
            #     ax.scatter(x=t,y=pvalt,s=20)
            #     texts += [ax.text(t,pvalt,text,fontsize=20)]
            # adjust_text(texts,only_move={'points': 'y', 'text': 'y', 'objects': 'y'})
                
        ax.legend(fontsize=20)
        ax.set_xlabel('Truncation',fontsize=30)
        ax.set_ylabel('Errors or pval',fontsize=30)
        n1,n2,n = self.get_n1n2n()
        samples = list(self.get_index().keys())
        ax.set_title(f'n{samples[0]}={n1} vs n{samples[1]}={n2}',fontsize=30)
        pval = self.df_pval[column][1]
        text=  f'{pval:.2f}' if pval >=.01 else f'{pval:1.0e}'
        replace_label(ax,0,f'p-value')
        
        if outliers_in_obs is not None:
            self.set_outliers_in_obs()
        return(fig,ax)

    def what_if_we_ignored_cells_by_condition(self,threshold,orientation,t='1',column_in_dataframe='kfda',proj='proj_kfda',outliers_in_obs=None):
        oname = f"{proj}[{column_in_dataframe}][{t}]{orientation}{threshold}"
    #     print(oname_)
    #     oname = f'outliers_kfdat1_{threshold}'
        outliers = self.determine_outliers_from_condition(threshold = threshold,
                                    orientation =orientation, 
                                    t=t,
                                    column_in_dataframe=column_in_dataframe,
                                    proj=proj,
                                    outliers_in_obs=outliers_in_obs)
        
        print(f'{oname} : {len(outliers)} outliers')

        self.add_outliers_in_obs(outliers,name_outliers=oname)
        self.set_outliers_in_obs(outliers_in_obs=oname)
        self.kfdat()    
        self.projections(t=20)
        self.set_outliers_in_obs()

        # a remplacer par self.compare_two_stats
        fig,axes = plt.subplots(ncols=4,figsize=(48,8))
        ax = axes[0]
        self.density_proj(t=int(t),labels='MF',proj=proj,name=column_in_dataframe,fig=fig,ax=ax)
        ax.axvline(threshold,ls='--',c='crimson')
        ax.set_title(column_in_dataframe,fontsize=20)

        ax = axes[1]
        self.plot_kfdat(fig,ax,t=20,columns = [column_in_dataframe,oname])
        
        ax = axes[2]
        self.plot_pvalue(fig=fig,ax=ax,t=20)
        self.plot_between_covariance_reconstruction_error_with_respect_to_t(r'$\mu_2 - \mu_1$ error',fig,ax,t=20)
        self.plot_within_covariance_reconstruction_error_with_respect_to_t(r'$\Sigma_W$ error',fig,ax,t=20)
        ax.legend()
        ax.set_xlabel('Truncation',fontsize=30)
        ax.set_ylabel('Errors or pval',fontsize=30)
        replace_label(ax,0,'p-value')
        ax.set_title('Before',fontsize=30)
        
        ax = axes[3]
        self.set_outliers_in_obs(outliers_in_obs=oname)
        self.plot_pvalue(fig=fig,ax=ax,t=20)
        self.plot_between_covariance_reconstruction_error_with_respect_to_t(r'$\mu_2 - \mu_1$ error',fig,ax,t=20)
        self.plot_within_covariance_reconstruction_error_with_respect_to_t(r'$\Sigma_W$ error',fig,ax,t=20)
        ax.legend()
        ax.set_xlabel('Truncation',fontsize=30)
        ax.set_ylabel('Errors or pval',fontsize=30)
        replace_label(ax,0,'p-value')
        ax.set_title(f'After ({oname})',fontsize=30)
        fig.tight_layout()
        self.set_outliers_in_obs()
        return(oname)
    

    def fit_tester_with_ignored_outliers(self,outliers_list,outliers_name):
        print(f'{outliers_name} : {len(outliers_list)} outliers')
        self.add_outliers_in_obs(outliers_list,name_outliers=outliers_name)
        self.set_outliers_in_obs(outliers_in_obs=outliers_name)
        self.kfdat() 
        self.projections(t=20)
        self.set_outliers_in_obs()

    def what_if_we_ignored_cells_by_outliers_list(self,outliers_list,outliers_name,t_errors=20,t_discriminant=None,t_residuals=None,nkpca=0):
        
        # if outliers_in_obs is not None:
        #     df_outliers = self.obs[outliers_in_obs]
        #     old_outliers    = df_outliers[df_outliers].index
        #     outliers = outliers.append(old_outliers)

        self.set_outliers_in_obs(outliers_in_obs=outliers_name)
        fig,axes = self.summary_plots_of_tester(title=outliers_name,
                                    t_errors=t_errors,
                                    t_discriminant=t_discriminant,
                                    t_residuals=t_residuals,
                                    nkpca=nkpca,
                                    highlight=outliers_list)
        self.set_outliers_in_obs()

        return(outliers_name,fig,axes)

    def summary_plots_of_tester(self,title,t_errors=None,t_discriminant=None,t_residuals=None,t_kpca=None,t_nextPC=None,color=None,marker=None,highlight=None):
        
        dict_t = {}
        dict_nt = {}
        
        errors = False if t_errors is None else True 
        ncols = errors
        for t,tname in zip([t_discriminant,t_residuals,t_kpca,t_nextPC],
                           ['discriminant','residuals','kpca','nextPC']):
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
        if dict_nt['residuals']: 
            for t in dict_t['residuals']:
                ax = axes[i]
                i+=1
                self.residuals(t)
                self.plot_residuals(t,fig=fig,ax=ax,color=color,marker=marker,highlight=highlight)

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
                