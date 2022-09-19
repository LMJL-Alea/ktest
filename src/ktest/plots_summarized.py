
import matplotlib.pyplot as plt


from .plots_wb_error import Plot_WBerrors
from .plots_standard import Plot_Standard
from .utils_plot import text_truncations_of_interest,replace_label


class Plot_Summarized(Plot_Standard,Plot_WBerrors):
        
    def __init__(self):        
        super(Plot_Summarized, self).__init__()


    # reconstructions error 
    def plot_pval_and_errors(self,truncations_of_interest=[1],t=20,fig=None,ax=None):
        column = self.get_kfdat_name()
        if fig is None:
            fig,ax = plt.subplots(ncols=1,figsize=(12,8))
        self.plot_pvalue(fig=fig,ax=ax,t=t,column = column,)
        self.plot_between_covariance_reconstruction_error_with_respect_to_t(r'explained difference',
                                                                            fig,ax,t=t,scatter=False)
        self.plot_within_covariance_reconstruction_error_with_respect_to_t(r'explained variance',
                                                                        fig,ax,t=t,scatter=False)
        if truncations_of_interest is not None:
            values = self.df_pval[column]
            text_truncations_of_interest(truncations_of_interest,ax,values)
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
        ax.set_title(f'n1={n1} vs n2={n2}',fontsize=30)
        pval = self.df_pval[column][1]
        text=  f'{pval:.2f}' if pval >=.01 else f'{pval:1.0e}'
        replace_label(ax,0,f'p-value')
        
        return(fig,ax)

    def what_if_we_ignored_cells_by_condition(self,threshold,orientation,t='1',column_in_dataframe='kfda',which='proj_kfda',outliers_in_obs=None):
        oname = f"{which}[{column_in_dataframe}][{t}]{orientation}{threshold}"
    #     print(oname_)
    #     oname = f'outliers_kfdat1_{threshold}'
        outliers = self.determine_outliers_from_condition(threshold = threshold,
                                    orientation =orientation, 
                                    t=t,
                                    column_in_dataframe=column_in_dataframe,
                                    which=which,
                                    outliers_in_obs=outliers_in_obs)
        
        print(f'{oname} : {len(outliers)} outliers')

        self.add_outliers_in_obs(outliers,name_outliers=oname)

        self.kfdat(outliers_in_obs=oname)    
        self.compute_proj_kfda(t=20,outliers_in_obs=oname)

        # a remplacer par self.compare_two_stats
        fig,axes = plt.subplots(ncols=4,figsize=(48,8))
        
        ax = axes[0]
        self.density_proj(t=int(t),labels='MF',which=which,name=column_in_dataframe,fig=fig,ax=ax)
        ax.axvline(threshold,ls='--',c='crimson')
        ax.set_title(column_in_dataframe,fontsize=20)
        ax = axes[1]
        self.plot_kfdat(fig,ax,t=20,columns = [column_in_dataframe,oname])
        
        ax = axes[2]
        self.plot_pvalue(fig=fig,ax=ax,t=20,column = column_in_dataframe,)
        self.plot_between_covariance_reconstruction_error_with_respect_to_t(r'$\mu_2 - \mu_1$ error',fig,ax,t=20)
        self.plot_within_covariance_reconstruction_error_with_respect_to_t(r'$\Sigma_W$ error',fig,ax,t=20)
        ax.legend()
        ax.set_xlabel('Truncation',fontsize=30)
        ax.set_ylabel('Errors or pval',fontsize=30)
        replace_label(ax,0,'p-value')
        ax.set_title('Before',fontsize=30)
        
        ax = axes[3]
        self.plot_pvalue(fig=fig,ax=ax,t=20,column = oname,)
        self.plot_between_covariance_reconstruction_error_with_respect_to_t(r'$\mu_2 - \mu_1$ error',fig,ax,t=20)
        self.plot_within_covariance_reconstruction_error_with_respect_to_t(r'$\Sigma_W$ error',fig,ax,t=20)
        ax.legend()
        ax.set_xlabel('Truncation',fontsize=30)
        ax.set_ylabel('Errors or pval',fontsize=30)
        replace_label(ax,0,'p-value')
        ax.set_title(f'After ({oname})',fontsize=30)
        fig.tight_layout()
        return(oname)
    
    def what_if_we_ignored_cells_by_outliers_list(self,outliers,oname,column_in_dataframe='kfda',outliers_in_obs=None):
        
    #     print(oname_)
    #     oname = f'outliers_kfdat1_{threshold}'

        if outliers_in_obs is not None:
            df_outliers = self.obs[outliers_in_obs]
            old_outliers    = df_outliers[df_outliers].index
            outliers = outliers.append(old_outliers)

        print(f'{oname} : {len(outliers)} outliers')

        self.add_outliers_in_obs(outliers,name_outliers=oname)
        self.kfdat(outliers_in_obs=oname)    
    #     self.diagonalize_residual_covariance(t=1,outliers_in_obs=oname)
    #     self.proj_residus(t=1,ndirections=20,outliers_in_obs=oname)
        self.compute_proj_kfda(t=20,outliers_in_obs=oname)

        fig,axes = plt.subplots(ncols=3,figsize=(35,8))
        
        ax = axes[0]
        self.plot_kfdat(fig,ax,t=20,columns = [column_in_dataframe,oname])
        
        ax = axes[1]
        self.plot_pvalue(fig=fig,ax=ax,t=20,column = column_in_dataframe,)
        self.plot_between_covariance_reconstruction_error_with_respect_to_t(r'$\mu_2 - \mu_1$ error',fig,ax,t=20)
        self.plot_within_covariance_reconstruction_error_with_respect_to_t(r'$\Sigma_W$ error',fig,ax,t=20)
        ax.legend()
        ax.set_xlabel('Truncation',fontsize=30)
        ax.set_ylabel('Errors or pval',fontsize=30)
        replace_label(ax,0,'p-value')
        ax.set_title('Before',fontsize=30)
        
        ax = axes[2]
        self.plot_pvalue(fig=fig,ax=ax,t=20,column = oname,)
        self.plot_between_covariance_reconstruction_error_with_respect_to_t(r'$\mu_2 - \mu_1$ error',fig,ax,t=20)
        self.plot_within_covariance_reconstruction_error_with_respect_to_t(r'$\Sigma_W$ error',fig,ax,t=20)
        ax.legend()
        ax.set_xlabel('Truncation',fontsize=30)
        ax.set_ylabel('Errors or pval',fontsize=30)
        replace_label(ax,0,'p-value')
        ax.set_title(f'After ({oname})',fontsize=30)
        fig.tight_layout()

        return(oname)

