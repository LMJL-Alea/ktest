import matplotlib.pyplot as plt

from .plots_summarized import Plot_Summarized
from .truncation_selection import TruncationSelection
from .univariate_testing import Univariate



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
            c = self.color_volcano_plot(var_prefix,pval.index,color=color)

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

