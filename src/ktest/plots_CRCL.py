
from time import time 
import matplotlib.pyplot as plt
from .tester import Ktest

class Ktest_CRCL(Ktest):
    def prepare_visualization(self,t,outliers_in_obs=None):
        t0 = time()
        kfda_name = self.kfdat(outliers_in_obs=outliers_in_obs)  
        self.compute_pval() 
        print('calcul KFDA :',time()-t0)
        
        t0 = time()
        self.diagonalize_residual_covariance(t=t,outliers_in_obs=outliers_in_obs)
        print('diagonalize_residual_covariance :',time()-t0)
        
        t0 = time()
        residuals_name = self.proj_residus(t=t,ndirections=20,outliers_in_obs=outliers_in_obs)
        proj_kfda_name = self.compute_proj_kfda(t=20,outliers_in_obs=outliers_in_obs,name=kfda_name)
        print('projections :',time()-t0)
        return({'kfda_name':kfda_name,
                'residuals_name':residuals_name,
                'proj_kfda_name':proj_kfda_name
                })

    def visualize_patient_celltypes_CRCL(self,t,title,outliers_in_obs=None):
        dict_names = self.prepare_visualization(t=t,outliers_in_obs=outliers_in_obs)
        
        xname = dict_names['kfda_name']    
        yname = dict_names['residuals_name']
        
        
        fig,axes = plt.subplots(ncols=4,figsize=(35,10))
        ax = axes[0]
        self.density_proj(t=t,labels='MF',name=xname,fig=fig,ax=ax)
        ax.set_title(f'{title}',fontsize=30)

        ax = axes[1]
        self.scatter_proj(xproj='proj_kfda',xname = xname,yproj='proj_residuals',yname =yname,
                        projection = [t,1],color='celltype',fig=fig,ax=ax,show_conditions=False)
        ax.set_title(f'{title} \n cell type',fontsize=30)


        ax = axes[2]
        self.scatter_proj(xproj='proj_kfda',xname = xname,yproj='proj_residuals',yname =yname,
                        projection = [t,1],color='patient',fig=fig,ax=ax,show_conditions=False)
        ax.set_title(f'{title} \n patient',fontsize=30)

        ax = axes[3]
        self.scatter_proj(xproj='proj_kfda',xname = xname,yproj='proj_residuals',yname =yname,
                        projection = [t,1],color='patient',marker='celltype',fig=fig,ax=ax,)
        ax.set_title(f'{title} \n patient and cell type',fontsize=30)

        fig.tight_layout()
        
    def visualize_quality_CRCL(self,t,outliers_in_obs=None):
        dict_names = self.prepare_visualization(t=t,outliers_in_obs=outliers_in_obs)
        
        xname = dict_names['kfda_name']    
        yname = dict_names['residuals_name']

        
        fig,axes = plt.subplots(ncols=4,figsize=(35,10))
        ax = axes[0]
        self.scatter_proj(xproj='proj_kfda',xname = xname,yproj='proj_residuals',yname =yname,
                        projection = [t,1],color='percent.mt',fig=fig,ax=ax,show_conditions=False,outliers_in_obs=outliers_in_obs)
        ax.set_title(f'percent.mt',fontsize=30)#,y=1.04)

    
        ax = axes[1]
        self.scatter_proj(xproj='proj_kfda',xname = xname,yproj='proj_residuals',yname =yname,
                        projection = [t,1],color='nCount_RNA',fig=fig,ax=ax,show_conditions=False,
                        outliers_in_obs=outliers_in_obs)
        ax.set_title(f'nCount_RNA',fontsize=30)#,y=1.04)

    
        ax = axes[2]
        self.scatter_proj(xproj='proj_kfda',xname = xname,yproj='proj_residuals',yname =yname,
                        projection = [t,1],color='nFeature_RNA',fig=fig,ax=ax,show_conditions=False,
                        outliers_in_obs=outliers_in_obs)
        ax.set_title(f'nFeature_RNA',fontsize=30)#,y=1.04)

        ax = axes[3]
        self.scatter_proj(xproj='proj_kfda',xname = xname,yproj='proj_residuals',yname =yname,
                        projection = [t,1],color='percent.ribo',marker='celltype',fig=fig,ax=ax,
                        outliers_in_obs=outliers_in_obs)
        ax.set_title(f'percent.ribo',fontsize=30)#,y=1.04)

        
        fig.tight_layout()
        
    def visualize_effect_graph_CRCL(self,t,title,
                                    effects=['celltype','patient',['patient','celltype']],
                                    outliers_in_obs=None,
                                    labels='MF'):
        
        dict_names = self.prepare_visualization(t=t,outliers_in_obs=outliers_in_obs)
        
        xname = dict_names['kfda_name']    
        yname = dict_names['residuals_name']
        
        ncols = len(effects)+1
        
        fig,axes = plt.subplots(ncols=ncols,figsize=(10*ncols,10))
        
        
        ax = axes[0]
        self.density_proj(t=t,labels=labels,name=xname,fig=fig,ax=ax)
        ax.set_title(f'{title}',fontsize=30)

        for effect,ax in zip(effects,axes[1:]):
            if type(effect) == str:
                self.scatter_proj(xproj='proj_kfda',xname = xname,yproj='proj_residuals',yname =yname,
                        projection = [t,1],color=effect,fig=fig,ax=ax,show_conditions=False)
                ax.set_title(f'{title} \n {effect}',fontsize=30)
            elif type(effect)== list:
                self.scatter_proj(xproj='proj_kfda',xname = xname,yproj='proj_residuals',yname =yname,
                        projection = [t,1],color=effect[0],marker=effect[1],fig=fig,ax=ax,)
                ax.set_title(f'{title} \n {effect[0]} and {effect[1]}',fontsize=30)
        fig.tight_layout()

