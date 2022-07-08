import numpy as np
import matplotlib.pyplot as plt


def plot_density_of_variable(self,variable,fig=None,ax=None,data_name ='data',color=None,condition_mean=True):
    if fig is None:
        fig,ax =plt.subplots(figsize=(10,6))
# dernière version dans endothelial_univariate        
#     proj = self.init_df_proj(variable,name=data_name)
#     cond = self.obs['sample']
#     xv = proj[cond=='x'][variable]
#     yv = proj[cond=='y'][variable]
    

#     self.density_proj(t=0,which=variable,name=data_name,fig=fig,ax=ax,color=color)
    
#     if condition_mean:
#         ax.axvline(xv.mean(),c='blue')
#         ax.axvline(yv.mean(),c='orange')

#         ax.axvline(xv[xv>0].mean(),c='blue',ls='--',alpha=.5)
#         ax.axvline(yv[yv>0].mean(),c='orange',ls='--',alpha=.5)

    
# #     for xy,sxy,color in zip([x,y],'xy',['blue','orange']):
        
# #         bins=int(np.floor(np.sqrt(len(xy))))
# #         ax.hist(xy,density=True,histtype='bar',label=f'{sxy}({len(xy)})',alpha=.3,bins=bins,color=color)
# #         ax.hist(xy,density=True,histtype='step',bins=bins,lw=3,edgecolor='black')
# #         ax.legend(fontsize=20)

#     zp = get_zero_proportions_of_variable(self,variable)
#     nzx = zp['x']['nzero']
#     nzy = zp['y']['nzero']
#     px = zp['x']['rzero']
#     py = zp['y']['rzero']
    
#     ax.set_title(f'{variable} {data_name}  nzx = {nzx} ({px:.2f}) nzy = {nzy} ({py:.2f})')
    return(fig,ax)

