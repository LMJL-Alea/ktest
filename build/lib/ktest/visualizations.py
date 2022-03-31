import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from scipy.stats import chi2
import numpy as np
import torch
from torch import mv,dot


def init_plot_kfdat(fig=None,ax=None,ylim=None,t=None,label=False,title=None,title_fontsize=40,asymp_arg=None):

    assert(t is not None)
    trunc=range(1,t)

    if ax is None:
        fig,ax = plt.subplots(figsize=(10,10))
    if asymp_arg is None:
        asymp_arg = {'label':r'$q_{\chi^2}(0.95)$' if label else '',
                    'ls':'--','c':'crimson','lw': 4}
    if title is not None:
        ax.set_title(title,fontsize=title_fontsize)
    
    ax.set_xlabel(r'regularization $t$',fontsize= 20)
    ax.set_ylabel(r'$\frac{n_1 n_2}{n} \Vert \widehat{\Sigma}_{W}^{-1/2}(t)(\widehat{\mu}_2 - \widehat{\mu}_1) \Vert _\mathcal{H}^2$',fontsize= 20)
    ax.set_xlim(0,trunc[-1])
    
    yas = [chi2.ppf(0.95,t) for t in trunc] 
    ax.plot(trunc,yas,**asymp_arg)

    if ylim is not None:
        ax.set_ylim(ylim)

    return(fig,ax)

def init_plot_pvalue(fig=None,ax=None,ylim=None,t=None,label=False,title=None,title_fontsize=40,asymp_arg=None,log=False):

    assert(t is not None)
    trunc=range(1,t)

    if ax is None:
        fig,ax = plt.subplots(figsize=(10,10))
    if asymp_arg is None:
        asymp_arg = {'label':r'$q_{\chi^2}(0.95)$' if label else '',
                    'ls':'--','c':'crimson','lw': 4}
    if title is not None:
        ax.set_title(title,fontsize=title_fontsize)
    
    ax.set_xlabel(r'regularization $t$',fontsize= 20)
    ax.set_ylabel(r'$\frac{n_1 n_2}{n} \Vert \widehat{\Sigma}_{W}^{-1/2}(t)(\widehat{\mu}_2 - \widehat{\mu}_1) \Vert _\mathcal{H}^2$',fontsize= 20)
    ax.set_xlim(0,trunc[-1])
    
    yas = [-np.log(.05) for t in trunc] if log else [.05 for t in trunc]
    ax.plot(trunc,yas,**asymp_arg)

    if ylim is not None:
        ax.set_ylim(ylim)

    return(fig,ax)

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

def plot_pvalue(self,fig=None,ax=None,ylim=None,t=None,columns=None,title=None,title_fontsize=40,mean=False,mean_label='mean',mean_color = 'xkcd: indigo',contrib=False,label_asymp=False,asymp_arg=None,legend=True,log=False):
    # on a pas le choix avec BH pour l'instant 
    # try:
        if columns is None:
            columns = self.df_kfdat.columns
        kfdat = self.df_pval[columns].copy()
        if log :
            kfdat = -np.log(kfdat)

        t = max([(~kfdat[c].isna()).sum() for c in columns]) if t is None and len(columns)==0 else \
            100 if t is None else t 
        trunc = range(1,t)  
        no_data_to_plot = len(columns)==0

        fig,ax = init_plot_pvalue(fig=fig,ax=ax,ylim=ylim,t=t,label=label_asymp,title=title,title_fontsize=title_fontsize,asymp_arg=asymp_arg,log=log)
        
        # if mean:
        #     ax.plot(kfdat.mean(axis=1),label=mean_label,c=mean_color)
        #     ax.plot(kfdat.mean(axis=1)- 2* kfdat.std(axis=1)/(~kfdat[columns[0]].isna()).sum(),c=mean_color,ls = '--',alpha=.5)
        #     ax.plot(kfdat.mean(axis=1)+ 2* kfdat.std(axis=1)/(~kfdat[columns[0]].isna()).sum(),c=mean_color,ls = '--',alpha=.5)

        if len(self.df_kfdat.columns)>0:
            kfdat.plot(ax=ax)

        # if ylim is None and not no_data_to_plot:
            # if log :
            # # probleme je voulais pas yas dans cette fonction 
            
            # yas = [chi2.ppf(0.95,t) for t in trunc] 
            # ymax = np.max([yas[-1], np.nanmax(np.isfinite(kfdat[kfdat.index == trunc[-1]]))]) # .max(numeric_only=True)
            # ylim = (-5,ymax)
            # ax.set_ylim(ylim)
        if contrib and len(columns)==1:
            self.plot_kfdat_contrib(fig,ax,t,columns[0])
        if legend:
           ax.legend()  
        return(fig,ax)

#    
def plot_kfdat_contrib(self,fig=None,ax=None,t=None,name=None):
    
    cov = self.approximation_cov    
    sp,ev = self.spev['xy'][cov]['sp'],self.spev['xy'][cov]['ev']
    n1,n2 = self.n1, self.n2 
    n = n1+n2 
    
    pkom = self.compute_pkm()
    om = self.compute_omega()
    K = self.compute_gram()
    mmd = dot(mv(K,om),om)
   
    # yp = n1*n2/n * 1/(sp[:t]*n) * mv(ev.T[:t],pkom)**2 #1/np.sqrt(n*sp[:t])*
    # ysp = sp[:t]    

    xp = range(1,t+1)
    yspnorm = sp[:t]/torch.sum(sp)
    ypnorm = 1/(sp[:t]*n) * mv(ev.T[:t],pkom)**2 /mmd
    # il y a des erreurs numériques sur les f donc je n'ai pas une somme totale égale au MMD


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
#    
def plot_spectrum(self,fig=None,ax=None,t=None,title=None,sample='xy',label=None):
    if ax is None:
        fig,ax = plt.subplots(figsize=(10,10))
    if title is not None:
        ax.set_title(title,fontsize=40)
    
    sp = self.spev[sample][self.approximation_cov]['sp']
    
    t = len(sp) if t is None else t
    trunc = range(1,t)
    ax.plot(trunc,sp[:trunc[-1]],label=label)
    ax.set_xlabel('t',fontsize= 20)

    return(fig,ax)
#
def density_proj(self,ax,projection,which='proj_kfda',name=None,orientation='vertical',sample='xy',labels='CT'):
    
    df_proj= self.init_df_proj(which,name)

    for xy,l in zip(sample,labels):
        
        dfxy = df_proj.loc[df_proj['sample']==xy][str(projection)]
        if len(dfxy)>0:
            color = 'blue' if xy =='x' else 'orange'
            bins=int(np.floor(np.sqrt(len(dfxy))))
            ax.hist(dfxy,density=True,histtype='bar',label=f'{l}({len(dfxy)})',alpha=.3,bins=bins,color=color,orientation=orientation)
            ax.hist(dfxy,density=True,histtype='step',bins=bins,lw=3,edgecolor=color,orientation=orientation)
            if orientation =='vertical':
                ax.axvline(dfxy.mean(),c=color)
            else:
                ax.axhline(dfxy.mean(),c=color)

    ax.set_xlabel(f't={projection}',fontsize=20)    
    ax.legend()
#       

# cette fonction peut servir a afficher plein de figures usuelles, que je dois définir et nommer pour créer une fonction par figure usuelle

def set_color_for_scatter(self,color):
    # pas de couleur
    if color is None:
        color_ = {'x':'xkcd:cerulean',
                'y':'xkcd:light orange'}
    # couleur de variable
    if  isinstance(color,str):
        color_ = {'title':'color'}
        if color in list(self.variables):
            x,y = self.get_xy()
            color_ = {'x':x[:,self.variables.get_loc(color)], 
                    'y':y[:,self.variables.get_loc(color)]}
        elif color in list(self.obs.columns):
            if self.obs[color].dtype == 'category':
                color_ = {pop:self.obs[self.obs[color]==pop].index for pop in self.obs[color].cat.categories}
            else:
                color_ = {'x':self.obs.loc[self.obs['sample']=='x'][color],
                        'y':self.obs.loc[self.obs['sample']=='y'][color]}
    # assert('x' in color)
        else:
            color_=color
            print(f'{color} not found in obs and variables')

    if isinstance(color_,dict) and 'x' in color_ and len(color_['x'])>1:
        color_['mx'] = 'xkcd:cerulean'
        color_['my'] = 'xkcd:light orange'

    return(color_)
    
def scatter_proj(self,ax,projection,xproj='proj_kfda',yproj=None,xname=None,yname=None,highlight=None,color=None,sample='xy',labels='CT'):

    p1,p2 = projection
    yproj = xproj if yproj is None else yproj
    if xproj == yproj and yname is None:
        yname = xname
    df_abscisse = self.init_df_proj(xproj,xname)
    df_ordonnee = self.init_df_proj(yproj,yname)
    color = self.set_color_for_scatter(color)
    
    for xy,l in zip(sample,labels):
        
        df_abscisse_xy = df_abscisse.loc[df_abscisse['sample']==xy]
        df_ordonnee_xy = df_ordonnee.loc[df_ordonnee['sample']==xy]
        m = 'x' if xy =='x' else '+'
        if len(df_abscisse_xy)>0 and len(df_ordonnee_xy)>0 :
            if xy in color :
                
                x_ = df_abscisse_xy[f'{p1}'] #[df_abscisse_xy.index.isin(ipop)]
                y_ = df_ordonnee_xy[f'{p2}'] #[df_ordonnee_xy.index.isin(ipop)]
                # ax.scatter(x_,y_,c=c,s=30,label=,alpha=.8,marker =m)
                ax.scatter(x_,y_,s=30,c=color[xy],label=f'{l}({len(x_)})', alpha=.8,marker =m)
            else:
                for pop,ipop in color.items():
                    x_ = df_abscisse_xy[f'{p1}'][df_abscisse_xy.index.isin(ipop)]
                    y_ = df_ordonnee_xy[f'{p2}'][df_ordonnee_xy.index.isin(ipop)]
                    if len(x_)>0:
                        ax.scatter(x_,y_,s=30,label=f'{pop} {l}({len(x_)})',alpha=.8,marker =m)
            

            # if color is None or (isinstance(color,str) and color in list(self.variables)): # list vraiment utile ? 
                # c = 'xkcd:cerulean' if xy =='x' else 'xkcd:light orange'
                # if color in list(self.variables):
                #     x,y = self.get_xy()
                #     c = x[:,self.variables.get_loc(color)] if xy=='x' else y[:,self.variables.get_loc(color)]   
            
                # x_ = df_abscisse_xy[f'{p1}']
                # y_ = df_ordonnee_xy[f'{p2}']

                # ax.scatter(x_,y_,c=c,s=30,label=f'{l}({len(x_)})',alpha=.8,marker =m)
            
            # else:
            #     if xy in color: # a complexifier si besoin (nystrom ou mask) 
            #         x_ = df_abscisse_xy[f'{p1}'] #[df_abscisse_xy.index.isin(ipop)]
            #         y_ = df_ordonnee_xy[f'{p2}'] #[df_ordonnee_xy.index.isin(ipop)]
            #         ax.scatter(x_,y_,s=30,c=color[xy], alpha=.8,marker =m)


            
    
    for xy,l in zip(sample,labels):

        df_abscisse_xy = df_abscisse.loc[df_abscisse['sample']==xy]
        df_ordonnee_xy = df_ordonnee.loc[df_ordonnee['sample']==xy]
        x_ = df_abscisse_xy[f'{p1}']
        y_ = df_ordonnee_xy[f'{p2}']
        if len(df_abscisse_xy)>0 and len(df_ordonnee_xy)>0 :
            if highlight is not None:
                x_ = df_abscisse_xy[f'{p1}']
                y_ = df_ordonnee_xy[f'{p2}']
                ax.scatter(x_[x_.index.isin(highlight)],y_[y_.index.isin(highlight)],c=color[xy],s=100,marker='*',edgecolor='black',linewidths=1)
            if xy in color:
                mx_ = x_.mean()
                my_ = y_.mean()
                c = color[f'm{xy}'] if f'm{xy}' in color else color[xy]
                ax.scatter(mx_,my_,edgecolor='black',linewidths=3,s=200,c=c)
            else:
                for pop,ipop in color.items():
                    x_ = df_abscisse_xy[f'{p1}'][df_abscisse_xy.index.isin(ipop)]
                    y_ = df_ordonnee_xy[f'{p2}'][df_ordonnee_xy.index.isin(ipop)]
                    if len(x_)>0:
                        mx_ = x_.mean()
                        my_ = y_.mean()
                        ax.scatter(mx_,my_,edgecolor='black',linewidths=3,s=200)

            
            
    
    if 'title' in color :
        ax.set_title(color['title'],fontsize=20)

        
    xlabel = xproj if xproj in self.variables else xproj.split(sep='_')[1]+f': t={p1}'
    ylabel = yproj if yproj in self.variables else yproj.split(sep='_')[1]+f': t={p2}'
    ax.set_xlabel(xlabel,fontsize=20)                    
    ax.set_ylabel(ylabel,fontsize=20)
    
    ax.legend()

def init_axes_projs(self,fig,axes,projections,sample,suptitle,kfda,kfda_ylim,t,kfda_title,spectrum,spectrum_label):
    if axes is None:
        rows=1;cols=len(projections) + kfda + spectrum
        fig,axes = plt.subplots(nrows=rows,ncols=cols,figsize=(6*cols,6*rows))
    if suptitle is not None:
        fig.suptitle(suptitle,fontsize=50)
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
        self.density_proj(ax,proj,which=which,name=name,labels=labels,sample=sample)
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

def find_cells_from_proj(self,which='proj_kfda',name=None,t=1,bound=0,side='left'):
    df_proj= self.init_df_proj(which,name=name)
    return df_proj[df_proj[str(t)] <= bound].index if side =='left' else df_proj[df_proj[str(t)] >= bound].index

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

