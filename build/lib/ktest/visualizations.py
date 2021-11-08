import matplotlib.pyplot as plt
from scipy.stats import chi2
import numpy as np


def init_plot_kfdat(fig=None,ax=None,ylim=None,figsize=(10,10),t=None,ls = '--',c='crimson',label=False,title=None,title_fontsize=40):
    assert(t is not None)

    if ax is None:
        fig,ax = plt.subplots(figsize=figsize)
    trunc=range(1,t)
    yas = [chi2.ppf(0.95,t) for t in trunc] 
    kwargs = {'label':r'$q_{\chi^2}(0.95)$'} if label else {}
    ax.plot(trunc,yas,ls=ls,c=c,lw=4,**kwargs)
    ax.set_xlabel(r'regularization $t$',fontsize= 20)
    ax.set_ylabel(r'$\frac{n_1 n_2}{n} \Vert \widehat{\Sigma}_{W}^{-1/2}(t)(\widehat{\mu}_2 - \widehat{\mu}_1) \Vert _\mathcal{H}^2$',fontsize= 20)
    ax.set_xlim(0,trunc[-1])
    if title is not None:
        ax.set_title(title,fontsize=title_fontsize)
    if ylim is not None:
        ax.set_ylim(ylim)
    return(fig,ax)

def plot_kfdat(self,fig=None,ax=None,ylim=None,figsize=(10,10),t=None,columns=None,asymp_ls='--',asymp_c = 'crimson',title=None,title_fontsize=40,mean=False,mean_label='mean',mean_color = 'xkcd: indigo',contrib=True):
    # try:
        if columns is None:
            columns = self.df_kfdat.columns
        kfdat = self.df_kfdat[columns].copy()

        t = max([(~kfdat[c].isna()).sum() for c in columns]) if t is None and len(columns)==0 else \
            100 if t is None else t 
        trunc = range(1,t)  
        no_data_to_plot = len(columns)==0

        fig,ax = init_plot_kfdat(fig=fig,ax=ax,ylim=ylim,figsize=figsize,t=t,ls=asymp_ls,c=asymp_c,label=no_data_to_plot,title=title,title_fontsize=title_fontsize)
        
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

        ax.legend()  
        return(fig,ax)
#       
def plot_spectrum(self,fig=None,ax=None,figsize=(10,10),t=None,title=None,generate_spectrum=True,approximation_cov='standard',sample='xy',label=None):
    if ax is None:
        fig,ax = plt.subplots(figsize=figsize)
    if title is not None:
        ax.set_title(title,fontsize=40)
    sp = self.spev[sample][approximation_cov]['sp']
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
def scatter_proj(self,ax,projection,xproj='proj_kfda',yproj=None,name=None,highlight=None,color=None,sample='xy',labels='CT'):

    p1,p2 = projection
    yproj = xproj if yproj is None else yproj
        
    df_abscisse = self.init_df_proj(xproj,name)
    df_ordonnee = self.init_df_proj(yproj,name)
    
    for xy,l in zip(sample,labels):
        df_abscisse_xy = df_abscisse.loc[df_abscisse['sample']==xy]
        df_ordonnee_xy = df_ordonnee.loc[df_ordonnee['sample']==xy]
        m = 'x' if xy =='x' else '+'
        if len(df_abscisse_xy)>0 and len(df_ordonnee_xy)>0 :
            if color is None or color in list(self.variables): # list vraiment utile ? 
                c = 'xkcd:cerulean' if xy =='x' else 'xkcd:light orange'
                if color in list(self.variables):
                    x,y = self.get_xy()
                    c = x[:,self.variables.get_loc(color)] if xy=='x' else y[:,self.variables.get_loc(color)]   
                x_ = df_abscisse_xy[f'{p1}']
                y_ = df_ordonnee_xy[f'{p2}']

                ax.scatter(x_,y_,c=c,s=30,label=f'{l}({len(x_)})',alpha=.8,marker =m)
            else:
                if xy in color: # a complexifier si besoin (nystrom ou mask) 
                    x_ = df_abscisse_xy[f'{p1}'] #[df_abscisse_xy.index.isin(ipop)]
                    y_ = df_ordonnee_xy[f'{p2}'] #[df_ordonnee_xy.index.isin(ipop)]
                    ax.scatter(x_,y_,s=30,c=color[xy], alpha=.8,marker =m)
                for pop,ipop in color.items():
                    x_ = df_abscisse_xy[f'{p1}'][df_abscisse_xy.index.isin(ipop)]
                    y_ = df_ordonnee_xy[f'{p2}'][df_ordonnee_xy.index.isin(ipop)]
                    if len(x_)>0:
                        ax.scatter(x_,y_,s=30,label=f'{pop} {l}({len(x_)})',alpha=.8,marker =m)

    
    for xy,l in zip(sample,labels):

        df_abscisse_xy = df_abscisse.loc[df_abscisse['sample']==xy]
        df_ordonnee_xy = df_ordonnee.loc[df_ordonnee['sample']==xy]
        x_ = df_abscisse_xy[f'{p1}']
        y_ = df_ordonnee_xy[f'{p2}']
        if len(df_abscisse_xy)>0 and len(df_ordonnee_xy)>0 :
            if highlight is not None:
                x_ = df_abscisse_xy[f'{p1}']
                y_ = df_ordonnee_xy[f'{p2}']
                c = 'xkcd:cerulean' if xy =='x' else 'xkcd:light orange'
                ax.scatter(x_[x_.index.isin(highlight)],y_[y_.index.isin(highlight)],c=c,s=100,marker='*',edgecolor='black',linewidths=1)

            mx_ = x_.mean()
            my_ = y_.mean()
            ax.scatter(mx_,my_,edgecolor='black',linewidths=3,s=200)

    if color in list(self.variables) :
        ax.set_title(color,fontsize=20)

        
    xlabel = xproj if xproj in self.variables else xproj.split(sep='_')[1]+f': t={p1}'
    ylabel = yproj if yproj in self.variables else yproj.split(sep='_')[1]+f': t={p2}'
    ax.set_xlabel(xlabel,fontsize=20)                    
    ax.set_ylabel(ylabel,fontsize=20)
    
    ax.legend()

def init_axes_projs(self,fig,axes,projections,approximation_cov,sample,suptitle,kfda,kfda_ylim,t,kfda_title,spectrum,spectrum_label):
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
        self.plot_spectrum(axes[0],t=t,title='spectrum',approximation_cov=approximation_cov,sample=sample,label=spectrum_label)
        axes = axes[1:]
    return(fig,axes)

def density_projs(self,fig=None,axes=None,which='proj_kfda',approximation_cov='standard',sample='xy',name=None,projections=range(1,10),suptitle=None,kfda=False,kfda_ylim=None,t=None,kfda_title=None,spectrum=False,spectrum_label=None,labels='CT'):
    fig,axes = self.init_axes_projs(fig=fig,axes=axes,projections=projections,approximation_cov=approximation_cov,sample=sample,suptitle=suptitle,kfda=kfda,
                                    kfda_ylim=kfda_ylim,t=t,kfda_title=kfda_title,spectrum=spectrum,spectrum_label=spectrum_label)
    if not isinstance(axes,np.ndarray):
        axes = [axes]
    for ax,proj in zip(axes,projections):
        self.density_proj(ax,proj,which=which,name=name,labels=labels,sample=sample)
    fig.tight_layout()
    return(fig,axes)

def scatter_projs(self,fig=None,axes=None,xproj='proj_kfda',approximation_cov='standard',sample='xy',yproj=None,name=None,projections=[(1,i+2) for i in range(10)],suptitle=None,
                    highlight=None,color=None,kfda=False,kfda_ylim=None,t=None,kfda_title=None,spectrum=False,spectrum_label=None,iterate_over='projections',labels='CT'):
    to_iterate = projections if iterate_over == 'projections' else color
    fig,axes = self.init_axes_projs(fig=fig,axes=axes,projections=to_iterate,approximation_cov=approximation_cov,sample=sample,suptitle=suptitle,kfda=kfda,
                                    kfda_ylim=kfda_ylim,t=t,kfda_title=kfda_title,spectrum=spectrum,spectrum_label=spectrum_label)
    if not isinstance(axes,np.ndarray):
        axes = [axes]
    for ax,obj in zip(axes,to_iterate):
        if iterate_over == 'projections':
            self.scatter_proj(ax,obj,xproj=xproj,yproj=yproj,name=name,highlight=highlight,color=color,labels=labels,sample=sample)
        elif iterate_over == 'color':
            self.scatter_proj(ax,projections,xproj=xproj,yproj=yproj,name=name,highlight=highlight,color=obj,labels=labels,sample=sample)
    fig.tight_layout()
    return(fig,axes)

def find_cells_from_proj(self,which='proj_kfda',name=None,t=1,bound=0,side='left'):
    df_proj= self.init_df_proj(which,name=name)
    return df_proj[df_proj[str(t)] <= bound].index if side =='left' else df_proj[df_proj[str(t)] >= bound].index

def plot_correlation_proj_var(self,ax=None,name=None,figsize=(10,10),nvar=30,projections=range(1,10),title=None,prefix_col=''):
    if name is None:
        name = self.get_names()['correlations'][0]
    if ax is None:
        fig,ax = plt.subplots(figsize=figsize)
    if title is not None:
        ax.set_title(title,fontsize=40)
    for proj in projections:
        col = f'{prefix_col}{proj}'
        val  = list(np.abs(self.corr[name][col]).sort_values(ascending=False).values)[:nvar]
        print(val)
        ax.plot(val,label=col)
    ax.legend()
    return(ax)
