import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from itertools import product
import matplotlib as mpl

def replace_label(ax,position,new_label):
    ax.legend_.texts[position]._text = new_label

def points_in_boxplot(df,ax,colors=None,vert=False):

    # ajouter les points 
    vals, ys = [], [] 
    for i,c in enumerate(df.columns):
        vals.append(df[c])
        ys.append(np.random.normal(i+1, 0.04, len(df)))
    ngroup = len(vals)
    clevels = np.linspace(0., 1., ngroup)
    colors = list(colors.values()) if colors is not None else [None]*len(ys)
    for x, val, color in zip(ys, vals, colors):
        if vert :
            ax.scatter(x,val, c=color, alpha=1)  
        else:
            ax.scatter(val,x,c=color,alpha=1)

def filled_boxplot(df,ax,colors=None,alpha=.5,vert=False):
    bp = df.boxplot(ax=ax,return_type='both',patch_artist=True,vert=vert)

    for i,patch in enumerate(bp[1]['boxes']):
        if colors is not None:
            color=list(colors.values())[i]
            patch.set(facecolor=color,edgecolor=color)

        patch.set(alpha=alpha,
                 fill=True,
                 linewidth=0)

def contours_boxplot(df,ax,colors=None,lw=3,vert=False):
    bp = df.boxplot(ax=ax,return_type='both',patch_artist=True,vert=vert)
    for i,patch in enumerate(bp[1]['boxes']):
        if colors is not None:
            color=list(colors.values())[i]
            patch.set(edgecolor=color)

        patch.set(alpha=1,
                 fill=False,
                 linewidth=lw)
    if colors is not None:
        for i,cap in enumerate(bp[1]['caps']):
            cap.set(color=list(colors.values())[i//2],linewidth=lw)

        for i,whisker in enumerate(bp[1]['whiskers']):
            whisker.set(color=list(colors.values())[i//2],linewidth=lw)
        
        
def custom_boxplot(df,colors=None,alpha=.5,lw=3,scatter=True,fig=None,ax=None,vert=True):
    if fig is None:
        fig,ax = plt.subplots(figsize=(20,7))
    
    if colors is not None:
        colors = {s:c for s,c in colors.items() if s in df} 
        df = df[list(colors.keys())]
    
    filled_boxplot(df=df,colors=colors,alpha=alpha,ax=ax,vert=vert)
    contours_boxplot(df=df,colors=colors,lw=lw,ax=ax,vert=vert)
    if scatter: 
        points_in_boxplot(df=df,colors=colors,ax=ax,vert=vert)
    return(fig,ax)


def custom_kde(data,
               normalize=True,
               coef=1,
               minmax=None,
               bw_method=.2,
               yshift=0,
               xshift=0
            ):
    if minmax is not None:
        min,max = minmax
        min,max = min - .1*(max-min),max +.1*(max-min)
    else:
        min,max = data.min(),data.max()
        min,max = min - .1*(max-min),max +.1*(max-min)

    
    x = np.linspace(min,max,200)
    density = gaussian_kde(data,bw_method=bw_method)
    y = density(x)
    if normalize : 
        y = y/np.max(y) 
    y = coef*y+yshift
    x = x+xshift
    return(x,y)


def custom_violin(data,
               fig=None,
               ax=None,
               orientation='vertical',
               alpha=.5,
               label=None,
               color=None, 
               lw=2,
               minmax=None,
               bw_method=.2,
               yshift=0,
               xshift=0
               ):
    if fig is None:
        fig,ax = plt.subplots(figsize=(12,6))
    
    for coef in [-1,1]:
        if coef == 1:
            label = None
        fig,ax,c = custom_rug(data,
                   fig=fig,
                   ax=ax,
                   orientation=orientation,
                   alpha=alpha,
                   label=label,
                   color=color, 
                   lw=lw,
                   normalize=True,
                   coef=coef,
                   minmax=minmax,
                   bw_method=bw_method,
                   yshift=yshift,
                   xshift=xshift
                   )
        if color is None: 
            color=c
    return(fig,ax,color)

def custom_rug(data,
               fig=None,
               ax=None,
               orientation='vertical',
               alpha=.5,
               label=None,
               color=None, 
               lw=2,
               normalize=True,
               coef=1,
               minmax=None,
               bw_method=.2,
               yshift=0,
               xshift=0
               ):
    if fig is None:
        fig,ax = plt.subplots(ncols=1,figsize=(12,6))

    x,y = custom_kde(data=data,
               normalize=normalize,
               coef=coef,
               minmax=minmax,
               bw_method=bw_method,
               yshift=yshift,
               xshift=xshift)

    if orientation == 'vertical':
        ax.plot(x,y,color=color,lw=lw,)
    else:
        ax.plot(y,x,color=color,lw=lw)

    if color is None:
        color = ax._children[-1]._color


    if orientation == 'vertical':
        ax.fill_between(x,y,y2=yshift,color=color,label=label,alpha=alpha)
    else:
        ax.fill_betweenx(x,y,x2=yshift,color=color,label=label,alpha=alpha)
        

    return(fig,ax,color)


def double_histogram(data,
                     fig=None,
                     ax=None,
                     alpha=.5,
                     orientation='vertical',
                     label=None,
                     color=None,
                     edgecolor=None,
                     coef_bins=3,
                     yshift=0
                     ):
    
    if fig is None:
        fig,ax = plt.subplots(ncols=1,figsize=(12,6))
    bins = coef_bins*int(np.floor(np.sqrt(len(data))))

    ax.hist(data,
        density=True,
        histtype='bar',
        bins=bins,
        alpha=alpha,
        orientation=orientation,
        label=label,color=color)
    
    if edgecolor is None:
        edgecolor=ax._children[-1]._facecolor
    if color is None:
        color = ax._children[-1]._facecolor

    ax.hist(data,
            density=True,
            histtype='step',
            bins=bins,
            lw=3,
            orientation=orientation,
            edgecolor=edgecolor)    

    return(fig,ax,color)

def custom_histogram(data,
                     fig=None,
                     ax=None,
                     orientation='vertical',
                     alpha=.5,
                     label=None,
                     color=None,
                     edgecolor=None,
                     lw=2,
                     coef_bins=3,
                     means=True,
                     hist_type='kde',
                     kde_bw=.2,
                     minmax=None,
                     xshift=0,
                     yshift=0,
                     normalize=True
                     ):
    """
    Parameters
    ----------
        type (default = 'kde') : in ['kde','violin','hist'] 
    """

    if hist_type=='kde':
        if not len(data[data==0])==len(data):
            fig,ax,color=custom_rug(data=data,
               fig=fig,
               ax=ax,
               orientation=orientation,
               alpha=alpha,
               label=label,
               color=color, 
               lw=lw,
               minmax=minmax,
               bw_method=kde_bw,
               xshift=xshift,
               yshift=yshift,
               normalize=normalize
               )

    if hist_type == 'violin':
        if not len(data[data==0])==len(data):
            fig,ax,color=custom_violin(data=data,
               fig=fig,
               ax=ax,
               orientation=orientation,
               alpha=alpha,
               label=label,
               color=color, 
               lw=lw,
               minmax=minmax,
               bw_method=kde_bw,
               xshift=xshift,
               yshift=yshift,
               )


    if hist_type == 'hist':
        fig,ax,color= double_histogram(data=data,
                     fig=fig,
                     ax=ax,
                     alpha=alpha,
                     orientation=orientation,
                     label=label,
                     color=color,
                     edgecolor=edgecolor,
                     coef_bins=coef_bins
                     )

    if means == 'line': 
        if orientation =='vertical':
            ax.axvline(data.mean(),c=color,lw=1.5)
        else:
            ax.axhline(data.mean(),c=color,lw=1.5)
    elif means:
        if orientation == 'vertical':
            ax.scatter(data.mean()+xshift,yshift,marker='+',s=100,c=color)
        else:
            ax.scatter(yshift,data.mean()+xshift,marker='+',s=100,color=color)            
    return(fig,ax)

def highlight_on_histogram(data,
                     fig=None,
                     ax=None,
                     orientation='vertical',
                     label=None,
                     color=None,
                     marker='*',
                     coef_bins=3,
                     linewidths=3,
                     means=True):
    if fig is None:
        fig,ax =plt.subplots(figsize=(12,6))
    nh = len(data)
    if label is None:
        label=f'({nh})'
    else:
        label = f'{label} ({nh})'

    if orientation =='vertical':
        x = data
        y = np.array([ax.get_ylim()[1]*.05]*nh)           
    else:
        y = data
        x = np.array([ax.get_ylim()[1]*.05]*nh)                              

    ax.scatter(x,y,color=color,s=100,marker=marker,edgecolor='black',linewidths=1)    
    ax.scatter(x.mean(),y.mean(),color=color,s=200,marker=marker,
               edgecolor='black',linewidths=linewidths,label=label)    


def plot_effectifs_matrix(effectifs,labels,fig=None,ax=None):
    

    n=len(labels)//2
    fig,ax=plt.subplots(figsize=(n,n))
    ax.matshow(effectifs, cmap=plt.cm.Blues)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels,rotation=90)#,rot)
    ax.set_yticklabels(labels)

    for i,j in product(range(len(labels)),range(len(labels))):
        ax.text(i, j, f'{effectifs[i,j]:.2f}', va='center', ha='center')

    return(fig,ax)

def plot_dendrogram_from_distance_matrix(s,labels=None,fig=None,ax=None):
    from scipy.cluster.hierarchy import dendrogram,linkage
    if fig is None:
        fig,ax = plt.subplots(figsize=(20,7))
    n=len(s)
    
    Z = linkage(s[np.triu_indices(n,k=1)])
    d = dendrogram(Z,labels=labels,ax=ax,)
    return(d,fig,ax)


def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)