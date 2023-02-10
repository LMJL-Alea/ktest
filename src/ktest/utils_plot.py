import matplotlib.pyplot as plt
from scipy.stats import chi2
from adjustText import adjust_text
import numpy as np

def text_truncations_of_interest(truncations_of_interest,ax,values,adjust=True,log=False):
    texts = []
    for t in set(truncations_of_interest):
        valt = values[t]
        if log:
            text = f'{valt:.2f}'
        else:
            text = f'{valt:.2f}' if valt >=.01 else f'{valt:1.0e}'
        ax.scatter(x=t,y=valt,s=20)
        texts += [ax.text(t,valt,text,fontsize=20)]
    if adjust:
        adjust_text(texts,only_move={'points': 'y', 'text': 'y', 'objects': 'y'})

def init_plot_kfdat(fig=None,ax=None,ylim=None,t=None,label=False,
                    title=None,title_fontsize=40,asymp_arg=None):

    assert(t is not None)
    trunc=range(1,t)

    if ax is None:
        fig,ax = plt.subplots(figsize=(12,6))
    if asymp_arg is None:
        asymp_arg = {'label':r'$q_{\chi^2}(0.95)$' if label else '',
                    'ls':'--','c':'crimson','lw': 4}
    if title is not None:
        ax.set_title(title,fontsize=title_fontsize)
    
    ax.set_xlabel(r'regularization $t$',fontsize= 20)
    ax.set_ylabel(r'Statistic',fontsize= 20)
    ax.set_xlim(0,trunc[-1])
    
    yas = [chi2.ppf(0.95,t) for t in trunc] 
    ax.plot(trunc,yas,**asymp_arg)

    if ylim is not None:
        ax.set_ylim(ylim)

    return(fig,ax)


def adjusted_xticks(xmax):
    xticks = np.arange(0,xmax) if xmax<20 else np.arange(0,xmax,2) if xmax<50 else np.arange(0,xmax,10) if xmax<200 else np.arange(0,xmax,20) if xmax < 500 else np.arange(0,xmax,50)
    return(xticks)

def init_plot_pvalue(fig=None,ax=None,ylim=None,t=None,label=False,
                    title=None,title_fontsize=40,log=False):

    assert(t is not None)
    trunc=range(1,t)

    if ax is None:
        fig,ax = plt.subplots(figsize=(12,6))
    asymp_arg = {'label':r'$q_{\chi^2}(0.95)$' if label else '',
                'ls':'--','c':'crimson','lw': 4}
    if title is not None:
        ax.set_title(title,fontsize=title_fontsize)
    
    ax.set_xlabel(r'$t$',fontsize= 30)
    ax.set_ylabel('log p-value' if log else 'p-value',fontsize= 30)
    ax.set_xlim(0,trunc[-1])
    ax.set_xticks(adjusted_xticks(t))
    
    yas = [np.log(.05) for t in trunc] if log else [.05 for t in trunc]
    ax.plot(trunc,yas,**asymp_arg)

    if ylim is not None and log == False:
        ax.set_ylim(ylim)
    return(fig,ax)

def replace_label(ax,position,new_label):
    ax.legend_.texts[position]._text = new_label