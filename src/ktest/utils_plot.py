import matplotlib.pyplot as plt
from scipy.stats import chi2
from adjustText import adjust_text

def text_truncations_of_interest(truncations_of_interest,ax,values):
    texts = []
    for t in set(truncations_of_interest):
        valt = values[t]
        text = f'{valt:.2f}' if valt >=.01 else f'{valt:1.0e}'
        ax.scatter(x=t,y=valt,s=20)
        texts += [ax.text(t,valt,text,fontsize=20)]
    adjust_text(texts,only_move={'points': 'y', 'text': 'y', 'objects': 'y'})

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

def init_plot_pvalue(fig=None,ax=None,ylim=None,t=None,label=False,title=None,title_fontsize=40,log=False):

    assert(t is not None)
    trunc=range(1,t)

    if ax is None:
        fig,ax = plt.subplots(figsize=(10,10))
    asymp_arg = {'label':r'$q_{\chi^2}(0.95)$' if label else '',
                'ls':'--','c':'crimson','lw': 4}
    if title is not None:
        ax.set_title(title,fontsize=title_fontsize)
    
    ax.set_xlabel(r'regularization $t$',fontsize= 20)
    ax.set_ylabel(r'$\frac{n_1 n_2}{n} \Vert \widehat{\Sigma}_{W}^{-1/2}(t)(\widehat{\mu}_2 - \widehat{\mu}_1) \Vert _\mathcal{H}^2$',fontsize= 20)
    ax.set_xlim(0,trunc[-1])
    ax.set_xticks(trunc)
    
    yas = [-np.log(.05) for t in trunc] if log else [.05 for t in trunc]
    ax.plot(trunc,yas,**asymp_arg)

    if ylim is not None:
        ax.set_ylim(ylim)

    return(fig,ax)

def replace_label(ax,position,new_label):
    ax.legend_.texts[position]._text = new_label