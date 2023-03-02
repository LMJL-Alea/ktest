

import matplotlib.cm as cm
import matplotlib as matplotlib

def filter_genes_wrt_pval(pval,exceptions=[],focus=None,zero_pvals=False,threshold=1):
    
    pval = pval if focus is None else pval[pval.index.isin(focus)]
    pval = pval[~pval.index.isin(exceptions)]
    pval = pval[pval<threshold]
    pval = pval[~pval.isna()]
    pval = pval[pval == 0] if zero_pvals else pval[pval>0]
    return(pval)

    
def color_map_color(value, cmap_name='viridis', vmin=0, vmax=1):
    # norm = plt.Normalize(vmin, vmax)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)  # PiYG
    rgb = cmap(norm(abs(value)))[:3]  # will return rgba, we take only first 3 so we get rgb
    color = matplotlib.colors.rgb2hex(rgb)
    return color

