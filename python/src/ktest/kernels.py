
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
from scipy.stats import norm

"""
Ces fonctions déterminent la fonction noyau 
"""




def torch_transformator(x):
    if (isinstance(x, np.ndarray)):
        x = torch.from_numpy(x).type(torch.double)
    if (isinstance(x,pd.DataFrame)) or (isinstance(x,pd.Series)) :
        x = torch.tensor(x)
    return(x)


def distances(x, y=None):
    """
    If Y=None, then this computes the distance between X and itself
    """
    x=torch_transformator(x)
    y=torch_transformator(y)
    
    assert(x.ndim == 2)

    if y is None:
        sq_dists = torch.cdist(x, x, compute_mode='use_mm_for_euclid_dist_if_necessary').pow(
            2)  # [sq_dists]_ij=||X_j - X_i \\^2
    else:
        assert(y.ndim == 2)
        assert(x.shape[1] == y.shape[1])
        sq_dists = torch.cdist(x, y, compute_mode='use_mm_for_euclid_dist_if_necessary').pow(
            2)  # [sq_dists]_ij=||x_j - y_i \\^2
    return sq_dists

def mediane(x, y=None,verbose=0):
    """
    Computes the median 
    """
    x=torch_transformator(x)
    y=torch_transformator(y)
    
    dxx = distances(x)
    if y == None:
        return dxx.median()
    dyy = distances(y)
    dxy = distances(x,y)
    dyx = dxy.t()
    dtot = torch.cat((torch.cat((dxx,dxy),dim=1),
                      torch.cat((dyx,dyy),dim=1)),dim=0)
    median = dtot.median()
    if median == 0: 
        if verbose>0 :
            print('warning: the median is null. To avoid a kernel with zero bandwidth, we replace the median by the mean')
        mean = dtot.mean()
        if mean == 0 : 
            print('warning: all your dataset is null')
        return mean
    else:
        return dtot.median()

def gauss_kernel(x, y, sigma=1):
    """
    Computes the standard Gaussian kernel k(x,y)=exp(- ||x-y||**2 / (2 * sigma**2))

    X - 2d array, samples on left hand side
    Y - 2d array, samples on right hand side, can be None in which case they are replaced by X

    returns: kernel matrix
    """
    
    d = distances(x, y)   # [sq_dists]_ij=||X_j - Y_i \\^2
    K = torch.exp(-d / (2 * sigma**2))  # Gram matrix
    return K

def gauss_kernel_mediane(x,y,bandwidth='median',median_coef=1,return_mediane=False,verbose=0):
    if bandwidth == 'median':
        computed_bandwidth = mediane(x, y,verbose=verbose) * median_coef
    else:
        computed_bandwidth = bandwidth * median_coef
    kernel = lambda x, y: gauss_kernel(x,y,computed_bandwidth)
    if return_mediane:
        return ( kernel, computed_bandwidth )
    else: 
        return kernel

def mediane_per_variable(x,y,verbose=0):
    x=torch_transformator(x)
    y=torch_transformator(y)
    z = torch.cat([x,y])
    medianes = torch.zeros(z.shape[1])
    for v in range(z.shape[1]):
        medianes[v] = mediane(z[:,[v]]).item()
    return(medianes)


def gauss_kernel_mediane_per_variable(x,y,bandwidth='p',median_coef=1,return_mediane=False,verbose=0):
    medianes = median_coef * mediane_per_variable(x,y,verbose=verbose)
    if bandwidth=='median':
        computed_bandwidth = median_coef*mediane(x/medianes,y/medianes)
    elif bandwidth =='p':
        computed_bandwidth = median_coef*len(medianes)
    elif bandwidth == 'pmedian':
        computed_bandwidth = median_coef*mediane(x/medianes,y/medianes)*len(medianes)
    else:
        computed_bandwidth = median_coef*bandwidth

    kernel = lambda x,y: gauss_kernel(x/medianes,y/medianes,computed_bandwidth)
    if return_mediane:
        return(kernel,computed_bandwidth)
    else:
        return(kernel)





def gauss_kernel_mediane_zi(x,y,median_coef=1,return_mediane=False,verbose=0):
    xnz=x[x[:,0]!=0]
    ynz=y[y[:,0]!=0]
    kernel,mediane= gauss_kernel_mediane(xnz,ynz,
                                         median_coef=median_coef,
                                         return_mediane=return_mediane,verbose=verbose)
    if return_mediane:
        return ( kernel, mediane )
    else: 
        return kernel



def fisher_zero_inflated_gaussian_kernel(x,y,pi1,pi2,bandwidth='median',median_coef=1,return_mediane=False,verbose=0):
    if bandwidth == 'median':
        kernel,computed_bandwidth=gauss_kernel_mediane_zi(x,y,median_coef=median_coef,return_mediane=True,verbose=verbose)  
    else:
        computed_bandwidth = bandwidth * median_coef
        kernel = lambda x, y: gauss_kernel(x,y,computed_bandwidth)
    
    def zi_kernel(x,y):
        k = kernel(x,y)
        n1,n2 = k.shape
        z_z = torch.ones([n1,n2])*pi1*pi2
        z_nz = pi1 * (1-pi2) * torch.tensor(norm.pdf(0,loc=y,scale=computed_bandwidth).repeat(len(x),1)).T
        nz_z = (1-pi1) * pi2 * torch.tensor(norm.pdf(0,loc=x,scale=computed_bandwidth).repeat(len(y),1))
        nz_nz = (1-pi1)*(1-pi2) * k
        return(z_z+z_nz+nz_z+nz_nz)
    
    if return_mediane:
        return ( zi_kernel, computed_bandwidth )
    else: 
        return zi_kernel

def corrected_variance_mediane( x, y,variance_per_gene,verbose=0):
    variance_per_gene = torch_transformator(variance_per_gene)
    correction = torch.sqrt(variance_per_gene)
    
    x = torch.div(x,correction)
    y = torch.div(y,correction)
    m = mediane(x, y,verbose=0)
    return(m)

def gauss_kernel_corrected_variance(x,y,variance_per_gene,sigma,verbose=0):
    variance_per_gene = torch_transformator(variance_per_gene)
    correction = torch.sqrt(variance_per_gene)
    x = torch.div(x,correction)
    y = torch.div(y,correction)
    d = distances(x, y)   # [sq_dists]_ij=||X_j - Y_i \\^2
    K = torch.exp(-d/(2*sigma**2))# / (2 * sigma**2))  # Gram matrix
    return K

def gauss_kernel_mediane_corrected_variance(x,y,variance_per_gene,return_mediane=False,verbose=0):
    m = corrected_variance_mediane(x,y,variance_per_gene,verbose)
    if return_mediane:
        return ( lambda x, y: gauss_kernel_corrected_variance(x,y,variance_per_gene,m,verbose), m.item() )
    else: 
        return lambda x, y: gauss_kernel_corrected_variance(x,y,variance_per_gene,m,verbose)



def log_corrected_variance_mediane( x, y,variance_per_gene,verbose=0):
    variance_per_gene = torch_transformator(variance_per_gene)
    correction = torch.sqrt(variance_per_gene)
    x = torch.log(torch.div(x,correction)+1)
    y = torch.log(torch.div(y,correction)+1)
    m = mediane(x, y,verbose=verbose)
    return(m)

def gauss_kernel_log_corrected_variance(x,y,variance_per_gene,sigma,verbose=0):
    variance_per_gene = torch_transformator(variance_per_gene)
    correction = torch.sqrt(variance_per_gene)
    x = torch.log(torch.div(x,correction)+1)
    y = torch.log(torch.div(y,correction)+1)
    d = distances(x, y)   # [sq_dists]_ij=||X_j - Y_i \\^2
    K = torch.exp(-d/(2*sigma**2))# / (2 * sigma**2))  # Gram matrix
    return K

def gauss_kernel_mediane_log_corrected_variance(x,y,variance_per_gene,return_mediane=False,verbose=0):
    m = log_corrected_variance_mediane(x,y,variance_per_gene,verbose)
    if return_mediane:
        return ( lambda x, y: gauss_kernel_log_corrected_variance(x,y,variance_per_gene,m,verbose), m.item() )
    else: 
        return lambda x, y: gauss_kernel_log_corrected_variance(x,y,variance_per_gene,m,verbose)



def linear_kernel(x,y):
    """
    Computes the standard linear kernel k(x,y)= <x,y> 

    X - 2d array, samples on left hand side
    Y - 2d array, samples on right hand side, can be None in which case they are replaced by X

    returns: kernel matrix
    """
    K = torch.matmul(x,y.T)
    return K

if __name__ == "__main__":
    from rv_gen import gen_couple

    x,y = gen_couple().values()
    print('\n *** distance \n',distances(x,y))
    print('\n *** gauss_kernek_mediane \n',gauss_kernel_mediane(x,y)(x,y))
    

