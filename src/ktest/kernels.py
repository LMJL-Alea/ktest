
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import pandas as pd

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

def gauss_kernel_mediane(x,y,return_mediane=False,verbose=0):
    m = mediane(x, y,verbose=verbose)
    if return_mediane:
        return ( lambda x, y: gauss_kernel(x,y,m), m.item() )
    else: 
        return lambda x, y: gauss_kernel(x,y,m)

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
    

