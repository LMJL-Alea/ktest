import torch
from apt.eigen_wrapper import eigsy
import pandas as pd
import numpy as np


def ordered_eigsy(matrix):
    # la matrice de vecteurs propres renvoyée a les vecteurs propres en colonnes.  
    sp,ev = eigsy(matrix)
    order = sp.argsort()[::-1]
    ev = torch.tensor(ev[:,order],dtype=torch.float64) 
    sp = torch.tensor(sp[order], dtype=torch.float64)
    return(sp,ev)

def pytorch_eigsy(matrix):
    # j'ai codé cette fonction pour tester les approches de nystrom 
    # avec la diag de pytorch mais ça semble marcher moins bien que la fonction eigsy
    # cpdt je devrais comparer sur le meme graphique 
    sp,ev = torch.symeig(matrix,eigenvectors=True)
    order = sp.argsort()
    ev = ev[:,order]
    sp = sp[order]
    return(sp,ev)


def convert_to_torch_tensor(X):
    token = True
    if isinstance(X,pd.Series):
        X = torch.from_numpy(X.to_numpy().reshape(-1,1)).double()
    if isinstance(X,pd.DataFrame):
        X = torch.from_numpy(X.to_numpy()).double()
    elif isinstance(X, np.ndarray):
        X = torch.from_numpy(X).double()
    elif isinstance(X,torch.Tensor):
        X = X.double()
    else : 
        token = False
        print(f'unknown data type {type(X)}')            

    return(X)

def convert_to_pandas_index(index):
    if isinstance(index,list) or isinstance(index,range):
        return(pd.Index(index))
    else:
        return(index)

def get_kernel_name(function,bandwidth,median_coef):
    n = ''
    if function in ['gauss','fisher_zero_inflated_gaussian']:
        n+=function
        if bandwidth == 'median':
            n+= f'_{median_coef}median' if median_coef != 1 else '_median' 
        else: 
            n+=f'_{bandwidth}'
    elif function == 'linear':
        n+=function
    elif function == 'gauss_kernel_mediane_per_variable':
        n+=function
    else:
        n='user_specified'
    return(n)

def init_test_params(stat='kfda',
                    nystrom=False,
                    n_landmarks=None,
                    n_anchors=None,
                    landmark_method='random',
                    anchor_basis='w',
                    permutation=False,
                    n_permutations=500,
                    seed_permutation=0):

    return({'stat':stat,
            'nystrom':nystrom,
            'n_landmarks':n_landmarks,
            'n_anchors':n_anchors,
            'landmark_method':landmark_method,
            'anchor_basis':anchor_basis,
            'permutation':permutation,
            'n_permutations':n_permutations,
            'seed_permutation':seed_permutation
            })

def init_kernel_params(function='gauss',
                       bandwidth='median',
                       median_coef=1,
                       weights=None,
                       weights_power=1,
                       kernel_name=None,
                       pi1=None,
                       pi2=None):
    """
    Returns an object that defines the kernel
    """
    return(
        {'function':function,
            'bandwidth':bandwidth,
            'median_coef':median_coef,
            'kernel_name':kernel_name,
            'weights':weights,
            'weights_power':weights_power,
            'pi1':pi1,
            'pi2':pi2
            }
    )