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

