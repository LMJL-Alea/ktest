import torch
from apt.eigen_wrapper import eigsy

def ordered_eigsy(matrix):
    # la matrice de vecteurs propres renvoyée a les vecteurs propres en colonnes.  
    sp,ev = eigsy(matrix)
    order = sp.argsort()[::-1]
    ev = torch.tensor(ev[:,order],dtype=torch.float64) 
    sp = torch.tensor(sp[order], dtype=torch.float64)
    return(sp,ev)
