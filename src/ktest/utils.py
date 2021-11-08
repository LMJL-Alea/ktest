import torch
from apt.eigen_wrapper import eigsy
import apt.kmeans # For kmeans
from kmeans_pytorch import kmeans

def ordered_eigsy(matrix):
    sp,ev = eigsy(matrix)
    order = sp.argsort()[::-1]
    ev = torch.tensor(ev[:,order],dtype=torch.float64) 
    sp = torch.tensor(sp[order], dtype=torch.float64)
    return(sp,ev)
