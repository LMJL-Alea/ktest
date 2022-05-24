import torch
import numpy as np
from torch import mv,ones,cat,eye,zeros,ger
from .utils import ordered_eigsy



def compute_Jn_by_n(n):
    return(1/n*ones(n,n,dtype = torch.float64))

def compute_diag_Jn_by_n(effectifs):
    cumul_effectifs = np.cumsum([0]+effectifs)
    L,n             = len(effectifs),np.sum(effectifs)
    diag_Jn_by_n = cat([
                        cat([
                                zeros(nprec,nell,dtype = torch.float64),
                                compute_Jn_by_n(nell),
                                zeros(n-nprec-nell,nell)
                            ],dim=0) 
                            for nell,nprec in zip(effectifs,cumul_effectifs)
                        ],dim=1)
    return(diag_Jn_by_n)

def compute_effectifs(s):
    """
    s is a  pandas.Series
    """
    categories  = s.unique()
    effectifs   = [len(s.loc[s == c]) for c in categories]
    return(effectifs)
  
def permute_matrix_to_respect_index_order(M,col):    
    # Pw a été calculé en supposant que les catégories étaient rangées dans l'ordre, 
    # si ça n'est pas le cas il faut permuter les lignes et les colonnes de Pw

    effectifs     = compute_effectifs(col)
    categories    = col.unique()
    permutation   = np.array([0]*len(M))
    cum_effectifs = np.cumsum([0]+effectifs)
    
    for i,c in enumerate(categories):
        li              = [col.index.get_loc(i) for i in col.loc[col==c].index]
        permutation[li] = range(cum_effectifs[i],cum_effectifs[i+1])
    return(M[permutation,:][:,permutation])    

def compute_centering_matrix_with_respect_to_some_effects(self):

    n  = len(self.obs)
    Pw = eye(n,dtype = torch.float64)

    if self.center_by[0] == '#':
        for center_by in self.center_by[1:].split(sep='_'):
            
            operation,effect = center_by[0],center_by[1:]
            col              = self.obs[effect].astype('category')
            effectifs        = compute_effectifs(col)
            diag_Jn          = compute_diag_Jn_by_n(effectifs)
            diag_Jn          = permute_matrix_to_respect_index_order(diag_Jn,col)
            
            if operation == '-':
                Pw -= diag_Jn
            elif operation == '+':
                Pw += diag_Jn

    else:
        effect           = self.center_by
        col              = self.obs[effect].astype('category')
        effectifs        = compute_effectifs(col)
        diag_Jn          = compute_diag_Jn_by_n(effectifs)
        diag_Jn          = permute_matrix_to_respect_index_order(diag_Jn,col)
        
        Pw -= diag_Jn

    return Pw    

def compute_within_covariance_centering_matrix(self,sample='xy',quantization=False,landmarks=False):
    """
    Computes the bicentering Gram matrix Pn. 
    Let I1,I2 the identity matrix of size n1 and n2 (or nxanchors and nyanchors if nystrom).
        J1,J2 the squared matrix full of one of size n1 and n2 (or nxanchors and nyanchors if nystrom).
        012, 021 the matrix full of zeros of size n1 x n2 and n2 x n1 (or nxanchors x nyanchors and nyanchors x nxanchors if nystrom)
    
    Pn = [I1 - 1/n1 J1 ,    012     ]
            [     021     ,I2 - 1/n2 J2]

    Returns
    sample in 'x','y','xy'
    -------
    torch.Tensor of size (nxanchors+nyanchors)**2 if quantization else (n1+n2)**2 
    """


    if landmarks:
        if self.anchors_basis.lower() == 'k':
            m = self.nxlandmarks if sample=='x' else self.nylandmarks if sample=='y' else self.m
            Im = eye(m, dtype=torch.float64)
            return(Im)
        elif self.anchors_basis.lower() == 's':
            m = self.nxlandmarks if sample=='x' else self.nylandmarks if sample=='y' else self.m
            Im = eye(m, dtype=torch.float64)
            Jm = ones(m, m, dtype=torch.float64)
            Pm = Im - 1/m * Jm
            return(Pm)
        elif self.anchors_basis.lower() == 'w':
            assert(sample=='xy')
            m1,m2 = self.nxlandmarks, self.nylandmarks
            Im1,Im2 = eye(m1, dtype=torch.float64),eye(m2, dtype=torch.float64)
            Jm1,Jm2 = ones(m1, m1, dtype=torch.float64),ones(m2, m2, dtype=torch.float64)
            Pm1,Pm2 = Im1 - 1/m1 * Jm1, Im2 - 1/m2 * Jm2
            z12 = zeros(m1, m2, dtype=torch.float64)
            z21 = zeros(m2, m1, dtype=torch.float64)
            return(cat((cat((Pm1, z12), dim=1), cat((z21, Pm2), dim=1)), dim=0)) 
        else:
            print('invalid anchor basis')  

    if 'x' in sample:
        n1 = self.nxlandmarks if quantization else self.n1 
        In1 = eye(n1, dtype=torch.float64)
        Jn1 = ones(n1, n1, dtype=torch.float64)
        if quantization: 
            a1 = self.compute_quantization_weights(sample='x',power=.5,diag=False)
            Pn1 = (In1 - 1/self.n2 * torch.ger(a1,a1))
            # A1 = self.compute_quantization_weights(sample='x')
            # pn1 = np.sqrt(self.n1/(self.n1+self.n2))*(idn1 - torch.matmul(A1,onen1))
        else:
            Pn1 = In1 - 1/n1 * Jn1

    if 'y' in sample:
        n2 = self.nylandmarks if quantization else self.n2
        In2 = eye(n2, dtype=torch.float64)
        Jn2 = ones(n2, n2, dtype=torch.float64)
        if quantization: 
            a2 = self.compute_quantization_weights(sample='y',power=.5,diag=False)
            Pn2 = (In2 - 1/self.n2 * torch.ger(a2,a2))
            # A2 = self.compute_quantization_weights(sample='y')
            # pn2 = np.sqrt(self.n2/(self.n1+self.n2))*(idn2 - torch.matmul(A2,onen2))
        else:
            Pn2 = In2 - 1/n2 * Jn2

    if sample == 'xy':
        z12 = zeros(n1, n2, dtype=torch.float64)
        z21 = zeros(n2, n1, dtype=torch.float64)
        return(cat((cat((Pn1, z12), dim=1), cat(
        (z21, Pn2), dim=1)), dim=0))  # bloc diagonal
    else:
        return(Pn1 if sample=='x' else Pn2)  

def compute_omega(self,sample='xy',quantization=False):
    n1,n2 = (self.n1,self.n2)
    if sample =='xy':
        if quantization:
            return(torch.cat((-1/n1*torch.bincount(self.xassignations),1/n2*torch.bincount(self.yassignations))).double())
        else:
            m_mu1    = -1/n1 * ones(n1, dtype=torch.float64) # , device=device)
            m_mu2    = 1/n2 * ones(n2, dtype=torch.float64) # , device=device) 
            return(torch.cat((m_mu1, m_mu2), dim=0)) #.to(device)
    elif sample=='x':
        return(1/n1 * ones(n1, dtype=torch.float64))
    elif sample=='y':
        return(1/n2 * ones(n2, dtype=torch.float64))


