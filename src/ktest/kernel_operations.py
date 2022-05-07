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


def center_gram_matrix_with_respect_to_some_effects(self,K):
    if self.center_by is None:
        return(K)
    else:
        P = compute_centering_matrix_with_respect_to_some_effects(self)
        return(torch.chain_matmul(P,K,P))
        # retutn(torch.linalg.multi_dot([P,K,P]))

def compute_gram(self,sample='xy',landmarks=False): 
    """
    Computes Gram matrix, on anchors if nystrom is True, else on data. 
    This function is called everytime the Gram matrix is needed but I could had an option to keep it in memory in case of a kernel function 
    that makes it difficult to compute

    If we want a gram matrix centered by some categorical variable, this variable should have a 
    column in self.obs and this columns should by categorical. 
    and the attribute self.center_by should have the name of the column.
    Then the gram matrix given by this function is centered according to this categorical variable. 

    Returns
    -------
    torch.Tensor of size (nxanchors+nyanchors)**2 if nystrom else (n1+n2)**2
    """

    kernel = self.kernel
   
    x,y = self.get_xy(landmarks=landmarks)
    
    if 'x' in sample:
        kxx = kernel(x,x)
    if 'y' in sample:
        kyy = kernel(y,y)

    if sample == 'xy':
        kxy = kernel(x, y)
        K = torch.cat((torch.cat((kxx, kxy), dim=1),
                        torch.cat((kxy.t(), kyy), dim=1)), dim=0)
        K = self.center_gram_matrix_with_respect_to_some_effects(K)

        return(K)
    else:
        return(kxx if sample =='x' else kyy)


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

def compute_kmn(self,sample='xy'):
    """
    Computes an (nxanchors+nyanchors)x(n1+n2) conversion gram matrix
    """
    assert(self.has_landmarks)
    kernel = self.kernel
    
    x,y = self.get_xy()
    z1,z2 = self.get_xy(landmarks=True)
    if 'x' in sample:
        kz1x = kernel(z1,x)
    if 'y' in sample:
        kz2y = kernel(z2,y)
    
    if sample =='xy':
        kz2x = kernel(z2,x)
        kz1y = kernel(z1,y)
        return(cat((cat((kz1x, kz1y), dim=1),
                        cat((kz2x, kz2y), dim=1)), dim=0))
    else:
        return(kz1x if sample =='x' else kz2y)

def compute_centering_matrix(self,sample='xy',quantization=False,landmarks=False):
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



        



def compute_centered_gram(self,approximation='standard',sample='xy',verbose=0):
    """ 
    Computes the bicentered Gram matrix which shares its spectrom with the 
    within covariance operator. 
    Returns the matrix because it is only used in diagonalize_bicentered_gram
    I separated this function because I want to assess the computing time and 
    simplify the code 

    approximation in 'standard','nystrom','quantization'
    # contre productif de choisir 'nystrom' car cela est aussi cher que standard pour une qualité d'approx de la matrice moins grande. 
    # pour utiliser nystrom, mieux vaux calculer la SVD de BB^T pas encore fait. 
    """

    self.verbosity(function_name='compute_centered_gram',
            dict_of_variables={'approximation':approximation,
                            'sample':sample},
            start=True,
            verbose = verbose)    
    
    quantization = approximation == 'quantization'
    P = self.compute_centering_matrix(sample=sample,quantization=quantization).double()
    
    n=0
    if 'x' in sample:
        n1 = self.n1 
        n+=n1     
    if 'y' in sample:
        n2 = self.n2
        n+=n2
    if 'nystrom' in approximation:
        r = self.r if sample=='xy' else self.nxanchors if sample =='x' else self.nyanchors
        anchors_basis = self.anchors_basis
    if approximation == 'quantization':
        if self.quantization_with_landmarks_possible:
            Kmm = self.compute_gram(sample=sample,landmarks=True)
            A = self.compute_quantization_weights(sample=sample,power=.5)
            Kw = 1/n * torch.chain_matmul(P,A,Kmm,A,P)
            # Kw = 1/n * torch.linalg.multi_dot([P,A,Kmm,A,P])
        else:
            print("quantization impossible, you need to call 'compute_nystrom_landmarks' with landmark_method='kmeans'")


    elif approximation == 'nystrom1':
        # version brute mais a terme utiliser la svd ?? 
        if self.has_landmarks and "anchors" in self.spev[sample]:
            Kmn = self.compute_kmn(sample=sample)
            Lp_inv = torch.diag(self.spev[sample]['anchors'][anchors_basis]['sp']**(-1))
            Up = self.spev[sample]['anchors'][anchors_basis]['ev']
            Pm = self.compute_centering_matrix(sample='xy',landmarks=True)
            Kw = 1/(n*r*2) * torch.chain_matmul(P,Kmn.T,Pm,Up,Lp_inv,Up.T,Pm,Kmn,P)            
            # Kw = 1/(n*r*2) * torch.linalg.multi_dot([P,Kmn.T,Pm,Up,Lp_inv,Up.T,Pm,Kmn,P])            
            # Kw = 1/(n) * torch.linalg.multi_dot([P,Kmn.T,Pm,Up,Lp_inv,Up.T,Pm,Kmn,P])            
            
        else:
            print("nystrom impossible, you need compute landmarks and/or anchors")
    
    elif approximation in ['nystrom2','nystrom3']:
        if self.has_landmarks and "anchors" in self.spev[sample]:
            Kmn = self.compute_kmn(sample=sample)
            Lp_inv_12 = torch.diag(self.spev[sample]['anchors'][anchors_basis]['sp']**(-(1/2)))
            # on est pas censé avoir de nan, il y en avait quand les ancres donnaient des spectres négatifs à cause des abérations numérique en univarié 
            # assert(not any(torch.isnan(Lp_inv_12)))
            Up = self.spev[sample]['anchors'][anchors_basis]['ev']
            Pm = self.compute_centering_matrix(sample='xy',landmarks=True)
            # print(f'Lp_inv_12{Lp_inv_12.shape},Up{Up.shape},Pm{Pm.shape},Kmn{Kmn.shape}')
            Kw = 1/(n*r**2) * torch.chain_matmul(Lp_inv_12,Up.T,Pm,Kmn,P,Kmn.T,Pm,Up,Lp_inv_12)            
            # Kw = 1/(n*r**2) * torch.linalg.multi_dot([Lp_inv_12,Up.T,Pm,Kmn,P,Kmn.T,Pm,Up,Lp_inv_12])            
            
            # Kw = 1/(n) * torch.chain_matmul(Lp_inv_12,Up.T,Pm,Kmn,P,Kmn.T,Pm,Up,Lp_inv_12)            
            # Kw = 1/(n) * torch.linalg.multi_dot([Lp_inv_12,Up.T,Pm,Kmn,P,Kmn.T,Pm,Up,Lp_inv_12])            
            # else:
            #     Kw = 1/n * torch.linalg.multi_dot([Lp_inv_12,Up.T,Kmn,P,Kmn.T,Up,Lp_inv_12])

        else:
            print("nystrom new version impossible, you need compute landmarks and/or anchors")
                

    elif approximation == 'standard':
        K = self.compute_gram(landmarks=False,sample=sample)
        Kw = 1/n * torch.chain_matmul(P,K,P)
        # Kw = 1/n * torch.linalg.multi_dot([P,K,P])

    self.verbosity(function_name='compute_centered_gram',
            dict_of_variables={'approximation':approximation,'sample':sample},
            start=False,
            verbose = verbose)    

    return Kw

def diagonalize_centered_gram(self,approximation='standard',sample='xy',verbose=0):
    """
    Diagonalizes the bicentered Gram matrix which shares its spectrum with the Withon covariance operator in the RKHS.
    Stores eigenvalues (sp or spny) and eigenvectors (ev or evny) as attributes
    """
    

    # if approximation in self.spev[sample]:
    #     if verbose:
    #         print('No need to diagonalize')
    # else:
    self.verbosity(function_name='diagonalize_centered_gram',
            dict_of_variables={'approximation':approximation,
            'sample':sample},
            start=True,
            verbose = verbose)
    
    Kw = self.compute_centered_gram(approximation=approximation,sample=sample,verbose=verbose)
    
    sp,ev = ordered_eigsy(Kw)
    # print('Kw',Kw,'sp',sp,'ev',ev)
    suffix_nystrom = self.anchors_basis if 'nystrom' in approximation else ''
    self.spev[sample][approximation+suffix_nystrom] = {'sp':sp,'ev':ev}
    
    self.verbosity(function_name='diagonalize_centered_gram',
            dict_of_variables={'approximation':approximation,
                                'sample':sample},
            start=False,
            verbose = verbose)
