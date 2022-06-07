import torch
import numpy as np
from torch import mv,ones,cat,eye,zeros,ger
from .utils import ordered_eigsy


def compute_gram(self,sample='xy',landmarks=False,outliers_in_obs=None): 
    """
    Computes the Gram matrix of the data corresponding to the parameters sample and landmarks. 
    
    The kernel used is the kernel stored in the attribute `kernel`. 
    ( The attribute `kernel` can be initialized with the method init_kernel() ) 

    The computed Gram matrix is centered with respect to the attribute `center_by`.
    ( The attribute `center_by` can be initialized with the method init_center_by())
    
    The Gram matrix is not stored in memory because it is usually large and fast to compute. 

    Parameters
    ----------
        sample (default = 'xy') : str,
            if 'x' : Returns the Gram matrix corresponding to the first sample.
            if 'y' : Returns the Gram matrix corresponding to the second sample. 
            if 'xy': Returns the Gram matrix corresponding to both samples concatenated. 
                    Such that the Gram is a bloc matrix with diagonal blocks corresponding to 
                    the gram matrix of each sample and the non-diagonal block correspond to the 
                    crossed-gram matrix between both samples. 

        landmarks (default = False) : boolean,
            if False : Returns the Gram matrix of the observation samples. 
            if True : Returns the Gram matrix of a set of landmarks stored in the attributes
                `xlandmarks`, `ylandmarks` or both. It is used in the nystrom approach. 
                ( The landmarks can be initialized with the method compute_nystrom_landmarks())

    Returns
    -------
        K : torch.Tensor,
            Gram matrix corresponding to the parameters centered with respect to the attribute `center_by`
    """

    kernel = self.kernel
   
    x,y = self.get_xy(landmarks=landmarks,outliers_in_obs=outliers_in_obs)
    
    if 'x' in sample:
        kxx = kernel(x,x)
    if 'y' in sample:
        kyy = kernel(y,y)

    if sample == 'xy':
        kxy = kernel(x, y)
        K = torch.cat((torch.cat((kxx, kxy), dim=1),
                        torch.cat((kxy.t(), kyy), dim=1)), dim=0)
        K = self.center_gram_matrix_with_respect_to_some_effects(K,outliers_in_obs=outliers_in_obs)

        return(K)
    else:
        return(kxx if sample =='x' else kyy)

def center_gram_matrix_with_respect_to_some_effects(self,K,outliers_in_obs=None):
    if self.center_by is None:
        return(K)
    else:
        P = self.compute_centering_matrix_with_respect_to_some_effects(outliers_in_obs=None)
        return(torch.chain_matmul(P,K,P))
        # retutn(torch.linalg.multi_dot([P,K,P]))

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

def compute_within_covariance_centered_gram(self,approximation='standard',sample='xy',verbose=0,outliers_in_obs=None):
    """ 
    Computes the bicentered Gram matrix which shares its spectrom with the 
    within covariance operator. 
    Returns the matrix because it is only used in diagonalize_bicentered_gram
    I separated this function because I want to assess the computing time and 
    simplify the code 

    approximation in 'standard','nystrom','quantization'
    # contre productif de choisir 'nystrom' car cela est aussi cher que standard pour une qualité d'approx de la matrice moins grande. 
    # pour utiliser nystrom, mieux vaux calculer la SVD de BB^T pas encore fait. 

    # pour l'instant les outliers ne sont pas compatibles avec nystrom
    """

    self.verbosity(function_name='compute_centered_gram',
            dict_of_variables={'approximation':approximation,
                            'sample':sample},
            start=True,
            verbose = verbose)    
    
    quantization = approximation == 'quantization'
    P = self.compute_covariance_centering_matrix(sample=sample,quantization=quantization,outliers_in_obs=outliers_in_obs).double()
    
    n=0
    n1,n2,_ = self.get_n1n2n(outliers_in_obs=outliers_in_obs)
    if 'x' in sample:
        n+=n1     
    if 'y' in sample:
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
            Pm = self.compute_covariance_centering_matrix(sample='xy',landmarks=True)
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
            Pm = self.compute_covariance_centering_matrix(sample='xy',landmarks=True)
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
        K = self.compute_gram(landmarks=False,sample=sample,outliers_in_obs=outliers_in_obs)
        Kw = 1/n * torch.chain_matmul(P,K,P)
        # Kw = 1/n * torch.linalg.multi_dot([P,K,P])

    self.verbosity(function_name='compute_centered_gram',
            dict_of_variables={'approximation':approximation,'sample':sample},
            start=False,
            verbose = verbose)    

    return Kw

def diagonalize_centered_gram(self,approximation='standard',sample='xy',verbose=0,outliers_in_obs=None):
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
    
    Kw = self.compute_within_covariance_centered_gram(approximation=approximation,sample=sample,verbose=verbose,outliers_in_obs=outliers_in_obs)
    
    sp,ev = ordered_eigsy(Kw)
    # print('Kw',Kw,'sp',sp,'ev',ev)
    suffix_nystrom = self.anchors_basis if 'nystrom' in approximation else ''
    suffix_outliers = outliers_in_obs if outliers_in_obs is not None else ''
    self.spev[sample][f'{approximation}{suffix_nystrom}{suffix_outliers}'] = {'sp':sp,'ev':ev}
    
    self.verbosity(function_name='diagonalize_centered_gram',
            dict_of_variables={'approximation':approximation,
                                'sample':sample,
                                'outliers_in_obs':outliers_in_obs},
            start=False,
            verbose = verbose)


