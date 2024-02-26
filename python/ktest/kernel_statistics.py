import numpy as np
import pandas as pd
from torch import cdist, cat, matmul, exp, mv, dot, ones, eye, zeros, tensor, float64
from torch.linalg import multi_dot
import warnings
from apt.eigen_wrapper import eigsy


"""

Ce fichier contient toutes les fonctions nécessaires au calcul des statistiques,
Les quantités pkm et upk sont des quantités génériques qui apparaissent dans beaucoup de calculs. 
Elles n'ont pas d'interprêtation facile, ces fonctions centrales permettent d'éviter les répétitions. 

Les fonctions initialize_kfdat et kfdat font simplement appel a plusieurs fonctions en une fois.

"""

def distances(x, y=None):
    """
    If Y=None, then this computes the distance between X and itself
    """
    if y is None:
        sq_dists = cdist(x, x, compute_mode='use_mm_for_euclid_dist_if_necessary').pow(
            2)  # [sq_dists]_ij=||X_j - X_i \\^2
    else:
        assert(y.ndim == 2)
        assert(x.shape[1] == y.shape[1])
        sq_dists = cdist(x, y, compute_mode='use_mm_for_euclid_dist_if_necessary').pow(
            2)  # [sq_dists]_ij=||x_j - y_i \\^2
    return sq_dists

def mediane(x, y=None, verbose=0):
    """
    Computes the median 
    """
    
    dxx = distances(x)
    if y == None:
        return dxx.median()
    dyy = distances(y)
    dxy = distances(x,y)
    dyx = dxy.t()
    dtot = cat((cat((dxx,dxy),dim=1),
                     cat((dyx,dyy),dim=1)),dim=0)
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
    
def linear_kernel(x,y):
    """
    Computes the standard linear kernel k(x,y)= <x,y> 

    X - 2d array, samples on left hand side
    Y - 2d array, samples on right hand side, can be None in which case they 
    are replaced by X

    returns: kernel matrix
    """
    K = matmul(x,y.T)
    return K

def gauss_kernel(x, y, sigma=1):
    """
    Computes the standard Gaussian kernel k(x,y)=exp(- ||x-y||**2 / (2 * sigma**2))

    X - 2d array, samples on left hand side
    Y - 2d array, samples on right hand side, can be None in which case they 
    are replaced by X

    returns: kernel matrix
    """
    
    d = distances(x, y)   # [sq_dists]_ij=||X_j - Y_i \\^2
    K = exp(-d / (2 * sigma**2))  # Gram matrix
    return K

def gauss_kernel_mediane(x,y,bandwidth='median', median_coef=1, 
                         return_mediane=False, verbose=0):
    if bandwidth == 'median':
        computed_bandwidth = mediane(x, y,verbose=verbose) * median_coef
    else:
        computed_bandwidth = bandwidth * median_coef
    K = gauss_kernel(x, y, computed_bandwidth)
    if return_mediane:
        return (K, computed_bandwidth )
    else: 
        return K

class Statistics():
    '''
    Attributes: 
    ------------------------
    data: instance of class Data !!!
        data matrix initialized from data 
        
    kernel name (default = None): str
        The name of the kernel function specified by the call of the function.
        if None: the kernel name is automatically generated through the function
        get_kernel_name 
        
    function (default = 'gauss'): function or str in ['gauss','linear','fisher_zero_inflated_gaussian','gauss_kernel_mediane_per_variable'] 
        if str: specifies the kernel function
        if function: kernel function specified by user

    bandwidth (default = 'median'): 'median' or float
        value of the bandwidth for kernels using a bandwidth
        if 'median': the bandwidth will be set as the median or a multiple of it is
            according to the value of parameter `median_coef`
        if float: value of the bandwidth

    median_coef (default = 1): float
        multiple of the median to use as bandwidth if bandwidth=='median' 
        
    '''     
    def __init__(self, data, kernel_function='gauss', bandwidth='median', 
                 median_coef=1, verbose=0):
        '''

        Parameters
        ----------
            stat (default=): str in ['kfda','mmd']
                Test statistic to be computed.
        '''
        
        self.data = data
        
        ### Kernel:
        self.kernel_function = kernel_function
        self.bandwidth = bandwidth
        self.median_coef = median_coef

        ### TO DO: add an explanation for the computed bandwidth thing
        if self.kernel_function == 'gauss':
            self.kernel = lambda x, y: gauss_kernel_mediane(x=x, y=y, 
                                                            bandwidth=bandwidth,  
                                                            median_coef=median_coef,
                                                            return_mediane=False,
                                                            verbose=verbose)
        elif self.kernel_function == 'linear':
            self.kernel = linear_kernel
        else:
            self.kernel = self.kernel_function
        
    @staticmethod
    def ordered_eigsy(matrix):
        # la matrice de vecteurs propres renvoyée a les vecteurs propres en colonnes.  
        sp,ev = eigsy(matrix)
        order = sp.argsort()[::-1]
        ev = tensor(ev[:,order],dtype=float64) 
        sp = tensor(sp[order], dtype=float64)
        return(sp,ev)

    def compute_centering_matrix(self):
        """
        Computes a projection matrix usefull for the kernel trick. 

        Example fir the within-group covariance :
            Let I1,I2 the identity matrix of size n1 and n2 (or nxanchors and 
             nyanchors if nystrom). J1,J2 the squared matrix full of one of 
            size n1 and n2 (or nxanchors and nyanchors if nystrom).
            012, 021 the matrix full of zeros of size n1 x n2 and n2 x n1 
            (or nxanchors x nyanchors and nyanchors x nxanchors if nystrom).
        
        Pn = [I1 - 1/n1 J1 ,    012     ]
                [     021     ,I2 - 1/n2 J2]

        Parameters
        ----------
            sample : string,
                if sample = 'xy' : returns the bicentering matrix corresponding
                to the within group covariance operator.
                if sample = 'x' (resp. 'y') : returns the centering matrix 
                corresponding to the covariance operator of sample 'x' (resp. 'y')

        Returns
        ------- 
            P : torch.tensor, 
                the centering matrix corresponding to the parameters
        """
        In = eye(self.data.ntot)
        effectifs = list(self.data.nobs.values())
        
        cumul_effectifs = np.cumsum([0]+effectifs)
        _,n = len(effectifs),np.sum(effectifs)
        
        # For comments: check compute_diag_Jn_by_n
        diag_Jn_by_n = cat([
                            cat([
                                    zeros(nprec,nell,dtype = float64),
                                    1/nell*ones(nell,nell,dtype = float64),
                                    zeros(n-nprec-nell,nell,dtype = float64)
                                ],dim=0) 
                                for nell,nprec in zip(effectifs,cumul_effectifs)
                            ],dim=1)
        return(In - diag_Jn_by_n)

    def compute_omega(self):
        '''
        Returns the weights vector to compute a mean. 
        
        Parameters
        ----------
            sample : str,
            if sample = 'x' : returns a vector of size n1 = len(self.x) full of 1/n1
            if sample = 'y' : returns a vector of size n2 = len(self.y) full of 1/n2
            if sample = 'xy' : returns a vector of size n1 + n2 with n1 1/-n1 followed by n2 1/n2 to compute (\mu_2 - \mu_1) 
            
        Returns
        ------- 
            omega : torch.tensor 
            a vector of size corresponding to the group of which we compute the mean. 
        '''

        n1, n2 = self.data.nobs.values()
        m_mu1 = -1/n1 * ones(n1, dtype=float64)
        m_mu2 = 1/n2 * ones(n2, dtype=float64)
        return(cat((m_mu1, m_mu2), dim=0))

    def compute_gram(self): 
        """
        Computes the Gram matrix of the data corresponding to the parameters 
        sample and landmarks. 

        The kernel used is the kernel stored in the attribute `kernel`. 
        ( The attribute `kernel` can be initialized with the method kernel() ) 

        The Gram matrix is not stored in memory because it is usually large 
        and fast to compute. 

        Returns
        -------
            K : torch.Tensor,
                Gram matrix of interest
        """
        data = cat([x for x in self.data.data.values()], axis=0)
        K = self.kernel(data, data)
        return(K)

    def diagonalize_centered_gram(self, verbose=0):
        """
        Diagonalizes the bicentered Gram matrix which shares its spectrum 
        with the Withon covariance operator in the RKHS.
        Stores eigenvalues and eigenvectors in the dict spev 

        Parameters 
        ----------
            verbose (default = 0): 
                Dans l'idée, plus verbose est grand, plus 
                la fonction détaille ce qu'elle fait

        """

        # calcul de la matrice a diagonaliserif verbose>0:
        if verbose>0:
            print('- Compute within covariance centered gram')
        # Instantiation de la matrice de centrage P 
        P = self.compute_centering_matrix()
        K = self.compute_gram()
        Kw = 1/self.data.ntot * multi_dot([P,K,P])
        
        # diagonalisation par la fonction C++ codée par François puis tri 
        # décroissant des valeurs propres et vecteurs prores
        if verbose >0:
            print('- Diagonalize within covariance centered gram')
        return Statistics.ordered_eigsy(Kw) 

    def compute_pkm(self):
        '''

        This function computes the term corresponding to the matrix-matrix-vector 
        product PK omega of the KFDA statistic.
        
        See the description of the method compute_kfdat() for a brief 
        description of the computation of the KFDA statistic. 

        Returns
        ------- 
        pkm : torch.tensor 
        Correspond to the product PK omega in the KFDA statistic. 
        '''

        # instantiation du vecteur de bi-centrage omega et de la matrice de centrage Pbi 
        omega = self.compute_omega() # vecteur de 1/n1 et de -1/n2 
        Pbi = self.compute_centering_matrix() # matrice de centrage par block

        Kx = self.compute_gram() # matrice de gram 
        pkm = mv(Pbi,mv(Kx,omega)) # le vecteur que renvoie cette fonction 

        return(pkm) 

    def compute_kfdat(self,t=None,verbose=0):
        
        """ 
        From the former class:
        Ces fonctions calculent les coordonnées des projections des embeddings sur des sous-espaces d'intérêt 
        dans le RKHS. Sauf la projection sur ce qu'on appelle pour l'instant l'espace des résidus, comme cette 
        projection nécessite de nombreux calculs intermédiaires, elle a été encodée dans un fichier à part. 
        
        
        
        Computes the kfda truncated statistic of [Harchaoui 2009].
        9 methods : 
        approximation_cov in ['standard','nystrom',]
        approximation_mmd in ['standard','nystrom',]
        
        Stores the result as a column in the dataframe df_kfdat

        Parameters
        ----------
            self : Ktest,
            the model parameter attributes `approximation_cov`, `approximation_mmd` must be defined.
            if the nystrom method is used, the attribute `anchor_basis` should be defined and the anchors must have been computed. 

            t (default = None) : None or int,
            valeur maximale de troncature calculée. 
            Si None, t prend la plus grande valeur possible, soit n (nombre d'observations) pour la 
            version standard et n_anchors (nombre d'ancres) pour la version nystrom  

            name (default = None) : None or str, 
            nom de la colonne des dataframe df_kfdat et df_kfdat_contributions dans lesquelles seront stockés 
            les valeurs de la statistique pour les différentes troncatures calculées de 1 à t 

            verbose (default = 0) : Dans l'idée, plus verbose est grand, plus la fonction détaille ce qu'elle fait

        Description 
        -----------
        Here is a brief description of the computation of the statistic, for more details, refer to the article : 

        Let k(·,·) denote the kernel function, K denote the Gram matrix of the two  samples 
        and kx the vector of embeddings of the observations x1,...,xn1,y1,...,yn2 :
        
                kx = (k(x1,·), ... k(xn1,·),k(y1,·),...,k(yn2,·)) 
        
        Let Sw denote the within covariance operator and P denote the centering matrix such that 

                Sw = 1/n (kx P)(kx P)^T
        
        Let Kw = 1/n (kx P)^T(kx P) denote the dual matrix of Sw and (li) (ui) denote its eigenvalues (shared with Sw) 
        and eigenvectors. We have :

                ui = 1/(lp * n)^{1/2} kx P up 

        Let Swt denote the spectral truncation of Sw with t directions
        such that 
        
                Swt = l1 (e1 (x) e1) + l2 (e2 (x) e2) + ... + lt (et (x) et) 
                    = \sum_{p=1:t} lp (ep (x) ep)
        
        where (li) and (ei) are the first t eigenvalues and eigenvectors of Sw ordered by decreasing eigenvalues,
        and (x) stands for the tensor product. 

        Let d = mu2 - mu1 denote the difference of the two kernel mean embeddings of the two samples 
        of sizes n1 and n2 (with n = n1 + n2) and omega the weights vector such that 
        
                d = kx * omega 
        
        
        The standard truncated KFDA statistic is given by :
        
                F   = n1*n2/n || Swt^{-1/2} d ||_H^2

                    = \sum_{p=1:t} n1*n2 / ( lp*n) <ep,d>^2 

                    = \sum_{p=1:t} n1*n2 / ( lp*n)^2 up^T PK omega


        Projection
        ----------

        This statistic also defines a discriminant axis ht in the RKHS H. 
        
                ht  = n1*n2/n Swt^{-1/2} d 
                    
                    = \sum_{p=1:t} n1*n2 / ( lp*n)^2 [up^T PK omega] kx P up 

        To project the dataset on this discriminant axis, we compute : 

                h^T kx =  \sum_{p=1:t} n1*n2 / ( lp*n)^2 [up^T PK omega] up^T P K   

        """
        
        if verbose >0: 
            print('- Compute kfda statistic') 
        
        # Récupération des vecteurs propres et valeurs propres calculés par la fonction de classe 
        # diagonalize_within_covariance_centered_gram 
        sp, ev = self.diagonalize_centered_gram(verbose=verbose)

        # détermination de la troncature maximale à calculer 
        t = len(sp) if t is None else len(sp) if len(sp)<t else t # troncature maximale

        # calcul de la statistique pour chaque troncature 
        pkm = self.compute_pkm() # partie de la stat qui ne dépend pas de la troncature mais du modèle
        n1, n2 = self.data.nobs.values() # nombres d'observations dans les échantillons
        exposant = 2 
        
        # calcule la contribution de chaque troncature 
        kfda_contributions = ((n1 * n2) / (self.data.ntot ** exposant
                                           * sp[:t]**exposant) 
                              * mv(ev.T[:t], pkm) ** 2).numpy()
        #somme les contributions pour avoir la stat correspondant à chaque troncature 
        kfda = kfda_contributions.cumsum(axis=0)

        # stockage des vecteurs propres et valeurs propres dans l'attribut spev
        trunc = range(1,t+1) # liste des troncatures possibles de 1 à t 

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kfdat = pd.Series(kfda,index=trunc)
            kfdat_contributions = pd.Series(kfda_contributions,index=trunc)
        return kfdat, kfdat_contributions
 
    def compute_mmd(self, unbiaised=False, verbose=0):
        m = self.compute_omega()
        K = self.compute_gram()
        if unbiaised:
            K.masked_fill_(eye(K.shape[0],K.shape[0]).byte(), 0)
        mmd = dot(mv(K,m),m)**2 # je crois qu'il n'y a pas besoin de carré
        return(mmd.item())    
                
                