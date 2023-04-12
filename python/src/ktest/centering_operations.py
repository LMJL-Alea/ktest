import torch
import numpy as np
from torch import mv,ones,cat,eye,zeros,ger
from .utils import ordered_eigsy

from .base import Base
from .nystrom_operations import NystromOps

"""
La plupart des quantités d'intérêt à calculer numériquement s'écrivent Ka ou PKP grace au kernel trick, avec 
a un vecteur de poids ou P une matrice de centrage. Les fonctions de ce fichier déterminent ces vecteurs 
de poids et matrices de centrage. 
"""

def compute_Jn_by_n(n):
    '''
    returns the (n x n) matrix full of 1/n called Jn 

    Parameters
    ----------
    n : int, the size of the matrix Jn

    Returns
    ------- 
        Jn : torch.tensor, a (n x n) matrix full of 1/n
    '''
    return(1/n*ones(n,n,dtype = torch.float64))

def compute_diag_Jn_by_n(effectifs):
    '''
    Returns a bloc diagonal matrix where the ith diagonal bloc is J_ni, an (ni x ni)
    matrix full of 1/ni where ni is the ith value of the list effectifs.
    This matrix is used to a dataset with respect to the groups corresponding to effectifs.  

    Parameters
    ----------
        effectifs : list of int, corresponding to group sizes.
    
    Returns
    -------
        diag_Jn_by_n : torch.tensor, 
        a block diagonal matrix of size (n times n) where n is the sum of effectifs values
    '''

    cumul_effectifs = np.cumsum([0]+effectifs)
    L,n             = len(effectifs),np.sum(effectifs)
    diag_Jn_by_n = cat([
                        cat([
                                zeros(nprec,nell,dtype = torch.float64),
                                compute_Jn_by_n(nell),
                                zeros(n-nprec-nell,nell,dtype = torch.float64)
                            ],dim=0) 
                            for nell,nprec in zip(effectifs,cumul_effectifs)
                        ],dim=1)
    return(diag_Jn_by_n)

def compute_effectifs(col):
    """
    Given a dataset containing cells from I samples, this function returns a list of size I containing 
    the size of the ith sample at the ith position. 

    Parameters
    ----------
        col (pandas.Series) : the list of cell-ids and their respective sample is encoded in a pandas.series, 
            the index correspond to the cell-id and the value correspond to the sample of the cell. 
    
    Returns
    -------
        effectifs : a list of size I containing the sizes of the I samples. 

    """
    categories  = col.unique()
    effectifs   = [len(col.loc[col == c]) for c in categories]
    return(effectifs)
  
def permute_matrix_to_respect_index_order(M,col): 
    '''
    The matrix given by the function `compute_diag_Jn_by_n` with respect to 
    a list effectifs given by `compute_effectifs` is generally not ordered with respect to the cell-ids. 
    This function permutes the rows and the columns of such a matrix in order to correspond to the cell ids 
    stored in col. 

    Parameters
    ----------
        M : torch.tensor, typically the output of `compute_diag_Jn_by_n`

        col :pandas.series, typically the same input than in `compute_effectifs`

    Returns
    ------- 
        torch.tensor corresponding to M with rows and columns permuted with respect to the information sotred in col. 
    
    '''
        

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



def compute_covariance_centering_matrix_(n,dict_nobs,ab):
    if ab == 'k':
        return(eye(n,dtype=torch.float64))
    if ab == 's':
        In = eye(n, dtype=torch.float64)
        Jn = ones(n, n, dtype=torch.float64)
        return(In - 1/n * Jn)
    if ab == 'w':
        In = eye(n)
        effectifs = [v for k,v in dict_nobs.items() if k != 'ntot'] 
        diag_Jn_by_n = compute_diag_Jn_by_n(effectifs)
        return(In - diag_Jn_by_n)
    else:
        print('invalid anchor basis') 

class CenteringOps(NystromOps):

    def __init__(self):
        super(CenteringOps,self).__init__()

    def compute_centering_matrix_with_respect_to_some_effects(self):
        '''
        Computes the projection matrix corresponding to center the data with respect to the 
        meta information of center_by. 
        
        For example, if we have some metadata on the datasets to compare, such as the patient 
        of origin and the cell type, stored in the dataframe self.obs, we can model the data 
        such that each embeding is :
        
        k_ijk = a + bi + cij + eijk
        
        where a is the global mean, bi is the effect of patient i, and cij is the effect of 
        cell-type j of patient i, and eijk is a random noise specific to cell k in cell-type j in patient i. 

        If the patient and cell-types info are stored in self.obs, then, the function return a centering matrix 
        corresponding of center_by :
            if center_by = 'patient', then the centered embeddings are cij + eijk
            if center_by = 'celltype', then the centered embeddings are eijk
        
        One can retire the effect cij by substituting the center corresponding to 'cell-type' 
        and adding the center corresponding to 'patient' with :
            center_by = '#-celltype_+patient', then the centered embeddings are a + bi + eijk
        The output matrix is such that P = In - Jcelltype + Jpatient 
        where In is the Identity matrix, Jcelltype and Jpatient are the diagonal matrix corresponding to 
        celltype and patient given by `compute_diag_Jn_by_n`. 


        Parameters
        ----------
            self : Ktest, 
            should contain a metadata dataframe self.obs

            center_by (default = None) : str, 
                either a column of self.obs or a combination of columns with the following syntax
                - starts with '#' to specify that it is a combination of effects
                - each effect should be a column of self.obs, preceded by '+' is the effect is added and '-' if the effect is retired. 
                - the effects are separated by a '_'
                exemple : '#-celltype_+patient'

        Returns
        ------- 
            torch.tensor, the centering matrix corresponding to center_by
        '''
        
        center_by = self.center_by
        marked_obs_to_ignore = self.marked_obs_to_ignore
        n = self.get_ntot()
        Pw = eye(n,dtype = torch.float64)
        obs = self.obs if marked_obs_to_ignore is None else self.obs[~self.obs[marked_obs_to_ignore]]

        if center_by is None:
            pass
        
        elif self.center_by[0] == '#':
            for center_by in self.center_by[1:].split(sep='_'):
                
                operation,effect = center_by[0],center_by[1:]
                col              = obs[effect].astype('category')
                effectifs        = compute_effectifs(col)
                diag_Jn          = compute_diag_Jn_by_n(effectifs)
                diag_Jn          = permute_matrix_to_respect_index_order(diag_Jn,col)
                
                if operation == '-':
                    Pw -= diag_Jn
                elif operation == '+':
                    Pw += diag_Jn

        else:
            effect           = self.center_by
            col              = obs[effect].astype('category')
            effectifs        = compute_effectifs(col)
            diag_Jn          = compute_diag_Jn_by_n(effectifs)
            diag_Jn          = permute_matrix_to_respect_index_order(diag_Jn,col)
            
            Pw -= diag_Jn

        return Pw    

    def compute_covariance_centering_matrix(self,quantization=False,landmarks=False):
        """
        Computes a projection matrix usefull for the kernel trick. 

        Example fir the within-group covariance :
            Let I1,I2 the identity matrix of size n1 and n2 (or nxanchors and nyanchors if nystrom).
            J1,J2 the squared matrix full of one of size n1 and n2 (or nxanchors and nyanchors if nystrom).
            012, 021 the matrix full of zeros of size n1 x n2 and n2 x n1 (or nxanchors x nyanchors and nyanchors x nxanchors if nystrom)
        
        Pn = [I1 - 1/n1 J1 ,    012     ]
                [     021     ,I2 - 1/n2 J2]

        Parameters
        ----------
            sample : string,
                if sample = 'xy' : returns the bicentering matrix corresponding to the within group covariance operator
                if sample = 'x' (resp. 'y') : returns the centering matrix corresponding to the covariance operator of sample 'x' (resp. 'y')
            
            quantization : boolean, 
                if quantization is true, returns the centering matrix of the landmarks obtained for a quantification approach. (may be deprecated since nystrom works well better)

            landmarks : boolean, 
                if landmarks is true, returns the centering matrix corresponding to the landmarks and not to the data. 
                    this centering matrix is used in the nystrom method.

        Returns
        ------- 
            P : torch.tensor, 
                the centering matrix corresponding to the parameters
        """


        dict_nobs = self.get_nobs(landmarks=landmarks)
        n = dict_nobs['ntot']
        ab = self.anchor_basis.lower() if landmarks else 'w'

        if quantization and not landmarks:
            print('quantization not implemented yet')
#                 a1 = self.compute_quantization_weights(sample='x',power=.5,diag=False)
#                 Pn1 = (In1 - 1/n * torch.ger(a1,a1)) # a vérifier si c'est 1/n ou 1/n1
            
        P = compute_covariance_centering_matrix_(n,dict_nobs,ab)
        return(P)

    def compute_omega(self,quantization=False):
        '''
        Returns the weights vector to compute a mean. 
        
        Parameters
        ----------
            sample : str,
            if sample = 'x' : returns a vector of size n1 = len(self.x) full of 1/n1
            if sample = 'y' : returns a vector of size n2 = len(self.y) full of 1/n2
            if sample = 'xy' : returns a vector of size n1 + n2 with n1 1/-n1 followed by n2 1/n2 to compute (\mu_2 - \mu_1) 
            

            quantization : boolean,
            if true, the landmarks should have been determined as the centers of a kmeans algorithm. 
            each landmark is associated to a cluster containing m observations. 
            the weight associated to this landmark is then m/ni whith ni = n1 or n2. 

        Returns
        ------- 
            omega : torch.tensor 
            a vector of size corresponding to the group of which we compute the mean. 
        '''

        # Avant cette fonction était plus générique mais je ne l'utilise que pour le vecteur omega de deux groupes
        n1,n2,n = self.get_n1n2n()
        if quantization:
            return(torch.cat((-1/n1*torch.bincount(self.xassignations),1/n2*torch.bincount(self.yassignations))).double())
        else:
            m_mu1    = -1/n1 * ones(n1, dtype=torch.float64) # , device=device)
            m_mu2    = 1/n2 * ones(n2, dtype=torch.float64) # , device=device) 
            return(torch.cat((m_mu1, m_mu2), dim=0)) #.to(device)

        
        
        

# Test application 
# a = Ktest()
# a.center_by = '#-cat_+sexe'
# a.obs = pd.DataFrame({'cat':np.array([0,0,0,1,1,1,2,2,2])[[1,3,5,7,0,2,4,6,8]],
#                      'sexe':np.array(['M','M','M','W','W','W','W','W','W',])[[1,3,5,7,0,2,4,6,8]]})
# a.obs.index = range(9)
# P = compute_centering_matrix_with_respect_to_some_effects(a)
# plt.imshow(P)