from ktest.residuals import Residuals
from torch import cat,tensor,float64
import numpy as np
from numpy import where
   
"""
Le choix de la troncature est la question difficile des tests avec KFDA.
Ici, j'ai mis plusieurs procédures de sélection automatique de la troncature. 
Plus tard, une idée à implémenter est de selectionner t par permutation. 
C'est à dire qu'on crée à partir des données des couples d'échantillons sous H0 et on voit jusqu'à 
quelle troncature on peut aller avec une erreur de type I contrôllée. 
On choisit à terme la plus grande troncature qui contrôle cette erreur.  
"""

class TruncationSelection(Residuals):

    def __init__(self):
        super(TruncationSelection,self).__init__()

    def select_trunc_by_between_reconstruction_ratio(self,ratio):
        pe = self.get_between_covariance_projection_error()
        pe = cat([tensor([1],dtype =float64),pe])
        return(where(pe<ratio)[0][0])

        
    def select_trunc_by_between_reconstruction_ressaut(self,kmax=11,S=.5,which_ressaut='max'):
        pe = self.get_between_covariance_projection_error()
        pe = cat([tensor([1],dtype =float64),pe])
        kmax = kmax if len(pe)>kmax else len(pe)-1
        pen = 1+ (kmax-1)* (pe[kmax]-pe)/(pe[kmax] - pe[1])
        D2 = np.diff(np.diff(pen))
        sel = np.where(D2>S)[0]
        fil = sel[sel<kmax]

        if len(fil)>0:
            if which_ressaut == 'max':
                tressaut = fil[np.argmax(D2[fil])] +1
                
            elif which_ressaut == 'first':
                tressaut = fil[0] +1
            elif which_ressaut == 'second':
                if len(fil)>1:
                    tressaut = fil[1] +1
                else: 
                    tressaut = fil[np.argmax(D2[fil])] +1

            elif which_ressaut == 'third':
                if len(fil)>2:
                    tressaut = fil[2] +1
                else: 
                    tressaut = fil[np.argmax(D2[fil])] +1                
        else:
            tressaut = 1

        #     print('pe',np.diff(pe)[:10])#     print('selected',sel)
        #     print('filtered',fil)#     print('values',val)
        
        return(tressaut)


    def select_trunc(self,selection_procedure='ressaut',selection_params={}):
        if selection_procedure == 'ressaut':
            self.t = self.select_trunc_by_between_reconstruction_ressaut(**selection_params)
        if selection_procedure == 'ratio':
            self.t = self.select_trunc_by_between_reconstruction_ratio(**selection_params)


# def get_trunc(self):
#     '''
#     This function should select automatically the truncation parameter of the KFDA method corresponding to the 
#     number of eigenvectors of Sigma_W used to compute the discriminant axis. 

#     It stores the choosen truncation parameter in the attribute self.t

#     The different methods we tried so far are : 
#         - take only the eigenvectors that support more than a certain threshold of the variance individually (e.g. 5%)
#         - take enough eigenvectors to cumulate 95% of the variance. 
#         - take the first eigenvector
#     '''


#     # suffix_nystrom = self.anchors_basis if 'nystrom' in self.approximation_cov else ''
#     # sp = self.spev[sample][self.approximation_cov+suffix_nystrom]['sp']
#     # avec ce code je ne prennais pas 95% de la variance comme je l'ai cru au départ.
#     # mais je prenais seulement les valeurs propres portant plus de 5% de variance
#     # spp = sp/torch.sum(sp)
#     # t = len(spp[spp>=ratio])
#     # return(t if t >0 else 1 )
#     # le test issu de t tel qu'on porte 95% de la variance est trop sensible et prend des valeurs de troncature très grandes

#         # spp = self.get_explained_variance(sample)
#         # t = len(spp[spp<(1-ratio)])
#     self.t = 1

# def get_95variance_trunc(self,sample='xy'):
#     '''
#     This function returns the number of eigenvalues to take to support 95% of the variance of the covariance operator
#     of interest.

#     Parameters
#     ----------
#         sample : str,
#         if sample = 'x' : Focuses on the covariance operator of the first sample
#         if sample = 'y' : Focuses on the covariance operator of the second sample
#         if sample = 'xy' : Focuses on the within-group covariance operator 
        
#     Returns
#     ------- 
#         t : int,
#         The number of composants needed to have 95% of the variance
 
#     ''' 
#     spp = self.get_explained_variance(sample)
#     t = len(spp[spp<(1-.05)])
#     return(t)