
from torch import mv,diag,dot,sum
import torch
from .gram_matrices import GramMatrices

class KernelTrick(GramMatrices):

    # def __init__(self,data,obs=None,var=None,):
    #     super(KernelTrick,self).__init__(data,obs=obs,var=var,)

    def compute_pkm(self):
        '''

        This function computes the term corresponding to the matrix-matrix-vector product PK omega
        of the KFDA statistic.
        
        See the description of the method compute_kfdat() for a brief description of the computation 
        of the KFDA statistic. 


        Parameters
        ----------
            self : Ktest,
            the model parameter attributes `approximation_cov`, `approximation_mmd` must be defined.
            if the nystrom method is used, the attribute `anchor_basis` should be defined and the anchors must have been computed. 


        Returns
        ------- 
        pkm : torch.tensor 
        Correspond to the product PK omega in the KFDA statistic. 
        '''


        nystrom = self.nystrom

        if nystrom:
            n_landmarks = self.get_ntot(landmarks=True)
            Lz1,Uz = self.get_spev(slot='anchors') # vecteurs propres de l'operateur de covariance des 
            Lz = diag(Lz1**-1) # matrice diagonale des inverses des valeurs propres associées aux ancres. 

        # instantiation du vecteur de bi-centrage omega et de la matrice de centrage Pbi 
        omega = self.compute_omega() # vecteur de 1/n1 et de -1/n2 
        Pbi = self.compute_covariance_centering_matrix() # matrice de centrage par block
        

        if nystrom: 
            Lz,_ = self.get_spev(slot='anchors')
            Lz12 = diag(Lz**-(1/2))
            Kzx = self.compute_kmn()
            # print("statistics pkm: L-1 nan ",(torch.isnan(torch.diag(Lz12))))
            Pi = self.compute_covariance_centering_matrix(landmarks=True)
            pkm = 1/n_landmarks * mv(Lz12,mv(Uz.T,mv(Pi,mv(Kzx,omega)))) # le vecteur d'intérêt renvoyé par la fonction  

        else: 
            Kx = self.compute_gram(landmarks=False) # matrice de gram 
            pkm = mv(Pbi,mv(Kx,omega)) # le vecteur que renvoie cette fonction 

        return(pkm) #

    def compute_upk(self,t,proj_condition=None,
                    proj_samples=None,
                    proj_marked_obs_to_ignore=None):
        """
        epk is an alias for the product ePK that appears when projecting the data on the discriminant axis. 
        This functions computes the corresponding block with respect to the model parameters. 

        warning: some work remains to be done to :
            - normalize the vectors with respect to n_anchors as in pkm 
            - separate the different nystrom approaches 
        """
        
        nystrom = self.nystrom
        proj = False if (proj_condition is None and proj_samples is None) else True


        sp,ev = self.get_spev('covw')

        Pbi = self.compute_covariance_centering_matrix(landmarks=False)


        if nystrom: 
            Kzx = self.compute_kmn(condition=proj_condition,samples=proj_samples,marked_obs_to_ignore=proj_marked_obs_to_ignore)
            n_landmarks = self.get_ntot(landmarks=True)
            Lz,Uz = self.get_spev(slot='anchors')
            Lz12 = diag(Lz**-(1/2))
            # print(f'm:{m} evt:{ev.T[:t].shape} Lz12{Lz12.shape} Uz{Uz.shape} Kzx{Kzx.shape}')
            epk = 1/n_landmarks**(1/2) * torch.linalg.multi_dot([ev.T[:t],Lz12,Uz.T,Kzx]).T
        
        else:
            if proj:
                Kx = self.compute_rectangle_gram(
                                y_condition=proj_condition,y_samples=proj_samples,y_marked_obs_to_ignore=proj_marked_obs_to_ignore)
            else:    
                Kx = self.compute_gram(landmarks=False)
            epk = torch.linalg.multi_dot([ev.T[:t],Pbi,Kx]).T

        return(epk)
