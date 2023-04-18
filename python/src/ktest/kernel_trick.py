
from torch import mv,diag,dot,sum
import torch
from .gram_matrices import GramMatrices

class KernelTrick(GramMatrices):
           
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


        # récupération des paramètres du modèle dans les attributs 
        cov = self.approximation_cov # approximation de l'opérateur de covariance. 
        mmd = self.approximation_mmd # approximation du vecteur mu2 - mu1 

        # récupération des paramètres du modèle spécifique à l'approximation de nystrom (infos sur les ancres)    
        if 'nystrom' in cov or 'nystrom' in mmd :
            cov_anchors = 'shared' # ce paramètre ne sert à rien dans mon usage. 
            n_landmarks = self.get_ntot(landmarks=True)
            
            # récupération dans l'attribut spev (sp : spectrum, ev : eigenvectors) des vecteurs propres et valeurs propres de l'opérateur de covariance utilisé pour pour calculer les 
            # ancres à partir des landmarks. 
            
            Lz1,Uz = self.get_spev(slot='anchors') # vecteurs propres de l'operateur de covariance des 
            Lz = diag(Lz1**-1) # matrice diagonale des inverses des valeurs propres associées aux ancres. 

        # instantiation du vecteur de bi-centrage omega et de la matrice de centrage Pbi 
        omega = self.compute_omega(quantization=(mmd=='quantization')) # vecteur de 1/n1 et de -1/n2 
        Pbi = self.compute_covariance_centering_matrix(quantization=(cov=='quantization')) # matrice de centrage par block
        
        # instantiation de la matrice de passage des landmarks aux observations 
        # (ne sert pas que pour nystrom, aussi pour la statistique par quantification qu'on
        # a définie à un moment mais abandonnée pour ses mauvaises performances)
        if not (mmd == cov) or mmd == 'nystrom':
            Kzx = self.compute_kmn()
        

        # calcul de la statistique correspondant aux valeurs de cov et mmd.
        # au départ j'avais codé tous les cas de figure possibles mais beaucoup ne présentent 
        # aucun avantage ni en terme de coût computationnel, ni en termes d'approximation asymptotique. 
        # Les deux cas que j'utilise sont la statistique standard (cov='standard', mmd='standard')
        # et la 3eme version de ma statistique approximée par nystrom, (les cas cov='nystrom3', mmd='standard'
        # et cov='nystrom3',mmd='nystrom' sont équivalents). 
        # Pour l'instant je garde ces versions si jamais on a envie de parler des simulations que j'ai faites 
        # pour les comparer dans l'article, mais à terme je compte simplifier le code pour n'avoir que deux statistiques. 
        
        if cov == 'standard': 
            # cas standard 
            if mmd == 'standard': 
                Kx = self.compute_gram(landmarks=False) # matrice de gram 
                pkm = mv(Pbi,mv(Kx,omega)) # le vecteur que renvoie cette fonction 
            # aucun avantage
            elif mmd == 'nystrom': 
                Pi = self.compute_covariance_centering_matrix(quantization=False,landmarks=True)
                # print(f'Pbi{Pbi.shape}Kxz{Kzx.T.shape}Pi{Pi.shape}Uz{Uz.shape}Lz{Lz.shape}omega{omega.shape}')
                pkm = 1/n_landmarks * mv(Pbi,mv(Kzx.T,mv(Pi,mv(Uz,mv(Lz,mv(Uz.T,mv(Pi,mv(Kzx,omega))))))))
                # pkm = mv(Pbi,mv(Kzx.T,mv(Pi,mv(Uz,mv(Lz,mv(Uz.T,mv(Pi,mv(Kzx,omega))))))))
            # aucun avantage
            elif mmd == 'quantization':
                pkm = mv(Pbi,mv(Kzx.T,omega))

        if cov == 'nystrom1' and cov_anchors == 'shared':
            # aucun avantage
            if mmd in ['standard','nystrom']: # c'est exactement la même stat  
                Pi = self.compute_covariance_centering_matrix(quantization=False,landmarks=True)
                pkm = 1/n_landmarks**2 * mv(Pbi,mv(Kzx.T,mv(Pi,mv(Uz,mv(Lz,mv(Uz.T,mv(Pi,mv(Kzx,omega))))))))
                # pkm = mv(Pbi,mv(Kzx.T,mv(Pi,mv(Uz,mv(Lz,mv(Uz.T,mv(Pi,mv(Kzx,omega))))))))
            # aucun avantage
            elif mmd == 'quantization':
                Kz = self.compute_gram(landmarks=True)
                pkm = 1/n_landmarks**2 * mv(Pbi,mv(Kzx.T,mv(Uz,mv(Lz,mv(Uz.T,mv(Kz,omega))))))
                # pkm = mv(Pbi,mv(Kzx.T,mv(Uz,mv(Lz,mv(Uz.T,mv(Kz,omega))))))
        
        if cov == 'nystrom2' and cov_anchors == 'shared':
            # aucun avantage
            Lz,_ = self.get_spev(slot='anchors')
            Lz12 = diag(Lz**-(1/2))
            if mmd in ['standard','nystrom']: # c'est exactement la même stat  
                Pi = self.compute_covariance_centering_matrix(quantization=False,landmarks=True)
                pkm = 1/n_landmarks**3 * mv(Lz12,mv(Uz.T,mv(Pi,mv(Kzx,mv(Pbi,mv(Kzx.T,mv(Pi,mv(Uz,mv(Lz,mv(Uz.T,mv(Pi,mv(Kzx,omega))))))))))))
                # pkm = mv(Lz12,mv(Uz.T,mv(Pi,mv(Kzx,mv(Pbi,mv(Kzx.T,mv(Pi,mv(Uz,mv(Lz,mv(Uz.T,mv(Pi,mv(Kzx,omega))))))))))))
            # aucun avantage
            elif mmd == 'quantization': # pas à jour
                # il pourrait y avoir la dichotomie anchres centrees ou non ici. 
                Kz = self.compute_gram(landmarks=True)
                pkm = 1/n_landmarks**3 * mv(Lz12,mv(Uz.T,mv(Kzx,mv(Pbi,mv(Kzx.T,mv(Uz,mv(Lz,mv(Uz.T,mv(Kz,omega)))))))))
                # pkm = mv(Lz12,mv(Uz.T,mv(Kzx,mv(Pbi,mv(Kzx.T,mv(Uz,mv(Lz,mv(Uz.T,mv(Kz,omega)))))))))
        
        if cov == 'nystrom3' and cov_anchors == 'shared':
            Lz,_ = self.get_spev(slot='anchors')
            Lz12 = diag(Lz**-(1/2))
            # print("statistics pkm: L-1 nan ",(torch.isnan(torch.diag(Lz12))))
            Pi = self.compute_covariance_centering_matrix(quantization=False,landmarks=True)
            # cas nystrom 
            if mmd in ['standard','nystrom']: # c'est exactement la même stat  
                
                pkm = 1/n_landmarks * mv(Lz12,mv(Uz.T,mv(Pi,mv(Kzx,omega)))) # le vecteur d'intérêt renvoyé par la fonction 
                
                # pkm = mv(Lz12,mv(Uz.T,mv(Pi,mv(Kzx,omega))))
                # print(f'in compute pkm: \n\t\
                #    Lz12{Lz12}\n Uz{Uz}\n Kzx{Kzx}')

            elif mmd == 'quantization': # pas à jour 
                # il faut ajouter Pi ici . 
                Kz = self.compute_gram(landmarks=True)
                pkm = 1/n_landmarks**2 * mv(Lz12,mv(Uz.T,mv(Pi,mv(Kzx,mv(Pbi,mv(Kzx.T,mv(Pi,mv(Uz,mv(Lz,mv(Uz.T,mv(Kz,omega)))))))))))
                # pkm = mv(Lz12,mv(Uz.T,mv(Kzx,mv(Pbi,mv(Kzx.T,mv(Uz,mv(Lz,mv(Uz.T,mv(Kz,omega)))))))))
        


        # aucun avantage 
        if cov == 'quantization': # pas à jour 
            A = self.compute_quantization_weights(power=1/2,sample='xy')
            if mmd == 'standard':
                pkm = mv(Pbi,mv(A,mv(Kzx,omega)))

            elif mmd == 'nystrom':
                Pi = self.compute_covariance_centering_matrix(quantization=False,landmarks=True)
                Kz = self.compute_gram(landmarks=True)
                pkm = 1/n_landmarks * mv(Pbi,mv(A,mv(Kz,mv(Uz,mv(Lz,mv(Uz.T,mv(Pi,mv(Kzx,omega))))))))

            elif mmd == 'quantization':
                Kz = self.compute_gram(landmarks=True)
                pkm = mv(Pbi,mv(A,mv(Kz,omega)))
        
        try:
            return(pkm) #
        except UnboundLocalError:
            print(f'UnboundLocalError: pkm was not computed for cov:{cov},mmd:{mmd}')


    def compute_upk(self,t,proj_condition=None,proj_samples=None,proj_marked_obs_to_ignore=None):
        """
        epk is an alias for the product ePK that appears when projecting the data on the discriminant axis. 
        This functions computes the corresponding block with respect to the model parameters. 

        warning: some work remains to be done to :
            - normalize the vectors with respect to n_anchors as in pkm 
            - separate the different nystrom approaches 
        """

        cov = self.approximation_cov
        quantization = cov=='quantization'
        proj = False if (proj_condition is None and proj_samples is None) else True


        sp,ev = self.get_spev('covw')

        Pbi = self.compute_covariance_centering_matrix(quantization=quantization,landmarks=False)

        if 'nystrom' in cov: 
            Kzx = self.compute_kmn(condition=proj_condition,samples=proj_samples,marked_obs_to_ignore=proj_marked_obs_to_ignore)
            n_landmarks = self.get_ntot(landmarks=True)
            _,Uz = self.get_spev(slot='anchors')


        if cov == 'standard':
            if proj:
                Kx = self.compute_rectangle_gram(
                                y_condition=proj_condition,y_samples=proj_samples,y_marked_obs_to_ignore=proj_marked_obs_to_ignore)
            else:    
                Kx = self.compute_gram(landmarks=False)
            epk = torch.linalg.multi_dot([ev.T[:t],Pbi,Kx]).T

        if cov == 'nystrom3':
            Lz,_ = self.get_spev(slot='anchors')
            Lz12 = diag(Lz**-(1/2))
            # print(f'm:{m} evt:{ev.T[:t].shape} Lz12{Lz12.shape} Uz{Uz.shape} Kzx{Kzx.shape}')
            epk = 1/n_landmarks**(1/2) * torch.linalg.multi_dot([ev.T[:t],Lz12,Uz.T,Kzx]).T

        # elif 'nystrom' in cov:
        #     Lz,_ = self.get_spev(slot='anchors')
        #     Lz1 = diag(Lz**-1)
        #     # print(f'r:{r} evt:{ev.T[:t].shape} Pbi{Pbi.shape} Kzx{Kzx.shape} Uz{Uz.shape} Lz{Lz.shape}  ')
        #     epk = 1/m*torch.linalg.multi_dot([ev.T[:t],Pbi,Kzx.T,Uz,Lz1,Uz.T,Kzx]).T
        # # pas à jour 
        # if cov == 'quantization':
        #     Kzx = self.compute_kmn()
        #     A_12 = self.compute_quantization_weights(power=1/2,sample='xy')
        #     epk = torch.linalg.multi_dot([ev.T[:t],A_12,Pbi,Kzx]).T

        return(epk)
