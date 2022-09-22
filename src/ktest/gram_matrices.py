import torch
import numpy as np
from torch import mv,ones,cat,eye,zeros,ger,isnan
from .utils import ordered_eigsy



from ktest.centering_operations import CenteringOps

"""
Les fonctions de ce fichier gèrent tout ce qui est évaluation de matrices de gram, centrage et 
diagonalisation. 
"""
class GramMatrices(CenteringOps):

    def __init__(self):
        super(GramMatrices,self).__init__()

    def compute_gram(self,landmarks=False): 
        """
        Computes the Gram matrix of the data corresponding to the parameters sample and landmarks. 
        
        The kernel used is the kernel stored in the attribute `kernel`. 
        ( The attribute `kernel` can be initialized with the method init_kernel() ) 

        The computed Gram matrix is centered with respect to the attribute `center_by`.
        ( The attribute `center_by` can be initialized with the method init_center_by())
        
        The Gram matrix is not stored in memory because it is usually large and fast to compute. 

        Parameters
        ----------


        Returns
        -------
            K : torch.Tensor,
                Gram matrix corresponding to the parameters centered with respect to the attribute `center_by`
        """


        dict_data = self.get_data(landmarks=landmarks)
        
        data = torch.cat([x for x in dict_data.values()],axis=0)
        K = self.kernel(data,data)
        if not landmarks : 
            K = self.center_gram_matrix_with_respect_to_some_effects(K)
        return(K)

    def center_gram_matrix_with_respect_to_some_effects(self,K):
        '''
        Faire ce calcul au moment de calculer la gram n'est pas optimal car il ajoute un produit de matrice
        qui peut être cher, surtout quand n est grand. 
        L'autre alternative plus optimale serait de calculer la matrice de centrage par effet à chaque fois 
        qu'on a besoin de faire une opération sur la gram. Cette solution diminuerait le nombre d'opération 
        mais augmenterait la complexité du code car il faudrait à chaque fois vérifier si on a un centrage à 
        faire et écrire la version du calcul avec et sans centrage (pour éviter de multiplier inutilement par une 
        matrice identité). 
        '''

        if self.center_by is None:
            return(K)
        else:
            P = self.compute_centering_matrix_with_respect_to_some_effects()
            return(torch.chain_matmul(P,K,P))
            # retutn(torch.linalg.multi_dot([P,K,P]))

    def compute_kmn(self):
        """
        Computes an (nxanchors+nyanchors)x(n1+n2) conversion gram matrix
        """
        assert(self.has_landmarks)
        
        dict_data = self.get_data(landmarks=False)
        dict_landmarks = self.get_data(landmarks=True)
        
        data = torch.cat([x for x in dict_data.values()],axis=0)
        landmarks = torch.cat([x for x in dict_landmarks.values()],axis=0)
        
        kmn = self.kernel(landmarks,data)        
        kmn = self.center_kmn_matrix_with_respect_to_some_effects(kmn)
        return(kmn)

    def center_kmn_matrix_with_respect_to_some_effects(self,kmn):
        '''
        Cf commentaire dans center_gram_matrix_with_respect_to_some_effects 
        '''
        if self.center_by is None:
            return(kmn)
        else:
            P = self.compute_centering_matrix_with_respect_to_some_effects()
            return(torch.matmul(kmn,P))
            # retutn(torch.linalg.multi_dot([K,P]))

    def compute_within_covariance_centered_gram(self,approximation='standard',verbose=0):
        """ 
        Computes and returns the bicentered Gram matrix which shares its spectrum with the 
        within covariance operator. 

        Parameters 
        ----------
            approximation (default='standard') : str,  
            Which matrix to diagonalize.  

            sample (default='xy') : str, on which data do we compute the matrix

            verbose (default = 0) : Dans l'idée, plus verbose est grand, plus la fonction détaille ce qu'elle fait

            outliers_in_obs : None ou string,
            nom de la colonne de l'attribut obs qui dit quelles cellules doivent être considérées comme des outliers et ignorées.    

        """

        # fonction qui gère la verbosité de la fonction si verbose >0 
        self.verbosity(function_name='compute_centered_gram',
                dict_of_variables={},
                start=True,
                verbose = verbose)    

        # Instantiation de la matrice de centrage P 
        quantization = approximation == 'quantization'
        P = self.compute_covariance_centering_matrix(quantization=quantization,landmarks=False)

        # Ici n ne correspond pas au nombre d'observations total 
        # mais au nombre d'observations sur lequel est calculé la matrice, déterminé par 
        # le paramètre sample. C'est le 1/n devant la structure de covariance. 
        dict_nobs = self.get_nobs(landmarks=False)
        
        n = dict_nobs['ntot']

        # récupération des paramètres du modèle spécifique à l'approximation de nystrom (infos sur les ancres)    
        if 'nystrom' in approximation:
            dict_nlandmarks = self.get_nobs(landmarks=True)

            m = dict_nlandmarks['ntot']
            anchor_name = self.get_anchors_name()
            
        # plus utilisé 
        # récupération des paramètres du modèle spécifique à l'approximation par quantization (infos sur les landmarks et les poids associés)    
        # pas à jour
        if approximation == 'quantization':
            if self.quantization_with_landmarks_possible:
                Kmm = self.compute_gram(landmarks=True)
                A = self.compute_quantization_weights(sample=sample,power=.5)
                Kw = 1/n * torch.chain_matmul(P,A,Kmm,A,P)
                # Kw = 1/n * torch.linalg.multi_dot([P,A,Kmm,A,P])
            else:
                print("quantization impossible, you need to call 'compute_nystrom_landmarks' with landmark_method='kmeans'")


        # plus utilisé, aucun gain de temps car matrice n x n 
        elif approximation == 'nystrom1':
            # version brute mais a terme utiliser la svd ?? 
            if self.has_landmarks: 
                Kmn = self.compute_kmn()
                Lp,Up = self.get_spev(slot='anchors')
                Lp_inv = torch.diag(Lp**(-1))
                
                Pm = self.compute_covariance_centering_matrix(quantization=False,landmarks=True)
                Kw = 1/(n*m*2) * torch.chain_matmul(P,Kmn.T,Pm,Up,Lp_inv,Up.T,Pm,Kmn,P)            
                # Kw = 1/(n*r*2) * torch.linalg.multi_dot([P,Kmn.T,Pm,Up,Lp_inv,Up.T,Pm,Kmn,P])            
                # Kw = 1/(n) * torch.linalg.multi_dot([P,Kmn.T,Pm,Up,Lp_inv,Up.T,Pm,Kmn,P])            
                
            else:
                print("nystrom impossible, you need compute landmarks and/or anchors")
        
        # calcul de la matrice correspondant à l'approximation de nystrm. 
        elif approximation in ['nystrom2','nystrom3']:
            if self.has_landmarks:
                Kmn = self.compute_kmn()

                Lp,Up = self.get_spev(slot='anchors')

                Lp_inv_12 = torch.diag(Lp**(-(1/2)))

                # on est pas censé avoir de nan, il y en avait quand les ancres donnaient des spectres négatifs à cause des abérations numérique,
                # problème réglé en amont 
                # assert(not any(torch.isnan(Lp_inv_12)))

                
                
                Pm = self.compute_covariance_centering_matrix(quantization=False,landmarks=True)
                # print(f'Lp_inv_12{Lp_inv_12.shape},Up{Up.shape},Pm{Pm.shape},Kmn{Kmn.shape}')
                
                

                # Calcul de la matrice à diagonaliser avec Nystrom. 
                # Comme tu le disais, cette formule est symétrique et on pourrait utiliser une SVD en l'écrivant BB^T
                # où B = 1/(nm) Lp Up' Pm Kmn P  (car PP = P)
                Kw = 1/(n*m**2) * torch.chain_matmul(Lp_inv_12,Up.T,Pm,Kmn,P,Kmn.T,Pm,Up,Lp_inv_12)            

            else:
                print("nystrom new version impossible, you need compute landmarks and/or anchors")
                    

        # version standard 
        elif approximation == 'standard':
            K = self.compute_gram(landmarks=False)
            Kw = 1/n * torch.chain_matmul(P,K,P)
            # Kw = 1/n * torch.linalg.multi_dot([P,K,P])

        # appel de la fonction verbosity qui va afficher le temps qu'ont pris les calculs
        self.verbosity(function_name='compute_centered_gram',
                dict_of_variables={},
                start=False,
                verbose = verbose)    

        return Kw

    def diagonalize_within_covariance_centered_gram(self,approximation='standard',verbose=0):
        """
        Diagonalizes the bicentered Gram matrix which shares its spectrum with the Withon covariance operator in the RKHS.
        Stores eigenvalues and eigenvectors in the dict spev 

        Parameters 
        ----------
            approximation (default='standard') : str,  
            Which matrix to diagonalize.  

            sample (default='xy') : str, on which data do we compute the matrix

            verbose (default = 0) : Dans l'idée, plus verbose est grand, plus la fonction détaille ce qu'elle fait

            outliers_in_obs : None ou string,
            nom de la colonne de l'attribut obs qui dit quelles cellules doivent être considérées comme des outliers et ignorées. 

        """

        self.verbosity(function_name='diagonalize_within_covariance_centered_gram',
                dict_of_variables={},
                start=True,
                verbose = verbose)
        
        # calcul de la matrice a diagonaliser
        Kw = self.compute_within_covariance_centered_gram(approximation=approximation,verbose=verbose)
        
        # diagonalisation par la fonction C++ codée par François puis tri décroissant des valeurs propres 
        # et vecteurs prores
        sp,ev = ordered_eigsy(Kw) 

        # A un moment j'avais des abérations sur le spectre inversé qui faussaient tous mes calculs. 
        # J'ai réglé ce problème en amont. 
        # sp12 = sp**(-1/2)
        # ev = ev[:,~isnan(sp12)]
        # sp = sp[~isnan(sp12)]
        # print('Kw',Kw,'sp',sp,'ev',ev)


        # enregistrement des vecteurs propres et valeurs propres dans l'attribut spev
        spev_name = self.get_covw_spev_name()
        

        self.spev['covw'][spev_name] = {'sp':sp,'ev':ev}
        
        self.verbosity(function_name='diagonalize_within_covariance_centered_gram',
                dict_of_variables={},
                start=False,
                verbose = verbose)
        

