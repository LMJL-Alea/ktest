import torch
import numpy as np
from torch import mv,ones,cat,eye,zeros,ger,isnan
from .utils import ordered_eigsy



from .centering_operations import CenteringOps

"""
Les fonctions de ce fichier gèrent tout ce qui est évaluation de matrices de gram, centrage et 
diagonalisation. 
"""
class GramMatrices(CenteringOps):

    # def __init__(self,data,obs=None,var=None,):
    #     super(GramMatrices,self).__init__(data,obs=obs,var=var,)

    def compute_gram(self,landmarks=False,condition=None,samples=None,marked_obs_to_ignore=None): 
        """
        Computes the Gram matrix of the data corresponding to the parameters sample and landmarks. 

        The kernel used is the kernel stored in the attribute `kernel`. 
        ( The attribute `kernel` can be initialized with the method kernel() ) 

        The Gram matrix is not stored in memory because it is usually large and fast to compute. 

        Parameters
        ----------
            landmarks (default = False): bool 
                    Landmarks or observations ? 
            condition (default = None): str
                    Column of the metadata that specify the dataset  
            samples (default = None): str 
                    List of values to select in the column condition of the metadata
            marked_obs_to_ignore (default = None): str
                    Column of the metadata specifying the observations to ignore

        Returns
        -------
            K : torch.Tensor,
                Gram matrix of interest
        """


        dict_data = self.get_data(landmarks=landmarks,condition=condition,samples=samples,marked_obs_to_ignore=marked_obs_to_ignore)
        kernel = self.kernel
        data = torch.cat([x for x in dict_data.values()],axis=0)
        K = kernel(data,data)
        return(K)

    def compute_rectangle_gram(self,
                               x_landmarks=False,
                               x_condition=None,
                               x_samples=None,
                               x_marked_obs_to_ignore=None,
                               y_landmarks=False,
                               y_condition=None,
                               y_samples=None,
                               y_marked_obs_to_ignore=None):    
        """
        Computes the matrix K(X_i,Y_i) where X_i and Y_i are two subsets of the observations contained in the data.

        Parameters
        ----------
            x_landmarks,y_landmarks (default = False): bool 
                    Whether to use landmarks or observations  
            
            x_condition, y_condition (default = None): str
                    Column of the metadata that specify the dataset

            x_samples, y_samples (default = None): str 
                    List of values to select in the column x_condition/y_condition of the metadata
            
            x_marked_obs_to_ignore, y_marked_obs_to_ignore (default = None): str
                    Column of the metadata specifying the observations to ignore
        
        """
        
        xy_data = []
        for landmarks,condition,samples,marked_obs_to_ignore in zip([x_landmarks,y_landmarks],
                                                                    [x_condition,y_condition],
                                                                    [x_samples,y_samples],
                                                                    [x_marked_obs_to_ignore,y_marked_obs_to_ignore]):
            dict_data = self.get_data(landmarks=landmarks,condition=condition,samples=samples,marked_obs_to_ignore=marked_obs_to_ignore)
            kernel = self.kernel
            data = torch.cat([x for x in dict_data.values()],axis=0)
            xy_data += [data]
        K = kernel(xy_data[0],xy_data[1])
        return(K)



    def compute_kmn(self,condition=None,samples=None,marked_obs_to_ignore=None):
        """
        Computes an (nxanchors+nyanchors)x(ndata) conversion gram matrix

            Parameters
        ----------
            landmarks (default = False): bool 
                    Landmarks or observations ? 
            condition (default = None): str
                    Column of the metadata that specify the dataset  
            samples (default = None): str 
                    List of values to select in the column condition of the metadata
            marked_obs_to_ignore (default = None): str
                    Column of the metadata specifying the observations to ignore


        """
        assert(self.has_landmarks)
        dict_data = self.get_data(landmarks=False,condition=condition,samples=samples,marked_obs_to_ignore=marked_obs_to_ignore)
        dict_landmarks = self.get_data(landmarks=True)
        kernel = self.kernel

        data = torch.cat([x for x in dict_data.values()],axis=0)
        landmarks = torch.cat([x for x in dict_landmarks.values()],axis=0)

        kmn = kernel(landmarks,data)        
        return(kmn)



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

            marked_obs_to_ignore : None ou string,
            nom de la colonne de l'attribut obs qui dit quelles cellules doivent être considérées comme des outliers et ignorées.    

        """

        if verbose>0:
            print(f'- Compute within covariance centered gram')

        # Instantiation de la matrice de centrage P 
        P = self.compute_covariance_centering_matrix(landmarks=False)

        # Ici n ne correspond pas au nombre d'observations total 
        # mais au nombre d'observations sur lequel est calculé la matrice, déterminé par 
        # le paramètre sample. C'est le 1/n devant la structure de covariance. 
        dict_nobs = self.get_nobs(landmarks=False)
        
        n = dict_nobs['ntot']

        # récupération des paramètres du modèle spécifique à l'approximation de nystrom (infos sur les ancres)    
        if approximation == 'nystrom':
            dict_n_landmarks = self.get_nobs(landmarks=True)
            n_landmarks = dict_n_landmarks['ntot']
           
            # calcul de la matrice correspondant à l'approximation de nystrm. 
            if self.has_landmarks:
                Kmn = self.compute_kmn()

                Lp,Up = self.get_spev(slot='anchors')

                Lp_inv_12 = torch.diag(Lp**(-(1/2)))

                # on est pas censé avoir de nan, il y en avait quand les ancres donnaient des spectres négatifs à cause des abérations numérique,
                # problème réglé en amont 
                # assert(not any(torch.isnan(Lp_inv_12)))

                
                
                Pm = self.compute_covariance_centering_matrix(landmarks=True)
                # print(f'Lp_inv_12{Lp_inv_12.shape},Up{Up.shape},Pm{Pm.shape},Kmn{Kmn.shape}')
                
                

                # Calcul de la matrice à diagonaliser avec Nystrom. 
                # Comme tu le disais, cette formule est symétrique et on pourrait utiliser une SVD en l'écrivant BB^T
                # où B = 1/(nm) Lp Up' Pm Kmn P  (car PP = P)         
                Kw = 1/(n*n_landmarks**2) * torch.linalg.multi_dot([Lp_inv_12,Up.T,Pm,Kmn,P,Kmn.T,Pm,Up,Lp_inv_12])  
            else:
                print("nystrom impossible, you need compute landmarks and/or anchors")
                    

        # version standard 
        elif approximation == 'standard':
            K = self.compute_gram(landmarks=False)
            Kw = 1/n * torch.linalg.multi_dot([P,K,P])
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

        """

        if verbose >0:
            print(f'- Diagonalize within covariance centered gram')
        
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



