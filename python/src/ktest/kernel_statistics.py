import torch
import pandas as pd
from torch import mv,diag,dot,sum
import numpy as np
from .projection_operations import ProjectionOps
import warnings

"""

Ce fichier contient toutes les fonctions nécessaires au calcul des statistiques,
Les quantités pkm et upk sont des quantités génériques qui apparaissent dans beaucoup de calculs. 
Elles n'ont pas d'interprêtation facile, ces fonctions centrales permettent d'éviter les répétitions. 

Les fonctions initialize_kfdat et kfdat font simplement appel a plusieurs fonctions en une fois.

"""


class Statistics(ProjectionOps):

    def __init__(self):
        super(Statistics,self).__init__()


    def get_trace(self):
        sp,_ = self.get_spev('covw')
        return(sum(sp))
 
    def compute_kfdat(self,t=None,verbose=0):
        
        """ 
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
        # récupération des paramètres du modèle dans les attributs 
        cov = self.approximation_cov # approximation de l'opérateur de covariance. 
        mmd = self.approximation_mmd # approximation du vecteur mu2 - mu1 
        
        # Récupération des vecteurs propres et valeurs propres calculés par la fonction de classe 
        # diagonalize_within_covariance_centered_gram 
        sp,ev = self.get_spev('covw')


        # définition du nom de la colonne dans laquelle seront stockées les valeurs de la stat 
        # dans l'attribut df_kfdat (une DataFrame Pandas)   
        
        kfdat_name = self.get_kfdat_name()
        
        if kfdat_name in self.df_kfdat:
            if verbose >0:
                print(f"écrasement de {kfdat_name} dans df_kfdat and df_kfdat_contributions")


        # détermination de la troncature maximale à calculer 
        t = len(sp) if t is None else len(sp) if len(sp)<t else t # troncature maximale



        # instancier le calcul GPU si possible, date du tout début et je pense que c'est la raison pour 
        # laquelle mon code est en torch, mais ça n'a jamais représenté un gain de temps. 
        # maintenant je ne l'utilise plus, ça légitime la question de voir si ça ne vaut pas le coup de passer 
        # sur du full numpy.  
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # calcul de la statistique pour chaque troncature 
        pkm = self.compute_pkm() # partie de la stat qui ne dépend pas de la troncature mais du modèle
        n1,n2,n = self.get_n1n2n() # nombres d'observations dans les échantillons
        exposant = 2 if cov=='standard' else 1 #if cov == 'nystrom' 
        kfda_contributions = ((n1*n2)/(n**exposant*sp[:t]**exposant)*mv(ev.T[:t],pkm)**2).numpy() # calcule la contribution de chaque troncature 
        kfda = kfda_contributions.cumsum(axis=0) #somme les contributions pour avoir la stat correspondant à chaque troncature 
        
        
        # print('\n\nstat compute kfdat\n\n''sp',sp,'kfda',kfda)

        # stockage des vecteurs propres et valeurs propres dans l'attribut spev
        trunc = range(1,t+1) # liste des troncatures possibles de 1 à t 

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.df_kfdat[kfdat_name] = pd.Series(kfda,index=trunc)
            self.df_kfdat_contributions[kfdat_name] = pd.Series(kfda_contributions,index=trunc)

        self.has_kfda_statistic = True        
        # appel de la fonction verbosity qui va afficher le temps qu'ont pris les calculs

        
        # les valeurs de la statistique ont été stockées dans une colonne de la dataframe df_kfdat. 
        # pour ne pas avoir à chercher le nom de cette colonne difficilement, il est renvoyé ici
        return(kfdat_name)

    def compute_kfdat_contrib(self,t):    
        
        sp,ev = self.get_spev('covw')
        n = self.get_ntot() 
        
        pkom = self.compute_pkm()
        om = self.compute_omega()
        K = self.compute_gram()
        mmd = dot(mv(K,om),om)
    
        # yp = n1*n2/n * 1/(sp[:t]*n) * mv(ev.T[:t],pkom)**2 #1/np.sqrt(n*sp[:t])*
        # ysp = sp[:t]    
        t = len(sp) if t is None else t
        xp = range(1,t+1)
        yspnorm = sp[:t]/torch.sum(sp)
        ypnorm = 1/(sp[:t]*n) * mv(ev.T[:t],pkom)**2 /mmd
        # il y a des erreurs numériques sur les f donc je n'ai pas une somme totale égale au MMD

        return(xp,yspnorm,ypnorm)
    
    def initialize_kfdat(self,verbose=0,**kwargs):
        """
        This function prepares the computation of the kfda statistic by precomputing everithing that 
        should be needed. 
        If a nystrom approximation is in the model, it computes the landmarks and anchors if not computed yet. 
        It also diagonalize the within covariance centered gram if not diagonalized yet. 

        """
        
        # récupération des paramètres du modèle dans les attributs 
        cov = self.approximation_cov # approximation de l'opérateur de covariance. 
        mmd = self.approximation_mmd # approximation du vecteur mu2 - mu1 
        # nystrom n'est pas autorisé si l'un des dataset a moins de 100 observations. 
        if verbose>0:
            nystr = 'with nystrom approximation' if self.nystrom else ''
            print(f'- Initialize kfdat {nystr}')

        #calcul des landmarks et des ancres 
        if 'nystrom' in [cov,mmd]:
            self.compute_nystrom_landmarks(verbose=verbose)
            self.compute_nystrom_anchors(verbose=verbose) 

        # diagonalisation de la matrice d'intérêt pour calculer la statistique 
        if (self.nystrom and self.has_anchors) or (not self.nystrom) :
            self.diagonalize_within_covariance_centered_gram(approximation=cov,verbose=verbose)

    def initialize_mmd(self,shared_anchors=True,verbose=0):

        """
        Calculs preliminaires pour lancer le MMD.
        approximation: determine les calculs a faire en amont du calcul du mmd
                    full : aucun calcul en amont puisque la Gram et m seront calcules dans mmd
                    nystrom : 
                            si il n'y a pas de landmarks deja calcules, on calcule nloandmarks avec la methode landmark_method
                            si shared_anchors = True, alors on calcule un seul jeu d'ancres de taille n_anchors pour les deux echantillons
                            si shared_anchors = False, alors on determine un jeu d'ancre par echantillon de taille n_anchors//2
                                        attention : le parametre n_anchors est divise par 2 pour avoir le meme nombre total d'ancres, risque de poser probleme si les donnees sont desequilibrees
                     
        n_landmarks : nombre de landmarks a calculer si approximation='nystrom' ou 'kmeans'
        landmark_method : dans ['random','kmeans'] methode de choix des landmarks
        verbose : booleen, vrai si les methodes appellees renvoies des infos sur ce qui se passe.  
        """
            # verbose -1 au lieu de verbose ? 

        approx = self.approximation_mmd

        if approx == 'nystrom':
            if not self.has_landmarks:
                    self.compute_nystrom_landmarks(verbose=verbose)
            
            if shared_anchors:
                if self.get_anchors_name() not in self.spev['anchors']:
                    self.compute_nystrom_anchors(verbose=verbose)
            # pas à jour 
            else:
                for xy in 'xy':
                    if 'anchors' not in self.spev[xy]:
                        assert(self.n_anchors is not None,"n_anchors not specified")
                        self.compute_nystrom_anchors(verbose=verbose)
 
    def compute_mmd(self,unbiaised=False,verbose=0):
        
        approx = self.approximation_mmd
        self.verbosity(function_name='compute_mmd',
                dict_of_variables={'unbiaised':unbiaised,
                                    'approximation':approx,
                                    },
                start=True,
                verbose = verbose)

        if approx == 'standard':
            m = self.compute_omega()
            K = self.compute_gram()
            if unbiaised:
                K.masked_fill_(torch.eye(K.shape[0],K.shape[0]).byte(), 0)
            mmd = dot(mv(K,m),m)**2 #je crois qu'il n'y a pas besoin de carré
        
        if approx == 'nystrom':

            m = self.compute_omega()
            Lp,Up = self.get_spev(slot='anchors')
            Lp12 = diag(Lp**-(1/2))
            Pm = self.compute_covariance_centering_matrix(landmarks=True)
            Kmn = self.compute_kmn()
            psi_m = mv(Lp12,mv(Up.T,mv(Pm,mv(Kmn,m))))
            mmd = dot(psi_m,psi_m)**2

        mmd_name = self.get_mmd_name()
        
        self.dict_mmd[mmd_name] = mmd.item()
        
        self.verbosity(function_name='compute_mmd',
                dict_of_variables={'unbiaised':unbiaised,
                                    'approximation':approx,
                                    },
                start=False,
                verbose = verbose)
        return(mmd.item())

    # def compute_kfdat_with_different_order(self,order='between'):
    #     '''
    #     Computes a truncated kfda statistic which is defined as the original truncated kfda statistic but 
    #     the eigenvectors and eigenvalues of the within covariance operator are not ordered by decreasing eigenvalues. 
        
    #     Parameters
    #     ----------
    #         order : str, in 'between', ...? 
    #         specify the rule to order the eigenvectors
    #         so far there is only one choice but I want to add a second one which would 
    #         be a compromize between the reconstruction of (\mu_2 - \mu_1) and the 
    #         reconstruction of the within covariance operator. 
        
    #     Returns
    #     -------
    #         The attribute `df_kfdat` is updated with new columns corresponding to the new kfda statistic. 
    #         Each new column is a column of the attribute `df_kfdat_contributions` with a '_between' at the end. 
    #     '''
    #     if order == 'between':
    #         projection_error,ordered_truncations = self.get_ordered_spectrum_wrt_between_covariance_projection_error()
            
    #         kfda_contrib = self.df_kfdat_contributions
    #         kfda_between = kfda_contrib.T[ordered_truncations.tolist()].T.cumsum()
    #         kfda_between.index = range(1,len(ordered_truncations)+1)
    #         for c in kfda_contrib.columns:
    #             self.df_kfdat[f'{c}_between'] = kfda_between[c]
