import torch
import pandas as pd
from torch import mv,diag,chain_matmul,dot,sum

from .kernel_trick import KernelTrick

"""

Ce fichier contient toutes les fonctions nécessaires au calcul des statistiques,
Les quantités pkm et upk sont des quantités génériques qui apparaissent dans beaucoup de calculs. 
Elles n'ont pas d'interprêtation facile, ces fonctions centrales permettent d'éviter les répétitions. 

Les fonctions initialize_kfdat et kfdat font simplement appel a plusieurs fonctions en une fois.

"""


class Statistics(KernelTrick):

    def __init__(self):
        super(Statistics,self).__init__()


    def get_explained_variance(self,sample='xy',outliers_in_obs=None):
        '''
        This function returns a list of percentages of supported variance, the ith element contain the 
        variance supported by the first i eigenvectors of the covariance operator of interest. 

        Parameters
        ----------
            sample : str,
            if sample = 'x' : Focuses on the covariance operator of the first sample
            if sample = 'y' : Focuses on the covariance operator of the second sample
            if sample = 'xy' : Focuses on the within-group covariance operator 
                    
        Returns
        ------- 
            spp : torch.tensor,
            the list of cumulated variances ordered in decreasing order.  

        '''

        cov = self.approximation_cov
        suffix_nystrom = self.anchors_basis if 'nystrom' in self.approximation_cov else ''
        suffix_outliers = outliers_in_obs if outliers_in_obs is not None else ''
        sp = self.spev[sample][f'{cov}{suffix_nystrom}{suffix_outliers}']['sp']
        spp = (sp/torch.sum(sp)).cumsum(0)
        return(spp)

    def get_trace(self,sample='xy',outliers_in_obs=None):
        cov = self.approximation_cov
        suffix_nystrom = self.anchors_basis if 'nystrom' in self.approximation_cov else ''
        suffix_outliers = outliers_in_obs if outliers_in_obs is not None else ''
        sp = self.spev[sample][f'{cov}{suffix_nystrom}{suffix_outliers}']['sp']
        return(sum(sp))
 
    def compute_kfdat(self,t=None,name=None,verbose=0,outliers_in_obs=None):
        
        """ 
        Computes the kfda truncated statistic of [Harchaoui 2009].
        9 methods : 
        approximation_cov in ['standard','nystrom1','quantization']
        approximation_mmd in ['standard','nystrom','quantization']
        
        Stores the result as a column in the dataframe df_kfdat

        Parameters
        ----------
            self : tester,
            the model parameter attributes `approximation_cov`, `approximation_mmd` must be defined.
            if the nystrom method is used, the attribute `anchor_basis` should be defined and the anchors must have been computed. 

            t (default = None) : None or int,
            valeur maximale de troncature calculée. 
            Si None, t prend la plus grande valeur possible, soit n (nombre d'observations) pour la 
            version standard et r (nombre d'ancres) pour la version nystrom  

            name (default = None) : None or str, 
            nom de la colonne des dataframe df_kfdat et df_kfdat_contributions dans lesquelles seront stockés 
            les valeurs de la statistique pour les différentes troncatures calculées de 1 à t 

            verbose (default = 0) : Dans l'idée, plus verbose est grand, plus la fonction détaille ce qu'elle fait

            outliers_in_obs : None ou string,
            nom de la colonne de l'attribut obs qui dit quelles cellules doivent être considérées comme des outliers et ignorées. 


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
        
        # récupération des paramètres du modèle dans les attributs 
        cov = self.approximation_cov # approximation de l'opérateur de covariance. 
        mmd = self.approximation_mmd # approximation du vecteur mu2 - mu1 
        anchors_basis = self.anchors_basis # 'nystrom) quel opérateur de covariance des landmarks est utilisé pour calculer les ancres 
        
        # à un moment, j'ai exploré plusieurs façons de définir les ancres, 
        # shared correspond à des ancres partagées pour les deux échantillons 
        # et separated correspon à des ancres partagées entre les deux échantillons. 
        # je n'ai jamais été au bout de cette démarche et ça n'apportait pas grand chose dans 
        # les quelques simus qu'on a faites. A nettoyer. 
        cov_anchors='shared' 
        mmd_anchors='shared'


        # Récupération des vecteurs propres et valeurs propres calculés par la fonction de classe 
        # diagonalize_within_covariance_centered_gram 
        suffix_nystrom = anchors_basis if 'nystrom' in cov else ''
        suffix_outliers = outliers_in_obs if outliers_in_obs is not None else ''
        sp,ev = self.spev['xy'][f'{cov}{suffix_nystrom}{suffix_outliers}']['sp'],self.spev['xy'][f'{cov}{suffix_nystrom}{suffix_outliers}']['ev']


        # définition du nom de la colonne dans laquelle seront stockées les valeurs de la stat 
        # dans l'attribut df_kfdat (une DataFrame Pandas)   
        name = name if name is not None else outliers_in_obs if outliers_in_obs is not None else f'{cov}{mmd}' 
        if name in self.df_kfdat:
            print(f"écrasement de {name} dans df_kfdat and df_kfdat_contributions")


        # détermination de la troncature maximale à calculer 
        t = len(sp) if t is None else len(sp) if len(sp)<t else t # troncature maximale
        
        # fonction qui gère la verbosité de la fonction si verbose >0 
        self.verbosity(function_name='compute_kfdat',
                dict_of_variables={
                't':t,
                'approximation_cov':cov,
                'approximation_mmd':mmd,
                'name':name},
                start=True,
                verbose = verbose)


        # instancier le calcul GPU si possible, date du tout début et je pense que c'est la raison pour 
        # laquelle mon code est en torch, mais ça n'a jamais représenté un gain de temps. 
        # maintenant je ne l'utilise plus, ça légitime la question de voir si ça ne vaut pas le coup de passer 
        # sur du full numpy.  
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # calcul de la statistique pour chaque troncature 
        pkm = self.compute_pkm(outliers_in_obs=outliers_in_obs) # partie de la stat qui ne dépend pas de la troncature mais du modèle
        n1,n2,n = self.get_n1n2n(outliers_in_obs=outliers_in_obs) # nombres d'observations dans les échantillons
        exposant = 2 if cov in ['standard','nystrom1','quantization'] else 3 if cov == 'nystrom2' else 1 if cov == 'nystrom3' else 'erreur exposant' # l'exposant dépend du modèle
        kfda_contributions = ((n1*n2)/(n**exposant*sp[:t]**exposant)*mv(ev.T[:t],pkm)**2).numpy() # calcule la contribution de chaque troncature 
        kfda = kfda_contributions.cumsum(axis=0) #somme les contributions pour avoir la stat correspondant à chaque troncature 
        
        
        # print('\n\nstat compute kfdat\n\n''sp',sp,'kfda',kfda)

        # stockage des vecteurs propres et valeurs propres dans l'attribut spev
        trunc = range(1,t+1) # liste des troncatures possibles de 1 à t 
        self.df_kfdat[name] = pd.Series(kfda,index=trunc)
        self.df_kfdat_contributions[name] = pd.Series(kfda_contributions,index=trunc)
        
        # appel de la fonction verbosity qui va afficher le temps qu'ont pris les calculs
        self.verbosity(function_name='compute_kfdat',
                                dict_of_variables={
                't':t,
                'approximation_cov':cov,
                'approximation_mmd':mmd,
                'name':name},
                start=False,
                verbose = verbose)
        
        # les valeurs de la statistique ont été stockées dans une colonne de la dataframe df_kfdat. 
        # pour ne pas avoir à chercher le nom de cette colonne difficilement, il est renvoyé ici
        return(name)
    
    def compute_kfdat_contrib(self,t):    
        cov = self.approximation_cov    
        sp,ev = self.spev['xy'][cov]['sp'],self.spev['xy'][cov]['ev']
        n1,n2,n = self.get_n1n2n() 
        
        pkom = self.compute_pkm()
        om = self.compute_omega()
        K = self.compute_gram()
        mmd = dot(mv(K,om),om)
    
        # yp = n1*n2/n * 1/(sp[:t]*n) * mv(ev.T[:t],pkom)**2 #1/np.sqrt(n*sp[:t])*
        # ysp = sp[:t]    

        xp = range(1,t+1)
        yspnorm = sp[:t]/torch.sum(sp)
        ypnorm = 1/(sp[:t]*n) * mv(ev.T[:t],pkom)**2 /mmd
        # il y a des erreurs numériques sur les f donc je n'ai pas une somme totale égale au MMD

        return(xp,yspnorm,ypnorm)
    def initialize_kfdat(self,sample='xy',verbose=0,outliers_in_obs=None,**kwargs):
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

        # la quantisation était la troisième approche après nystrom et standard mais je ne l'utilise plus car 
        # son coût computationnel est faible mais ses performances le sont aussi. 
        # cette approche a besoin d'avoir un poids associé aux landmarks pour savoir combien ils représentent d'observations. 
        if 'quantization' in [cov,mmd] and not self.quantization_with_landmarks_possible: # besoin d'avoir des poids des ancres de kmeans en quantization
            self.compute_nystrom_landmarks(outliers_in_obs=outliers_in_obs,verbose=verbose) 

        #calcul des landmarks et des ancres 
        if any([ny in [cov,mmd] for ny in ['nystrom1','nystrom2','nystrom3']]):
            print('nystrom detected')
            self.compute_nystrom_landmarks(verbose=verbose,outliers_in_obs=outliers_in_obs)
            self.compute_nystrom_anchors(sample=sample,verbose=verbose,outliers_in_obs=outliers_in_obs) 

        # diagonalisation de la matrice d'intérêt pour calculer la statistique 
        self.diagonalize_within_covariance_centered_gram(approximation=cov,sample=sample,verbose=verbose,outliers_in_obs=outliers_in_obs)

    def initialize_mmd(self,shared_anchors=True,verbose=0,outliers_in_obs=None):

        """
        Calculs preliminaires pour lancer le MMD.
        approximation: determine les calculs a faire en amont du calcul du mmd
                    full : aucun calcul en amont puisque la Gram et m seront calcules dans mmd
                    nystrom : 
                            si il n'y a pas de landmarks deja calcules, on calcule nloandmarks avec la methode landmark_method
                            si shared_anchors = True, alors on calcule un seul jeu d'ancres de taille r pour les deux echantillons
                            si shared_anchors = False, alors on determine un jeu d'ancre par echantillon de taille r//2
                                        attention : le parametre r est divise par 2 pour avoir le meme nombre total d'ancres, risque de poser probleme si les donnees sont desequilibrees
                    quantization : m sont determines comme les centroides de l'algo kmeans 
        shared_anchors : si approximation='nystrom' alors shared anchors determine si les ancres sont partagees ou non
        m : nombre de landmarks a calculer si approximation='nystrom' ou 'kmeans'
        landmark_method : dans ['random','kmeans'] methode de choix des landmarks
        verbose : booleen, vrai si les methodes appellees renvoies des infos sur ce qui se passe.  
        """
            # verbose -1 au lieu de verbose ? 

        approx = self.approximation_mmd

        if approx == 'quantization' and not self.quantization_with_landmarks_possible: # besoin des poids des ancres de kmeans en quantization
            self.compute_nystrom_landmarks(verbose=verbose,outliers_in_obs=outliers_in_obs)
        
        if approx == 'nystrom':
            if not self.has_landmarks:
                    self.compute_nystrom_landmarks(verbose=verbose,outliers_in_obs=outliers_in_obs)
            
            if shared_anchors:
                if "anchors" not in self.spev['xy']:
                    self.compute_nystrom_anchors(sample='xy',verbose=verbose,outliers_in_obs=outliers_in_obs)
            else:
                for xy in 'xy':
                    if 'anchors' not in self.spev[xy]:
                        assert(self.r is not None,"r not specified")
                        self.compute_nystrom_anchors(sample=xy,verbose=verbose,outliers_in_obs=outliers_in_obs)
 
    def compute_mmd(self,unbiaised=False,shared_anchors=True,name=None,verbose=0,outliers_in_obs=None):
        
        approx = self.approximation_mmd
        anchors_basis=self.anchors_basis
        suffix_outliers = '' if outliers_in_obs is None else outliers_in_obs 
        anchors_name = f'{anchors_basis}{suffix_outliers}'
        self.verbosity(function_name='compute_mmd',
                dict_of_variables={'unbiaised':unbiaised,
                                    'approximation':approx,
                                    'shared_anchors':shared_anchors,
                                    'name':name},
                start=True,
                verbose = verbose)

        if approx == 'standard':
            m = self.compute_omega(sample='xy',quantization=False,outliers_in_obs=outliers_in_obs)
            K = self.compute_gram(outliers_in_obs=outliers_in_obs)
            if unbiaised:
                K.masked_fill_(torch.eye(K.shape[0],K.shape[0]).byte(), 0)
            mmd = dot(mv(K,m),m)**2 #je crois qu'il n'y a pas besoin de carré
        
        if approx == 'nystrom' and shared_anchors:
            anchors_basis=self.anchors_basis
            suffix_outliers = '' if outliers_in_obs is None else outliers_in_obs 
            anchors_name = f'{anchors_basis}{suffix_outliers}'
            m = self.compute_omega(sample='xy',quantization=False,outliers_in_obs=outliers_in_obs)
            Up = self.spev['xy']['anchors'][anchors_name]['ev']
            Lp_inv2 = diag(self.spev['xy']['anchors'][anchors_name]['sp']**-(1/2))
            Pm = self.compute_covariance_centering_matrix(sample='xy',landmarks=True,outliers_in_obs=outliers_in_obs)
            Kmn = self.compute_kmn(sample='xy',outliers_in_obs=outliers_in_obs)
            psi_m = mv(Lp_inv2,mv(Up.T,mv(Pm,mv(Kmn,m))))
            mmd = dot(psi_m,psi_m)**2
        
        if approx == 'nystrom' and not shared_anchors:
            # utile ? a mettre à jour
            mx = self.compute_omega(sample='x',quantization=False)
            my = self.compute_omega(sample='y',quantization=False)
            Upx = self.spev['x']['anchors'][anchors_basis]['ev']
            Upy = self.spev['y']['anchors'][anchors_basis]['ev']
            Lpx_inv2 = diag(self.spev['x']['anchors'][anchors_basis]['sp']**-(1/2))
            Lpy_inv2 = diag(self.spev['y']['anchors'][anchors_basis]['sp']**-(1/2))
            Lpy_inv = diag(self.spev['y']['anchors'][anchors_basis]['sp']**-1)
            Pmx = self.compute_covariance_centering_matrix(sample='x',landmarks=True)
            Pmy = self.compute_covariance_centering_matrix(sample='y',landmarks=True)
            Kmnx = self.compute_kmn(sample='x',outliers_in_obs=outliers_in_obs)
            Kmny = self.compute_kmn(sample='y',outliers_in_obs=outliers_in_obs)
            
            Km = self.compute_gram(sample='xy',landmarks=True)
            m1 = Kmnx.shape[0]
            m2 = Kmny.shape[0]
            Kmxmy = Km[:m1,m2:]

            psix_mx = mv(Lpx_inv2,mv(Upx.T,mv(Pmx,mv(Kmnx,mx))))
            psiy_my = mv(Lpy_inv2,mv(Upy.T,mv(Pmy,mv(Kmny,my))))
            Cpsiy_my = mv(Lpx_inv2,mv(Upx.T,mv(Pmx,mv(Kmxmy,\
                mv(Pmy,mv(Upy,mv(Lpy_inv,mv(Upy.T,mv(Pmy,mv(Kmny,my))))))))))
            mmd = dot(psix_mx,psix_mx)**2 + dot(psiy_my,psiy_my)**2 - 2*dot(psix_mx,Cpsiy_my)
        
        if approx == 'quantization':
            mq = self.compute_omega(sample='xy',quantization=True)
            Km = self.compute_gram(sample='xy',landmarks=True)
            mmd = dot(mv(Km,mq),mq) **2


        if name is None:
            name=f'{approx}'
            if approx == 'nystrom':
                name += 'shared' if shared_anchors else 'diff'
        
        self.dict_mmd[name] = mmd.item()
        
        self.verbosity(function_name='compute_mmd',
                dict_of_variables={'unbiaised':unbiaised,
                                    'approximation':approx,
                                    'shared_anchors':shared_anchors,
                                    'name':name},
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
