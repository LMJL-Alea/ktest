import torch
import pandas as pd
from torch import mv,diag,chain_matmul,dot,sum


"""

Ce fichier contient toutes les fonctions nécessaires au calcul des statistiques,
Les quantités pkm et upk sont des quantités génériques qui apparaissent dans beaucoup de calculs. 
Elles n'ont pas d'interprêtation facile, ces fonctions centrales permettent d'éviter les répétitions. 

Les fonctions initialize_kfdat et kfdat font simplement appel a plusieurs fonctions en une fois.

"""



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
    

def compute_pkm(self,outliers_in_obs=None):
    '''

    This function computes the term corresponding to the matrix-matrix-vector product PK omega
    of the KFDA statistic.
    
    See the description of the method compute_kfdat() for a brief description of the computation 
    of the KFDA statistic. 


    Parameters
    ----------
        self : tester,
        the model parameter attributes `approximation_cov`, `approximation_mmd` must be defined.
        if the nystrom method is used, the attribute `anchor_basis` should be defined and the anchors must have been computed. 

        outliers_in_obs (default = None) : None or string,
        nom de la colonne de l'attribut obs qui dit quelles cellules doivent être considérées comme des outliers et ignorées. 
                
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
        m1,m2,m = self.get_n1n2n(landmarks=True,outliers_in_obs=outliers_in_obs) # nombre de landmarks
        suffix_outliers = '' if outliers_in_obs is None else outliers_in_obs  # cellules à ignorer 
        
        # récupération dans l'attribut spev (sp : spectrum, ev : eigenvectors) des vecteurs propres et valeurs propres de l'opérateur de covariance utilisé pour pour calculer les 
        # ancres à partir des landmarks. 
        anchors_basis = self.anchors_basis # quel opérateur de covariance des landmarks est utilisé pour calculer les ancres 
        anchor_name = f'{anchors_basis}{suffix_outliers}'  # où sont rangés les objets dans spev
        Uz = self.spev['xy']['anchors'][anchor_name]['ev'] # vecteurs propres de l'operateur de covariance des 
        Lz = diag(self.spev['xy']['anchors'][anchor_name]['sp']**-1) # matrice diagonale des inverses des valeurs propres associées aux ancres. 

    # instantiation du vecteur de bi-centrage omega et de la matrice de centrage Pbi 
    omega = self.compute_omega(quantization=(mmd=='quantization'),outliers_in_obs=outliers_in_obs) # vecteur de 1/n1 et de -1/n2 
    Pbi = self.compute_covariance_centering_matrix(sample='xy',quantization=(cov=='quantization'),outliers_in_obs=outliers_in_obs) # matrice de centrage par block
    
    # instantiation de la matrice de passage des landmarks aux observations 
    # (ne sert pas que pour nystrom, aussi pour la statistique par quantification qu'on
    # a définie à un moment mais abandonnée pour ses mauvaises performances)
    if not (mmd == cov) or mmd == 'nystrom':
        Kzx = self.compute_kmn(sample='xy',outliers_in_obs=outliers_in_obs)
    

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
            Kx = self.compute_gram(outliers_in_obs=outliers_in_obs) # matrice de gram 
            pkm = mv(Pbi,mv(Kx,omega)) # le vecteur que renvoie cette fonction 
        # aucun avantage
        elif mmd == 'nystrom': 
            Pi = self.compute_covariance_centering_matrix(sample='xy',landmarks=True,outliers_in_obs=outliers_in_obs)
            pkm = 1/m * mv(Pbi,mv(Kzx.T,mv(Pi,mv(Uz,mv(Lz,mv(Uz.T,mv(Pi,mv(Kzx,omega))))))))
            # pkm = mv(Pbi,mv(Kzx.T,mv(Pi,mv(Uz,mv(Lz,mv(Uz.T,mv(Pi,mv(Kzx,omega))))))))
        # aucun avantage
        elif mmd == 'quantization':
            pkm = mv(Pbi,mv(Kzx.T,omega))

    if cov == 'nystrom1' and cov_anchors == 'shared':
        # aucun avantage
        if mmd in ['standard','nystrom']: # c'est exactement la même stat  
            Pi = self.compute_covariance_centering_matrix(sample='xy',landmarks=True,outliers_in_obs=outliers_in_obs)
            pkm = 1/m**2 * mv(Pbi,mv(Kzx.T,mv(Pi,mv(Uz,mv(Lz,mv(Uz.T,mv(Pi,mv(Kzx,omega))))))))
            # pkm = mv(Pbi,mv(Kzx.T,mv(Pi,mv(Uz,mv(Lz,mv(Uz.T,mv(Pi,mv(Kzx,omega))))))))
        # aucun avantage
        elif mmd == 'quantization':
            Kz = self.compute_gram(landmarks=True,outliers_in_obs=outliers_in_obs)
            pkm = 1/m**2 * mv(Pbi,mv(Kzx.T,mv(Uz,mv(Lz,mv(Uz.T,mv(Kz,omega))))))
            # pkm = mv(Pbi,mv(Kzx.T,mv(Uz,mv(Lz,mv(Uz.T,mv(Kz,omega))))))
    
    if cov == 'nystrom2' and cov_anchors == 'shared':
        # aucun avantage
        Lz12 = diag(self.spev['xy']['anchors'][anchor_name]['sp']**-(1/2))
        if mmd in ['standard','nystrom']: # c'est exactement la même stat  
            Pi = self.compute_covariance_centering_matrix(sample='xy',landmarks=True,outliers_in_obs=outliers_in_obs)
            pkm = 1/m**3 * mv(Lz12,mv(Uz.T,mv(Pi,mv(Kzx,mv(Pbi,mv(Kzx.T,mv(Pi,mv(Uz,mv(Lz,mv(Uz.T,mv(Pi,mv(Kzx,omega))))))))))))
            # pkm = mv(Lz12,mv(Uz.T,mv(Pi,mv(Kzx,mv(Pbi,mv(Kzx.T,mv(Pi,mv(Uz,mv(Lz,mv(Uz.T,mv(Pi,mv(Kzx,omega))))))))))))
        # aucun avantage
        elif mmd == 'quantization': # pas à jour
            # il pourrait y avoir la dichotomie anchres centrees ou non ici. 
            Kz = self.compute_gram(landmarks=True,outliers_in_obs=outliers_in_obs)
            pkm = 1/m**3 * mv(Lz12,mv(Uz.T,mv(Kzx,mv(Pbi,mv(Kzx.T,mv(Uz,mv(Lz,mv(Uz.T,mv(Kz,omega)))))))))
            # pkm = mv(Lz12,mv(Uz.T,mv(Kzx,mv(Pbi,mv(Kzx.T,mv(Uz,mv(Lz,mv(Uz.T,mv(Kz,omega)))))))))
    
    if cov == 'nystrom3' and cov_anchors == 'shared':
        Lz12 = diag(self.spev['xy']['anchors'][anchor_name]['sp']**-(1/2))
        # print("statistics pkm: L-1 nan ",(torch.isnan(torch.diag(Lz12))))
        Pi = self.compute_covariance_centering_matrix(sample='xy',landmarks=True,outliers_in_obs=outliers_in_obs)
        # cas nystrom 
        if mmd in ['standard','nystrom']: # c'est exactement la même stat  
            pkm = 1/m * mv(Lz12,mv(Uz.T,mv(Pi,mv(Kzx,omega)))) # le vecteur d'intérêt renvoyé par la fonction 
            # pkm = mv(Lz12,mv(Uz.T,mv(Pi,mv(Kzx,omega))))
            # print(f'in compute pkm: \n\t\
            #    Lz12{Lz12}\n Uz{Uz}\n Kzx{Kzx}')

        elif mmd == 'quantization': # pas à jour 
            # il faut ajouter Pi ici . 
            Kz = self.compute_gram(landmarks=True,outliers_in_obs=outliers_in_obs)
            pkm = 1/m**2 * mv(Lz12,mv(Uz.T,mv(Pi,mv(Kzx,mv(Pbi,mv(Kzx.T,mv(Pi,mv(Uz,mv(Lz,mv(Uz.T,mv(Kz,omega)))))))))))
            # pkm = mv(Lz12,mv(Uz.T,mv(Kzx,mv(Pbi,mv(Kzx.T,mv(Uz,mv(Lz,mv(Uz.T,mv(Kz,omega)))))))))
    
    # exploration jamais terminée 
    if cov == 'nystrom1' and cov_anchors == 'separated':
        # utile ?  A mettre à jour
        if mmd == 'standard':
            x,y = self.get_xy(outlier_in_obs=outliers_in_obs)
            z1,z2 = self.get_xy(landmarks=True,outlier_in_obs=outliers_in_obs)
            Kz1x = self.kerne(z1,x)
            Kz1y = self.kerne(z1,y)
            Kz2x = self.kerne(z2,x)
            Kz2y = self.kerne(z2,y)
            Uz1 = self.spev['x']['anchors'][anchors_basis]['ev']
            Lz1 = diag(self.spev['x']['anchors'][anchors_basis]['sp']**-1)
            Uz2 = self.spev['y']['anchors'][anchors_basis]['ev']
            Lz2 = diag(self.spev['y']['anchors'][anchors_basis]['sp']**-1)
            omega1 = self.compute_omega(sample='x',quantization=False)
            omega2 = self.compute_omega(sample='y',quantization=False)
            Pn1 = self.compute_covariance_centering_matrix(sample='x')
            Pn2 = self.compute_covariance_centering_matrix(sample='y')
            haut = mv(Lz1,mv(Uz1,mv(Kz1x,mv(Pn1,mv(Kz1x,mv(Uz1,mv(Lz1,mv(Uz1.T,mv(Kz1y,omega2) -mv(Kz1x,omega1)))))))))
            bas = mv(Lz2,mv(Uz2,mv(Kz2y,mv(Pn2,mv(Kz2y,mv(Uz2,mv(Lz2,mv(Uz2.T,mv(Kz2y,omega2) -mv(Kz2x,omega1)))))))))
            

    # aucun avantage 
    if cov == 'quantization': # pas à jour 
        A = self.compute_quantization_weights(power=1/2,sample='xy')
        if mmd == 'standard':
            pkm = mv(Pbi,mv(A,mv(Kzx,omega)))

        elif mmd == 'nystrom':
            Pi = self.compute_covariance_centering_matrix(sample='xy',landmarks=True,outliers_in_obs=outliers_in_obs)
            Kz = self.compute_gram(landmarks=True,outliers_in_obs=outliers_in_obs)
            pkm = 1/m * mv(Pbi,mv(A,mv(Kz,mv(Uz,mv(Lz,mv(Uz.T,mv(Pi,mv(Kzx,omega))))))))

        elif mmd == 'quantization':
            Kz = self.compute_gram(landmarks=True,outliers_in_obs=outliers_in_obs)
            pkm = mv(Pbi,mv(A,mv(Kz,omega)))
    return(pkm) #

def compute_upk(self,t,outliers_in_obs=None):
    """
    epk is an alias for the product ePK that appears when projecting the data on the discriminant axis. 
    This functions computes the corresponding block with respect to the model parameters. 
    
    warning: some work remains to be done to :
        - normalize the vecters with respect to r as in pkm 
        - separate the different nystrom approaches 
    """
    
    cov = self.approximation_cov
    anchors_basis = self.anchors_basis
    suffix_nystrom = anchors_basis if 'nystrom' in cov else ''
    suffix_outliers = outliers_in_obs if outliers_in_obs is not None else ''
    sp = self.spev['xy'][f'{cov}{suffix_nystrom}{suffix_outliers}']['sp']
    ev = self.spev['xy'][f'{cov}{suffix_nystrom}{suffix_outliers}']['ev']
        

    Pbi = self.compute_covariance_centering_matrix(sample='xy',quantization=(cov=='quantization'),outliers_in_obs=outliers_in_obs)
      
    if not (cov == 'standard'):
        Kzx = self.compute_kmn(sample='xy',outliers_in_obs=outliers_in_obs)
    
    if cov == 'standard':
        Kx = self.compute_gram(outliers_in_obs=outliers_in_obs)
        epk = chain_matmul(ev.T[:t],Pbi,Kx).T
        # epk = torch.linalg.multi_dot([ev.T[:t],Pbi,Kx]).T
    if cov == 'nystrom3':
        anchor_name = f'{anchors_basis}{suffix_outliers}'
        m1,m2,m = self.get_n1n2n(landmarks=True,outliers_in_obs=outliers_in_obs)
        Uz = self.spev['xy']['anchors'][anchor_name]['ev']
        Lz = diag(self.spev['xy']['anchors'][anchor_name]['sp']**-1)
        Lz12 = diag(self.spev['xy']['anchors'][anchor_name]['sp']**-(1/2))
        # print(f'm:{m} evt:{ev.T[:t].shape} Lz12{Lz12.shape} Uz{Uz.shape} Kzx{Kzx.shape}')
        
        epk = 1/m**(1/2) * chain_matmul(ev.T[:t],Lz12,Uz.T,Kzx).T

    elif 'nystrom' in cov:
        anchor_name = f'{anchors_basis}{suffix_outliers}'
        Uz = self.spev['xy']['anchors'][anchor_name]['ev']
        Lz = diag(self.spev['xy']['anchors'][anchor_name]['sp']**-1)
        r = self.r
        m1,m2,m = self.get_n1n2n(landmarks=True,outliers_in_obs=outliers_in_obs)

        # print(f'r:{r} evt:{ev.T[:t].shape} Pbi{Pbi.shape} Kzx{Kzx.shape} Uz{Uz.shape} Lz{Lz.shape}  ')
        epk = 1/m*chain_matmul(ev.T[:t],Pbi,Kzx.T,Uz,Lz,Uz.T,Kzx).T
        # epk = 1/r*torch.linalg.multi_dot([ev.T[:t],Pbi,Kzx.T,Uz,Lz,Uz.T,Kzx]).T
    if cov == 'quantization':
        A_12 = self.compute_quantization_weights(power=1/2,sample='xy')
        epk = chain_matmul(ev.T[:t],A_12,Pbi,Kzx).T
        # epk = torch.linalg.multi_dot([ev.T[:t],A_12,Pbi,Kzx]).T
    
    return(epk)
#
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

def compute_kfdat_with_different_order(self,order='between'):
    '''
    Computes a truncated kfda statistic which is defined as the original truncated kfda statistic but 
    the eigenvectors and eigenvalues of the within covariance operator are not ordered by decreasing eigenvalues. 
    
    Parameters
    ----------
        order : str, in 'between', ...? 
        specify the rule to order the eigenvectors
        so far there is only one choice but I want to add a second one which would 
        be a compromize between the reconstruction of (\mu_2 - \mu_1) and the 
        reconstruction of the within covariance operator. 
    
    Returns
    -------
        The attribute `df_kfdat` is updated with new columns corresponding to the new kfda statistic. 
        Each new column is a column of the attribute `df_kfdat_contributions` with a '_between' at the end. 
    '''
    if order == 'between':
        projection_error,ordered_truncations = self.get_ordered_spectrum_wrt_between_covariance_projection_error()
        
        kfda_contrib = self.df_kfdat_contributions
        kfda_between = kfda_contrib.T[ordered_truncations.tolist()].T.cumsum()
        kfda_between.index = range(1,len(ordered_truncations)+1)
        for c in kfda_contrib.columns:
            self.df_kfdat[f'{c}_between'] = kfda_between[c]


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

def kfdat(self,t=None,name=None,verbose=0,outliers_in_obs=None):
    """"
    This functions computes the truncated kfda statistic from scratch, if needed, it computes landmarks and 
    anchors for the nystrom approach and diagonalize the matrix of interest for the computation fo the statistic. 
    It also computes the asymptotic pvalues for each truncation and determine automatically a truncation of interest. 

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

    """

    # récupération des paramètres du modèle dans les attributs 
    cov = self.approximation_cov # approximation de l'opérateur de covariance. 
    mmd = self.approximation_mmd # approximation du vecteur mu2 - mu1 


    # définition du nom de la colonne dans laquelle seront stockées les valeurs de la stat 
    # dans l'attribut df_kfdat (une DataFrame Pandas)   
    # je devrais définir une fonction spécifique pour ces lignes qui apparaissent dans plusieurs fonctions. 
    name = name if name is not None else outliers_in_obs if outliers_in_obs is not None else f'{cov}{mmd}' 
    
    # inutile de calculer la stat si elle est déjà calculée (le name doit la caractériser)
    if name in self.df_kfdat :
        if verbose : 
            print(f'kfdat {name} already computed')

    else:

        self.initialize_kfdat(sample='xy',verbose=verbose,outliers_in_obs=outliers_in_obs) # landmarks, ancres et diagonalisation           
        self.compute_kfdat(t=t,name=name,verbose=verbose,outliers_in_obs=outliers_in_obs) # caclul de la stat 
        self.select_trunc() # selection automatique de la troncature 
        self.compute_pval() # calcul des troncatures asymptotiques 
        self.kfda_stat = self.df_kfdat[name][self.t] # stockage de la valeur de la stat pour la troncature selectionnées 
    
    # les valeurs de la statistique ont été stockées dans une colonne de la dataframe df_kfdat. 
    # pour ne pas avoir à chercher le nom de cette colonne difficilement, il est renvoyé ici
    return(name)

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
#
def mmd(self,shared_anchors=True,name=None,unbiaised=False,verbose=0):
    """
    appelle la fonction initialize mmd puis la fonction compute_mmd si le mmd n'a pas deja ete calcule. 
    """
    approx = self.approximation_mmd
    
    if name is None:
        name=f'{approx}'
        if approx == 'nystrom':
            name += 'shared' if shared_anchors else 'diff'
    
    if name in self.dict_mmd :
        if verbose : 
            print(f'mmd {name} already computed')
    else:
        self.initialize_mmd(shared_anchors=shared_anchors,verbose=verbose)
        self.compute_mmd(shared_anchors=shared_anchors,
                        name=name,unbiaised=unbiaised,verbose=0)

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

def kpca(self,t=None,approximation_cov='standard',sample='xy',name=None,verbose=0):
    
    cov = approximation_cov
    name = name if name is not None else f'{cov}{sample}' 
    if name in self.df_proj_kpca :
        if verbose : 
            print(f'kfdat {name} already computed')
    else:
        self.initialize_kfda(approximation_cov=cov,sample=sample,verbose=verbose)            
        self.compute_proj_kpca(t=t,approximation_cov=cov,sample=sample,name=name,verbose=verbose)


