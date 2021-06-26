from typing_extensions import Literal
from typing import Optional,Callable,Union,List

import numpy as np
from numpy.lib.function_base import kaiser
import pandas as pd
import torch
import os
from time import time

import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.linalg import svd
from numpy.random import permutation
from sklearn.model_selection import train_test_split

from kernels import gauss_kernel_mediane
from apt.eigen_wrapper import eigsy
import apt.kmeans # For kmeans
from kmeans_pytorch import kmeans




# Choix à faire ( trouver les bonnes pratiques )

# est ce que je range la gram ou la calcule à chaque besoin ? 
# Je calcule la gram a chaque fois mais la diagonalise une fois par setting 
# nouvelle question : est ce que je garde en mémoire toute les matrices diag ou pas ( si je fais des tests avec Nystrom)
# est ce que je crée deux variables, une pour la gram et une pour la gram nystrom ?
# reponse : au moment de la calculer, ce uqi est calculé est selectionné automatiquement 

# les assignations du nystrom anchors pourront servir à calculer une somme de carrés résiduels par ex.

# voir comment ils gèrent les plot dans scanpy

# initialiser le mask aussi au moment de tracer des figures
# ranger les dataframes de proj dans des dict pour en avoir plusieurs en mémoire en même temps. 
# faire les plot par defaut sur toutes les proj en mémoire 


# tracer les spectres 
# tracer l'evolution des corrélations par rapport aux gènes ordonnés

# tout rendre verbose 

# ecrire les docstring de chaque fonction 

# faire des dict pour sp et ev ? pour : quand on veut tracer les spectres. 

# faire des fonctions compute kfdat et proj qui font appelle au scheduleur "compute and load" (a renommer) 
# le name de corr doit spécifier en plus du nom la projection auxquel il fiat référence (acp ou fda)

# enregistrer le coef de la projection kfda de chaque axe de l'acp 
# faire une fonction __print__ (?)
 
# calculer les p values par direction 
# liste de gène 
# lsmeans least square means (dans le cours de Franck pour ANOVA deux facteurs p32 test moyennes ajustées )
# mmd et test par permutation 

# plot proj :on a dfx et dfy pour tracer le result en fct d'un axe de la pca 
# on peut aussi vouloir tracer en fonction de ll'expression d'un gène 

# ajouter un get_xy pour gérer les masks 
# a partir d'un moment, j'ai supposé que test data n'était plus d'actualité et cessé de l'utiliser. A vérifier et supprimer si oui 
# def des fonction type get pour les opérations d'initialisation redondantes
# acces facile aux noms des dict de dataframes. 
# faire en sorte de pouvoir calculer corr kfda et kpca 

# mettre une limite globale a 100 pour les tmax des projections (éviter d'enregistrer des structures de données énormes)
# mieux gérer la projection de gènes 

class Tester:
    """
    Tester is a class that performs kernels tests such that MMD and the test based on Kernel Fisher Discriminant Analysis. 
    It also provides a range of visualisations based on the discrimination between two groups.  
    """
    def __init__(self,
        x:Union[np.array,torch.tensor]=None,
        y:Union[np.array,torch.tensor]=None,
        kernel:Callable[[torch.Tensor,torch.Tensor],torch.Tensor]=None, 
        x_index:List = None,
        y_index:List = None,
        variables:List = None):
        """\
        Parameters
        ----------
        x,y:                torch.Tensor or numpy.array of sizes n1 x p and n2 x p 
        kernel:             kernel function to apply on (x,y)
        x_index,y_index:    pd.Index or list of index to identify observations from x and y
        variables:          pd.Index or list of index to identify variables from x and y
        
        Returns
        -------
        :obj:`Tester`
        """
        self.has_data = False   
        self.has_landmarks = False
        self.quantization_with_landmarks_possible = False
        self.has_anchors = False
        
        # attributs initialisés 
        self.df_kfdat = pd.DataFrame()
        self.df_proj_kfda = {}
        self.df_proj_kpca = {}
        self.corr = {}     
        self.dict_mmd = {}
        self.spev = {'x':{},'y':{}} # dict containing every result of diagonalization
        # les vecteurs propres sortant de eigsy sont rangés en colonnes

        # for verbosity 
        self.start_times = {}

        if x is not None and y is not None:
            self.init_data(x=x,y=y,kernel=kernel,x_index=x_index,y_index=y_index,variables=variables)
            
    def init_data(self,x,y,kernel=None, x_index=None, y_index=None,variables=None):
        # Tester works with torch tensor objects 
        self.x = torch.from_numpy(x).double() if (isinstance(x, np.ndarray)) else x
        self.y = torch.from_numpy(y).double() if (isinstance(y, np.ndarray)) else y

        self.n1_initial = x.shape[0]
        self.n2_initial = y.shape[0]
        
        self.n1 = x.shape[0]
        self.n2 = y.shape[0]
    
        # generates range index if no index
        self.x_index=pd.Index(range(1,self.n1+1)) if x_index is None else pd.Index(x_index) if isinstance(x_index,list) else x_index 
        self.y_index=pd.Index(range(self.n1,self.n1+self.n2)) if y_index is None else pd.Index(y_index) if isinstance(y_index,list) else y_index
        self.index = self.x_index.append(self.y_index) 
        self.variables = range(x.shape[1]) if variables is None else variables

        self.xmask = self.x_index.isin(self.x_index)
        self.ymask = self.y_index.isin(self.y_index)
        self.imask = self.index.isin(self.index)
        self.ignored_obs = None
        if kernel is None:
            self.kernel,self.mediane = gauss_kernel_mediane(x,y,return_mediane=True)        
        else:
            self.kernel = kernel
            
        if self.df_kfdat.empty:
            self.df_kfdat = pd.DataFrame(index= list(range(1,self.n1+self.n2)))
        self.has_data = True        

    def __str__(self):
        if self.has_data:
            s = f"View of Tester object with n1 = {self.n1}, n2 = {self.n2}\n"
            
        else: 
            s = "View of Tester object with no data"  

        s += "kfdat : "
        if not self.df_kfdat.empty:
            for c in self.df_kfdat.columns:
                s += f"'{c}' "
        s += '\n'

        s += 'proj kfdat : '
        if  len(self.df_proj_kfda)>0:
            for c in self.df_proj_kfda.keys():
                s += f"'{c}'"
        s += '\n'    

        s += 'proj kpcaw : '
        if  len(self.df_proj_kpca)>0:
            for c in self.df_proj_kpca.keys():
                s += f"'{c}'"
        s += '\n'    

        s += 'correlations : '
        if  len(self.corr)>0:
            for c in self.corr.keys():
                s += f"'{c}'"
        s += '\n'   


        s += 'mmd : '
        if  len(self.corr)>0:
            for c in self.corr.keys():
                s += f"'{c}'"
        s += '\n'    


        return(s) 

    def __repr__(self):
        return(self.__str__())
 
    def verbosity(self,function_name,dict_of_variables=None,start=True,verbose=0):
        if verbose >0:
            end = ' ' if verbose == 1 else '\n'
            if start:  # pour verbose ==1 on start juste le chrono mais écris rien     
                self.start_times[function_name] = time()
                if verbose >1 : 
                    print(f"Starting {function_name} ...",end= end)
                    if dict_of_variables is not None:
                        for k,v in dict_of_variables.items():
                            if verbose ==2:
                                print(f'\t {k}:{v}', end = '') 
                            else:
                                print(f'\t {k}:{v}')
                    
            else: 
                start_time = self.start_times[function_name]
                print(f"Done {function_name} in  {time() - start_time:.2f}")

    def compute_nystrom_landmarks(self,nlandmarks=None,landmarks_method='random',verbose=0):
        # anciennement compute_nystrom_anchors(self,nanchors,nystrom_method='kmeans',split_data=False,test_size=.8,on_other_data=False,x_other=None,y_other=None,verbose=0): # max_iter=1000, (pour kmeans de François)
        
        """
        We distinguish the nystrom landmarks and nystrom anchors.
        The nystrom landmarks are the points from the observation space that will be used to compute the nystrom anchors. 
        The nystrom anchors are the points in the RKHS computed from the nystrom landmarks on which we will apply the nystrom method. 

        Parameters
        ----------
        nlandmarks:    (1/10 * n if None)  number of landmarks in total (proportionnaly sampled according to the data)
        landmarks_method: 'kmeans' or 'random' (in the future : 'kmeans ++, greedy...)
        """

        self.verbosity(function_name='compute_nystrom_landmarks',
                       dict_of_variables={'nlandmarks':nlandmarks,'landmarks_method':landmarks_method},
                       start=True,
                       verbose = verbose)
            
            
        if nlandmarks is None:
            nlandmarks = 1/10 * (self.n1 + self.n2)

        xratio,yratio = self.n1/(self.n1 + self.n2), self.n2/(self.n1 + self.n2)
        self.nlandmarks = nlandmarks
        self.nxlandmarks=np.int(np.floor(xratio * nlandmarks)) 
        self.nylandmarks=np.int(np.floor(yratio * nlandmarks))

        xmask_ny = self.xmask
        ymask_ny = self.ymask  

        if landmarks_method == 'kmeans':
            # self.xanchors,self.xassignations = apt.kmeans.spherical_kmeans(self.x[self.xmask,:], nxanchors, max_iter)
            # self.yanchors,self.yassignations = apt.kmeans.spherical_kmeans(self.y[self.ymask,:], nyanchors, max_iter)
            self.xassignations,self.xlandmarks = kmeans(X=self.x[xmask_ny,:], num_clusters=self.nxlandmarks, distance='euclidean', tqdm_flag=False) #cuda:0')
            self.yassignations,self.ylandmarks = kmeans(X=self.y[ymask_ny,:], num_clusters=self.nylandmarks, distance='euclidean', tqdm_flag=False) #cuda:0')
            self.xlandmarks = self.xlandmarks.double()
            self.ylandmarks = self.ylandmarks.double()
            self.quantization_with_landmarks_possible = True
            

        elif landmarks_method == 'random':
            self.xlandmarks = self.x[xmask_ny,:][np.random.choice(self.x[xmask_ny,:].shape[0], size=self.nxlandmarks, replace=False)]
            self.ylandmarks = self.y[ymask_ny,:][np.random.choice(self.y[ymask_ny,:].shape[0], size=self.nylandmarks, replace=False)]
            
            # Necessaire pour remettre a false au cas ou on a déjà utilisé 'kmeans' avant 
            self.quantization_with_landmarks_possible = False

        self.has_landmarks= True

        self.verbosity(function_name='compute_nystrom_landmarks',
                       dict_of_variables={'nlandmarks':nlandmarks,'landmarks_method':landmarks_method},
                       start=False,
                       verbose = verbose)

    def compute_nystrom_anchors(self,nanchors=None,verbose=0):
        """
        Determines the nystrom anchors using 
        Stores the results as a list of eigenvalues and the 
        
        Parameters
        ----------
        nanchors:      <= nlandmarks (= by default). Number of anchors to determine in total (proportionnaly according to the data)
        """
        
        
        self.verbosity(function_name='compute_nystrom_anchors',
                        dict_of_variables={'nanchors':nanchors},
                        start=True,
                        verbose = verbose)



        # bien faire l'arbre des possibles ici 
        if nanchors is None:
            self.nanchors = self.nlandmarks
            # self.nxanchors, self.nyanchors = self.nxlandmarks, self.nylandmarks

        else:
            self.nanchors = nanchors
            assert(self.nanchors <= self.nlandmarks)

        Km = self.compute_gram_matrix(landmarks=True)

        sp_anchors,ev_anchors = eigsy(Km)
                
        order = sp_anchors.argsort()[::-1]
        sp_anchors = torch.tensor(sp_anchors[order][:self.nanchors], dtype=torch.float64)
        ev_anchors = torch.tensor(ev_anchors[:,order][:,:self.nanchors],dtype=torch.float64) 
        self.spev['anchors'] = {'sp':sp_anchors,'ev':ev_anchors}
        self.has_anchors = True

        self.verbosity(function_name='compute_nystrom_anchors',
                        dict_of_variables={'nanchors':nanchors},
                        start=False,
                        verbose = verbose)

    def compute_nystrom_anchors_per_sample(self,nanchors=None,covariance='x',verbose=0):
        """
        Determines the nystrom anchors using 
        Stores the results as a list of eigenvalues and the 
        
        Parameters
        ----------
        nanchors:      <= nlandmarks (= by default). Number of anchors to determine in total (proportionnaly according to the data)
        """
        
        
        self.verbosity(function_name='compute_nystrom_anchors_per_sample',
                        dict_of_variables={'nanchors':nanchors,
                                           'covariance':covariance},
                        start=True,
                        verbose = verbose)


        kernel = self.kernel
        x = covariance =='x'
        # bien faire l'arbre des possibles ici 
        if x:
            self.nxanchors = self.nxlandmarks if nanchors is None else nanchors
            assert(self.nxanchors <= self.nxlandmarks)
            landmarks = self.xlandmarks
        else:
            self.nyanchors = self.nylandmarks if nanchors is None else nanchors
            assert(self.nyanchors <= self.nylandmarks)
            landmarks = self.ylandmarks
            # self.nxanchors, self.nyanchors = self.nxlandmarks, self.nylandmarks


        Km = kernel(landmarks,landmarks)

        sp_anchors,ev_anchors = eigsy(Km)
                
        order = sp_anchors.argsort()[::-1]
        sp_anchors = torch.tensor(sp_anchors[order][:self.nanchors], dtype=torch.float64)
        ev_anchors = torch.tensor(ev_anchors[:,order][:,:self.nanchors],dtype=torch.float64) 
        self.spev[covariance]['anchors'] = {'sp':sp_anchors,'ev':ev_anchors}

        self.verbosity(function_name='compute_nystrom_anchors_per_sample',
                        dict_of_variables={'nanchors':nanchors},
                        start=False,
                        verbose = verbose)


    def compute_gram_matrix(self,landmarks=False): 
        """
        Computes Gram matrix, on anchors if nystrom is True, else on data. 
        This function is called everytime the Gram matrix is needed but I could had an option to keep it in memory in case of a kernel function 
        that makes it difficult to compute

        Returns
        -------
        torch.Tensor of size (nxanchors+nyanchors)**2 if nystrom else (n1+n2)**2
        """


        x,y = (self.xlandmarks,self.ylandmarks) if landmarks else \
              (self.x[self.xmask,:],self.y[self.ymask,:])

        kernel = self.kernel
                         
        kxx = kernel(x, x)
        kyy = kernel(y, y)
        kxy = kernel(x, y)

        return(torch.cat((torch.cat((kxx, kxy), dim=1),
                            torch.cat((kxy.t(), kyy), dim=1)), dim=0))

    def compute_m(self,quantization=False):
        n1,n2 = (self.n1,self.n2)
        if quantization:
            return(torch.cat((-1/n1*torch.bincount(self.xassignations),1/n2*torch.bincount(self.yassignations))).double())
        else:
            m_mu1    = -1/n1 * torch.ones(n1, dtype=torch.float64) # , device=device)
            m_mu2    = 1/n2 * torch.ones(n2, dtype=torch.float64) # , device=device) 
            return(torch.cat((m_mu1, m_mu2), dim=0)) #.to(device)
        
    def compute_kmn(self):
        """
        Computes an (nxanchors+nyanchors)x(n1+n2) conversion gram matrix
        """
        assert(self.has_landmarks)
        x,y = (self.x[self.xmask,:],self.y[self.ymask,:])
        z1,z2 = self.xlandmarks,self.ylandmarks
        kernel = self.kernel
        

        kz1x = kernel(z1,x)
        kz2x = kernel(z2,x)
        kz1y = kernel(z1,y)
        kz2y = kernel(z2,y)
        
        return(torch.cat((torch.cat((kz1x, kz1y), dim=1),
                            torch.cat((kz2x, kz2y), dim=1)), dim=0))
    
    def compute_centering_matrix(self,covariance='x',quantization=False):
                
        

        if quantization:
            assignations = self.xassignations if x else self.yassignations
            ngroupe =  self.n1 if x else self.n2
            A = 1/ngroupe*torch.diag(torch.bincount(assignations)).double()
            pn = idn - torch.matmul(A,onen)

        else:
            pn = idn - 1/n * onen
        return(pn)

    def compute_centering_matrix(self,quantization=False,sample='xy'):
        """
        Computes the bicentering Gram matrix Pn. 
        Let I1,I2 the identity matrix of size n1 and n2 (or nxanchors and nyanchors if nystrom).
            J1,J2 the squared matrix full of one of size n1 and n2 (or nxanchors and nyanchors if nystrom).
            012, 021 the matrix full of zeros of size n1 x n2 and n2 x n1 (or nxanchors x nyanchors and nyanchors x nxanchors if nystrom)
        
        Pn = [I1 - 1/n1 J1 ,    012     ]
             [     021     ,I2 - 1/n2 J2]

        Returns
        sample in 'x','y','xy'
        -------
        torch.Tensor of size (nxanchors+nyanchors)**2 if quantization else (n1+n2)**2 
        """

        
        if 'x' in sample:
            n1 = self.nxlandmarks if quantization else self.n1 
            idn1 = torch.eye(n1, dtype=torch.float64)

        if 'y' in sample:
            n2 = self.nylandmarks if quantization else self.n2
            idn2 = torch.eye(n2, dtype=torch.float64)
       

        x = covariance =='x'
        n1,n2 = (self.nxlandmarks,self.nylandmarks)  if quantization else (self.n1,self.n2) 
        n = n1 if x else n2
        idn = torch.eye(n, dtype=torch.float64)
        onen = torch.ones(n, n, dtype=torch.float64)


        n1,n2 = (self.nxlandmarks,self.nylandmarks)  if quantization else (self.n1,self.n2) 
        
        idn1 = torch.eye(n1, dtype=torch.float64)
        idn2 = torch.eye(n2, dtype=torch.float64)

        onen1 = torch.ones(n1, n1, dtype=torch.float64)
        onen2 = torch.ones(n2, n2, dtype=torch.float64)

        if quantization: 
            A1 = torch.diag(1/self.n1*torch.bincount(self.xassignations)).double()
            A2 = torch.diag(1/self.n2*torch.bincount(self.yassignations)).double()
            pn1 = np.sqrt(self.n1/(self.n1+self.n2))*(idn1 - torch.matmul(A1,onen1))
            pn2 = np.sqrt(self.n2/(self.n1+self.n2))*(idn2 - torch.matmul(A2,onen2))
        else:
            pn1 = idn1 - 1/n1 * onen1
            pn2 = idn2 - 1/n2 * onen2

        z12 = torch.zeros(n1, n2, dtype=torch.float64)
        z21 = torch.zeros(n2, n1, dtype=torch.float64)

        return(torch.cat((torch.cat((pn1, z12), dim=1), torch.cat(
            (z21, pn2), dim=1)), dim=0))  # bloc diagonal

    def compute_centered_gram(self,approximation='full',covariance='x',verbose=0):

        self.verbosity(function_name='compute_centered_gram',
                dict_of_variables={'approximation':approximation,
                                'covariance':covariance},
                start=True,
                verbose = verbose)    
        
        x = covariance =='x'
        quantization = approximation == 'quantization'
        n1,n2 =  (self.n1,self.n2)
        n1,n2 = (self.nxlandmarks,self.nylandmarks)  if quantization else (self.n1,self.n2) 
        n = n1 if x else n2
        kernel = self.kernel
        sample = self.x if x else self.y

        # nystrom et standard : c'est la même (pbi par blocs n1 n2) 
        P = self.compute_centering_matrix(covariance=covariance,quantization=quantization).double()
        
            
        if approximation == 'quantization':
            if self.quantization_with_landmarks_possible:
                Z = self.xlandmarks if x else self.ylandmarks 
                K = kernel(Z,Z)
                assignations = self.xassignations if covariance =='x' else self.yassignations
                ngroupe =  self.n1 if x else self.n2
                A = 1/ngroupe*torch.diag(torch.bincount(assignations)).double()
                K = torch.chain_matmul(A**(1/2),P, K,P,A**(1/2))
            else:
                print("quantization impossible, you need to call 'compute_nystrom_landmarks' with landmarks_method='kmeans'")

        elif approximation == 'nystrom':
            # version brute mais a terme utiliser la svd ?? 
            if self.has_landmarks and self.has_anchors:
                Z = self.xlandmarks if x else self.ylandmarks 
                Kmn = kernel(Z,sample)
                Lp_inv = torch.diag(self.spev[covariance]['anchors']['sp']**(-1))
                Up = self.spev[covariance]['anchors']['ev']
                K = 1/n * torch.chain_matmul(P,Kmn.T,Up,Lp_inv,Up.T,Kmn,P)

                # print(f"rectangle pour svd ")
                # Lp_inv = torch.diag(self.spev['anchors']['sp']**(-1/2))
                # B = torch.chain_matmul(Lp_inv,Up.T,Kmn,Pbi)
                # print(f"B {B.shape} {B[:3,:3]}")
                # print(f"1/sqrt(n) B {B.shape} {1/np.sqrt(n1+n2)*B[:3,:3]}")

            else:
                print("nystrom impossible, you need compute landmarks and/or anchors")
        
        
        elif approximation == 'full':
            K = kernel(sample,sample)
            K = 1/(n) * torch.chain_matmul(P,K,P)


        self.verbosity(function_name='compute_centered_gram',
                dict_of_variables={'approximation':approximation},
                start=False,
                verbose = verbose)    

        return K

    def compute_bicentered_gram(self,approximation='full',verbose=0):
        """ 
        Computes the bicentered Gram matrix which shares its spectrom with the 
        within covariance operator. 
        Returns the matrix because it is only used in diagonalize_bicentered_gram
        I separated this function because I want to assess the computing time and 
        simplify the code 

        approximation in 'full','nystrom','quantization'
        # contre productif de choisir 'nystrom' car cela est aussi cher que standard pour une qualité d'approx de la matrice moins grande. 
        # pour utiliser nystrom, mieux vaux calculer la SVD de BB^T pas encore fait. 
        """



        self.verbosity(function_name='compute_bicentered_gram',
                dict_of_variables={'approximation':approximation},
                start=True,
                verbose = verbose)    

        n1,n2 =  (self.n1,self.n2)


        # nystrom et standard : c'est la même (pbi par blocs n1 n2) 
        quantization = approximation == 'quantization'
        Pbi = self.compute_bicentering_matrix(quantization=quantization).double()
        
        if approximation == 'quantization':
            if self.quantization_with_landmarks_possible:
                K = self.compute_gram_matrix(landmarks=True) # on utilise les landmarks uniquement dans le cas quantization
                A = torch.diag(torch.cat((1/n1*torch.bincount(self.xassignations),1/n2*torch.bincount(self.yassignations)))).double()
                Kw = torch.chain_matmul(A**(1/2),Pbi, K,Pbi,A**(1/2))
            else:
                print("quantization impossible, you need to call 'compute_nystrom_landmarks' with landmarks_method='kmeans'")

        elif approximation == 'nystrom':
            # version brute mais a terme utiliser la svd ?? 
            if self.has_landmarks and self.has_anchors:
                Kmn = self.compute_kmn()
                Lp_inv = torch.diag(self.spev['anchors']['sp']**(-1))
                Up = self.spev['anchors']['ev']
                Kw = 1/(n1+n2) * torch.chain_matmul(Pbi,Kmn.T,Up,Lp_inv,Up.T,Kmn,Pbi)

                # print(f"rectangle pour svd ")
                # Lp_inv = torch.diag(self.spev['anchors']['sp']**(-1/2))
                # B = torch.chain_matmul(Lp_inv,Up.T,Kmn,Pbi)
                # print(f"B {B.shape} {B[:3,:3]}")
                # print(f"1/sqrt(n) B {B.shape} {1/np.sqrt(n1+n2)*B[:3,:3]}")



            else:
                print("nystrom impossible, you need compute landmarks and/or anchors")
        
        
        elif approximation == 'full':
            K = self.compute_gram_matrix(landmarks=False)
            Kw = 1/(n1+n2) * torch.chain_matmul(Pbi,K,Pbi)


        self.verbosity(function_name='compute_bicentered_gram',
                dict_of_variables={'approximation':approximation},
                start=False,
                verbose = verbose)    

        return Kw

    def diagonalize_centered_gram(self,approximation='full',covariance='x',overwrite=False,verbose=0):
        """
        Diagonalizes the bicentered Gram matrix which shares its spectrum with the Withon covariance operator in the RKHS.
        Stores eigenvalues (sp or spny) and eigenvectors (ev or evny) as attributes
        """
        if approximation in self.spev[covariance] and not overwrite:
            if verbose:
                print('No need to diagonalize')
        else:
            self.verbosity(function_name='diagonalize_centered_gram',
                    dict_of_variables={'approximation':approximation,
                    'covariance':covariance},
                    start=True,
                    verbose = verbose)
            
            K = self.compute_centered_gram(approximation=approximation,covariance=covariance,verbose=verbose)
            sp,ev = eigsy(K)
            order = sp.argsort()[::-1]
            
            ev = torch.tensor(ev[:,order],dtype=torch.float64) 
            sp = torch.tensor(sp[order], dtype=torch.float64)

            self.spev[covariance][approximation] = {'sp':sp,'ev':ev}
            
            self.verbosity(function_name='diagonalize_centered_gram',
                    dict_of_variables={'approximation':approximation,
                                     'covariance':covariance},
                    start=False,
                    verbose = verbose)
        
    def diagonalize_bicentered_gram(self,approximation='full',overwrite=False,verbose=0):
        """
        Diagonalizes the bicentered Gram matrix which shares its spectrum with the Withon covariance operator in the RKHS.
        Stores eigenvalues (sp or spny) and eigenvectors (ev or evny) as attributes
        """
        if approximation in self.spev and not overwrite:
            if verbose:
                print('No need to diagonalize')
        else:
            self.verbosity(function_name='diagonalize_bicentered_gram',
                    dict_of_variables={'approximation':approximation},
                    start=True,
                    verbose = verbose)
            
            Kw = self.compute_bicentered_gram(approximation=approximation,verbose=verbose)
            sp,ev = eigsy(Kw)
            order = sp.argsort()[::-1]
            
            ev = torch.tensor(ev[:,order],dtype=torch.float64) 
            sp = torch.tensor(sp[order], dtype=torch.float64)

            self.spev[approximation] = {'sp':sp,'ev':ev}
            
            self.verbosity(function_name='diagonalize_bicentered_gram',
                    dict_of_variables={'approximation':approximation},
                    start=False,
                    verbose = verbose)
        
    def compute_kfdat(self,trunc=None,approximation_cov='full',approximation_mmd='full',name=None,verbose=0):
        # je n'ai plus besoin de trunc, seulement d'un t max 
        """ 
        Computes the kfda truncated statistic of [Harchaoui 2009].
        9 methods : 
        approximation_cov in ['full','nystrom','quantization']
        approximation_mmd in ['full','nystrom','quantization']
        
        Stores the result as a column in the dataframe df_kfdat
        """
        
        name = name if name is not None else f'{approximation_cov}{approximation_mmd}' 
        sp,ev = self.spev[approximation_cov]['sp'],self.spev[approximation_cov]['ev']
        tmax = len(sp)+1
        t = tmax if (trunc is None or trunc[-1]>tmax) else trunc[-1]
                
        self.verbosity(function_name='compute_kfdat',
                dict_of_variables={
                't':t,
                'approximation_cov':approximation_cov,
                'approximation_mmd':approximation_mmd,
                'name':name},
                start=True,
                verbose = verbose)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        n1,n2 = (self.n1,self.n2) 
        n = n1+n2

        m = self.compute_m(quantization=(approximation_mmd=='quantization'))
        Pbi = self.compute_bicentering_matrix(quantization=(approximation_cov=='quantization'))

        if 'nystrom' in [approximation_mmd,approximation_cov]:
            Up = self.spev['anchors']['ev']
            Lp_inv = torch.diag(self.spev['anchors']['sp']**-1)

        if not (approximation_mmd == approximation_cov) or approximation_mmd == 'nystrom':
            Kmn = self.compute_kmn()
        
        if approximation_cov == 'full':
            if approximation_mmd == 'full':
                K = self.compute_gram_matrix()
                pkm = torch.mv(Pbi,torch.mv(K,m))
                kfda = ((n1*n2)/(n**2*sp[:t]**2)*torch.mv(ev.T[:t],pkm)**2).cumsum(axis=0).numpy() 

            elif approximation_mmd == 'nystrom':
                pkuLukm = torch.mv(Pbi,torch.mv(Kmn.T,torch.mv(Up,torch.mv(Lp_inv,torch.mv(Up.T,torch.mv(Kmn,m))))))
                kfda = ((n1*n2)/(n**2*sp[:t]**2)*torch.mv(ev.T[:t],pkuLukm)**2).cumsum(axis=0).numpy() 

            elif approximation_mmd == 'quantization':
                pkm = torch.mv(Pbi,torch.mv(Kmn.T,m))
                kfda = ((n1*n2)/(n**2*sp[:t]**2)*torch.mv(ev.T[:t],pkm)**2).cumsum(axis=0).numpy() 
    
        if approximation_cov == 'nystrom':
            if approximation_mmd in ['full','nystrom']: # c'est exactement la même stat  
                pkuLukm = torch.mv(Pbi,torch.mv(Kmn.T,torch.mv(Up,torch.mv(Lp_inv,torch.mv(Up.T,torch.mv(Kmn,m))))))
                kfda = ((n1*n2)/(n**2*sp[:t]**2)*torch.mv(ev.T[:t],pkuLukm)**2).cumsum(axis=0).numpy() 

            elif approximation_mmd == 'quantization':
                Kmm = self.compute_gram_matrix(landmarks=True)
                pkuLukm = torch.mv(Pbi,torch.mv(Kmn.T,torch.mv(Up,torch.mv(Lp_inv,torch.mv(Up.T,torch.mv(Kmm,m))))))
                kfda = ((n1*n2)/(n**2*sp[:t]**2)*torch.mv(ev.T[:t],pkuLukm)**2).cumsum(axis=0).numpy() 
        
        if approximation_cov == 'quantization':
            A_12 = torch.diag(torch.cat((1/n1*torch.bincount(self.xassignations),1/n2*torch.bincount(self.yassignations)))**(1/2)).double()
            if approximation_mmd == 'full':
                apkm = torch.mv(A_12,torch.mv(Pbi,torch.mv(Kmn,m)))
                # le n n'est pas au carré ici
                kfda = ((n1*n2)/(n*sp[:t]**2)*torch.mv(ev.T[:t],apkm)**2).cumsum(axis=0).numpy() 

            elif approximation_mmd == 'nystrom':
                Kmm = self.compute_gram_matrix(landmarks=True)
                apkuLukm = torch.mv(A_12,torch.mv(Pbi,torch.mv(Kmm,torch.mv(Up,torch.mv(Lp_inv,torch.mv(Up.T,torch.mv(Kmn,m)))))))
                # le n n'est pas au carré ici
                kfda = ((n1*n2)/(n*sp[:t]**2)*torch.mv(ev.T[:t],apkuLukm)**2).cumsum(axis=0).numpy() 

            elif approximation_mmd == 'quantization':
                Kmm = self.compute_gram_matrix(landmarks=True)
                apkm = torch.mv(A_12,torch.mv(Pbi,torch.mv(Kmm,m)))
                kfda = ((n1*n2)/(n*sp[:t]**2)*torch.mv(ev.T[:t],apkm)**2).cumsum(axis=0).numpy() 

        name = name if name is not None else f'{approximation_cov}{approximation_mmd}' 
        if name in self.df_kfdat:
            print(f"écrasement de {name} dans df_kfdat")
        self.df_kfdat[name] = pd.Series(kfda,index=trunc)

        self.verbosity(function_name='compute_kfdat',
                                dict_of_variables={
                't':t,
                'approximation_cov':approximation_cov,
                'approximation_mmd':approximation_mmd,
                'name':name},
                start=False,
                verbose = verbose)

    def kfdat(self,trunc=None,approximation_cov='full',approximation_mmd='full',name=None,
            overwrite_kfdat=False,overwrite_cov = False,overwrite_landmarks=False,overwrite_anchors=False,
            nlandmarks=None,landmarks_method='random',nanchors=None,verbose=0):
        #nystrom=False,nanchors=None,nystrom_method='kmeans',name=None,main=False,obs_to_ignore=None,save=False,path=None,verbose=0):
        # mettre des verbose - 1 dans les fonctions intermediaires si j'ajoute verbosity ici 
        
        cov,mmd = approximation_cov,approximation_mmd
        name = name if name is not None else f'{cov}{mmd}' 
        if name in self.df_kfdat and not overwrite_kfdat :
            if verbose : 
                print(f'kfdat {name} already computed')
        else:
            if 'quantization' in [cov,mmd]: # besoin des poids des ancres de kmeans en quantization
                if not self.quantization_with_landmarks_possible or overwrite_landmarks: # on lance kmeans si il n'a pas été lancé ou qu'on veut le refaire avec un nlandmarks différent
                    self.compute_nystrom_landmarks(nlandmarks=nlandmarks,landmarks_method='kmeans',verbose=verbose)
                    if cov=='quantization': # on calcule les vp de ce qu'on vient de calculer si on en a besoin
                        self.diagonalize_bicentered_gram(approximation='quantization',overwrite=True,verbose=verbose)
                elif cov=='quantization':
                    if 'quantization' not in self.spev or overwrite_cov: 
                        self.diagonalize_bicentered_gram(approximation='quantization',overwrite=overwrite_cov,verbose=verbose)

            if 'nystrom' in [cov,mmd]:
                if not self.has_anchors or overwrite_anchors:
                    if not self.has_landmarks or overwrite_landmarks:
                        self.compute_nystrom_landmarks(nlandmarks=nlandmarks,landmarks_method=landmarks_method,verbose=verbose)
                    self.compute_nystrom_anchors(nanchors=nanchors,verbose=verbose)
                    self.diagonalize_bicentered_gram(approximation='nystrom',overwrite=overwrite_cov,verbose=verbose)
                elif cov == 'nystrom':
                    print('première fois ici')
                    if 'nystrom' not in self.spev or overwrite_cov: 
                        self.diagonalize_bicentered_gram(approximation='nystrom',overwrite=overwrite_cov,verbose=verbose)

            if cov=='full':
                if 'full' not in self.spev or overwrite_cov: 
                    self.diagonalize_bicentered_gram(approximation='full',overwrite=overwrite_cov,verbose=verbose)


            self.compute_kfdat(trunc=trunc,approximation_cov=cov,approximation_mmd=mmd,name=name,verbose=verbose)
        
        
        # which_dict={'kfdat':path if save else ''}
        # self.test(trunc=trunc,which_dict=which_dict,nystrom=nystrom,nanchors=nanchors,nystrom_method=nystrom_method,name=name,main=main,
        # obs_to_ignore=obs_to_ignore,save=save,verbose=verbose)

    def compute_proj_kfda(self,trunc=None,approximation_cov='full',approximation_mmd='full',overwrite=False,name=None,verbose=0):
        # je n'ai plus besoin de trunc, seulement d'un t max 
        """ 
        Projections of the embeddings of the observation onto the discriminant axis
        9 methods : 
        approximation_cov in ['full','nystrom','quantization']
        approximation_mmd in ['full','nystrom','quantization']
        
        Stores the result as a column in the dataframe df_proj_kfda
        """


        name = name if name is not None else f'{approximation_cov}{approximation_mmd}' 
        if name in self.df_proj_kfda and not overwrite :
            if verbose : 
                print('Proj on discriminant axis Already computed')
        else:

            sp,ev = self.spev[approximation_cov]['sp'],self.spev[approximation_cov]['ev']
            tmax = len(sp)+1
            t = tmax if (trunc is None or trunc[-1]>tmax) else trunc[-1]
            trunc = range(1,tmax) if (trunc is None or trunc[-1]>tmax) else trunc
            self.verbosity(function_name='compute_proj_kfda',
                    dict_of_variables={
                    't':t,
                    'approximation_cov':approximation_cov,
                    'approximation_mmd':approximation_mmd,
                    'name':name},
                    start=True,
                    verbose = verbose)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            n1,n2 = (self.n1,self.n2) 
            n = n1+n2

            m = self.compute_m(quantization=(approximation_mmd=='quantization'))
            Pbi = self.compute_bicentering_matrix(quantization=(approximation_cov=='quantization'))

            if 'nystrom' in [approximation_mmd,approximation_cov]:
                Up = self.spev['anchors']['ev']
                Lp_inv = torch.diag(self.spev['anchors']['sp']**-1)

            if not (approximation_mmd == approximation_cov == 'full'):
                Kmn = self.compute_kmn()
            if approximation_cov == 'full':
                K = self.compute_gram_matrix()

            if approximation_cov == 'full':
                if approximation_mmd == 'full':
                    pkm = torch.mv(Pbi,torch.mv(K,m))
                    proj = (n**-1*sp[:t]**(-3/2)*torch.mv(ev.T[:t],pkm)*torch.chain_matmul(ev.T[:t],Pbi,K).T).cumsum(axis=1).numpy()
                    
                elif approximation_mmd == 'nystrom':
                    pkuLukm = torch.mv(Pbi,torch.mv(Kmn.T,torch.mv(Up,torch.mv(Lp_inv,torch.mv(Up.T,torch.mv(Kmn,m))))))
                    proj = (n**-1*sp[:t]**(-3/2)*torch.mv(ev.T[:t],pkuLukm)*torch.chain_matmul(ev.T[:t],Pbi,K).T).cumsum(axis=1).numpy()

                elif approximation_mmd == 'quantization':
                    pkm = torch.mv(Pbi,torch.mv(Kmn.T,m))
                    proj = (n**-1*sp[:t]**(-3/2)*torch.mv(ev.T[:t],pkm)*torch.chain_matmul(ev.T[:t],Pbi,K).T).cumsum(axis=1).numpy()
                    
            if approximation_cov == 'nystrom':
                if approximation_mmd == 'full':
                    pkuLukm = torch.mv(Pbi,torch.mv(Kmn.T,torch.mv(Up,torch.mv(Lp_inv,torch.mv(Up.T,torch.mv(Kmn,m))))))
                    proj = (n**-1*sp[:t]**(-3/2)*torch.mv(ev.T[:t],pkuLukm)*torch.chain_matmul(ev.T[:t],Pbi,Kmn.T,Up,Lp_inv,Up.T,Kmn).T).cumsum(axis=1).numpy()

                elif approximation_mmd == 'nystrom':
                    pkuLukm = torch.mv(Pbi,torch.mv(Kmn.T,torch.mv(Up,torch.mv(Lp_inv,torch.mv(Up.T,torch.mv(Kmn,m))))))
                    proj = (n**-1*sp[:t]**(-3/2)*torch.mv(ev.T[:t],pkuLukm)*torch.chain_matmul(ev.T[:t],Pbi,Kmn.T,Up,Lp_inv,Up.T,Kmn).T).cumsum(axis=1).numpy()

                elif approximation_mmd == 'quantization':
                    Kmm = self.compute_gram_matrix(landmarks=True)
                    pkuLukm = torch.mv(Pbi,torch.mv(Kmn.T,torch.mv(Up,torch.mv(Lp_inv,torch.mv(Up.T,torch.mv(Kmm,m))))))
                    proj = (n**-1*sp[:t]**(-3/2)*torch.mv(ev.T[:t],pkuLukm)*torch.chain_matmul(ev.T[:t],Pbi,Kmn.T,Up,Lp_inv,Up.T,Kmn).T).cumsum(axis=1).numpy()

            if approximation_cov == 'quantization':
                A_12 = torch.diag(torch.cat((1/n1*torch.bincount(self.xassignations),1/n2*torch.bincount(self.yassignations)))**(1/2)).double()
                if approximation_mmd == 'full':
                    apkm = torch.mv(A_12,torch.mv(Pbi,torch.mv(Kmn,m)))
                    # pas de n ici
                    proj = (sp[:t]**(-3/2)*torch.mv(ev.T[:t],apkm)*torch.chain_matmul(ev.T[:t],A_12,Pbi,Kmn).T).cumsum(axis=1).numpy()

                elif approximation_mmd == 'nystrom':
                    Kmm = self.compute_gram_matrix(landmarks=True)
                    apkuLukm = torch.mv(A_12,torch.mv(Pbi,torch.mv(Kmm,torch.mv(Up,torch.mv(Lp_inv,torch.mv(Up.T,torch.mv(Kmn,m)))))))
                    # pas de n ici
                    proj = (sp[:t]**(-3/2)*torch.mv(ev.T[:t],apkuLukm)*torch.chain_matmul(ev.T[:t],A_12,Pbi,Kmn).T).cumsum(axis=1).numpy()

                elif approximation_mmd == 'quantization':
                    Kmm = self.compute_gram_matrix(landmarks=True)
                    apkm = torch.mv(A_12,torch.mv(Pbi,torch.mv(Kmm,m)))
                    proj = (sp[:t]**(-3/2)*torch.mv(ev.T[:t],apkm)*torch.chain_matmul(ev.T[:t],A_12,Pbi,Kmn).T).cumsum(axis=1).numpy()

            name = name if name is not None else f'{approximation_cov}{approximation_mmd}' 
            if name in self.df_proj_kfda:
                print(f"écrasement de {name} dans df_proj_kfda")
            self.df_proj_kfda[name] = pd.DataFrame(proj,index= self.index[self.imask],columns=[str(t) for t in trunc])
            self.df_proj_kfda[name]['sample'] = ['x']*n1 + ['y']*n2
            
            self.verbosity(function_name='compute_proj_kfda',
                                    dict_of_variables={
                    't':t,
                    'approximation_cov':approximation_cov,
                    'approximation_mmd':approximation_mmd,
                    'name':name},
                    start=False,
                    verbose = verbose)

    def compute_proj_kpca(self,trunc=None,approximation_cov='full',covariance='x',overwrite=False,name=None,verbose=0):
        # je n'ai plus besoin de trunc, seulement d'un t max 
        """ 
        
        """
        name = name if name is not None else f'{approximation_cov}{covariance}' 
        if name in self.df_proj_kfda and not overwrite :
            if verbose : 
                print('Proj on principal componant axis Already computed')
        else:
            w = covariance.lower() =='w'
            x = covariance.lower() =='x'
            quantization = approximation_cov =='quantization'
            if w:
                sp,ev = self.spev[approximation_cov]['sp'],self.spev[approximation_cov]['ev']
            else:
                sp,ev = self.spev[covariance][approximation_cov]['sp'],self.spev[covariance][approximation_cov]['ev']
            
            tmax = len(sp)+1
            t = tmax if (trunc is None or trunc[-1]>tmax) else trunc[-1]
            trunc = range(1,tmax) if (trunc is None or trunc[-1]>tmax) else trunc
            self.verbosity(function_name='compute_proj_kpca',
                    dict_of_variables={
                    't':t,
                    'approximation_cov':approximation_cov,
                    'covariance':covariance,
                    'name':name},
                    start=True,
                    verbose = verbose)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            n1,n2 = (self.n1,self.n2) 
            n = n1+n2
            kernel = self.kernel        
            

            # faire une seule fct de bicentering-centering
            # faire une seule fct diagonalize gram (centered/bicentered)
            # faire une seule fct de compute kmn 
            if w:
                P = self.compute_bicentering_matrix(quantization=quantization)
                if approximation_cov in ['nystrom','quantization']:
                    Kmn = self.compute_kmn()
                    if approximation_cov == 'nystrom':
                        Up = self.spev['anchors']['ev']
                        Lp_inv = torch.diag(self.spev['anchors']['sp']**-1)
                    else:
                        A_12 = torch.diag(torch.cat((1/n1*torch.bincount(self.xassignations),1/n2*torch.bincount(self.yassignations)))**(1/2)).double()
                
                else:
                    K = self.compute_gram_matrix()
            else:
                P = self.compute_centering_matrix(covariance=covariance,quantization=quantization)
                sample = self.x if x else self.y 
                if approximation_cov in ['nystrom','quantization']:
                    Z = self.xlandmarks if x else self.ylandmarks 
                    Kmn = kernel(Z,sample)
                    if approximation_cov == 'nystrom':
                        Up = self.spev[covariance]['anchors']['ev']
                        Lp_inv = torch.diag(self.spev[covariance]['anchors']['sp']**-1)
                    else:
                        if x:
                            A_12 = torch.diag(1/n1*torch.bincount(self.xassignations)**(1/2)).double()
                        else:
                            A_12 = torch.diag(1/n2*torch.bincount(self.yassignations)**(1/2)).double()
               
                else:
                    K = kernel(sample,sample)

        

            if approximation_cov == 'full':
                proj = (  n**(-1/2)*sp[:t]**(-1/2)*torch.chain_matmul(ev.T[:t],P,K).T).numpy()
            if approximation_cov == 'nystrom':
                proj = (  n**(-1/2)*sp[:t]**(-1/2)*torch.chain_matmul(ev.T[:t],P,Kmn.T,Up,Lp_inv,Up.T,Kmn).T).cumsum(axis=1).numpy()
            if approximation_cov =='quantization':
                proj = ( sp[:t]**(-1/2)*torch.chain_matmul(ev.T[:t],A_12,P,Kmn).T)

            name = name if name is not None else f'{approximation_cov}{covariance}' 
            if name in self.df_proj_kpca:
                print(f"écrasement de {name} dans df_proj_kfda")
            if w:
                self.df_proj_kpca[name] = pd.DataFrame(proj,index= self.index[self.imask],columns=[str(t) for t in trunc])
                self.df_proj_kpca[name]['sample'] = ['x']*n1 + ['y']*n2
            else:
                if x:
                    self.df_proj_kpca[name] = pd.DataFrame(proj,index= self.x_index[self.xmask],columns=[str(t) for t in trunc])
                    self.df_proj_kpca[name]['sample'] = ['x']*n1 
                else:
                    self.df_proj_kpca[name] = pd.DataFrame(proj,index= self.y_index[self.ymask],columns=[str(t) for t in trunc])
                    self.df_proj_kpca[name]['sample'] = ['y']*n2 
                    
            self.verbosity(function_name='compute_proj_kpca',
                                    dict_of_variables={
                    't':t,
                    'approximation_cov':approximation_cov,
                    'covariance':covariance,
                    'name':name},
                    start=False,
                    verbose = verbose)

    def compute_corr_proj_var(self,trunc=None,
                            covariance='w',
                            which='proj_kfda',
                            name_corr=None,
                            name_proj=None,prefix_col='',verbose=0): # df_array,df_proj,csvfile,pathfile,trunc=range(1,60)):
        
        self.verbosity(function_name='compute_corr_proj_var',
                dict_of_variables={'trunc':trunc,
                            'covariance':covariance,'which':which,'name_corr':name_corr,'name_proj':name_proj,'prefix_col':prefix_col},
                start=True,
                verbose = verbose)

        self.prefix_col=prefix_col

        df_proj= self.init_df_proj(which,name_proj)
        if trunc is None:
            trunc = range(1,df_proj.shape[1] - 1) # -1 pour la colonne sample
        w = covariance =='w'
        x = covariance =='x'
        if w:
            df_array = pd.DataFrame(torch.cat((self.x[self.xmask,:],self.y[self.ymask,:]),dim=0).numpy(),index=self.index[self.imask],columns=self.variables)
        else:
            df_array = pd.DataFrame((self.x[self.xmask,:]).numpy(),index=self.x_index[self.xmask],columns=self.variables) if x else \
                       pd.DataFrame((self.y[self.ymask,:]).numpy(),index=self.y_index[self.ymask],columns=self.variables)
        for t in trunc:
            df_array[f'{prefix_col}{t}'] = pd.Series(df_proj[f'{t}'])
        
        name_corr = name_corr if name_corr is not None else which.split(sep='_')[1]+name_proj if name_proj is not None else which.split(sep='_')[1] + covariance
        # if w:
        self.corr[name_corr] = df_array.corr().loc[self.variables,[f'{prefix_col}{t}' for t in trunc]]
        # else:
        #     print(df_array.head(20))
        #     df = df_array.corr()
        #     print(df.shape)
        self.verbosity(function_name='compute_corr_proj_var',
                dict_of_variables={'trunc':trunc,'covariance':covariance,'which':which,'name_corr':name_corr,'name_proj':name_proj,'prefix_col':prefix_col},
                start=False,
                verbose = verbose)


    def compute_mmd(self,unbiaised=False,nystrom=False,name='',verbose=0):
        
        self.verbosity(function_name='compute_mmd',
                dict_of_variables={'unbiaised':unbiaised,'nystrom':nystrom,'name':name},
                start=True,
                verbose = verbose)

        n1,n2 = (self.n1,self.n2) 
        ntot = n1+n2
        if nystrom >=1:
            m1,m2 = (self.nxanchors,self.nyanchors)
            mtot = m1+m2
        npoints = mtot if nystrom else ntot
        
        if nystrom:
            m1,m2 = (self.nxanchors,self.nyanchors)
            m_mu1   = -1/m1 * torch.ones(m1, dtype=torch.float64) #, device=device) 
            m_mu2   = 1/m2 * torch.ones(m2, dtype=torch.float64) # , device=device)
        else:
            m_mu1    = -1/n1 * torch.ones(n1, dtype=torch.float64) # , device=device)
            m_mu2    = 1/n2 * torch.ones(n2, dtype=torch.float64) # , device=device) 
        m_mu12 = torch.cat((m_mu1, m_mu2), dim=0) #.to(device)
        
        K = self.compute_gram_matrix(landmarks=nystrom)
        
        if name is None:
            name=''
        self.dict_mmd['B'+name] = torch.dot(torch.mv(K,m_mu12),m_mu12)

        if unbiaised:                
            mask = torch.eye(npoints,npoints).byte()
            K.masked_fill_(mask, 0)
            self.dict_mmd['U'+name] = torch.dot(torch.mv(K,m_mu12),m_mu12)
        
        self.verbosity(function_name='compute_mmd',
                dict_of_variables={'unbiaised':unbiaised,'nystrom':nystrom,'name':name},
                start=False,
                verbose = verbose)

    def name_generator(self,trunc=None,nystrom=0,nystrom_method='kmeans',nanchors=None,obs_to_ignore=None):
        
        if obs_to_ignore is not None:
            name_ = f'~{obs_to_ignore[0]} n={len(obs_to_ignore)}'
        else:
            name_ = ""
            if trunc is not None:
                name_ += f"tmax{trunc[-1]}"
            if nystrom:
                name_ +=f'ny{nystrom}{nystrom_method}na{nanchors}'
        return(name_)
    


    def proj_kfda(self,trunc=None,nystrom=False,nanchors=None,nystrom_method='kmeans',name=None,main=False,obs_to_ignore=None,save=False,path=None,verbose=0):
        which_dict={'proj_kfda':path if save else ''}
        self.test(trunc=trunc,which_dict=which_dict,nystrom=nystrom,nanchors=nanchors,nystrom_method=nystrom_method,name=name,main=main,
        obs_to_ignore=obs_to_ignore,save=save,verbose=verbose)

    def proj_kpca(self,trunc=None,nystrom=False,nanchors=None,nystrom_method='kmeans',name=None,main=False,obs_to_ignore=None,save=False,path=None,verbose=0):
        which_dict={'proj_kpca':path if save else ''}
        self.test(trunc=trunc,which_dict=which_dict,nystrom=nystrom,nanchors=nanchors,nystrom_method=nystrom_method,name=name,main=main,
        obs_to_ignore=obs_to_ignore,save=save,verbose=verbose)

    def correlations(self,trunc=None,corr_which='proj_kfda',nystrom=False,nanchors=None,nystrom_method='kmeans',name=None,main=False,obs_to_ignore=None,save=False,path=None,verbose=0):
        which_dict={'corr':path if save else ''}
        self.test(trunc=trunc,which_dict=which_dict,nystrom=nystrom,nanchors=nanchors,nystrom_method=nystrom_method,name=name,main=main,
        obs_to_ignore=obs_to_ignore,save=save,verbose=verbose,corr_which=corr_which,corr_prefix_col='')

    def mmd(self,unbiaised=True,nystrom=False,nanchors=None,nystrom_method='kmeans',name=None,main=False,obs_to_ignore=None,save=False,path=None,verbose=0):
        which_dict={'mmd':path if save else ''}
        self.test(which_dict=which_dict,nystrom=nystrom,nanchors=nanchors,nystrom_method=nystrom_method,name=name,main=main,
        obs_to_ignore=obs_to_ignore,mmd_unbiaised=unbiaised,save=save,verbose=verbose)


    def test(self,trunc=None,which_dict=['kfdat','proj_kfda','proj_kpca','corr','mmd'],
             nystrom=False,nanchors=None,nystrom_method='kmeans',
             name=None,main=False,corr_which='proj_kfda',corr_prefix_col='',obs_to_ignore=None,mmd_unbiaised=False,save=False,verbose=0):

        # for output,path in which.items()
        name_ = "main" if not hasattr(self,'main_name') and name is None else \
                self.name_generator(trunc=trunc,nystrom=nystrom,nystrom_method=nystrom_method,nanchors=nanchors,
                obs_to_ignore=obs_to_ignore) if name is None else \
                name


        if main or not hasattr(self,'main_name'):
            self.main_name = name_
        
        # if verbose >0:
        #     none = 'None'
        #     datastr = f'n1:{self.n1} n2:{self.n2} trunc:{none if trunc is None else len(trunc)}'
        #     datastr += f'\nname:{name}\n' 
        #     inwhich = ' and '.join(which_dict.keys()) if len(which_dict)>1 else list(which_dict.keys())[0]
        #     ny=''
        #     if nystrom:
        #         ny += f' nystrom:{nystrom} {nystrom_method} nanchors={nanchors}' 
        #         if split_data:
        #             ny+=f' split{test_size}' 
 
        #     print(f'{datastr}Compute {inwhich} {ny}') #  of {self.n1} and {self.n2} points{ny} ')
        # if verbose >1:
        #     print(f"trunc:{len(trunc)} \n which:{which_dict} nystrom:{nystrom} nanchors:{nanchors} nystrom_method:{nystrom_method} split:{split_data} test_size:{test_size}\n")
        #     print(f"main:{main} corr:{corr_which} mmd_unbiaised:{mmd_unbiaised} seva:{save}")
        
        loaded = []    
        if save:
            if 'kfdat' in which_dict and os.path.isfile(which_dict['kfdat']):
                loaded_kfdat = pd.read_csv(which_dict['kfdat'],header=0,index_col=0)
                if len(loaded_kfdat.columns)==1 and name is not None:
                    c= loaded_kfdat.columns[0]
                    self.df_kfdat[name] = loaded_kfdat[c]
                else:
                    for c in loaded_kfdat.columns:
                        if c not in self.df_kfdat.columns:
                            self.df_kfdat[c] = loaded_kfdat[c]
                loaded += ['kfdat']

            if 'proj_kfda' in which_dict and os.path.isfile(which_dict['proj_kfda']):
                self.df_proj_kfda[name_] = pd.read_csv(which_dict['proj_kfda'],header=0,index_col=0)
                loaded += ['proj_kfda']

            if 'proj_kpca' in which_dict and os.path.isfile(which_dict['proj_kpca']):
                self.df_proj_kpca[name_] = pd.read_csv(which_dict['proj_kpca'],header=0,index_col=0)
                loaded += ['proj_kpca']
            
            if 'corr' in which_dict and os.path.isfile(which_dict['corr']):
                self.corr[name_] =pd.read_csv(which_dict['corr'],header=0,index_col=0)
                loaded += ['corr']

            # if 'mmd' in which_dict and os.path.isfile(which_dict['mmd']):
            #     self.mmd[name_] =pd.read_csv(which_dict['mmd'],header=0,index_col=0)
            #     loaded += ['mmd']


            if verbose >0:
                print('loaded:',loaded)

        if len(loaded) < len(which_dict):
            
            missing = [k for k in which_dict.keys() if k not in loaded]
            # try:

            
            self.ignore_obs(obs_to_ignore=obs_to_ignore)
            
            if any([m in ['kfdat','proj_kfda','proj_kpca'] for m in missing]):
                if nystrom:
                    self.nystrom_method = nystrom_method
                    self.compute_nystrom_anchors(nanchors=nanchors,nystrom_method=nystrom_method,verbose=verbose) # max_iter=1000,

                if 'kfdat' in missing and nystrom==3 and not hasattr(self,'sp'):
                    self.diagonalize_bicentered_gram(nystrom=False,verbose=verbose)
                else:
                    self.diagonalize_bicentered_gram(nystrom,verbose=verbose)

            if 'kfdat' in which_dict and 'kfdat' not in loaded:
                self.compute_kfdat(trunc=trunc,nystrom=nystrom,name=name_,verbose=verbose)  
                loaded += ['kfdat']
                if save and obs_to_ignore is None:
                    self.df_kfdat.to_csv(which_dict['kfdat'],index=True)    
    
            if 'proj_kfda' in which_dict and 'proj_kfda' not in loaded:
                self.compute_proj_kfda(trunc=trunc,nystrom=nystrom,name=name_,verbose=verbose)    
                loaded += ['proj_kfda']
                if save and obs_to_ignore is None:
                    self.df_proj_kfda[name_].to_csv(which_dict['proj_kfda'],index=True)

            if 'proj_kpca' in which_dict and 'proj_kpca' not in loaded:
                self.compute_proj_kpca(trunc=trunc,nystrom=nystrom,name=name_,verbose=verbose)    
                loaded += ['proj_kpca']
                if save and obs_to_ignore is None:
                    self.df_proj_kpca[name_].to_csv(which_dict['proj_kpca'],index=True)
            
            if 'corr' in which_dict and 'corr' not in loaded:
                self.compute_corr_proj_var(trunc=trunc,nystrom=nystrom,which=corr_which,name_corr=name_,prefix_col=corr_prefix_col,verbose=verbose)
                loaded += ['corr']
                if save and obs_to_ignore is None:
                    self.corr[name_].to_csv(which_dict['corr'],index=True)
            
            if 'mmd' in which_dict and 'mmd' not in loaded:
                self.compute_mmd(unbiaised=mmd_unbiaised,nystrom=nystrom,name=name_,verbose=verbose)
                loaded += ['mmd']
                if save and obs_to_ignore is None:
                    self.corr[name_].to_csv(which_dict['mmd'],index=True)
            
            if verbose>0:
                print('computed:',missing)
            if obs_to_ignore is not None:
                self.unignore_obs()
        
            # except:
            #     print('No computed')        

    def ignore_obs(self,obs_to_ignore=None,reinitialize_ignored_obs=False):
        
        if self.ignored_obs is None or reinitialize_ignored_obs:
            self.ignored_obs = obs_to_ignore
        else:
            self.ignored_obs.append(obs_to_ignore)

        if obs_to_ignore is not None:
            self.xmask = ~self.x_index.isin(obs_to_ignore)
            self.ymask = ~self.y_index.isin(obs_to_ignore)
            self.imask = ~self.index.isin(obs_to_ignore)
            self.n1 = self.x[self.xmask,:].shape[0]
            self.n2 = self.y[self.ymask,:].shape[0]

    def unignore_obs(self):
        self.xmask = [True]*len(self.x)
        self.ymask = [True]*len(self.y)
        self.imask = [True]*len(self.index)
        self.n1 = self.x.shape[0]
        self.n2 = self.y.shape[0]
    
    def infer_nobs(self,which ='proj_kfda',name=None):
        if not hasattr(self,'n1'):
            if name is None:
                name = self.main_name
            df_proj= self.init_df_proj(which,name)
            self.n1 =  df_proj[df_proj['sample']=='x'].shape[0]
            self.n2 =  df_proj[df_proj['sample']=='y'].shape[0]

    def plot_kfdat(self,ax=None,ylim=None,figsize=(10,10),trunc=None,columns=None,asymp_ls='--',asymp_c = 'crimson',title=None,title_fontsize=40,highlight=False,highlight_main=False,mean=False,mean_label='mean',mean_color = 'xkcd: indigo'):
            
        # try:
            if columns is None:
                columns = self.df_kfdat.columns
            kfdat = self.df_kfdat[columns].copy()
            
            if self.main_name in columns and highlight_main and len(columns)>1:
                kfdat[self.main_name].plot(ax=ax,lw=4,c='black')
                kfdat = kfdat.drop(columns=self.main_name) 
            elif highlight and len(columns)==1:
                kfdat.plot(ax=ax,lw=4,c='black')
            # print(self.df_kfdat.columns,'\n',kfdat.columns)
            if ax is None:
                fig,ax = plt.subplots(figsize=figsize)
            if mean:
                ax.plot(kfdat.mean(axis=1),label=mean_label,c=mean_color)
                ax.plot(kfdat.mean(axis=1)- 2* kfdat.std(axis=1)/(~kfdat[columns[0]].isna()).sum(),c=mean_color,ls = '--',alpha=.5)
                ax.plot(kfdat.mean(axis=1)+ 2* kfdat.std(axis=1)/(~kfdat[columns[0]].isna()).sum(),c=mean_color,ls = '--',alpha=.5)
            else:
                kfdat.plot(ax=ax)
            if trunc is None:
                trunc = range(1,max([(~kfdat[c].isna()).sum() for c in columns]))
            
            yas = [chi2.ppf(0.95,t) for t in trunc]    
            ax.plot(trunc,yas,ls=asymp_ls,c=asymp_c,lw=4)
            ax.set_xlabel('t',fontsize= 20)
            ax.set_xlim(0,trunc[-1])
            if title is not None:
                ax.set_title(title,fontsize=title_fontsize)
            if ylim is None:
                ymax = np.max([yas[-1], np.nanmax(np.isfinite(kfdat[kfdat.index == trunc[-1]]))]) # .max(numeric_only=True)
                ylim = (-5,ymax)
            ax.set_ylim(ylim)
            ax.legend()  
            return(ax)
        # except Exception as e:
        #     print(type(e),e)
        
    def plot_spectrum(self,ax=None,figsize=(10,10),trunc=None,title=None,generate_spectrum=True,approximation_cov='full',covariance='w'):
        if ax is None:
            fig,ax = plt.subplots(figsize=figsize)
        if title is not None:
            ax.set_title(title,fontsize=40)
        if covariance == 'w':
            sp = self.spev[approximation_cov]['sp']
        else:
            sp = self.spev[covariance][approximation_cov]['sp']
        if trunc is None:
            trunc = range(1,len(sp))
        ax.plot(trunc,sp[:trunc[-1]])
        ax.set_xlabel('t',fontsize= 20)

        return(ax)

    def density_proj(self,ax,projection,which='proj_kfda',name=None):
        
        df_proj= self.init_df_proj(which,name)

        for xy,l in zip('xy','CT'):
            
            dfxy = df_proj.loc[df_proj['sample']==xy][str(projection)]
            if len(dfxy)>0:
                ax.hist(dfxy,density=True,histtype='bar',label=f'{l}({len(dfxy)})',alpha=.3,bins=int(np.floor(np.sqrt(len(dfxy)))),color='blue' if xy =='x' else 'orange')
                ax.hist(dfxy,density=True,histtype='step',bins=int(np.floor(np.sqrt(len(dfxy)))),lw=3,edgecolor='blue' if xy =='x' else 'orange')
                ax.axvline(dfxy.mean(),c='blue' if xy=='x' else 'orange')
        
        ax.set_xlabel(f't={projection}',fontsize=20)    
        ax.legend()
        
    def scatter_proj(self,ax,projection,xproj='proj_kfda',yproj=None,name=None,highlight=None,color=None):
        p1,p2 = projection
        yproj = xproj if yproj is None else yproj
        df_abscisse = self.init_df_proj(xproj,name)
        df_ordonnee = self.init_df_proj(yproj,name)
        
        for xy,l in zip('xy','CT'):
            df_abscisse_xy = df_abscisse.loc[df_abscisse['sample']==xy]
            df_ordonnee_xy = df_ordonnee.loc[df_ordonnee['sample']==xy]
            m = 'x' if xy =='x' else '+'
            if len(df_abscisse_xy)>0 and len(df_ordonnee_xy)>0 :
                if color is None or color in list(self.variables): # list vraiment utile ? 
                    c = 'xkcd:cerulean' if xy =='x' else 'xkcd:light orange'
                    if color in list(self.variables):
                        c = self.x[self.xmask,:][:,self.variables.get_loc(color)] if xy=='x' else self.y[self.ymask,:][:,self.variables.get_loc(color)]   
                    x_ = df_abscisse_xy[f'{p1}']
                    y_ = df_ordonnee_xy[f'{p2}']

                    ax.scatter(x_,y_,c=c,s=30,label=f'{l}({len(x_)})',alpha=.8,marker =m)
                else:
                    if xy in color: # a complexifier si besoin (nystrom ou mask) 
                        x_ = df_abscisse_xy[f'{p1}'] #[df_abscisse_xy.index.isin(ipop)]
                        y_ = df_ordonnee_xy[f'{p2}'] #[df_ordonnee_xy.index.isin(ipop)]
                        ax.scatter(x_,y_,s=30,c=color[xy], alpha=.8,marker =m)
                    for pop,ipop in color.items():
                        x_ = df_abscisse_xy[f'{p1}'][df_abscisse_xy.index.isin(ipop)]
                        y_ = df_ordonnee_xy[f'{p2}'][df_ordonnee_xy.index.isin(ipop)]
                        if len(x_)>0:
                            ax.scatter(x_,y_,s=30,label=f'{pop} {l}({len(x_)})',alpha=.8,marker =m)

        
        for xy,l in zip('xy','CT'):

            df_abscisse_xy = df_abscisse.loc[df_abscisse['sample']==xy]
            df_ordonnee_xy = df_ordonnee.loc[df_ordonnee['sample']==xy]
            x_ = df_abscisse_xy[f'{p1}']
            y_ = df_ordonnee_xy[f'{p2}']
            if len(df_abscisse_xy)>0 and len(df_ordonnee_xy)>0 :
                if highlight is not None:
                    x_ = df_abscisse_xy[f'{p1}']
                    y_ = df_ordonnee_xy[f'{p2}']
                    c = 'xkcd:cerulean' if xy =='x' else 'xkcd:light orange'
                    ax.scatter(x_[x_.index.isin(highlight)],y_[y_.index.isin(highlight)],c=c,s=100,marker='*',edgecolor='black',linewidths=1)

                mx_ = x_.mean()
                my_ = y_.mean()
                ax.scatter(mx_,my_,edgecolor='black',linewidths=3,s=200)

        if color in list(self.variables) :
            ax.set_title(color,fontsize=20)
        ax.set_xlabel(xproj.split(sep='_')[1]+f': t={p1}',fontsize=20)                    
        ax.set_ylabel(yproj.split(sep='_')[1]+f': t={p2}',fontsize=20)
        
        ax.legend()

    def init_df_proj(self,which,name=None):
        # if name is None:
        #     name = self.main_name
        
        dict_df_proj = self.df_proj_kfda if which=='proj_kfda' else self.df_proj_kpca

        nproj = len(dict_df_proj)
        names = list(dict_df_proj.keys())

        if nproj == 0:
            print('Proj_kfda has not been computed yet')
        if nproj == 1:
            if name is not None and name != names[0]:
                print(f'{name} not corresponding to {names[0]}')
            else:
                df_proj = dict_df_proj[names[0]]
        if nproj >1:
            if name is not None and name not in names:
                print(f'{name} not found in {names}')
            if name is None and self.main_name not in names:
                print("the default name {self.main_name} is not in {names} so you need to specify 'name' argument")
            if name is None and self.main_name in names:
                df_proj = dict_df_proj[self.main_name]
            else: 
                df_proj = dict_df_proj[name]

        return(df_proj)

    def init_axes_projs(self,fig,axes,projections,approximation_cov,covariance,suptitle,kfda,kfda_ylim,trunc,kfda_title,spectrum):
        if axes is None:
            rows=1;cols=len(projections) + kfda + spectrum
            fig,axes = plt.subplots(nrows=rows,ncols=cols,figsize=(6*cols,6*rows))
        if suptitle is not None:
            fig.suptitle(suptitle,fontsize=50)
        if kfda:
            self.plot_kfdat(axes[0],ylim=kfda_ylim,trunc = trunc,title=kfda_title)
            axes = axes[1:]
        if spectrum:
            self.plot_spectrum(axes[0],trunc=trunc,title='spectrum',approximation_cov=approximation_cov,covariance=covariance)
            axes = axes[1:]
        return(fig,axes)

    def density_projs(self,fig=None,axes=None,which='proj_kfda',approximation_cov='full',covariance='w',name=None,projections=range(1,10),suptitle=None,kfda=False,kfda_ylim=None,trunc=None,kfda_title=None,spectrum=False):
        fig,axes = self.init_axes_projs(fig=fig,axes=axes,projections=projections,approximation_cov=approximation_cov,covariance=covariance,suptitle=suptitle,kfda=kfda,
                                        kfda_ylim=kfda_ylim,trunc=trunc,kfda_title=kfda_title,spectrum=spectrum)
        if not isinstance(axes,np.ndarray):
            axes = [axes]
        for ax,proj in zip(axes,projections):
            self.density_proj(ax,proj,which=which,name=name)
        return(fig,axes)

    def scatter_projs(self,fig=None,axes=None,xproj='proj_kfda',approximation_cov='full',covariance='w',yproj=None,name=None,projections=[(1,i+2) for i in range(10)],suptitle=None,
                        highlight=None,color=None,kfda=False,kfda_ylim=None,trunc=None,kfda_title=None,spectrum=False,iterate_over='projections'):
        to_iterate = projections if iterate_over == 'projections' else color
        fig,axes = self.init_axes_projs(fig=fig,axes=axes,projections=to_iterate,approximation_cov=approximation_cov,covariance=covariance,suptitle=suptitle,kfda=kfda,
                                        kfda_ylim=kfda_ylim,trunc=trunc,kfda_title=kfda_title,spectrum=spectrum)
        if not isinstance(axes,np.ndarray):
            axes = [axes]
        for ax,obj in zip(axes,to_iterate):
            if iterate_over == 'projections':
                self.scatter_proj(ax,obj,xproj=xproj,yproj=yproj,name=name,highlight=highlight,color=color)
            elif iterate_over == 'color':
                self.scatter_proj(ax,projections,xproj=xproj,yproj=yproj,name=name,highlight=highlight,color=obj)
        return(fig,axes)

    def find_cells_from_proj(self,which='proj_kfda',name=None,t=1,bound=0,side='left'):
        df_proj= self.init_df_proj(which,name=name)
        return df_proj[df_proj[str(t)] <= bound].index if side =='left' else df_proj[df_proj[str(t)] >= bound].index

    def find_correlated_variables(self,name=None,nvar=1,t=1,prefix_col=''):
        if name is None:
            name = self.main_name
        if nvar==0:
            return(np.abs(self.corr[name][f'{prefix_col}{t}']).sort_values(ascending=False)[:])
        else: 
            return(np.abs(self.corr[name][f'{prefix_col}{t}']).sort_values(ascending=False)[:nvar])
        
    def plot_correlation_proj_var(self,ax=None,name=None,figsize=(10,10),nvar=30,projections=range(1,10),title=None,prefix_col=''):
        if name is None:
            name = self.main_name
        if ax is None:
            fig,ax = plt.subplots(figsize=figsize)
        if title is not None:
            ax.set_title(title,fontsize=40)
        for proj in projections:
            col = f'{prefix_col}{proj}'
            val  = list(np.abs(self.corr[name][col]).sort_values(ascending=False).values)[:nvar]
            print(val)
            ax.plot(val,label=col)
        ax.legend()
        return(ax)
        
    def eval_nystrom_trace(self):
        """
        Returns the squared difference between the spectrum in ``self.sp`` and the nystrom spectrum in ``self.spny``
        """
        # je pourrais choisir jusqu'où je somme mais pour l'instant je somme sur l'ensemble du spectre 
        return((self.sp.sum() - self.spny.sum())**2)
    
    def eval_nystrom_spectrum(self):
        """
        Returns the squared difference between each eigenvalue of the spectrum in ``self.sp`` and each eigenvalue of the nystrom spectrum in ``self.spny``
        """
        return([ (lny.item() - l.item())**2 for lny,l in zip(self.spny,self.sp)])

    def eval_nystrom_discriminant_axis(self,nystrom=1,t=None,m=None):
        # a adapter au GPU
        # j'ai pas du tout réfléchi à la version test_data, j'ai juste fait en sorte que ça marche au niveau des tailles de matrices donc il y a peut-être des erreurs de logique

        n1,n2 = (self.n1,self.n2)
        ntot = n1+n2
        m1,m2 = (self.nxlandmarks,self.nylandmarks)
        mtot = m1+m2

        mtot=self.nxlandmarks + self.nylandmarks
        ntot= self.n1 + self.n2
        
        t = 60   if t is None else t # pour éviter un calcul trop long # t= self.n1 + self.n2
        m = mtot if m is None else m
        
        kmn   = self.compute_kmn(test_data=False)
        kmn_test   = self.compute_kmn(test_data=True)
        kny   = self.compute_gram_matrix(landmarks=True)
        k     = self.compute_gram_matrix(landmarks=False,test_data=False)
        Pbiny = self.compute_bicentering_matrix(quantization=True)
        Pbi   = self.compute_bicentering_matrix(quantization=False,test_data=False)
        
        Vny  = self.evny[:m]
        V    = self.ev[:t] 
        spny = self.spny[:m]
        sp   = self.sp[:t]

        mny1   = -1/m1 * torch.ones(m1, dtype=torch.float64) #, device=device) 
        mny2   = 1/m2 * torch.ones(m2, dtype=torch.float64) # , device=device)
        m_mtot = torch.cat((mny1, mny2), dim=0) # .to(device)
        

        mn1    = -1/n1 * torch.ones(n1, dtype=torch.float64) # , device=device)
        mn2    = 1/n2 * torch.ones(n2, dtype=torch.float64) # , device=device) 
        m_ntot = torch.cat((mn1, mn2), dim=0) #.to(device)
        
        # mn1_test    = -1/n1_test * torch.ones(n1_test, dtype=torch.float64) # , device=device)
        # mn2_test    = 1/n2_test * torch.ones(n2_test, dtype=torch.float64) # , device=device) 
        # m_ntot_test = torch.cat((mn1_test, mn2_test), dim=0) #.to(device)
        
        vpkm    = torch.mv(torch.chain_matmul(V,Pbi,k),m_ntot)
        vpkm_ny = torch.mv(torch.chain_matmul(Vny,Pbiny,kmn_test),m_ntot_test) if nystrom==1 else \
                  torch.mv(torch.chain_matmul(Vny,Pbiny,kny),m_mtot)

        norm_h   = (ntot**-1 * sp**-2   * torch.mv(torch.chain_matmul(V,Pbi,k),m_ntot)**2).sum()**(1/2)
        norm_hny = (mtot**-1 * spny**-2 * torch.mv(torch.chain_matmul(Vny,Pbiny,kmn_test),m_ntot_test)**2).sum()**(1/2) if nystrom==1 else \
                   (mtot**-1 * spny**-2 * torch.mv(torch.chain_matmul(Vny,Pbiny,kny),m_mtot)**2).sum()**(1/2)

        # print(norm_h,norm_hny)

        A = torch.zeros(m,t,dtype=torch.float64).addr(1/mtot*spny,1/ntot*sp) # A = outer(1/mtot*self.spny,1/ntot*self.sp)
        B = torch.zeros(m,t,dtype=torch.float64).addr(vpkm_ny,vpkm) # B = torch.outer(vpkm_ny,vpkm)
        C = torch.chain_matmul(Vny,Pbiny,kmn,Pbi,V.T)

        return(norm_h**-1*norm_hny**-1*(A*B*C).sum().item()) # <h_ny^* , h^*>/(||h_ny^*|| ||h^*||)



    def eval_nystrom_eigenfunctions(self,t=None,m=None):
        # a adapter au GPU


        n1,n2 = (self.n1,self.n2)
        ntot = n1+n2
        m1,m2 = (self.nxlandmarks,self.nylandmarks)
        mtot = m1+m2

        mtot=self.nxlandmarks + self.nylandmarks
        ntot= self.n1 + self.n2
        
        t = 60   if t is None else t # pour éviter un calcul trop long # t= self.n1 + self.n2
        m = mtot if m is None else m
        
        kmn   = self.compute_kmn()
        Pbiny = self.compute_bicentering_matrix(quantization=True)
        Pbi   = self.compute_bicentering_matrix(quantization=False)
        
        Vny  = self.evny[:m]
        V    = self.ev[:t] 
        spny = self.spny[:m]
        sp   = self.sp[:t]

        # print(((spny * mtot)**(-1/2))[:3],'\n',((sp*ntot)**-(1/2))[:3])
        return( ((((spny * mtot)**(-1/2))*((sp*ntot)**-(1/2)*torch.chain_matmul(Vny,Pbiny,kmn,Pbi, V.T)).T).T).diag())
        


        # cev=(ev[:trunc[-1]].t()*coefs).t()
        

        # hip_wny_w = torch.matmul() 
    # trouver comment inserer un test sous H0 sans prendre trop de mémoire 
    # def split_one_sample_to_simulate_H0(self,sample='x'):
    #     z = self.x if sample =='x' else self.y
    #     nH0 = self.n1 if sample == 'x' else self.n2
    #     p = permutation(np.arange(nH0))
    #     self.x0,self.y0 = z[p[:nH0//2]],z[p[nH0//2:]]




