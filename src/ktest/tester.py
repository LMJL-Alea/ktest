from typing_extensions import Literal
from typing import Optional,Callable,Union,List

import numpy as np
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

# est ce que je range la gram ou la calcule à chaque besoin ? Je calcule la gram a chaque fois mais la diagonalise une fois par setting 
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


# a partir d'un moment, j'ai supposé que test data n'était plus d'actualité et cessé de l'utiliser. A vérifier et supprimer si oui 

# def des fonction type get pour les opérations d'initialisation redondantes
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
        # attributs initialisés 
        self.df_kfdat = pd.DataFrame()
        self.df_proj_kfda = {}
        self.df_proj_kpca = {}
        self.corr = {}     
        self.dict_mmd = {}

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


    def compute_nystrom_landmarks(self,nlandmarks,landmarks_method,verbose=0):
        # anciennement compute_nystrom_anchors(self,nanchors,nystrom_method='kmeans',split_data=False,test_size=.8,on_other_data=False,x_other=None,y_other=None,verbose=0): # max_iter=1000, (pour kmeans de François)
        
        """
        We distinguish the nystrom landmarks and nystrom anchors.
        The nystrom landmarks are the points from the observation space that will be used to compute the nystrom anchors. 
        The nystrom anchors are the points in the RKHS computed from the nystrom landmarks on which we will apply the nystrom method. 

        Parameters
        ----------
        nlandmarks:      number of landmarks in total (proportionnaly sampled according to the data)
        landmarks_method: 'kmeans' or 'random' (in the future : 'kmeans ++, greedy...)
        """
        if verbose >0:
            start = time()
            print(f'compute_nystrom_landmarks(nlandmarks={nlandmarks},landmarks_method={landmarks_method})...',end=' ')

        # Commentted bc thought to be useless, to suppress if confirmed

        # if verbose >1:
        #     print(f"nanchors{nanchors} nystrom_method:{nystrom_method} split:{split_data} test_size:{test_size} on_other_data:{on_other_data}")

        # if on_other_data:

        #     xmask_ny = self.xmask
        #     ymask_ny = self.ymask  
        #     xratio,yratio = self.n1/(self.n1 + self.n2), self.n2/(self.n1 + self.n2)
        #     self.nxanchors=np.int(np.floor(xratio * nanchors)) 
        #     self.nyanchors=np.int(np.floor(yratio * nanchors))

        #     if nystrom_method == 'kmeans':
        #         # self.xanchors,self.xassignations = apt.kmeans.spherical_kmeans(self.x[self.xmask,:], nxanchors, max_iter)
        #         # self.yanchors,self.yassignations = apt.kmeans.spherical_kmeans(self.y[self.ymask,:], nyanchors, max_iter)
        #         self.xassignations,self.xanchors = kmeans(X=x_other, num_clusters=self.nxanchors, distance='euclidean', tqdm_flag=False) #cuda:0')
        #         self.yassignations,self.yanchors = kmeans(X=y_other, num_clusters=self.nyanchors, distance='euclidean', tqdm_flag=False) #cuda:0')
        #         self.xanchors = self.xanchors.double()
        #         self.yanchors = self.yanchors.double()
                
        #     elif nystrom_method == 'random':
        #         self.xanchors = x_other[np.random.choice(x_other.shape[0], size=self.nxanchors, replace=False)]
        #         self.yanchors = y_other[np.random.choice(y_other.shape[0], size=self.nyanchors, replace=False)]
            
            
        
        # else:
        xratio,yratio = self.n1/(self.n1 + self.n2), self.n2/(self.n1 + self.n2)
        self.nxlandmarks=np.int(np.floor(xratio * nlandmarks)) 
        self.nylandmarks=np.int(np.floor(yratio * nlandmarks))

            # if split_data:

            #     #split data
            #     # Say a = 1 - test_size
            #     # We determine the nanchors = nxanchors + nyanchors on n1_ny = |_a*n1_| and n2_ny = |_a*n2_| data. 
            #     # To keep proportion we have nxanchors = |_ nanchors * n1/(n1+n2) _|and nyanchors = |_ nanchors * n2/(n1+n2) _| (strictly positive numbers)
            #     # Thus, we need to have n1_ny >= nxanchors and n2_ny >= nyanchors 
            #     # We use a*n1 >= |_ a*n1 _| and find the condition a >= 1/n1 |_ nanchors* n1/(n1+n2) _| and  a >= 1/n2 |_ nanchors* n2/(n1+n2) _| 
            #     # in order to implement a simple rule, we raise an error if these conditions are not fulfilled:
            #     assert (1-test_size) >= 1/self.n1 * np.int(np.floor(nanchors * xratio)) and \
            #         (1-test_size) >= 1/self.n2 * np.int(np.floor(nanchors * yratio)) 
            #     assert self.nxanchors >0 and self.nyanchors >0
                    
            #     # print(self.xmask.sum(),len(self.x_index))
            #     xindex_nystrom,xindex_test = train_test_split(self.x_index[self.xmask],test_size=.8)
            #     yindex_nystrom,yindex_test = train_test_split(self.y_index[self.ymask],test_size=.8)
                
            #     xmask_ny = self.x_index.isin(xindex_nystrom)
            #     ymask_ny = self.y_index.isin(yindex_nystrom)

            #     self.xmask_test = self.x_index.isin(xindex_test)
            #     self.ymask_test = self.y_index.isin(yindex_test)
            #     self.n1_test = len(xindex_test)
            #     self.n2_test = len(yindex_test)


            # else:
        xmask_ny = self.xmask
        ymask_ny = self.ymask  

        if landmarks_method == 'kmeans':
            # self.xanchors,self.xassignations = apt.kmeans.spherical_kmeans(self.x[self.xmask,:], nxanchors, max_iter)
            # self.yanchors,self.yassignations = apt.kmeans.spherical_kmeans(self.y[self.ymask,:], nyanchors, max_iter)
            self.xassignations,self.xlandmarks = kmeans(X=self.x[xmask_ny,:], num_clusters=self.nxlandmarks, distance='euclidean', tqdm_flag=False) #cuda:0')
            self.yassignations,self.ylandmarks = kmeans(X=self.y[ymask_ny,:], num_clusters=self.nylandmarks, distance='euclidean', tqdm_flag=False) #cuda:0')
            self.xlandmarks = self.xlandmarks.double()
            self.ylandmarks = self.ylandmarks.double()
        elif landmarks_method == 'random':
            self.xlandmarks = self.x[xmask_ny,:][np.random.choice(self.x[xmask_ny,:].shape[0], size=self.nxlandmarks, replace=False)]
            self.ylandmarks = self.y[ymask_ny,:][np.random.choice(self.y[ymask_ny,:].shape[0], size=self.nylandmarks, replace=False)]

        if verbose > 0:
            print(time() - start)


    def compute_nystrom_anchors(self,nanchors,nystrom_method='raw',split_data=False,test_size=.8,on_other_data=False,x_other=None,y_other=None,verbose=0): # max_iter=1000, (pour kmeans de François)
        """
        Determines the nystrom anchors using ``nystrom_method`` which can be 'raw' or 'kPCA'
        
        Parameters
        ----------
        nystrom_method: 'raw' project all the observation on Span(landmarks) 
                        'kPCA' determines the subspace generated by the first directions of a kPCA applied to the landmarks
                        note that when nanchors == nlandmarks 'kPCA' and 'raw' are equivalent. 
        nanchors:      = nlandmarks by default if nystrom_method is 'raw'. Number of anchors to determine in total (proportionnaly according to the data)
        """
        
        if verbose >0:
            start = time()
            print(f'Computing anchors by {nystrom_method} nystrom',end=' ')
            if nystrom_method == 'kPCA':
                print(f'for {nanchors} anchors ...', end=' ')
            else:         
                print('...',end=' ')


        # bien faire l'arbre des possibles ici 
        if nystrom_method =='raw':
            self.nxanchors, self.nyanchors = self.nxlandmarks, self.nylandmarks

        if nystrom_method =='kPCA':
            xratio,yratio = self.n1/(self.n1 + self.n2), self.n2/(self.n1 + self.n2) 
            self.nxanchors=np.int(np.floor(xratio * nanchors)) 
            self.nyanchors=np.int(np.floor(yratio * nanchors))
            assert(self.nxanchors <= self.nxlandmarks)
            assert(self.nyanchors <= self.nylandmarks)

        spx,evx = eigsy(self.kernel(self.xlandmarks,self.xlandmarks))
        spx = torch.diag(torch.tensor(spx[:self.nxanchors]))
        evx = torch.tensor(evx).T[:self.nxanchors]
        self.k_rectx = torch.chain_matmul(spx**(-1/2), evx,self.kernel(self.xlandmarks,self.x))


        spy,evy = eigsy(self.kernel(self.ylandmarks,self.ylandmarks))
        spy = torch.diag(torch.tensor(spy[:self.nyanchors]))
        evy = torch.tensor(evy).T[:self.nyanchors]
        self.k_recty = torch.chain_matmul(spy**(-1/2), evy,self.kernel(self.ylandmarks,self.y)) 

        if verbose > 0:
            print(time() - start)
        
        # def compute_nystrom_kmn(self,test_data=False):
        #     """
        #     Computes an (nxanchors+nyanchors)x(n1+n2) conversion gram matrix
        #     """
        #     x,y = (self.x[self.xmask_test,:],self.y[self.ymask_test,:]) if test_data else (self.x[self.xmask,:],self.y[self.ymask,:])
        #     z1,z2 = self.xanchors,self.yanchors
        #     kernel = self.kernel
            

        #     kz1x = kernel(z1,x)
        #     kz2x = kernel(z2,x)
        #     kz1y = kernel(z1,y)
        #     kz2y = kernel(z2,y)
            
        #     return(torch.cat((torch.cat((kz1x, kz1y), dim=1),
        #                         torch.cat((kz2x, kz2y), dim=1)), dim=0))
        
        # def compute_nystrom_kntestn(self):
        #     """
        #     Computes an (nxanchors+nyanchors)x(n1+n2) conversion gram matrix
        #     """
        #     x,y = (self.x[self.xmask,:],self.y[self.ymask,:])
        #     z1,z2 = self.x[self.xmask_test,:],self.y[self.ymask_test,:]
        #     kernel = self.kernel
            
        #     kz1x = kernel(z1,x)
        #     kz2x = kernel(z2,x)
        #     kz1y = kernel(z1,y)
        #     kz2y = kernel(z2,y)
            
        #     return(torch.cat((torch.cat((kz1x, kz1y), dim=1),
        #                         torch.cat((kz2x, kz2y), dim=1)), dim=0))
        

    
    def compute_gram_matrix(self,nystrom=False,test_data=False):
        """
        Computes Gram matrix, on anchors if nystrom is True, else on data. 
        This function is called everytime the Gram matrix is needed but I could had an option to keep it in memory in case of a kernel function 
        that makes it difficult to compute

        Returns
        -------
        torch.Tensor of size (nxanchors+nyanchors)**2 if nystrom else (n1+n2)**2
        """
        x,y = (self.xanchors,self.yanchors) if nystrom else \
              (self.x[self.xmask_test,:],self.y[self.ymask_test,:]) if test_data else \
              (self.x[self.xmask,:],self.y[self.ymask,:])

        kernel = self.kernel
                         
        kxx = kernel(x, x)
        kyy = kernel(y, y)
        kxy = kernel(x, y)

        return(torch.cat((torch.cat((kxx, kxy), dim=1),
                            torch.cat((kxy.t(), kyy), dim=1)), dim=0))

    def compute_bicentering_matrix(self,nystrom=False,test_data=False):
        """
        Computes the bicentering Gram matrix Pn. 
        Let I1,I2 the identity matrix of size n1 and n2 (or nxanchors and nyanchors if nystrom).
            J1,J2 the squared matrix full of one of size n1 and n2 (or nxanchors and nyanchors if nystrom).
            012, 021 the matrix full of zeros of size n1 x n2 and n2 x n1 (or nxanchors x nyanchors and nyanchors x nxanchors if nystrom)
        
        Pn = [I1 - 1/n1 J1 ,    012     ]
             [     021     ,I2 - 1/n2 J2]

        Returns
        -------
        torch.Tensor of size (nxanchors+nyanchors)**2 if nystrom else (n1+n2)**2 
        """

        
        n1,n2 = (self.nxanchors,self.nyanchors)  if nystrom else (self.n1_test,self.n2_test) if test_data else (self.n1,self.n2) 
        
        idn1 = torch.eye(n1, dtype=torch.float64)
        idn2 = torch.eye(n2, dtype=torch.float64)

        onen1 = torch.ones(n1, n1, dtype=torch.float64)
        onen2 = torch.ones(n2, n2, dtype=torch.float64)

        if nystrom in [4,5]:
            A1 = 1/self.n1*torch.diag(torch.bincount(self.xassignations)).double()
            A2 = 1/self.n2*torch.diag(torch.bincount(self.yassignations)).double()
            pn1 = np.sqrt(self.n1/(self.n1+self.n2))*(idn1 - torch.matmul(A1,onen1))
            pn2 = np.sqrt(self.n2/(self.n1+self.n2))*(idn2 - torch.matmul(A2,onen2))
        else:
            pn1 = idn1 - 1/n1 * onen1
            pn2 = idn2 - 1/n2 * onen2

        z12 = torch.zeros(n1, n2, dtype=torch.float64)
        z21 = torch.zeros(n2, n1, dtype=torch.float64)

        return(torch.cat((torch.cat((pn1, z12), dim=1), torch.cat(
            (z21, pn2), dim=1)), dim=0))  # bloc diagonal

    def compute_pkm(self,nystrom=False,test_data=False):
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        n1,n2 = (self.n1_test,self.n2_test) if test_data else (self.n1,self.n2) 

        if nystrom in [0,3]:
            Pbi   = self.compute_bicentering_matrix(nystrom=False,test_data=test_data)
        elif nystrom in [1,2]:    
            Pbi = self.compute_bicentering_matrix(nystrom=True,test_data=False)
        elif nystrom in [4,5]:
            Pbi = self.compute_bicentering_matrix(nystrom=nystrom,test_data=False)

        if nystrom in [2,3,5]:
            m1,m2 = (self.nxanchors,self.nyanchors)
            m_mu1   = -1/m1 * torch.ones(m1, dtype=torch.float64) #, device=device) 
            m_mu2   = 1/m2 * torch.ones(m2, dtype=torch.float64) # , device=device)
        elif nystrom in [0,1,4]:
            m_mu1    = -1/n1 * torch.ones(n1, dtype=torch.float64) # , device=device)
            m_mu2    = 1/n2 * torch.ones(n2, dtype=torch.float64) # , device=device) 
        
        m_mu12 = torch.cat((m_mu1, m_mu2), dim=0) #.to(device)
        
        if nystrom in [4,5]:
            A = torch.diag(torch.cat((1/n1*torch.bincount(self.xassignations),1/n2*torch.bincount(self.yassignations)))).double()
        if nystrom ==0:
            K = self.compute_gram_matrix(nystrom=False,test_data=test_data).to(device)
            pk = torch.matmul(Pbi,K)
        elif nystrom == 1:
            kmn = self.compute_nystrom_kmn(test_data=test_data).to(device)
            pk = torch.matmul(Pbi,kmn)
        elif nystrom == 2:
            kny = self.compute_gram_matrix(nystrom=nystrom).to(device)
            pk = torch.matmul(Pbi,kny)
        elif nystrom == 3:
            kmn = self.compute_nystrom_kmn(test_data=test_data).to(device)
            pk = torch.matmul(kmn,Pbi).T
        elif nystrom == 4:
            kmn = self.compute_nystrom_kmn(test_data=test_data).to(device)
            pk = torch.chain_matmul(A**(1/2),Pbi.T,kmn)
        elif nystrom == 5:
            kny = self.compute_gram_matrix(nystrom=nystrom).to(device)
            pk = torch.chain_matmul(A**(1/2),Pbi.T,kny)
            # pk = torch.chain_matmul(Pbi,A,kny,A)
            
        return(torch.mv(pk,m_mu12))  
        
    def diagonalize_bicentered_gram(self,nystrom=False,test_data=False,verbose=0):
        """
        Diagonalizes the bicentered Gram matrix which shares its spectrum with the Withon covariance operator in the RKHS.
        Stores eigenvalues (sp or spny) and eigenvectors (ev or evny) as attributes
        """
        if verbose >0:
            start = time()
            ny = ' nystrom' if nystrom else '' 
            print(f'Diagonalizing the{ny} Gram matrix ...',end=' ')
        if verbose >1:
            print(f'nystrom:{nystrom} test_data:{test_data}')

        n1,n2 =  (self.n1_test,self.n2_test) if test_data else (self.n1,self.n2)
        if nystrom:
            m1,m2 = (self.nxanchors,self.nyanchors)  
            # if nystrom in [1,2,3] else \
            #     (self.n1_test,self.n2_test) if test_data else \
            #     (self.n1,self.n2) # nystrom = False or nystrom in [4,5]

        pn = self.compute_bicentering_matrix(nystrom=nystrom,test_data=test_data).double()
        
        if nystrom in [4,5]:
            A = torch.diag(torch.cat((1/n1*torch.bincount(self.xassignations),1/n2*torch.bincount(self.yassignations)))).double()
            sp,ev = eigsy(torch.chain_matmul(A**(1/2),pn, self.compute_gram_matrix(nystrom=nystrom,test_data=test_data),pn,A**(1/2)).cpu().numpy())
        elif nystrom in [1,2]:
            sp,ev = eigsy(1/(m1+m2) * torch.chain_matmul(pn, self.compute_gram_matrix(nystrom=nystrom,test_data=test_data), pn).cpu().numpy())  # eigsy uses numpy
        else:
            sp,ev = eigsy(1/(n1+n2) * torch.chain_matmul(pn, self.compute_gram_matrix(nystrom=nystrom,test_data=test_data), pn).cpu().numpy())  # eigsy uses numpy
        
        order = sp.argsort()[::-1]
        
        if nystrom: # la distinction est utile pour calculer les metriques sur nystrom, mais on ne garde en mémoire que la dernière version de la diag nystrom
            self.evny = torch.tensor(ev.T[order],dtype=torch.float64) 
            self.spny = torch.tensor(sp[order], dtype=torch.float64)
        elif test_data:
            self.ev_test = torch.tensor(ev.T[order],dtype=torch.float64) 
            self.sp_test = torch.tensor(sp[order], dtype=torch.float64)
        else:
            self.ev = torch.tensor(ev.T[order],dtype=torch.float64) 
            self.sp = torch.tensor(sp[order], dtype=torch.float64)
        
        if verbose > 0:
            print(time() - start)
            
        # def compute_kfdat(self,trunc=None,nystrom=False,name=None,verbose=0):
        #     """ 
        #     Computes the kfda truncated statistic of [Harchaoui 2009].
        #     Two ways of using Nystrom: 
        #     nystrom = False or 0 -> Statistic computed without nystrom
        #     nystrom = True or 1  -> Statistic computed using nystrom for the diagonalized bicentered gram matrix (~Sigma_W) and not for mn (\mu_2 - \mu_1)
        #     nystrom = 2          -> Statistic computed using nystrom for the diagonalized bicentered gram matrix (~Sigma_W) and for mn (\mu_2 - \mu_1)
        #     Depending on the situation, the coeficient of the statistic changes. 


        #     Stores the result as a column in the dataframe df_kfdat
        #     """
        #     if verbose >0:
        #         start = time()
        #         ny = ' nystrom' if nystrom else '' 
        #         print(f'Computing the{ny} kfdat statistic ...',end=' ')



        #     n1,n2 = (self.nxanchors,self.nyanchors)  if nystrom else (self.n1,self.n2)
        #     ntot = self.n1 + self.n2  # tot nobs in data
        #     mtot = n1 + n2 # either tot nobs in data or tot nanchors 
            
        #     if trunc is None:
        #         trunc = np.arange(1,mtot+1)
            
        #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #     sp,ev = (self.spny.to(device),self.evny.to(device)) if nystrom else (self.sp.to(device),self.ev.to(device))  
        #     pn = self.compute_bicentering_matrix(nystrom).to(device)
            

        #     mn1 = -1/self.n1 * torch.ones(self.n1, dtype=torch.float64, device=device) if nystrom < 2 else -1/n1 * torch.ones(n1, dtype=torch.float64, device=device)
        #     mn2 = 1/self.n2 * torch.ones(self.n2, dtype=torch.float64, device=device)  if nystrom <2 else 1/n2 * torch.ones(n2, dtype=torch.float64, device=device)
        #     mn = torch.cat((mn1, mn2), dim=0).to(device)
            
            
        #     gram = self.compute_nystrom_kmn().to(device) if nystrom==1 else self.compute_gram_matrix(nystrom=nystrom).to(device) 
        #     pk = torch.matmul(pn,gram)
        #     pkm = torch.mv(pk,mn)

        #     kfda = 0
        #     kfda_dict = []
        #     if trunc[-1] >mtot:
        #         trunc=trunc[:mtot]
        #     for i,t in enumerate(trunc):
        #         if t <= mtot: 
        #             evi = ev[i]
        #             # kfda +=  (n1*n2)/(ntot * mtot *sp[i]**2)* torch.dot(evi,pkm)**2 if nystrom <2 else (n1*n2)/(mtot * mtot *sp[i]**2)* torch.dot(evi,pkm)**2
        #             kfda += (self.n1*self.n2)/(mtot * ntot *sp[i]**2)* torch.dot(evi,pkm)**2 if nystrom <2  else \
        #                     (n1*n2)/(mtot * mtot *sp[i]**2)* torch.dot(evi,pkm)**2
        #             kfda_dict += [kfda.item()] # [f'O_{t}']
        #     name = name if name is not None else self.name_generator(trunc,nystrom)
        #     self.df_kfdat[name] = pd.Series(kfda_dict,index=trunc)

        #     if verbose > 0:
        #         print(time() - start)
     
    def compute_kfdat(self,trunc=None,nystrom=False,test_data=False,name=None,verbose=0):
        """ 
        Computes the kfda truncated statistic of [Harchaoui 2009].
        Two ways of using Nystrom: 
        nystrom = False or 0 -> Statistic computed without nystrom
        nystrom = True or 1  -> Statistic computed using nystrom for the diagonalized bicentered gram matrix (~Sigma_W) and not for mn (\mu_2 - \mu_1)
        nystrom = 2          -> Statistic computed using nystrom for the diagonalized bicentered gram matrix (~Sigma_W) and for mn (\mu_2 - \mu_1)
        Depending on the situation, the coeficient of the statistic changes. 


        Stores the result as a column in the dataframe df_kfdat
        """
        if verbose >0:
            start = time()
            ny = ' nystrom' if nystrom else '' 
            print(f'Computing the{ny} kfdat statistic ...',end=' ')

        n1,n2 = (self.n1_test,self.n2_test) if test_data else (self.n1,self.n2) 
        ntot = n1+n2
        if nystrom >=1:
            m1,m2 = (self.nxanchors,self.nyanchors)
            mtot = m1+m2

        maxtrunc = ntot if nystrom ==0 else mtot
        if trunc is None:
            trunc = np.arange(1,ntot+1) if nystrom==False else np.arange(1,mtot+1)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if nystrom in [0,3]:
            sp,ev = (self.sp.to(device),self.ev.to(device))  
        else:    
            sp,ev = (self.spny.to(device),self.evny.to(device)) 
        
        pkm = self.compute_pkm(nystrom=nystrom,test_data=test_data)
        
        if trunc[-1] >maxtrunc:
            trunc=trunc[:maxtrunc]
        
        t=trunc[-1]
        kfda = ((n1*n2)/(ntot*ntot*sp[:t]**2)*torch.mv(ev[:t],pkm)**2).cumsum(axis=0).numpy() if nystrom ==0  else \
               ((n1*n2)/(ntot*mtot*sp[:t]**2)*torch.mv(ev[:t],pkm)**2).cumsum(axis=0).numpy() if nystrom ==1  else \
               ((m1*m2)/(mtot*mtot*sp[:t]**2)*torch.mv(ev[:t],pkm)**2).cumsum(axis=0).numpy() if nystrom ==2  else \
               ((m1*m2)/(mtot*ntot*sp[:t]**2)*torch.mv(ev[:t],pkm)**2).cumsum(axis=0).numpy() if nystrom ==3  else \
               ((n1*n2)/(ntot*sp[:t]**2)*torch.mv(ev[:t],pkm)**2).cumsum(axis=0).numpy() if nystrom ==4  else \
               ((m1*m2)/(mtot*sp[:t]**2)*torch.mv(ev[:t],pkm)**2).cumsum(axis=0).numpy() 
                        
        name = name if name is not None else self.name_generator(trunc,nystrom)
        self.df_kfdat[name] = pd.Series(kfda,index=trunc)

        if verbose > 0:
            print(time() - start)

    def compute_proj_kfda(self,trunc = None,nystrom=False,test_data=False,name=None,verbose=0):
        # ajouter nystrom dans m et dans la colonne sample
        if verbose >0:
            start = time()
            ny = ' nystrom' if nystrom else '' 
            print(f'Computing{ny} proj on kernel Fisher discriminant axis ...',end=' ')

        n1,n2 = (self.n1_test,self.n2_test) if test_data else (self.n1,self.n2) 
        ntot = n1+n2
        if nystrom >=1:
            m1,m2 = (self.nxanchors,self.nyanchors)
            mtot = m1+m2
        
        maxtrunc = ntot if nystrom ==0 else mtot
        if trunc is None:
            trunc = np.arange(1,ntot+1) if nystrom==False else np.arange(1,mtot+1)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if nystrom in [0,3]:
            Pbi   = self.compute_bicentering_matrix(nystrom=False,test_data=test_data)
            sp,ev     = (self.sp.to(device),self.ev.to(device))  
            if test_data:
                kntestn = self.compute_nystrom_kntestn()
                pk2 = torch.matmul(Pbi,kntestn)
        
        else:    
            Pbi = self.compute_bicentering_matrix(nystrom=True,test_data=False)
            sp,ev = (self.spny.to(device),self.evny.to(device)) 
            if test_data:
                kmn = self.compute_nystrom_kmn(test_data=False).to(device)
                pk2 = torch.matmul(Pbi,kmn)
        
        if nystrom ==0:
            K = self.compute_gram_matrix(nystrom=False,test_data=test_data).to(device)
            pk2 = torch.matmul(Pbi,K)
        elif nystrom == 1:
            kmn = self.compute_nystrom_kmn(test_data=test_data).to(device)
            pk2 = torch.matmul(Pbi,kmn)
        elif nystrom == 2:
            kny = self.compute_gram_matrix(nystrom=nystrom).to(device)
            pk2 = torch.matmul(Pbi,kny)
        else:
            kmn = self.compute_nystrom_kmn(test_data=test_data).to(device)
            pk2 = torch.matmul(kmn,Pbi).T
        
        pkm = self.compute_pkm(nystrom=nystrom,test_data=test_data)
           
            
        t=trunc[-1]
        proj = (ntot**-1*sp[:t]**(-3/2)*torch.mv(ev[:t],pkm)*torch.matmul(ev[:t],pk2).T).cumsum(axis=1).numpy() if nystrom ==0  else \
               (mtot**-1*sp[:t]**(-3/2)*torch.mv(ev[:t],pkm)*torch.matmul(ev[:t],pk2).T).cumsum(axis=0).numpy() if nystrom ==1  else \
               (mtot**-1*sp[:t]**(-3/2)*torch.mv(ev[:t],pkm)*torch.matmul(ev[:t],pk2).T).cumsum(axis=0).numpy() if nystrom ==2  else \
               (ntot**-1*sp[:t]**(-3/2)*torch.mv(ev[:t],pkm)*torch.matmul(ev[:t],pk2).T).cumsum(axis=0).numpy() 
        
        name = name if name is not None else self.name_generator(trunc,nystrom)
        self.df_proj_kfda[name] = pd.DataFrame(proj,index= self.index[self.imask],columns=[str(t) for t in trunc])
        self.df_proj_kfda[name]['sample'] = ['x']*n1 + ['y']*n2
        
        if verbose > 0:
            print(time() - start)

        # résidus du passé
        # n1,n2 = (self.nxanchors,self.nyanchors)  if nystrom else (self.n1,self.n2)
        # ntot = self.n1 + self.n2
        # mtot = n1 + n2
        
        # if trunc is None:
        #     trunc = np.arange(1,ntot+1)
        
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # sp,ev = (self.spny.to(device),self.evny.to(device)) if nystrom else (self.sp.to(device),self.ev.to(device))
        
        
        # pn = self.compute_bicentering_matrix(nystrom).to(device)
        
        # # m est construit sur toute les obs, pas sur les ancres même si Nystrom 
        # mn1 = -1/n1 * torch.ones(n1, dtype=torch.float64, device=device)
        # mn2 = 1/n2 * torch.ones(n2, dtype=torch.float64, device=device)
        # mn = torch.cat((mn1, mn2), dim=0).to(device)
        # pk = torch.matmul(pn,gram)


        # pkm = torch.mv(pk,mn)

        # coefs = []
        
       #         print(mtot)
        # for i,t in enumerate(trunc):
        #     if t<=mtot:
        #         evi = ev[i]
        #         coefs += [1/(mtot *sp[i]**(3/2))* torch.dot(evi,pkm).item()]
        # lvpkm =   
        # print(mn.shape)
        # coefs = torch.tensor(coefs)
        # cev=(ev[:trunc[-1]].t()*coefs).t()
        # cevpk = torch.matmul(cev,pk)
        # cevpk= cevpk.cumsum(dim=0)
        # print(cevpk[:2,:2])

    def compute_proj_kpca(self,trunc=None,nystrom=False,test_data=False,name=None,verbose=0):

        if verbose >0:
            start = time()
            ny = ' nystrom' if nystrom else '' 
            print(f'Computing{ny} proj on kernel principal componant axis ...',end=' ')


        n1,n2 = (self.n1_test,self.n2_test) if test_data else (self.n1,self.n2) 
        ntot = n1+n2
        if nystrom >=1:
            m1,m2 = (self.nxanchors,self.nyanchors)
            mtot = m1+m2
        
        maxtrunc = ntot if nystrom ==0 else mtot
        if trunc is None:
            trunc = np.arange(1,ntot+1) if nystrom==False else np.arange(1,mtot+1)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if nystrom in [0,3]:
            Pbi   = self.compute_bicentering_matrix(nystrom=False,test_data=test_data)
            sp,ev     = (self.sp.to(device),self.ev.to(device))  
        else:    
            Pbi = self.compute_bicentering_matrix(nystrom=True,test_data=False)
            sp,ev = (self.spny.to(device),self.evny.to(device)) 
        
        K = self.compute_kmn() if nystrom in [1,2] else \
            self.compute_nystrom_kntestn() if nystrom ==3 and test_data else \
            self.compute_gram_matrix()   


        t = trunc[-1]
        proj = (  ntot**(-1/2)*sp[:t]**(-1/2)*torch.chain_matmul(ev[:t],Pbi,K).T).numpy()
        
        name = name if name is not None else self.name_generator(trunc,nystrom)
        self.df_proj_kpca[name] = pd.DataFrame(proj,index=self.index[self.imask],columns=[str(t) for t in trunc])
        self.df_proj_kpca[name]['sample'] = ['x']*self.n1 + ['y']*self.n2 

        if verbose > 0:
            print(time() - start)
        
    def compute_corr_proj_var(self,trunc=None,nystrom=False,which='proj_kfda',name=None,prefix_col='',verbose=0): # df_array,df_proj,csvfile,pathfile,trunc=range(1,60)):
        if verbose >0:
            start = time()
            ny = ' nystrom' if nystrom else '' 
            print(f'Computing the{ny} corr matrix between projections and variables ...',end=' ')


        self.prefix_col=prefix_col
        df_proj= self.init_df_proj(which,name)
        if trunc is None:
            trunc = range(1,df_proj.shape[1] - 1) # -1 pour la colonne sample
        df_array = pd.DataFrame(torch.cat((self.x[self.xmask,:],self.y[self.ymask,:]),dim=0).numpy(),index=self.index[self.imask],columns=self.variables)
        for t in trunc:
            df_array[f'{prefix_col}{t}'] = pd.Series(df_proj[f'{t}'])
        
        name = name if name is not None else self.name_generator(trunc,nystrom)
        self.corr[name] = df_array.corr().loc[self.variables,[f'{prefix_col}{t}' for t in trunc]]
        
        if verbose > 0:
            print(time() - start)

    def compute_mmd(self,unbiaised=False,nystrom=False,test_data=False,name='',verbose=0):
        
        if verbose >0:
            start = time()
            ny = ' nystrom' if nystrom else '' 
            print(f'Computing the{ny} kfdat statistic ...',end=' ')

        n1,n2 = (self.n1_test,self.n2_test) if test_data else (self.n1,self.n2) 
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
        
        K = self.compute_gram_matrix(nystrom=nystrom,test_data=test_data)
        
        if name is None:
            name=''
        self.dict_mmd['B'+name] = torch.dot(torch.mv(K,m_mu12),m_mu12)

        if unbiaised:                
            mask = torch.eye(npoints,npoints).byte()
            K.masked_fill_(mask, 0)
            self.dict_mmd['U'+name] = torch.dot(torch.mv(K,m_mu12),m_mu12)
        
        if verbose > 0:
            print(time() - start)

    def name_generator(self,trunc=None,nystrom=0,nystrom_method='kmeans',nanchors=None,
                        split_data=False,test_size=.8,obs_to_ignore=None):
        
        if obs_to_ignore is not None:
            name_ = f'~{obs_to_ignore[0]} n={len(obs_to_ignore)}'
        else:
            name_ = ""
            if trunc is not None:
                name_ += f"tmax{trunc[-1]}"
            if nystrom:
                name_ +=f'ny{nystrom}{nystrom_method}na{nanchors}'
                if split_data:
                    name_ +=f'split{test_size}'
        return(name_)
    

    def kfdat(self,trunc=None,nystrom=False,nanchors=None,nystrom_method='kmeans',split_data=False,test_size=.8,name=None,main=False,obs_to_ignore=None,save=False,path=None,verbose=0):
        which_dict={'kfdat':path if save else ''}
        self.test(trunc=trunc,which_dict=which_dict,nystrom=nystrom,nanchors=nanchors,nystrom_method=nystrom_method,split_data=split_data,test_size=test_size,name=name,main=main,
        obs_to_ignore=obs_to_ignore,save=save,verbose=verbose)

    def proj_kfda(self,trunc=None,nystrom=False,nanchors=None,nystrom_method='kmeans',split_data=False,test_size=.8,name=None,main=False,obs_to_ignore=None,save=False,path=None,verbose=0):
        which_dict={'proj_kfda':path if save else ''}
        self.test(trunc=trunc,which_dict=which_dict,nystrom=nystrom,nanchors=nanchors,nystrom_method=nystrom_method,split_data=split_data,test_size=test_size,name=name,main=main,
        obs_to_ignore=obs_to_ignore,save=save,verbose=verbose)

    def proj_kpca(self,trunc=None,nystrom=False,nanchors=None,nystrom_method='kmeans',split_data=False,test_size=.8,name=None,main=False,obs_to_ignore=None,save=False,path=None,verbose=0):
        which_dict={'proj_kpca':path if save else ''}
        self.test(trunc=trunc,which_dict=which_dict,nystrom=nystrom,nanchors=nanchors,nystrom_method=nystrom_method,split_data=split_data,test_size=test_size,name=name,main=main,
        obs_to_ignore=obs_to_ignore,save=save,verbose=verbose)

    def correlations(self,trunc=None,nystrom=False,nanchors=None,nystrom_method='kmeans',split_data=False,test_size=.8,name=None,main=False,obs_to_ignore=None,save=False,path=None,verbose=0):
        which_dict={'corr':path if save else ''}
        self.test(trunc=trunc,which_dict=which_dict,nystrom=nystrom,nanchors=nanchors,nystrom_method=nystrom_method,split_data=split_data,test_size=test_size,name=name,main=main,
        obs_to_ignore=obs_to_ignore,save=save,verbose=verbose,corr_which='proj_kfda',corr_prefix_col='')

    def mmd(self,unbiaised=True,nystrom=False,nanchors=None,nystrom_method='kmeans',split_data=False,test_size=.8,name=None,main=False,obs_to_ignore=None,save=False,path=None,verbose=0):
        which_dict={'mmd':path if save else ''}
        self.test(which_dict=which_dict,nystrom=nystrom,nanchors=nanchors,nystrom_method=nystrom_method,split_data=split_data,test_size=test_size,name=name,main=main,
        obs_to_ignore=obs_to_ignore,mmd_unbiaised=unbiaised,save=save,verbose=verbose)


    def test(self,trunc=None,which_dict=['kfdat','proj_kfda','proj_kpca','corr','mmd'],
             nystrom=False,nanchors=None,nystrom_method='kmeans',split_data=False,test_size=.8,
             name=None,main=False,corr_which='proj_kfda',corr_prefix_col='',obs_to_ignore=None,mmd_unbiaised=False,save=False,verbose=0):

        # for output,path in which.items()
        name_ = "main" if not hasattr(self,'main_name') and name is None else \
                self.name_generator(trunc=trunc,nystrom=nystrom,nystrom_method=nystrom_method,nanchors=nanchors,
                split_data=split_data,test_size=test_size,obs_to_ignore=obs_to_ignore) if name is None else \
                name


        if main or not hasattr(self,'main_name'):
            self.main_name = name_
        
        if verbose >0:
            none = 'None'
            datastr = f'n1:{self.n1} n2:{self.n2} trunc:{none if trunc is None else len(trunc)}'
            datastr += f'\nname:{name}\n' 
            inwhich = ' and '.join(which_dict.keys()) if len(which_dict)>1 else list(which_dict.keys())[0]
            ny=''
            if nystrom:
                ny += f' nystrom:{nystrom} {nystrom_method} nanchors={nanchors}' 
                if split_data:
                    ny+=f' split{test_size}' 
 
            print(f'{datastr}Compute {inwhich} {ny}') #  of {self.n1} and {self.n2} points{ny} ')
        if verbose >1:
            print(f"trunc:{len(trunc)} \n which:{which_dict} nystrom:{nystrom} nanchors:{nanchors} nystrom_method:{nystrom_method} split:{split_data} test_size:{test_size}\n")
            print(f"main:{main} corr:{corr_which} mmd_unbiaised:{mmd_unbiaised} seva:{save}")
        
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

            test_data = split_data
            self.ignore_obs(obs_to_ignore=obs_to_ignore)
            
            if any([m in ['kfdat','proj_kfda','proj_kpca'] for m in missing]):
                if nystrom:
                    self.nystrom_method = nystrom_method
                    self.compute_nystrom_anchors(nanchors=nanchors,nystrom_method=nystrom_method,split_data=split_data,test_size=test_size,verbose=verbose) # max_iter=1000,

                if 'kfdat' in missing and nystrom==3 and not hasattr(self,'sp'):
                    self.diagonalize_bicentered_gram(nystrom=False,verbose=verbose)
                else:
                    self.diagonalize_bicentered_gram(nystrom,verbose=verbose)

            if 'kfdat' in which_dict and 'kfdat' not in loaded:
                self.compute_kfdat(trunc=trunc,nystrom=nystrom,test_data=test_data,name=name_,verbose=verbose)  
                loaded += ['kfdat']
                if save and obs_to_ignore is None:
                    self.df_kfdat.to_csv(which_dict['kfdat'],index=True)    
    
            if 'proj_kfda' in which_dict and 'proj_kfda' not in loaded:
                self.compute_proj_kfda(trunc=trunc,nystrom=nystrom,test_data=test_data,name=name_,verbose=verbose)    
                loaded += ['proj_kfda']
                if save and obs_to_ignore is None:
                    self.df_proj_kfda[name_].to_csv(which_dict['proj_kfda'],index=True)

            if 'proj_kpca' in which_dict and 'proj_kpca' not in loaded:
                self.compute_proj_kpca(trunc=trunc,nystrom=nystrom,test_data=test_data,name=name_,verbose=verbose)    
                loaded += ['proj_kpca']
                if save and obs_to_ignore is None:
                    self.df_proj_kpca[name_].to_csv(which_dict['proj_kpca'],index=True)
            
            if 'corr' in which_dict and 'corr' not in loaded:
                self.compute_corr_proj_var(trunc=trunc,nystrom=nystrom,which=corr_which,name=name_,prefix_col=corr_prefix_col,verbose=verbose)
                loaded += ['corr']
                if save and obs_to_ignore is None:
                    self.corr[name_].to_csv(which_dict['corr'],index=True)
            
            if 'mmd' in which_dict and 'mmd' not in loaded:
                self.compute_mmd(unbiaised=mmd_unbiaised,nystrom=nystrom,test_data=test_data,name=name_,verbose=verbose)
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
        
    def plot_spectrum(self,ax=None,figsize=(10,10),trunc=None,title=None,generate_spectrum=True):
        if ax is None:
            fig,ax = plt.subplots(figsize=figsize)
        if title is not None:
            ax.set_title(title,fontsize=40)
        if not hasattr(self,'sp') and generate_spectrum:
            self.diagonalize_bicentered_gram()
        if trunc is None:
            trunc = range(1,len(self.sp))
        ax.plot(trunc,self.sp[:trunc[-1]])
        ax.set_xlabel('t',fontsize= 20)

        return(ax)

    def density_proj(self,ax,projection,which='proj_kfda',name=None):
        
        df_proj= self.init_df_proj(which,name)

        for xy,l in zip('xy','CT'):
            
            dfxy = df_proj.loc[df_proj['sample']==xy][str(projection)]
            
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

            if color is None or color in list(self.variables): # list vraiment utile ? 
                c = 'xkcd:cerulean' if xy =='x' else 'xkcd:light orange'
                if color in list(self.variables):
                    c = self.x[self.xmask,:][:,self.variables.get_loc(color)] if xy=='x' else self.y[self.ymask,:][:,self.variables.get_loc(color)]   
                x_ = df_abscisse_xy[f'{p1}']
                y_ = df_ordonnee_xy[f'{p2}']

                ax.scatter(x_,y_,c=c,s=30,label=f'{l}({len(x_)})',alpha=.8,marker =m)
            else:
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

    def init_df_proj(self,which,name):
        if name is None:
            name = self.main_name
        if which == 'proj_kfda':
            df_proj = self.df_proj_kfda[name]
        elif which =='proj_kpca':
            df_proj = self.df_proj_kpca[name]
        else:
            print('pb df_proj',which)
        return(df_proj)

    def init_axes_projs(self,fig,axes,projections,suptitle,kfda,kfda_ylim,trunc,kfda_title,spectrum):
        if axes is None:
            rows=1;cols=len(projections) + kfda + spectrum
            fig,axes = plt.subplots(nrows=rows,ncols=cols,figsize=(6*cols,6*rows))
        if suptitle is not None:
            fig.suptitle(suptitle,fontsize=50)
        if kfda:
            self.plot_kfdat(axes[0],ylim=kfda_ylim,trunc = trunc,title=kfda_title)
            axes = axes[1:]
        if spectrum:
            self.plot_spectrum(axes[0],trunc=trunc,title='spectrum')
            axes = axes[1:]
        return(fig,axes)

    def density_projs(self,fig=None,axes=None,which='proj_kfda',name=None,projections=range(1,10),suptitle=None,kfda=False,kfda_ylim=None,trunc=None,kfda_title=None,spectrum=False):
        fig,axes = self.init_axes_projs(fig=fig,axes=axes,projections=projections,suptitle=suptitle,kfda=kfda,
                                        kfda_ylim=kfda_ylim,trunc=trunc,kfda_title=kfda_title,spectrum=spectrum)
        if not isinstance(axes,np.ndarray):
            axes = [axes]
        for ax,proj in zip(axes,projections):
            self.density_proj(ax,proj,which=which,name=name)
        return(fig,axes)

    def scatter_projs(self,fig=None,axes=None,xproj='proj_kfda',yproj=None,name=None,projections=[(i*2+1,i*2+2) for i in range(10)],suptitle=None,
                        highlight=None,color=None,kfda=False,kfda_ylim=None,trunc=None,kfda_title=None,spectrum=False,iterate_over='projections'):
        to_iterate = projections if iterate_over == 'projections' else color
        fig,axes = self.init_axes_projs(fig=fig,axes=axes,projections=to_iterate,suptitle=suptitle,kfda=kfda,
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

    def eval_nystrom_discriminant_axis(self,nystrom=1,t=None,m=None,test_data=False):
        # a adapter au GPU
        # j'ai pas du tout réfléchi à la version test_data, j'ai juste fait en sorte que ça marche au niveau des tailles de matrices donc il y a peut-être des erreurs de logique
        if test_data:
            n1_test,n2_test = (self.n1_test,self.n2_test)   
            ntot_test = n1_test+n2_test
        n1,n2 = (self.n1,self.n2)
        ntot = n1+n2
        m1,m2 = (self.nxanchors,self.nyanchors)
        mtot = m1+m2

        mtot=self.nxanchors + self.nyanchors
        ntot= self.n1 + self.n2
        
        t = 60   if t is None else t # pour éviter un calcul trop long # t= self.n1 + self.n2
        m = mtot if m is None else m
        
        kmn   = self.compute_nystrom_kmn(test_data=False)
        kmn_test   = self.compute_nystrom_kmn(test_data=True)
        kny   = self.compute_gram_matrix(nystrom=True)
        k     = self.compute_gram_matrix(nystrom=False,test_data=False)
        Pbiny = self.compute_bicentering_matrix(nystrom=True)
        Pbi   = self.compute_bicentering_matrix(nystrom=False,test_data=False)
        
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
        
        mn1_test    = -1/n1_test * torch.ones(n1_test, dtype=torch.float64) # , device=device)
        mn2_test    = 1/n2_test * torch.ones(n2_test, dtype=torch.float64) # , device=device) 
        m_ntot_test = torch.cat((mn1_test, mn2_test), dim=0) #.to(device)
        
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



    def eval_nystrom_eigenfunctions(self,t=None,m=None,test_data=True):
        # a adapter au GPU

        if test_data:
            n1_test,n2_test = (self.n1_test,self.n2_test)   
            ntot_test = n1_test+n2_test
        n1,n2 = (self.n1,self.n2)
        ntot = n1+n2
        m1,m2 = (self.nxanchors,self.nyanchors)
        mtot = m1+m2

        mtot=self.nxanchors + self.nyanchors
        ntot= self.n1 + self.n2
        
        t = 60   if t is None else t # pour éviter un calcul trop long # t= self.n1 + self.n2
        m = mtot if m is None else m
        
        kmn   = self.compute_nystrom_kmn(test_data=False)
        Pbiny = self.compute_bicentering_matrix(nystrom=True)
        Pbi   = self.compute_bicentering_matrix(nystrom=False)
        
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




