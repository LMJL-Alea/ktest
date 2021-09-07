from typing_extensions import Literal
from typing import Optional,Callable,Union,List

import numpy as np
from numpy.lib.function_base import kaiser
import pandas as pd
import torch
from torch import mv
import os
from time import time

import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.linalg import svd
from numpy.random import permutation
from sklearn.model_selection import train_test_split

# CCIPL 
# from kernels import gauss_kernel_mediane
# Local 
from ktest.kernels import gauss_kernel_mediane

from apt.eigen_wrapper import eigsy
import apt.kmeans # For kmeans
from kmeans_pytorch import kmeans


def ordered_eigsy(matrix):
    sp,ev = eigsy(matrix)
    order = sp.argsort()[::-1]
    ev = torch.tensor(ev[:,order],dtype=torch.float64) 
    sp = torch.tensor(sp[order], dtype=torch.float64)
    return(sp,ev)


# Choix à faire ( trouver les bonnes pratiques )

# voir comment ils gèrent les plot dans scanpy

# initialiser le mask aussi au moment de tracer des figures

# tracer l'evolution des corrélations par rapport aux gènes ordonnés

# ecrire les docstring de chaque fonction 

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

# def des fonction type get pour les opérations d'initialisation redondantes
# acces facile aux noms des dict de dataframes. 
# faire en sorte de pouvoir calculer corr kfda et kpca 

# mettre une limite globale a 100 pour les tmax des projections (éviter d'enregistrer des structures de données énormes)
# mieux gérer la projection de gènes et le param color
 
# verbosity devient aussi un verificateur de code 

# repenser les plots et df_init_proj

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
        
        # attributs initialisés 
        self.df_kfdat = pd.DataFrame()
        self.df_proj_kfda = {}
        self.df_proj_kpca = {}
        self.corr = {}     
        self.dict_mmd = {}
        self.spev = {'x':{},'y':{},'xy':{}} # dict containing every result of diagonalization
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

    def load_kfdat(self,path):
        df = pd.read_csv(path,header=0,index_col=0)
        for c in df.columns:
            if c not in self.df_kfdat.columns:
                self.df_kfdat[c] = df[c]
            else: 
                print(f'kfdat {c} already here')

    def load_proj_kfda(self,path,name,):
        df = pd.read_csv(path,header=0,index_col=0)
        if name not in self.df_proj_kfda:
            self.df_proj_kfda[name] = df
        else:
            print(f'proj kfda {name} already here')

    def load_proj_kpca(self,path,name):
        df = pd.read_csv(path,header=0,index_col=0)
        if name not in self.df_proj_kpca:
            self.df_proj_kpca[name] = df
        else:
            print(f'proj kpca {name} already here')

    def load_correlations(self,path,name):
        df = pd.read_csv(path,header=0,index_col=0)
        if name not in self.corr:
            self.corr[name] = df
        else:
            print(f'corr {name} already here')



    def save_kfdat(self,path):
        self.df_kfdat.to_csv(path,index=True)

    def save_proj_kfda(self,path,name):
        self.df_proj_kfda[name].to_csv(path,index=True)
    
    def save_proj_kpca(self,path,name):
        self.df_proj_kpca[name].to_csv(path,index=True)

    def save_correlations(self,path,name):
        self.corr[name].to_csv(path,index=True)

    def load_data(self,to_load):
        """
        to_load ={'kfdat':path_kfdat,
                    'proj_kfda':{name1:path1,name2,path2},
                    'proj_kpca':{name1:path1,name2,path2},
                    'correlations':{name1:path1,name2,path2}}
        """
        types_ref= {'proj_kfda':self.load_proj_kfda,
                    'proj_kpca':self.load_proj_kpca,
                    'correlations':self.load_correlations}
        
        for type in to_load:
            if type == 'kfdat':
                self.load_kfdat(to_load[type]) 
            else:
                for name,path in to_load[type].items():
                    types_ref[type](path,name)


    def get_names(self):
        names = {'kfdat':[c for c in self.df_kfdat],
                'proj_kfda':[name for name in self.df_proj_kfda.keys()],
                'proj_kpca':[name for name in self.df_proj_kpca.keys()],
                'correlations':[name for name in self.corr.keys()]}
        return(names)


    # def load_data(self,data_dict,):
    # def save_data():
    # def save_a_dataframe(self,path,which)

    def get_xy(self,landmarks=False):
        if landmarks: 
            x,y = self.xlandmarks,self.ylandmarks
        else:
            x,y = self.x[self.xmask,:],self.y[self.ymask,:]
        return(x,y)

    def compute_nystrom_landmarks(self,nlandmarks=None,landmarks_method=None,verbose=0):
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
            
        x,y = self.get_xy()
        n1,n2 = self.n1,self.n2 
        xratio,yratio = n1/(n1 + n2), n2/(n1 +n2)

        if nlandmarks is None:
            print("nlandmarks not specified, by default, nlandmarks = (n1+n2)//10")
            nlandmarks = (n1 + n2)//10

        if landmarks_method is None:
            print("landmarks_method not specified, by default, landmarks_method='random'")
            landmarks_method = 'random'
        
        self.nlandmarks = nlandmarks
        self.nxlandmarks=np.int(np.floor(xratio * nlandmarks)) 
        self.nylandmarks=np.int(np.floor(yratio * nlandmarks))

        

        if landmarks_method == 'kmeans':
            # self.xanchors,self.xassignations = apt.kmeans.spherical_kmeans(self.x[self.xmask,:], nxanchors, max_iter)
            # self.yanchors,self.yassignations = apt.kmeans.spherical_kmeans(self.y[self.ymask,:], nyanchors, max_iter)
            self.xassignations,self.xlandmarks = kmeans(X=x, num_clusters=self.nxlandmarks, distance='euclidean', tqdm_flag=False) #cuda:0')
            self.yassignations,self.ylandmarks = kmeans(X=y, num_clusters=self.nylandmarks, distance='euclidean', tqdm_flag=False) #cuda:0')
            self.xlandmarks = self.xlandmarks.double()
            self.ylandmarks = self.ylandmarks.double()
            self.quantization_with_landmarks_possible = True
            

        elif landmarks_method == 'random':
            self.xlandmarks = x[np.random.choice(x.shape[0], size=self.nxlandmarks, replace=False)]
            self.ylandmarks = y[np.random.choice(y.shape[0], size=self.nylandmarks, replace=False)]
            
            # Necessaire pour remettre a false au cas ou on a déjà utilisé 'kmeans' avant 
            self.quantization_with_landmarks_possible = False

        self.has_landmarks= True

        self.verbosity(function_name='compute_nystrom_landmarks',
                       dict_of_variables={'nlandmarks':nlandmarks,'landmarks_method':landmarks_method},
                       start=False,
                       verbose = verbose)

    def compute_quantization_weights(self,power=1,sample='xy',diag=True):
        if 'x' in sample:
            a1 = torch.tensor(torch.bincount(self.xassignations),dtype=torch.float64)**power
        if 'y' in sample:
            a2 = torch.tensor(torch.bincount(self.yassignations),dtype=torch.float64)**power
        
        if diag:
            if sample =='xy':
                return(torch.diag(torch.cat((a1,a2))).double())
            else:
                return(torch.diag(a1).double() if sample =='x' else torch.diag(a2).double())
        else:
            if sample=='xy':
                return(torch.cat(a1,a2))
            else:    
                return(a1 if sample =='x' else a2)

    def compute_nystrom_anchors(self,sample='xy',nanchors=None,verbose=0,center_anchors=False):
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

        if nanchors is None:
            print("nanchors not specified, by default, nanchors = nlandmarks" )

        if sample == 'xy':
            self.nanchors = self.nlandmarks if nanchors is None else nanchors
            assert(self.nanchors <= self.nlandmarks)
            
        elif sample =='x':
            self.nxanchors = self.nxlandmarks if nanchors is None else nanchors
            assert(self.nxanchors <= self.nxlandmarks)
        elif sample =='y':
            self.nyanchors = self.nylandmarks if nanchors is None else nanchors
            assert(self.nyanchors <= self.nylandmarks)

        nanchors = self.nanchors if sample =='xy' else self.nxanchors if sample=='x' else self.nyanchors
        
        Km = self.compute_gram(sample=sample,landmarks=True)
        if center_anchors:
            Pm = self.compute_centering_matrix(sample=sample,landmarks=True)
            sp_anchors,ev_anchors = ordered_eigsy(torch.chain_matmul(Pm,Km,Pm))        
        else:
            sp_anchors,ev_anchors = ordered_eigsy(Km)        
        
        self.spev[sample]['anchors'] = {'sp':sp_anchors[:nanchors],'ev':ev_anchors[:,:nanchors]}

        self.verbosity(function_name='compute_nystrom_anchors',
                        dict_of_variables={'nanchors':nanchors},
                        start=False,
                        verbose = verbose)

    def reinitialize_landmarks(self):
        if self.quantization_with_landmarks_possible:
            self.quantization_with_landmarks_possible = False
            delattr(self,'xassignations')
            delattr(self,'yassignations')
            for sample in ['x','y','xy']: 
                self.spev[sample].pop('quantization',None)

        if self.has_landmarks:
            self.has_landmarks = False
            delattr(self,'nlandmarks')
            delattr(self,'nxlandmarks')
            delattr(self,'nylandmarks')
            delattr(self,'xlandmarks')
            delattr(self,'ylandmarks')
        
    def reinitialize_anchors(self):
        for sample in ['x','y','xy']: 
            # self.spev[sample].pop('anchors',None)
            self.spev[sample].pop('nystrom',None)

    def compute_gram(self,sample='xy',landmarks=False): 
        """
        Computes Gram matrix, on anchors if nystrom is True, else on data. 
        This function is called everytime the Gram matrix is needed but I could had an option to keep it in memory in case of a kernel function 
        that makes it difficult to compute

        Returns
        -------
        torch.Tensor of size (nxanchors+nyanchors)**2 if nystrom else (n1+n2)**2
        """

        kernel = self.kernel
        
        x,y = self.get_xy(landmarks=landmarks)
        
        if 'x' in sample:
            kxx = kernel(x,x)
        if 'y' in sample:
            kyy = kernel(y,y)

        if sample == 'xy':
            kxy = kernel(x, y)
            return(torch.cat((torch.cat((kxx, kxy), dim=1),
                            torch.cat((kxy.t(), kyy), dim=1)), dim=0))
        else:
            return(kxx if sample =='x' else kyy)

    def compute_m(self,sample='xy',quantization=False):
        n1,n2 = (self.n1,self.n2)
        if sample =='xy':
            if quantization:
                return(torch.cat((-1/n1*torch.bincount(self.xassignations),1/n2*torch.bincount(self.yassignations))).double())
            else:
                m_mu1    = -1/n1 * torch.ones(n1, dtype=torch.float64) # , device=device)
                m_mu2    = 1/n2 * torch.ones(n2, dtype=torch.float64) # , device=device) 
                return(torch.cat((m_mu1, m_mu2), dim=0)) #.to(device)
        elif sample=='x':
            return(1/n1 * torch.ones(n1, dtype=torch.float64))
        elif sample=='y':
            return(1/n2 * torch.ones(n2, dtype=torch.float64))

    def compute_kmn(self,sample='xy'):
        """
        Computes an (nxanchors+nyanchors)x(n1+n2) conversion gram matrix
        """
        assert(self.has_landmarks)
        kernel = self.kernel
        
        x,y = self.get_xy()
        z1,z2 = self.get_xy(landmarks=True)
        if 'x' in sample:
            kz1x = kernel(z1,x)
        if 'y' in sample:
            kz2y = kernel(z2,y)
        
        if sample =='xy':
            kz2x = kernel(z2,x)
            kz1y = kernel(z1,y)
            return(torch.cat((torch.cat((kz1x, kz1y), dim=1),
                            torch.cat((kz2x, kz2y), dim=1)), dim=0))
        else:
            return(kz1x if sample =='x' else kz2y)

    def compute_centering_matrix(self,sample='xy',quantization=False,landmarks=False):
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

        if landmarks:
            n = self.nxlandmarks if sample=='x' else self.nylandmarks if sample=='y' else self.nlandmarks
            idn = torch.eye(n, dtype=torch.float64)
            onen = torch.ones(n, n, dtype=torch.float64)
            pn = idn - 1/n * onen
            return(pn)

        if 'x' in sample:
            n1 = self.nxlandmarks if quantization else self.n1 
            idn1 = torch.eye(n1, dtype=torch.float64)
            onen1 = torch.ones(n1, n1, dtype=torch.float64)
            if quantization: 
                a1 = self.compute_quantization_weights(sample='x',power=.5,diag=False)
                pn1 = (idn1 - 1/self.n2 * torch.ger(a1,a1))
                # A1 = self.compute_quantization_weights(sample='x')
                # pn1 = np.sqrt(self.n1/(self.n1+self.n2))*(idn1 - torch.matmul(A1,onen1))
            else:
                pn1 = idn1 - 1/n1 * onen1

        if 'y' in sample:
            n2 = self.nylandmarks if quantization else self.n2
            idn2 = torch.eye(n2, dtype=torch.float64)
            onen2 = torch.ones(n2, n2, dtype=torch.float64)
            if quantization: 
                a2 = self.compute_quantization_weights(sample='y',power=.5,diag=False)
                pn2 = (idn2 - 1/self.n2 * torch.ger(a2,a2))
                # A2 = self.compute_quantization_weights(sample='y')
                # pn2 = np.sqrt(self.n2/(self.n1+self.n2))*(idn2 - torch.matmul(A2,onen2))
            else:
                pn2 = idn2 - 1/n2 * onen2

        if sample == 'xy':
            z12 = torch.zeros(n1, n2, dtype=torch.float64)
            z21 = torch.zeros(n2, n1, dtype=torch.float64)
            return(torch.cat((torch.cat((pn1, z12), dim=1), torch.cat(
            (z21, pn2), dim=1)), dim=0))  # bloc diagonal
        else:
            return(pn1 if sample=='x' else pn2)  

    def compute_centered_gram(self,approximation='full',sample='xy',verbose=0,center_anchors=False):
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

        self.verbosity(function_name='compute_centered_gram',
                dict_of_variables={'approximation':approximation,
                                'sample':sample},
                start=True,
                verbose = verbose)    
        
        quantization = approximation == 'quantization'
        P = self.compute_centering_matrix(sample=sample,quantization=quantization).double()
        
        n=0
        if 'x' in sample:
            n1 = self.n1 
            n+=n1     
        if 'y' in sample:
            n2 = self.n2
            n+=n2
        
        if approximation == 'quantization':
            if self.quantization_with_landmarks_possible:
                Kmm = self.compute_gram(sample=sample,landmarks=True)
                A = self.compute_quantization_weights(sample=sample,power=.5)
                Kw = 1/n * torch.chain_matmul(P,A,Kmm,A,P)
            else:
                print("quantization impossible, you need to call 'compute_nystrom_landmarks' with landmarks_method='kmeans'")


        elif approximation == 'nystrom':
            # version brute mais a terme utiliser la svd ?? 
            if self.has_landmarks and "anchors" in self.spev[sample]:
                Kmn = self.compute_kmn(sample=sample)
                Lp_inv = torch.diag(self.spev[sample]['anchors']['sp']**(-1))
                Up = self.spev[sample]['anchors']['ev']
                if center_anchors:
                    Pm = self.compute_centering_matrix(sample='xy',landmarks=True)
                    Kw = 1/n * torch.chain_matmul(P,Kmn.T,Pm,Up,Lp_inv,Up.T,Pm,Kmn,P)            
                else:
                    Kw = 1/n * torch.chain_matmul(P,Kmn.T,Up,Lp_inv,Up.T,Kmn,P)
            
            else:
                print("nystrom impossible, you need compute landmarks and/or anchors")
        
        elif approximation == 'nystromnew':
            if self.has_landmarks and "anchors" in self.spev[sample]:
                Kmn = self.compute_kmn(sample=sample)
                Lp_inv_12 = torch.diag(self.spev[sample]['anchors']['sp']**(-(1/2)))
                Up = self.spev[sample]['anchors']['ev']

                if center_anchors:
                    Pm = self.compute_centering_matrix(sample='xy',landmarks=True)
                    Kw = 1/n * torch.chain_matmul(Lp_inv_12,Up.T,Pm,Kmn,P,Kmn.T,Pm,Up,Lp_inv_12)            
                else:
                    Kw = 1/n * torch.chain_matmul(Lp_inv_12,Up.T,Kmn,P,Kmn.T,Up,Lp_inv_12)

            else:
                print("nystrom new version impossible, you need compute landmarks and/or anchors")
                    

        elif approximation == 'full':
            K = self.compute_gram(landmarks=False,sample=sample)
            Kw = 1/n * torch.chain_matmul(P,K,P)

        self.verbosity(function_name='compute_centered_gram',
                dict_of_variables={'approximation':approximation,'sample':sample},
                start=False,
                verbose = verbose)    

        return Kw

    def diagonalize_centered_gram(self,approximation='full',sample='xy',verbose=0,center_anchors=False):
        """
        Diagonalizes the bicentered Gram matrix which shares its spectrum with the Withon covariance operator in the RKHS.
        Stores eigenvalues (sp or spny) and eigenvectors (ev or evny) as attributes
        """
        if approximation in self.spev[sample]:
            if verbose:
                print('No need to diagonalize')
        else:
            self.verbosity(function_name='diagonalize_centered_gram',
                    dict_of_variables={'approximation':approximation,
                    'sample':sample},
                    start=True,
                    verbose = verbose)
            
            Kw = self.compute_centered_gram(approximation=approximation,sample=sample,verbose=verbose,center_anchors=center_anchors)
            
            sp,ev = ordered_eigsy(Kw)
            self.spev[sample][approximation] = {'sp':sp,'ev':ev}
            
            self.verbosity(function_name='diagonalize_centered_gram',
                    dict_of_variables={'approximation':approximation,
                                     'sample':sample},
                    start=False,
                    verbose = verbose)
                
    def compute_kfdat(self,trunc=None,approximation_cov='full',approximation_mmd='full',name=None,verbose=0,center_anchors=False):
        # je n'ai plus besoin de trunc, seulement d'un t max 
        """ 
        Computes the kfda truncated statistic of [Harchaoui 2009].
        9 methods : 
        approximation_cov in ['full','nystrom','quantization']
        approximation_mmd in ['full','nystrom','quantization']
        
        Stores the result as a column in the dataframe df_kfdat
        """
        
        name = name if name is not None else f'{approximation_cov}{approximation_mmd}' 
        sp,ev = self.spev['xy'][approximation_cov]['sp'],self.spev['xy'][approximation_cov]['ev']
        t = len(sp)+1
        # t = tmax if (trunc is None or trunc[-1]>tmax) else trunc[-1]

        trunc = range(1,t)        
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
        Pbi = self.compute_centering_matrix(sample='xy',quantization=(approximation_cov=='quantization'))

        if 'nystrom' in [approximation_mmd,approximation_cov] or 'nystromnew' in [approximation_mmd,approximation_cov] :
            Up = self.spev['xy']['anchors']['ev']
            Lp_inv = torch.diag(self.spev['xy']['anchors']['sp']**-1)

        if not (approximation_mmd == approximation_cov) or approximation_mmd == 'nystrom':
            Kmn = self.compute_kmn(sample='xy')
        
        if approximation_cov == 'full':
            if approximation_mmd == 'full':
                K = self.compute_gram()
                pkm = mv(Pbi,mv(K,m))
                kfda = ((n1*n2)/(n**2*sp[:t]**2)*mv(ev.T[:t],pkm)**2).cumsum(axis=0).numpy() 

            elif approximation_mmd == 'nystrom':
                if center_anchors:
                    Pm = self.compute_centering_matrix(sample='xy',landmarks=True)
                    pkpuLupkm = mv(Pbi,mv(Kmn.T,mv(Pm,mv(Up,mv(Lp_inv,mv(Up.T,mv(Pm,mv(Kmn,m))))))))
                    kfda = ((n1*n2)/(n**2*sp[:t]**2)*mv(ev.T[:t],pkpuLupkm)**2).cumsum(axis=0).numpy() 

                else:
                    pkuLukm = mv(Pbi,mv(Kmn.T,mv(Up,mv(Lp_inv,mv(Up.T,mv(Kmn,m))))))
                    kfda = ((n1*n2)/(n**2*sp[:t]**2)*mv(ev.T[:t],pkuLukm)**2).cumsum(axis=0).numpy() 

            elif approximation_mmd == 'quantization':
                pkm = mv(Pbi,mv(Kmn.T,m))
                kfda = ((n1*n2)/(n**2*sp[:t]**2)*mv(ev.T[:t],pkm)**2).cumsum(axis=0).numpy() 
    
        if approximation_cov == 'nystrom':
            if approximation_mmd in ['full','nystrom']: # c'est exactement la même stat  
                if center_anchors:
                    Pm = self.compute_centering_matrix(sample='xy',landmarks=True)
                    pkpuLupkm = mv(Pbi,mv(Kmn.T,mv(Pm,mv(Up,mv(Lp_inv,mv(Up.T,mv(Pm,mv(Kmn,m))))))))
                    kfda = ((n1*n2)/(n**2*sp[:t]**2)*mv(ev.T[:t],pkpuLupkm)**2).cumsum(axis=0).numpy() 

                else:
                    pkuLukm = mv(Pbi,mv(Kmn.T,mv(Up,mv(Lp_inv,mv(Up.T,mv(Kmn,m))))))
                    kfda = ((n1*n2)/(n**2*sp[:t]**2)*mv(ev.T[:t],pkuLukm)**2).cumsum(axis=0).numpy() 

            elif approximation_mmd == 'quantization':
                Kmm = self.compute_gram(landmarks=True)
                pkuLukm = mv(Pbi,mv(Kmn.T,mv(Up,mv(Lp_inv,mv(Up.T,mv(Kmm,m))))))
                kfda = ((n1*n2)/(n**2*sp[:t]**2)*mv(ev.T[:t],pkuLukm)**2).cumsum(axis=0).numpy() 
        
        if approximation_cov == 'nystromnew':
            Lp12 = torch.diag(self.spev['xy']['anchors']['sp']**-(1/2))
            if approximation_mmd in ['full','nystrom']: # c'est exactement la même stat  
                if center_anchors:
                    Pm = self.compute_centering_matrix(sample='xy',landmarks=True)
                    LupkpkpuLupkm = mv(Lp12,mv(Up.T,mv(Pm,mv(Kmn,mv(Pbi,mv(Kmn.T,mv(Pm,mv(Up,mv(Lp_inv,mv(Up.T,mv(Pm,mv(Kmn,m))))))))))))
                    # pkpuLupkm = mv(Pbi,mv(Kmn.T,mv(Pm,mv(Up,mv(Lp_inv,mv(Up.T,mv(Pm,mv(Kmn,m))))))))
                    kfda = ((n1*n2)/(n**3*sp[:t]**3)*mv(ev.T[:t],LupkpkpuLupkm)**2).cumsum(axis=0).numpy() 

                else:
                    LupkpkpuLupkm = mv(Lp12,mv(Up.T,mv(Kmn,mv(Pbi,mv(Kmn.T,mv(Up,mv(Lp_inv,mv(Up.T,mv(Kmn,m)))))))))
                    kfda = ((n1*n2)/(n**3*sp[:t]**3)*mv(ev.T[:t],LupkpkpuLupkm)**2).cumsum(axis=0).numpy() 

            elif approximation_mmd == 'quantization':
                # il pourrait y avoir la dichotomie anchres centrees ou non ici. 
                Kmm = self.compute_gram(landmarks=True)
                LukpkuLukm = mv(Lp12,mv(Up.T,mv(Kmn,mv(Pbi,mv(Kmn.T,mv(Up,mv(Lp_inv,mv(Up.T,mv(Kmm,m)))))))))
                kfda = ((n1*n2)/(n**3*sp[:t]**3)*mv(ev.T[:t],LukpkuLukm)**2).cumsum(axis=0).numpy() 
        
        if approximation_cov == 'quantization':
            A_12 = self.compute_quantization_weights(power=1/2,sample='xy')
            if approximation_mmd == 'full':
                apkm = mv(Pbi,mv(A_12,mv(Kmn,m)))
                # le n n'est pas au carré ici
                kfda = ((n1*n2)/(n**2*sp[:t]**2)*mv(ev.T[:t],apkm)**2).cumsum(axis=0).numpy() 

            elif approximation_mmd == 'nystrom':
                Kmm = self.compute_gram(landmarks=True)
                apkuLukm = mv(Pbi,mv(A_12,mv(Kmm,mv(Up,mv(Lp_inv,mv(Up.T,mv(Kmn,m)))))))
                # le n n'est pas au carré ici
                kfda = ((n1*n2)/(n**2*sp[:t]**2)*mv(ev.T[:t],apkuLukm)**2).cumsum(axis=0).numpy() 

            elif approximation_mmd == 'quantization':
                Kmm = self.compute_gram(landmarks=True)
                apkm = mv(Pbi,mv(A_12,mv(Kmm,m)))
                kfda = ((n1*n2)/(n**2*sp[:t]**2)*mv(ev.T[:t],apkm)**2).cumsum(axis=0).numpy() 

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

    def initialize_kfdat(self,approximation_cov='full',approximation_mmd=None,sample='xy',nlandmarks=None,
                               nanchors=None,landmarks_method='random',verbose=0,center_anchors=False,**kwargs):
        # verbose -1 au lieu de verbose ? 
        cov,mmd = approximation_cov,approximation_mmd
        if 'quantization' in [cov,mmd] and not self.quantization_with_landmarks_possible: # besoin des poids des ancres de kmeans en quantization
            self.compute_nystrom_landmarks(nlandmarks=nlandmarks,landmarks_method='kmeans',verbose=verbose)
        
        if 'nystrom' in [cov,mmd] or 'nystromnew' in [cov,mmd] :
            if not self.has_landmarks:
                self.compute_nystrom_landmarks(nlandmarks=nlandmarks,landmarks_method=landmarks_method,verbose=verbose)
            if "anchors" not in self.spev[sample]:
                self.compute_nystrom_anchors(nanchors=nanchors,sample=sample,verbose=verbose,center_anchors=center_anchors)
            
        if cov not in self.spev[sample]:
            self.diagonalize_centered_gram(approximation=cov,sample=sample,verbose=verbose,center_anchors=center_anchors)

    def kfdat(self,trunc=None,approximation_cov='full',approximation_mmd='full',
                nlandmarks=None,nanchors=None,landmarks_method='random',
                name=None,verbose=0,center_anchors=False):
                
        cov,mmd = approximation_cov,approximation_mmd
        name = name if name is not None else f'{cov}{mmd}' 
        if name in self.df_kfdat :
            if verbose : 
                print(f'kfdat {name} already computed')
        else:
            self.initialize_kfdat(approximation_cov=cov,approximation_mmd=mmd,sample='xy',
                                nlandmarks=nlandmarks,nanchors=nanchors,landmarks_method=landmarks_method,
                                        verbose=verbose,center_anchors=center_anchors)            
            self.compute_kfdat(trunc=trunc,approximation_cov=cov,approximation_mmd=mmd,name=name,verbose=verbose,center_anchors=center_anchors)
        
        

    def compute_proj_kfda(self,trunc=None,approximation_cov='full',approximation_mmd='full',name=None,verbose=0):
        # je n'ai plus besoin de trunc, seulement d'un t max 
        """ 
        Projections of the embeddings of the observation onto the discriminant axis
        9 methods : 
        approximation_cov in ['full','nystrom','quantization']
        approximation_mmd in ['full','nystrom','quantization']
        
        Stores the result as a column in the dataframe df_proj_kfda
        """


        name = name if name is not None else f'{approximation_cov}{approximation_mmd}' 
        if name in self.df_proj_kfda :
            if verbose : 
                print('Proj on discriminant axis Already computed')
        else:

            sp,ev = self.spev['xy'][approximation_cov]['sp'],self.spev['xy'][approximation_cov]['ev']
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
            Pbi = self.compute_centering_matrix(sample='xy',quantization=(approximation_cov=='quantization'))

            if 'nystrom' in [approximation_mmd,approximation_cov]:
                Up = self.spev['xy']['anchors']['ev']
                Lp_inv = torch.diag(self.spev['xy']['anchors']['sp']**-1)

            if not (approximation_mmd == approximation_cov == 'full'):
                Kmn = self.compute_kmn()
            if approximation_cov == 'full':
                K = self.compute_gram()

            if approximation_cov == 'full':
                if approximation_mmd == 'full':
                    pkm = mv(Pbi,mv(K,m))
                    proj = (n**-1*sp[:t]**(-3/2)*mv(ev.T[:t],pkm)*torch.chain_matmul(ev.T[:t],Pbi,K).T).cumsum(axis=1).numpy()
                    
                elif approximation_mmd == 'nystrom':
                    pkuLukm = mv(Pbi,mv(Kmn.T,mv(Up,mv(Lp_inv,mv(Up.T,mv(Kmn,m))))))
                    proj = (n**-1*sp[:t]**(-3/2)*mv(ev.T[:t],pkuLukm)*torch.chain_matmul(ev.T[:t],Pbi,K).T).cumsum(axis=1).numpy()

                elif approximation_mmd == 'quantization':
                    pkm = mv(Pbi,mv(Kmn.T,m))
                    proj = (n**-1*sp[:t]**(-3/2)*mv(ev.T[:t],pkm)*torch.chain_matmul(ev.T[:t],Pbi,K).T).cumsum(axis=1).numpy()
                    
            if approximation_cov == 'nystrom':
                if approximation_mmd == 'full':
                    pkuLukm = mv(Pbi,mv(Kmn.T,mv(Up,mv(Lp_inv,mv(Up.T,mv(Kmn,m))))))
                    proj = (n**-1*sp[:t]**(-3/2)*mv(ev.T[:t],pkuLukm)*torch.chain_matmul(ev.T[:t],Pbi,Kmn.T,Up,Lp_inv,Up.T,Kmn).T).cumsum(axis=1).numpy()

                elif approximation_mmd == 'nystrom':
                    pkuLukm = mv(Pbi,mv(Kmn.T,mv(Up,mv(Lp_inv,mv(Up.T,mv(Kmn,m))))))
                    proj = (n**-1*sp[:t]**(-3/2)*mv(ev.T[:t],pkuLukm)*torch.chain_matmul(ev.T[:t],Pbi,Kmn.T,Up,Lp_inv,Up.T,Kmn).T).cumsum(axis=1).numpy()

                elif approximation_mmd == 'quantization':
                    Kmm = self.compute_gram(landmarks=True)
                    pkuLukm = mv(Pbi,mv(Kmn.T,mv(Up,mv(Lp_inv,mv(Up.T,mv(Kmm,m))))))
                    proj = (n**-1*sp[:t]**(-3/2)*mv(ev.T[:t],pkuLukm)*torch.chain_matmul(ev.T[:t],Pbi,Kmn.T,Up,Lp_inv,Up.T,Kmn).T).cumsum(axis=1).numpy()

            if approximation_cov == 'quantization':
                A_12 = self.compute_quantization_weights(power=1/2,sample='xy')
                if approximation_mmd == 'full':
                    apkm = mv(A_12,mv(Pbi,mv(Kmn,m)))
                    # pas de n ici
                    proj = (sp[:t]**(-3/2)*mv(ev.T[:t],apkm)*torch.chain_matmul(ev.T[:t],A_12,Pbi,Kmn).T).cumsum(axis=1).numpy()

                elif approximation_mmd == 'nystrom':
                    Kmm = self.compute_gram(landmarks=True)
                    apkuLukm = mv(A_12,mv(Pbi,mv(Kmm,mv(Up,mv(Lp_inv,mv(Up.T,mv(Kmn,m)))))))
                    # pas de n ici
                    proj = (sp[:t]**(-3/2)*mv(ev.T[:t],apkuLukm)*torch.chain_matmul(ev.T[:t],A_12,Pbi,Kmn).T).cumsum(axis=1).numpy()

                elif approximation_mmd == 'quantization':
                    Kmm = self.compute_gram(landmarks=True)
                    apkm = mv(A_12,mv(Pbi,mv(Kmm,m)))
                    proj = (sp[:t]**(-3/2)*mv(ev.T[:t],apkm)*torch.chain_matmul(ev.T[:t],A_12,Pbi,Kmn).T).cumsum(axis=1).numpy()

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

    def compute_proj_kpca(self,trunc=None,approximation_cov='full',sample='xy',name=None,verbose=0):
        # je n'ai plus besoin de trunc, seulement d'un t max 
        """ 
        
        """
        name = name if name is not None else f'{approximation_cov}{sample}' 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        quantization = approximation_cov =='quantization'
        sp,ev = self.spev[sample][approximation_cov]['sp'],self.spev[sample][approximation_cov]['ev']
        P = self.compute_centering_matrix(sample=sample,quantization=quantization)    
        n1,n2 = (self.n1,self.n2) 
        n = (n1*('x' in sample)+n2*('y' in sample))
        
        tmax = len(sp)+1
        t = tmax if (trunc is None or trunc[-1]>tmax) else trunc[-1]
        trunc = range(1,tmax) if (trunc is None or trunc[-1]>tmax) else trunc


        self.verbosity(function_name='compute_proj_kpca',
                dict_of_variables={
                't':t,
                'approximation_cov':approximation_cov,
                'sample':sample,
                'name':name},
                start=True,
                verbose = verbose)

    
        if approximation_cov =='quantization':
            Kmn = self.compute_kmn(sample=sample)
            A_12 = self.compute_quantization_weights(sample=sample,power=1/2)                
            proj = ( sp[:t]**(-1/2)*torch.chain_matmul(ev.T[:t],A_12,P,Kmn).T)
        elif approximation_cov == 'nystrom':
            Kmn = self.compute_kmn(sample=sample)
            Up = self.spev[sample]['anchors']['ev']
            Lp_inv = torch.diag(self.spev[sample]['anchors']['sp']**-1)
            proj = (  n**(-1/2)*sp[:t]**(-1/2)*torch.chain_matmul(ev.T[:t],P,Kmn.T,Up,Lp_inv,Up.T,Kmn).T).cumsum(axis=1).numpy()
        elif approximation_cov == 'full':
            K = self.compute_gram(sample=sample)
            proj = (  n**(-1/2)*sp[:t]**(-1/2)*torch.chain_matmul(ev.T[:t],P,K).T).numpy()

    
        if name in self.df_proj_kpca:
            print(f"écrasement de {name} dans df_proj_kfda")
        
        index = self.index[self.imask] if sample=='xy' else self.x_index[self.xmask] if sample =='x' else self.y_index[self.ymask]
        self.df_proj_kpca[name] = pd.DataFrame(proj,index=index,columns=[str(t) for t in trunc])
        self.df_proj_kpca[name]['sample'] = ['x']*n1*('x' in sample) + ['y']*n2*('y' in sample)
                
        self.verbosity(function_name='compute_proj_kpca',
                                dict_of_variables={
                't':t,
                'approximation_cov':approximation_cov,
                'sample':sample,
                'name':name},
                start=False,
                verbose = verbose)

    def kpca(self,trunc=None,approximation_cov='full',sample='xy',name=None,verbose=0):
        
        cov = approximation_cov
        name = name if name is not None else f'{cov}{sample}' 
        if name in self.df_proj_kpca :
            if verbose : 
                print(f'kfdat {name} already computed')
        else:
            self.initialize_kfda(approximation_cov=cov,sample=sample,verbose=verbose)            
            self.compute_proj_kpca(trunc=trunc,approximation_cov=cov,sample=sample,name=name,verbose=verbose)
        


    def compute_corr_proj_var(self,trunc=None,sample='xy',which='proj_kfda',name_corr=None,
                            name_proj=None,prefix_col='',verbose=0): 
            # df_array,df_proj,csvfile,pathfile,trunc=range(1,60)):
        
        self.verbosity(function_name='compute_corr_proj_var',
                dict_of_variables={'trunc':trunc,
                            'sample':sample,'which':which,'name_corr':name_corr,'name_proj':name_proj,'prefix_col':prefix_col},
                start=True,
                verbose = verbose)

        self.prefix_col=prefix_col

        df_proj= self.init_df_proj(which,name_proj)
        if trunc is None:
            trunc = range(1,df_proj.shape[1] - 1) # -1 pour la colonne sample

        x,y = self.get_xy()

        array = torch.cat((x,y),dim=0).numpy() if sample == 'xy' else x.numpy() if sample=='x' else y.numpy()
        index = self.index[self.imask] if sample=='xy' else self.x_index[self.xmask] if sample =='x' else self.y_index[self.ymask]
        
        df_array = pd.DataFrame(array,index=index,columns=self.variables)
        for t in trunc:
            df_array[f'{prefix_col}{t}'] = pd.Series(df_proj[f'{t}'])
        name_corr = name_corr if name_corr is not None else which.split(sep='_')[1]+name_proj if name_proj is not None else which.split(sep='_')[1] + covariance
        self.corr[name_corr] = df_array.corr().loc[self.variables,[f'{prefix_col}{t}' for t in trunc]]
        
        self.verbosity(function_name='compute_corr_proj_var',
                dict_of_variables={'trunc':trunc,'sample':sample,'which':which,'name_corr':name_corr,'name_proj':name_proj,'prefix_col':prefix_col},
                start=False,
                verbose = verbose)


    def compute_mmd(self,unbiaised=False,approximation='full',shared_anchors=True,name=None,verbose=0):
        
        self.verbosity(function_name='compute_mmd',
                dict_of_variables={'unbiaised':unbiaised,
                                   'approximation':approximation,
                                   'shared_anchors':shared_anchors,
                                   'name':name},
                start=True,
                verbose = verbose)

        if approximation == 'full':
            m = self.compute_m(sample='xy',quantization=False)
            K = self.compute_gram()
            if unbiaised:
                K.masked_fill_(torch.eye(K.shape[0],K.shape[0]).byte(), 0)
            mmd = torch.dot(mv(K,m),m)**2
        
        if approximation == 'nystrom' and shared_anchors:
            m = self.compute_m(sample='xy',quantization=False)
            Up = self.spev['xy']['anchors']['ev']
            Lp_inv2 = torch.diag(self.spev['xy']['anchors']['sp']**-(1/2))
            Pm = self.compute_centering_matrix(sample='xy',landmarks=True)
            Kmn = self.compute_kmn(sample='xy')
            psi_m = mv(Lp_inv2,mv(Up.T,mv(Pm,mv(Kmn,m))))
            mmd = torch.dot(psi_m,psi_m)**2
        
        if approximation == 'nystrom' and not shared_anchors:
            
            mx = self.compute_m(sample='x',quantization=False)
            my = self.compute_m(sample='y',quantization=False)
            Upx = self.spev['x']['anchors']['ev']
            Upy = self.spev['y']['anchors']['ev']
            Lpx_inv2 = torch.diag(self.spev['x']['anchors']['sp']**-(1/2))
            Lpy_inv2 = torch.diag(self.spev['y']['anchors']['sp']**-(1/2))
            Lpy_inv = torch.diag(self.spev['y']['anchors']['sp']**-1)
            Pmx = self.compute_centering_matrix(sample='x',landmarks=True)
            Pmy = self.compute_centering_matrix(sample='y',landmarks=True)
            Kmnx = self.compute_kmn(sample='x')
            Kmny = self.compute_kmn(sample='y')
            
            Km = self.compute_gram(sample='xy',landmarks=True)
            m1 = Kmnx.shape[0]
            m2 = Kmny.shape[0]
            Kmxmy = Km[:m1,m2:]

            psix_mx = mv(Lpx_inv2,mv(Upx.T,mv(Pmx,mv(Kmnx,mx))))
            psiy_my = mv(Lpy_inv2,mv(Upy.T,mv(Pmy,mv(Kmny,my))))
            Cpsiy_my = mv(Lpx_inv2,mv(Upx.T,mv(Pmx,mv(Kmxmy,\
                mv(Pmy,mv(Upy,mv(Lpy_inv,mv(Upy.T,mv(Pmy,mv(Kmny,my))))))))))
            mmd = torch.dot(psix_mx,psix_mx)**2 + torch.dot(psiy_my,psiy_my)**2 - 2*torch.dot(psix_mx,Cpsiy_my)
        
        if approximation == 'quantization':
            mq = self.compute_m(sample='xy',quantization=True)
            Km = self.compute_gram(sample='xy',landmarks=True)
            mmd = torch.dot(mv(Km,mq),mq)**2


        if name is None:
            name=f'{approximation}'
            if approximation == 'nystrom':
                name += 'shared' if shared_anchors else 'diff'
        
        self.dict_mmd[name] = mmd.item()
        
        self.verbosity(function_name='compute_mmd',
                dict_of_variables={'unbiaised':unbiaised,
                                   'approximation':approximation,
                                   'shared_anchors':shared_anchors,
                                   'name':name},
                start=False,
                verbose = verbose)

    def initialize_mmd(self,approximation='full',shared_anchors=True,nlandmarks=None,
                               nanchors=None,landmarks_method='random',verbose=0,center_anchors=False):
    
        """
        Calculs preliminaires pour lancer le MMD.
        approximation: determine les calculs a faire en amont du calcul du mmd
                    full : aucun calcul en amont puisque la Gram et m seront calcules dans mmd
                    nystrom : 
                            si il n'y a pas de landmarks deja calcules, on calcule nloandmarks avec la methode landmarks_method
                            si shared_anchors = True, alors on calcule un seul jeu d'ancres de taille nanchors pour les deux echantillons
                            si shared_anchors = False, alors on determine un jeu d'ancre par echantillon de taille nanchors//2
                                        attention : le parametre nanchors est divise par 2 pour avoir le meme nombre total d'ancres, risque de poser probleme si les donnees sont desequilibrees
                    quantization : nlandmarks sont determines comme les centroides de l'algo kmeans 
        shared_anchors : si approximation='nystrom' alors shared anchors determine si les ancres sont partagees ou non
        nlandmarks : nombre de landmarks a calculer si approximation='nystrom' ou 'kmeans'
        landmarks_method : dans ['random','kmeans'] methode de choix des landmarks
        verbose : booleen, vrai si les methodes appellees renvoies des infos sur ce qui se passe.  
        """
            # verbose -1 au lieu de verbose ? 

        if approximation == 'quantization' and not self.quantization_with_landmarks_possible: # besoin des poids des ancres de kmeans en quantization
            self.compute_nystrom_landmarks(nlandmarks=nlandmarks,landmarks_method='kmeans',verbose=verbose)
        
        if approximation == 'nystrom':
            if not self.has_landmarks:
                    self.compute_nystrom_landmarks(nlandmarks=nlandmarks,landmarks_method=landmarks_method,verbose=verbose)
            
            if shared_anchors:
                if "anchors" not in self.spev['xy']:
                    self.compute_nystrom_anchors(nanchors=nanchors,sample='xy',verbose=verbose,center_anchors=center_anchors)
            else:
                for xy in 'xy':
                    if 'anchors' not in self.spev[xy]:
                        assert(nanchors is not None,"nanchors not specified")
                        self.compute_nystrom_anchors(nanchors=nanchors//2,sample=xy,verbose=verbose,center_anchors=center_anchors)

    def mmd(self,approximation='full',shared_anchors=True,nlandmarks=None,
                               nanchors=None,landmarks_method='random',name=None,unbiaised=False,verbose=0):
        """
        appelle la fonction initialize mmd puis la fonction compute_mmd si le mmd n'a pas deja ete calcule. 
        """
        if name is None:
            name=f'{approximation}'
            if approximation == 'nystrom':
                name += 'shared' if shared_anchors else 'diff'
        
        if name in self.dict_mmd :
            if verbose : 
                print(f'mmd {name} already computed')
        else:
            self.initialize_mmd(approximation=approximation,shared_anchors=shared_anchors,
                    nlandmarks=nlandmarks,nanchors=nanchors,landmarks_method=landmarks_method,verbose=verbose)
            self.compute_mmd(approximation=approximation,shared_anchors=shared_anchors,
                            name=name,unbiaised=unbiaised,verbose=0)
        
       

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



    # def test(self,trunc=None,which_dict=['kfdat','proj_kfda','proj_kpca','corr','mmd'],
    #          nystrom=False,nanchors=None,nystrom_method='kmeans',
    #          name=None,main=False,corr_which='proj_kfda',corr_prefix_col='',obs_to_ignore=None,mmd_unbiaised=False,save=False,verbose=0):

    #     # for output,path in which.items()
    #     name_ = "main" if not hasattr(self,'main_name') and name is None else \
    #             self.name_generator(trunc=trunc,nystrom=nystrom,nystrom_method=nystrom_method,nanchors=nanchors,
    #             obs_to_ignore=obs_to_ignore) if name is None else \
    #             name


    #     if main or not hasattr(self,'main_name'):
    #         self.main_name = name_
        
    #     if verbose >0:
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
        
        # loaded = []    
        # if save:
        #     if 'kfdat' in which_dict and os.path.isfile(which_dict['kfdat']):
        #         loaded_kfdat = pd.read_csv(which_dict['kfdat'],header=0,index_col=0)
        #         if len(loaded_kfdat.columns)==1 and name is not None:
        #             c= loaded_kfdat.columns[0]
        #             self.df_kfdat[name] = loaded_kfdat[c]
        #         else:
        #             for c in loaded_kfdat.columns:
        #                 if c not in self.df_kfdat.columns:
        #                     self.df_kfdat[c] = loaded_kfdat[c]
        #         loaded += ['kfdat']

        #     if 'proj_kfda' in which_dict and os.path.isfile(which_dict['proj_kfda']):
        #         self.df_proj_kfda[name_] = pd.read_csv(which_dict['proj_kfda'],header=0,index_col=0)
        #         loaded += ['proj_kfda']

        #     if 'proj_kpca' in which_dict and os.path.isfile(which_dict['proj_kpca']):
        #         self.df_proj_kpca[name_] = pd.read_csv(which_dict['proj_kpca'],header=0,index_col=0)
        #         loaded += ['proj_kpca']
            
        #     if 'corr' in which_dict and os.path.isfile(which_dict['corr']):
        #         self.corr[name_] =pd.read_csv(which_dict['corr'],header=0,index_col=0)
        #         loaded += ['corr']

            # if 'mmd' in which_dict and os.path.isfile(which_dict['mmd']):
            #     self.mmd[name_] =pd.read_csv(which_dict['mmd'],header=0,index_col=0)
            #     loaded += ['mmd']


        #     if verbose >0:
        #         print('loaded:',loaded)

        # if len(loaded) < len(which_dict):
            
        #     missing = [k for k in which_dict.keys() if k not in loaded]
        #     # try:

            
        #     self.ignore_obs(obs_to_ignore=obs_to_ignore)
            
        #     if any([m in ['kfdat','proj_kfda','proj_kpca'] for m in missing]):
        #         if nystrom:
        #             self.nystrom_method = nystrom_method
        #             self.compute_nystrom_anchors(nanchors=nanchors,nystrom_method=nystrom_method,verbose=verbose,center_anchors=center_anchors) # max_iter=1000,

        #         if 'kfdat' in missing and nystrom==3 and not hasattr(self,'sp'):
        #             self.diagonalize_bicentered_gram(nystrom=False,verbose=verbose)
        #         else:
        #             self.diagonalize_bicentered_gram(nystrom,verbose=verbose)

        #     if 'kfdat' in which_dict and 'kfdat' not in loaded:
        #         self.compute_kfdat(trunc=trunc,nystrom=nystrom,name=name_,verbose=verbose)  
        #         loaded += ['kfdat']
        #         if save and obs_to_ignore is None:
        #             self.df_kfdat.to_csv(which_dict['kfdat'],index=True)    
    
        #     if 'proj_kfda' in which_dict and 'proj_kfda' not in loaded:
        #         self.compute_proj_kfda(trunc=trunc,nystrom=nystrom,name=name_,verbose=verbose)    
        #         loaded += ['proj_kfda']
        #         if save and obs_to_ignore is None:
        #             self.df_proj_kfda[name_].to_csv(which_dict['proj_kfda'],index=True)

        #     if 'proj_kpca' in which_dict and 'proj_kpca' not in loaded:
        #         self.compute_proj_kpca(trunc=trunc,nystrom=nystrom,name=name_,verbose=verbose)    
        #         loaded += ['proj_kpca']
        #         if save and obs_to_ignore is None:
        #             self.df_proj_kpca[name_].to_csv(which_dict['proj_kpca'],index=True)
            
        #     if 'corr' in which_dict and 'corr' not in loaded:
        #         self.compute_corr_proj_var(trunc=trunc,nystrom=nystrom,which=corr_which,name_corr=name_,prefix_col=corr_prefix_col,verbose=verbose)
        #         loaded += ['corr']
        #         if save and obs_to_ignore is None:
        #             self.corr[name_].to_csv(which_dict['corr'],index=True)
            
        #     if 'mmd' in which_dict and 'mmd' not in loaded:
        #         self.compute_mmd(unbiaised=mmd_unbiaised,nystrom=nystrom,name=name_,verbose=verbose)
        #         loaded += ['mmd']
        #         if save and obs_to_ignore is None:
        #             self.corr[name_].to_csv(which_dict['mmd'],index=True)
            
        #     if verbose>0:
        #         print('computed:',missing)
        #     if obs_to_ignore is not None:
        #         self.unignore_obs()
        
            # except:
            #     print('No computed')        

    def ignore_obs(self,obs_to_ignore=None,reinitialize_ignored_obs=False):
        
        if self.ignored_obs is None or reinitialize_ignored_obs:
            self.ignored_obs = obs_to_ignore
        else:
            self.ignored_obs.append(obs_to_ignore)

        if obs_to_ignore is not None:
            x,y = self.get_xy()
            self.xmask = ~self.x_index.isin(obs_to_ignore)
            self.ymask = ~self.y_index.isin(obs_to_ignore)
            self.imask = ~self.index.isin(obs_to_ignore)
            self.n1 = x.shape[0]
            self.n2 = y.shape[0]

    def unignore_obs(self):
        
        self.xmask = [True]*len(self.x)
        self.ymask = [True]*len(self.y)
        self.imask = [True]*len(self.index)
        self.n1 = self.x.shape[0]
        self.n2 = self.y.shape[0]
    
    def infer_nobs(self,which ='proj_kfda',name=None):
        if not hasattr(self,'n1'):
            if name is None:
                name = self.get_names()[which][0]
            df_proj= self.init_df_proj(which,name)
            self.n1 =  df_proj[df_proj['sample']=='x'].shape[0]
            self.n2 =  df_proj[df_proj['sample']=='y'].shape[0]

    def plot_kfdat(self,ax=None,ylim=None,figsize=(10,10),trunc=None,columns=None,asymp_ls='--',asymp_c = 'crimson',title=None,title_fontsize=40,highlight=False,highlight_main=False,mean=False,mean_label='mean',mean_color = 'xkcd: indigo'):
            
        # try:
            if columns is None:
                columns = self.df_kfdat.columns
            kfdat = self.df_kfdat[columns].copy()
            
            # if self.main_name in columns and highlight_main and len(columns)>1:
            #     kfdat[self.main_name].plot(ax=ax,lw=4,c='black')
            #     kfdat = kfdat.drop(columns=self.main_name) 
            # elif highlight and len(columns)==1:
            #     kfdat.plot(ax=ax,lw=4,c='black')
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
        
    def plot_spectrum(self,ax=None,figsize=(10,10),trunc=None,title=None,generate_spectrum=True,approximation_cov='full',sample='xy'):
        if ax is None:
            fig,ax = plt.subplots(figsize=figsize)
        if title is not None:
            ax.set_title(title,fontsize=40)
        sp = self.spev[sample][approximation_cov]['sp']
        if trunc is None:
            trunc = range(1,len(sp))
        ax.plot(trunc,sp[:trunc[-1]])
        ax.set_xlabel('t',fontsize= 20)

        return(ax)

    def density_proj(self,ax,projection,which='proj_kfda',name=None,orientation='vertical'):
        
        df_proj= self.init_df_proj(which,name)

        for xy,l in zip('xy','CT'):
            
            dfxy = df_proj.loc[df_proj['sample']==xy][str(projection)]
            if len(dfxy)>0:
                color = 'blue' if xy =='x' else 'orange'
                bins=int(np.floor(np.sqrt(len(dfxy))))
                ax.hist(dfxy,density=True,histtype='bar',label=f'{l}({len(dfxy)})',alpha=.3,bins=bins,color=color,orientation=orientation)
                ax.hist(dfxy,density=True,histtype='step',bins=bins,lw=3,edgecolor=color,orientation=orientation)
                if orientation =='vertical':
                    ax.axvline(dfxy.mean(),c=color)
                else:
                    ax.axhline(dfxy.mean(),c=color)

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
                        x,y = self.get_xy()
                        c = x[:,self.variables.get_loc(color)] if xy=='x' else y[:,self.variables.get_loc(color)]   
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
            # if name is None and self.main_name not in names:
            #     print("the default name {self.main_name} is not in {names} so you need to specify 'name' argument")
            # if name is None and self.main_name in names:
                df_proj = dict_df_proj[self.main_name]
            else: 
                df_proj = dict_df_proj[name]

        return(df_proj)

    def init_axes_projs(self,fig,axes,projections,approximation_cov,sample,suptitle,kfda,kfda_ylim,trunc,kfda_title,spectrum):
        if axes is None:
            rows=1;cols=len(projections) + kfda + spectrum
            fig,axes = plt.subplots(nrows=rows,ncols=cols,figsize=(6*cols,6*rows))
        if suptitle is not None:
            fig.suptitle(suptitle,fontsize=50)
        if kfda:
            self.plot_kfdat(axes[0],ylim=kfda_ylim,trunc = trunc,title=kfda_title)
            axes = axes[1:]
        if spectrum:
            self.plot_spectrum(axes[0],trunc=trunc,title='spectrum',approximation_cov=approximation_cov,sample=sample)
            axes = axes[1:]
        return(fig,axes)

    def density_projs(self,fig=None,axes=None,which='proj_kfda',approximation_cov='full',sample='xy',name=None,projections=range(1,10),suptitle=None,kfda=False,kfda_ylim=None,trunc=None,kfda_title=None,spectrum=False):
        fig,axes = self.init_axes_projs(fig=fig,axes=axes,projections=projections,approximation_cov=approximation_cov,sample=sample,suptitle=suptitle,kfda=kfda,
                                        kfda_ylim=kfda_ylim,trunc=trunc,kfda_title=kfda_title,spectrum=spectrum)
        if not isinstance(axes,np.ndarray):
            axes = [axes]
        for ax,proj in zip(axes,projections):
            self.density_proj(ax,proj,which=which,name=name)
        return(fig,axes)

    def scatter_projs(self,fig=None,axes=None,xproj='proj_kfda',approximation_cov='full',sample='xy',yproj=None,name=None,projections=[(1,i+2) for i in range(10)],suptitle=None,
                        highlight=None,color=None,kfda=False,kfda_ylim=None,trunc=None,kfda_title=None,spectrum=False,iterate_over='projections'):
        to_iterate = projections if iterate_over == 'projections' else color
        fig,axes = self.init_axes_projs(fig=fig,axes=axes,projections=to_iterate,approximation_cov=approximation_cov,sample=sample,suptitle=suptitle,kfda=kfda,
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
            name = self.get_names()['correlations'][0]
        if nvar==0:
            return(np.abs(self.corr[name][f'{prefix_col}{t}']).sort_values(ascending=False)[:])
        else: 
            return(np.abs(self.corr[name][f'{prefix_col}{t}']).sort_values(ascending=False)[:nvar])
        
    def plot_correlation_proj_var(self,ax=None,name=None,figsize=(10,10),nvar=30,projections=range(1,10),title=None,prefix_col=''):
        if name is None:
            name = self.get_names()['correlations'][0]
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
        kny   = self.compute_gram(landmarks=True)
        k     = self.compute_gram(landmarks=False,test_data=False)
        Pbiny = self.compute_centering_matrix(sample='xy',quantization=True)
        Pbi   = self.compute_centering_matrix(sample='xy',quantization=False,test_data=False)
        
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
        
        vpkm    = mv(torch.chain_matmul(V,Pbi,k),m_ntot)
        vpkm_ny = mv(torch.chain_matmul(Vny,Pbiny,kmn_test),m_ntot_test) if nystrom==1 else \
                  mv(torch.chain_matmul(Vny,Pbiny,kny),m_mtot)

        norm_h   = (ntot**-1 * sp**-2   * mv(torch.chain_matmul(V,Pbi,k),m_ntot)**2).sum()**(1/2)
        norm_hny = (mtot**-1 * spny**-2 * mv(torch.chain_matmul(Vny,Pbiny,kmn_test),m_ntot_test)**2).sum()**(1/2) if nystrom==1 else \
                   (mtot**-1 * spny**-2 * mv(torch.chain_matmul(Vny,Pbiny,kny),m_mtot)**2).sum()**(1/2)

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
        Pbiny = self.compute_centering_matrix(sample='xy',quantization=True)
        Pbi   = self.compute_centering_matrix(sample='xy',quantization=False)
        
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




