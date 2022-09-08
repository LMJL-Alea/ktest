from .utils import ordered_eigsy
import numpy as np
import torch

# import apt.kmeans 
from kmeans_pytorch import kmeans


from ktest.base import Data,Model
from ktest.outliers_operations import OutliersOps
from ktest.verbosity import Verbosity


"""
Ces fonctions déterminent permettent de déterminer les landmarks puis les ancres dans le cas de l'utilisation
de la méthode de nystrom. 
"""
class NystromOps(Model,OutliersOps,Verbosity):
    def __init__(self):
        super(NystromOps,self).__init__()

    def compute_nystrom_landmarks(self,outliers_in_obs=None,verbose=0):
        # anciennement compute_nystrom_anchors(self,r,nystrom_method='kmeans',split_data=False,test_size=.8,on_other_data=False,x_other=None,y_other=None,verbose=0): # max_iter=1000, (pour kmeans de François)
        
        """
        Problème des landmarks : on perd l'info 

        We distinguish the nystrom landmarks and nystrom anchors.
        The nystrom landmarks are the points from the observation space that will be used to compute the nystrom anchors. 
        The nystrom anchors are the points in the RKHS computed from the nystrom landmarks on which we will apply the nystrom method. 

        Parameters
        ----------
        m:    (1/10 * n if None)  number of landmarks in total (proportionnaly sampled according to the data)
        landmark_method: 'kmeans' or 'random' (in the future : 'kmeans ++, greedy...)
        """

        self.verbosity(function_name='compute_nystrom_landmarks',
                        dict_of_variables={'m':self.m,'landmark_method':self.landmark_method},
                        start=True,
                        verbose = verbose)
            
        landmarks_name = 'landmarks' if outliers_in_obs is None else f'landmarks{outliers_in_obs}'
        x,y = self.get_xy(outliers_in_obs=outliers_in_obs)
        n1,n2,n = self.get_n1n2n(outliers_in_obs=outliers_in_obs)
        xratio,yratio = n1/n, n2/n

        if self.m is None:
            if verbose >0:
                print("m not specified, by default, m = (n1+n2)//10")
            self.m = n//10

        if self.landmark_method is None:
            if verbose >0:
                print("landmark_method not specified, by default, landmark_method='random'")
            self.landmark_method = 'random'
        
        nxlandmarks=np.int(np.floor(xratio * self.m)) 
        nylandmarks=np.int(np.floor(yratio * self.m))
        

        if self.landmark_method == 'kmeans':
            xassignations,xlandmarks = kmeans(X=x, num_clusters=nxlandmarks, distance='euclidean', tqdm_flag=False) #cuda:0')
            yassignations,ylandmarks = kmeans(X=y, num_clusters=nylandmarks, distance='euclidean', tqdm_flag=False) #cuda:0')
            xlandmarks = xlandmarks.double()
            ylandmarks = ylandmarks.double()

            self.data['x'][landmarks_name] = {'X':xlandmarks,'assignations':xassignations,'n':nxlandmarks}
            self.data['y'][landmarks_name] = {'X':ylandmarks,'assignations':yassignations,'n':nylandmarks}

            self.quantization_with_landmarks_possible = True
            

        elif self.landmark_method == 'random':
            
            z1 = np.random.choice(n1, size=nxlandmarks, replace=False)
            z2 = np.random.choice(n2, size=nylandmarks, replace=False)
            
            xindex = self.get_index(sample='x')
            yindex = self.get_index(sample='y')

            self.add_outliers_in_obs(xindex[z1],f'x{landmarks_name}')
            self.add_outliers_in_obs(yindex[z2],f'y{landmarks_name}')

            # on pourrait s'arrêter là et construire la matrice des landmarks a chaque fois qu'on appelle get_xy
            # et calculer facilement nlandmarks quand on appelle get_n1n2n. 
            # mais ça se généralise mal à l'approche kmeans donc pour l'instant je garde les deux versions.  

            xlandmarks = x[z1]
            ylandmarks = y[z2]
            
            self.data['x'][landmarks_name] = {'X':xlandmarks,'n':nxlandmarks}
            self.data['y'][landmarks_name] = {'X':ylandmarks,'n':nylandmarks}

            # Necessaire pour remettre a false au cas ou on a déjà utilisé 'kmeans' avant 
            self.quantization_with_landmarks_possible = False

        self.has_landmarks= True

        self.verbosity(function_name='compute_nystrom_landmarks',
                        dict_of_variables={'m':self.m,'landmark_method':self.landmark_method},
                        start=False,
                        verbose = verbose)

    def compute_nystrom_anchors(self,sample='xy',verbose=0,outliers_in_obs=None):
        """
        Determines the nystrom anchors using 
        Stores the results as a list of eigenvalues and the 
        
        Parameters
        anchors_basis in ['K','S','W']
        ----------
        r:      <= m (= by default). Number of anchors to determine in total (proportionnaly according to the data)
        
        # le anchor basis est encore en param car je n'ai pas réfléchi à la version pour 1 groupe
        """
        
        assert(self.anchors_basis is not None)
        
        self.verbosity(function_name='compute_nystrom_anchors',
                        dict_of_variables={'r':self.r},
                        start=True,
                        verbose = verbose)
        anchors_basis=self.anchors_basis
        suffix_outliers = '' if outliers_in_obs is None else outliers_in_obs 
        anchor_name = f'{anchors_basis}{suffix_outliers}'
        
        if anchor_name not in self.spev[sample]['anchors']:

            if sample == 'xy':
                if self.r is None:
                    if verbose > 0:
                        print("r not specified, by default, r = m" )
                self.r = self.m if self.r is None else self.r
                anchors_basis = self.anchors_basis
                assert(self.r <= self.m)

            # a réfléchir dans les cas de sample in {'x','y'} 

            r = self.r if sample =='xy' else self.nxanchors if sample=='x' else self.nyanchors
            m = self.m
            Km = self.compute_gram(sample=sample,landmarks=True,outliers_in_obs=outliers_in_obs)
            P = self.compute_covariance_centering_matrix(sample=sample,landmarks=True,outliers_in_obs=outliers_in_obs)
            sp_anchors,ev_anchors = ordered_eigsy(1/m*torch.chain_matmul(P,Km,P))        
            # sp_anchors,ev_anchors = ordered_eigsy(1/r*torch.linalg.multi_dot([P,Km,P]))        
            if sum(sp_anchors>0)<r:

                self.r = sum(sp_anchors>0).item()
                r = self.r
                # ajout suite aux simu univariées ou le spectre était parfois négatif, ce qui provoquait des abérations quand on l'inversait. La solution que j'ai choisie est de tronquer le spectre uniquement aux valeurs positives et considérer les autres comme nulles. 
                if verbose>0:
                    print(f'due to a numerical aberation, the number of anchors is reduced from {self.r} to {sum(sp_anchors>0)}')

        # print(f'In compute nystrom anchors:\n\t Km\n{Km}\n\t P\n{P} \n\t sp_anchors\n{sp_anchors}\n\t ev_anchors\n{ev_anchors}')

            self.spev[sample]['anchors'][anchor_name] = {'sp':sp_anchors[:r],'ev':ev_anchors[:,:r]}

            self.verbosity(function_name='compute_nystrom_anchors',
                            dict_of_variables={'r':r},
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

    def reinitialize_landmarks(self):
        if self.quantization_with_landmarks_possible:
            self.quantization_with_landmarks_possible = False
            delattr(self,'xassignations')
            delattr(self,'yassignations')
            for sample in ['x','y','xy']: 
                self.spev[sample].pop('quantization',None)

        if self.has_landmarks:
            self.has_landmarks = False
            delattr(self,'m')
            delattr(self,'nxlandmarks')
            delattr(self,'nylandmarks')
            delattr(self,'xlandmarks')
            delattr(self,'ylandmarks')
        
    def reinitialize_anchors(self):
        for sample in ['x','y','xy']: 
            # self.spev[sample].pop('anchors',None)
            self.spev[sample].pop('nystrom',None)
