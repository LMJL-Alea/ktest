from .utils import ordered_eigsy
import numpy as np
import torch

import apt.kmeans 
from kmeans_pytorch import kmeans

def compute_nystrom_anchors(self,sample='xy',verbose=0,anchors_basis=None):
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

    if sample == 'xy':
        if self.r is None:
            if verbose > 0:
                print("r not specified, by default, r = m" )
        self.r = self.m if self.r is None else self.r
        anchors_basis = self.anchors_basis
        assert(self.r <= self.m)
        
    # a réfléchir dans le cas de 1 groupe 
    # elif sample =='x':
    #     self.nxanchors = self.nxlandmarks if r is None else r
    #     assert(self.nxanchors <= self.nxlandmarks)
    # elif sample =='y':
    #     self.nyanchors = self.nylandmarks if r is None else r
    #     assert(self.nyanchors <= self.nylandmarks)

    r = self.r if sample =='xy' else self.nxanchors if sample=='x' else self.nyanchors
    
    Km = self.compute_gram(sample=sample,landmarks=True)
    P = self.compute_centering_matrix(sample=sample,landmarks=True)
    sp_anchors,ev_anchors = ordered_eigsy(1/r*torch.chain_matmul(P,Km,P))        
    # sp_anchors,ev_anchors = ordered_eigsy(1/r*torch.linalg.multi_dot([P,Km,P]))        
    if any(sp_anchors<0):
        # ajout suite aux simu univariées ou le spectre était parfois négatif, ce qui provoquait des abérations quand on l'inversait. La solution que j'ai choisie est de tronquer le spectre uniquement aux valeurs positives et considérer les autres comme nulles. 
        if verbose>0:
            print(f'due to a numerical aberation, the number of anchors is reduced from {self.r} to {sum(sp_anchors>0)}')
        r = sum(sp_anchors>0)
        self.r = r.item()

    # print(f'In compute nystrom anchors:\n\t Km\n{Km}\n\t P\n{P} \n\t sp_anchors\n{sp_anchors}\n\t ev_anchors\n{ev_anchors}')

    if 'anchors' in self.spev[sample]:
        self.spev[sample]['anchors'][anchors_basis] = {'sp':sp_anchors[:r],'ev':ev_anchors[:,:r]}
    else:
        self.spev[sample]['anchors'] = {anchors_basis:{'sp':sp_anchors[:r],'ev':ev_anchors[:,:r]}}

    self.verbosity(function_name='compute_nystrom_anchors',
                    dict_of_variables={'r':r},
                    start=False,
                    verbose = verbose)

def compute_nystrom_landmarks(self,verbose=0):
    # anciennement compute_nystrom_anchors(self,r,nystrom_method='kmeans',split_data=False,test_size=.8,on_other_data=False,x_other=None,y_other=None,verbose=0): # max_iter=1000, (pour kmeans de François)
    
    """
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
        
    x,y = self.get_xy()
    n1,n2 = self.n1,self.n2 
    xratio,yratio = n1/(n1 + n2), n2/(n1 +n2)

    if self.m is None:
        if verbose >0:
            print("m not specified, by default, m = (n1+n2)//10")
        self.m = (n1 + n2)//10

    if self.landmark_method is None:
        if verbose >0:
            print("landmark_method not specified, by default, landmark_method='random'")
        self.landmark_method = 'random'
    
    self.nxlandmarks=np.int(np.floor(xratio * self.m)) 
    self.nylandmarks=np.int(np.floor(yratio * self.m))

    

    if self.landmark_method == 'kmeans':
        # self.xanchors,self.xassignations = apt.kmeans.spherical_kmeans(self.x[self.xmask,:], nxanchors, max_iter)
        # self.yanchors,self.yassignations = apt.kmeans.spherical_kmeans(self.y[self.ymask,:], nyanchors, max_iter)
        self.xassignations,self.xlandmarks = kmeans(X=x, num_clusters=self.nxlandmarks, distance='euclidean', tqdm_flag=False) #cuda:0')
        self.yassignations,self.ylandmarks = kmeans(X=y, num_clusters=self.nylandmarks, distance='euclidean', tqdm_flag=False) #cuda:0')
        self.xlandmarks = self.xlandmarks.double()
        self.ylandmarks = self.ylandmarks.double()
        self.quantization_with_landmarks_possible = True
        

    elif self.landmark_method == 'random':
        self.xlandmarks = x[np.random.choice(x.shape[0], size=self.nxlandmarks, replace=False)]
        self.ylandmarks = y[np.random.choice(y.shape[0], size=self.nylandmarks, replace=False)]
        
        # Necessaire pour remettre a false au cas ou on a déjà utilisé 'kmeans' avant 
        self.quantization_with_landmarks_possible = False

    self.has_landmarks= True

    self.verbosity(function_name='compute_nystrom_landmarks',
                    dict_of_variables={'m':self.m,'landmark_method':self.landmark_method},
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
#



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
