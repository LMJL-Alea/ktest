from .utils import ordered_eigsy
import numpy as np
import torch
import pandas as pd
# import apt.kmeans 
from kmeans_pytorch import kmeans


from ktest.base import Data,Model
from ktest.outliers_operations import OutliersOps
from ktest.verbosity import Verbosity

from .utils import get_landmarks_name,get_kmeans_landmarks_name_for_sample

"""
Ces fonctions déterminent permettent de déterminer les landmarks puis les ancres dans le cas de l'utilisation
de la méthode de nystrom. 
"""
class NystromOps(Model,OutliersOps,Verbosity):
    def __init__(self):
        super(NystromOps,self).__init__()

    def compute_nystrom_landmarks(self,groups='sample',samples='all',outliers_in_obs=None,verbose=0,data_name=None):
        """
        The nystrom landmarks are the vectors of the RKHS from which we determine the anchors in the nystrom method.  
        
        Parameters
        ----------
            groups (default:'sample') : str

            samples (default:'all'): str or iterable

            outliers_in_obs (default:None) : str 

            data_name (default:None) : str

        """

        self.verbosity(function_name='compute_nystrom_landmarks',
                        dict_of_variables={'groups':groups,'outliers_in_obs':outliers_in_obs},
                        start=True,
                        verbose = verbose)

        dict_index = self.get_index(groups=groups,samples=samples,landmarks=False,outliers_in_obs=outliers_in_obs)
        dict_nobs  =  self.get_nobs(groups=groups,samples=samples,landmarks=False,outliers_in_obs=outliers_in_obs)
        ntot = dict_nobs['ntot']
        

        if self.m is None and verbose >0:
                print("m not specified, by default, m = (n1+n2)//10")
        if self.landmark_method is None and verbose >0:
                print("landmark_method not specified, by default, landmark_method='random'")
        self.m = ntot//10 if self.m is None else self.m
        self.landmark_method = 'random' if self.landmark_method is None else self.landmark_method       
        
        dict_nlandmarks = {k:int(np.floor(n/ntot*self.m)) for k,n in dict_nobs.items() if k!='ntot'}
        landmarks_name = get_landmarks_name(outliers_in_obs,groups,samples,self.landmark_method)
        

        # for kmeans, a new dataset containing the centroïds is added to self.data 
        if self.landmark_method =='kmeans':
            # load data to determine kmeans centroids
            if data_name is None:
                data_name = self.current_data_name
            dict_data = self.get_data(data_name,groups=groups,samples=samples,landmarks=False,outliers_in_obs=outliers_in_obs)
            
            # determine kmeans centroids and assignations
            for sample in dict_data.keys():
                # determine centroids
                x,index,nlandmarks = dict_data[sample],dict_index[sample],dict_nlandmarks[sample]
                
                assignations,landmarks = kmeans(X=x, num_clusters=nlandmarks, distance='euclidean', tqdm_flag=False) #cuda:0')
                landmarks = landmarks.double()
                
                # save results 
                kmeans_landmarks_name = get_kmeans_landmarks_name_for_sample(outliers_in_obs,groups,samples,self.landmark_method,sample,data_name)
                self._update_dict_data(landmarks,kmeans_landmarks_name)
                self._update_index(nlandmarks,index=None,data_name=kmeans_landmarks_name)

                self.obs[kmeans_landmarks_name] = pd.DataFrame(assignations,index=index)

                self.quantization_with_landmarks_possible = True

        # for random, we only add a column of booleans to identify the sampled landmarks  
        elif self.landmark_method == 'random':
            for sample in dict_index.keys():
                ni,index,nlandmarks = dict_nobs[sample],dict_index[sample],dict_nlandmarks[sample]
                if ni < nlandmarks:
                    print(f'problem in nystrom landmarks : {sample} has {ni} obs for {nlandmarks} landmarks')
                z = np.random.choice(ni,size=nlandmarks,replace=False)
                
                self.add_outliers_in_obs(index[z],f'{sample}_{landmarks_name}')
                self.quantization_with_landmarks_possible = False

        self.has_landmarks= True

        self.verbosity(function_name='compute_nystrom_landmarks',
                        dict_of_variables={'groups':groups,'outliers_in_obs':outliers_in_obs},
                        start=False,
                        verbose = verbose)

    def compute_nystrom_landmarks_new(self,groups='sample',samples='all',outliers_in_obs=None,verbose=0,data_name=None):
        """
        The nystrom landmarks are the vectors of the RKHS from which we determine the anchors in the nystrom method.  
        
        Parameters
        ----------
            groups (default:'sample') : str

            samples (default:'all'): str or iterable

            outliers_in_obs (default:None) : str 

            data_name (default:None) : str

        """

        self.verbosity(function_name='compute_nystrom_landmarks',
                        dict_of_variables={'groups':groups,'outliers_in_obs':outliers_in_obs},
                        start=True,
                        verbose = verbose)

        dict_index = self.get_index(groups=groups,samples=samples,landmarks=False,outliers_in_obs=outliers_in_obs)
        dict_nobs  =  self.get_nobs(groups=groups,samples=samples,landmarks=False,outliers_in_obs=outliers_in_obs)
        ntot = dict_nobs['ntot']
        

        if self.m is None and verbose >0:
                print("m not specified, by default, m = (n1+n2)//10")
        if self.landmark_method is None and verbose >0:
                print("landmark_method not specified, by default, landmark_method='random'")
        self.m = ntot//10 if self.m is None else self.m
        self.landmark_method = 'random' if self.landmark_method is None else self.landmark_method       
        
        dict_nlandmarks = {k:int(np.floor(n/ntot*self.m)) for k,n in dict_nobs.items() if k!='ntot'}
        landmarks_name = get_landmarks_name(outliers_in_obs,groups,samples,self.landmark_method)
        

        # for kmeans, a new dataset containing the centroïds is added to self.data 
        if self.landmark_method =='kmeans':
            # load data to determine kmeans centroids
            if data_name is None:
                data_name = self.current_data_name
            dict_data = self.get_data(data_name,groups=groups,samples=samples,landmarks=False,outliers_in_obs=outliers_in_obs)
            
            # determine kmeans centroids and assignations
            for sample in dict_data.keys():
                # determine centroids
                x,index,nlandmarks = dict_data[sample],dict_index[sample],dict_nlandmarks[sample]
                
                assignations,landmarks = kmeans(X=x, num_clusters=nlandmarks, distance='euclidean', tqdm_flag=False) #cuda:0')
                landmarks = landmarks.double()
                
                # save results 
                kmeans_landmarks_name = get_kmeans_landmarks_name_for_sample(outliers_in_obs,groups,samples,self.landmark_method,sample,data_name)
                self._update_dict_data(landmarks,kmeans_landmarks_name)
                self._update_index(nlandmarks,index=None,data_name=kmeans_landmarks_name)

                self.obs[kmeans_landmarks_name] = pd.DataFrame(assignations,index=index)

                self.quantization_with_landmarks_possible = True

        # for random, we only add a column of booleans to identify the sampled landmarks  
        elif self.landmark_method == 'random':
            for sample in dict_index.keys():
                ni,index,nlandmarks = dict_nobs[sample],dict_index[sample],dict_nlandmarks[sample]
                if ni < nlandmarks:
                    print(f'problem in nystrom landmarks : {sample} has {ni} obs for {nlandmarks} landmarks')
                z = np.random.choice(ni,size=nlandmarks,replace=False)
                
                self.add_outliers_in_obs(index[z],f'{sample}_{landmarks_name}')
                self.quantization_with_landmarks_possible = False

        self.has_landmarks= True

        self.verbosity(function_name='compute_nystrom_landmarks',
                        dict_of_variables={'groups':groups,'outliers_in_obs':outliers_in_obs},
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
