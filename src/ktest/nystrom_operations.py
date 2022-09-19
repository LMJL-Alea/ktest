from .utils import ordered_eigsy
import numpy as np
import torch
import pandas as pd
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

    def init_nystrom(self,verbose=0):
        if not self.nystrom_initialized:
            ntot = self.get_ntot(landmarks=False)        

            if not self.m_initial is None and verbose >0:
                print("m not specified, by default, m = (n1+n2)//10")
            if self.landmark_method is None and verbose >0:
                print("landmark_method not specified, by default, landmark_method='random'")
            if self.r is None and verbose > 0:
                print("r not specified, by default, r = m" )
            if self.anchors_basis is None and verbose > 0:
                print("nystrom anchors_basis not specified, default : w")
            
            self.m_initial = self.m_initial if self.m_initial is not None else ntot//10
            dict_nobs  =  self.get_nobs(landmarks=False)
            try:
                dict_nlandmarks = {k:int(np.floor(n/ntot*self.m_initial)) for k,n in dict_nobs.items() if k!='ntot'}
            except ZeroDivisionError:
                print(f'ZeroDivisionError : the effectifs in dict_nobs are {dict_nobs}')
            nlandmarks_total = sum([v for k,v in dict_nlandmarks.items()])

            # sometimes an approximation error changes the number of landmarks
            # if self.m_initial != nlandmarks_total:
            self.m = nlandmarks_total        
            self.r = self.m if self.r is None else self.r     

            self.landmark_method = 'random' if self.landmark_method is None else self.landmark_method       
            self.anchors_basis = 'w' if self.anchors_basis is None else self.anchors_basis
            assert(self.r <= self.m)
            self.nystrom_initialized = True
    
    def compute_nystrom_landmarks(self,verbose=0):
        """
        The nystrom landmarks are the vectors of the RKHS from which we determine the anchors in the nystrom method.  
        
        Parameters
        ----------

        """

    
        self.verbosity(function_name='compute_nystrom_landmarks',
                        dict_of_variables={},
                        start=True,
                        verbose = verbose)
        self.init_nystrom(verbose=verbose)

        dict_index = self.get_index(landmarks=False)
        dict_nobs  =  self.get_nobs(landmarks=False)
        ntot = self.get_ntot(landmarks=False)        
        dict_nlandmarks = {k:int(np.floor(n/ntot*self.m_initial)) for k,n in dict_nobs.items() if k!='ntot'}
        # print(f'nylm dict_nlandmarks : {dict_nlandmarks}')
        landmarks_name = self.get_landmarks_name()
        
        # for kmeans, a new dataset containing the centroïds is added to self.data 
        if self.landmark_method =='kmeans':
            # load data to determine kmeans centroids

            dict_data = self.get_data(landmarks=False)
            
            # determine kmeans centroids and assignations
            for sample in dict_data.keys():
                # determine centroids
                kmeans_landmarks_name = self.get_kmeans_landmarks_name_for_sample(sample=sample)
                if kmeans_landmarks_name not in self.data:
                    x,index,nlandmarks = dict_data[sample],dict_index[sample],dict_nlandmarks[sample]
                    assignations,landmarks = kmeans(X=x, num_clusters=nlandmarks, distance='euclidean', tqdm_flag=False) #cuda:0')
                    landmarks = landmarks.double()
                    # save results 
                    kmeans_landmarks_name = self.get_kmeans_landmarks_name_for_sample(sample=sample)

                    self._update_dict_data(landmarks,kmeans_landmarks_name,False)
                    self._update_index(nlandmarks,index=None,data_name=kmeans_landmarks_name)
                    self.obs[kmeans_landmarks_name] = pd.DataFrame(assignations,index=index)

                    self.quantization_with_landmarks_possible = True
                else:
                    print(f'kmeans landmarks {kmeans_landmarks_name} already computed')
        
        # for random, we only add a column of booleans to identify the sampled landmarks  
        elif self.landmark_method == 'random':
            for sample in dict_index.keys():
                ni,index,nlandmarks = dict_nobs[sample],dict_index[sample],dict_nlandmarks[sample]
                
                if ni < nlandmarks:
                    print(f'problem in nystrom landmarks : {sample} has {ni} obs for {nlandmarks} landmarks')
                z = np.random.choice(ni,size=nlandmarks,replace=False)
                # print(f'###{sample} \n z{len(z)}{z} \n index{len(index)}{index} \n index[z] {len(index[z])}{index[z]}')
                # print(f'lm rd {len(index.isin(index[z]))}')
                self.add_outliers_in_obs(index[z],f'{sample}_{landmarks_name}')
                # self.quantization_with_landmarks_possible = False
        self.has_landmarks= True

        self.verbosity(function_name='compute_nystrom_landmarks',
                        dict_of_variables={},
                        start=False,
                        verbose = verbose)

    def compute_nystrom_anchors(self,verbose=0):
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
        anchors_name = self.get_anchors_name()
        
        self.verbosity(function_name='compute_nystrom_anchors',
                        dict_of_variables={'r':self.r},
                        start=True,
                        verbose = verbose)
        
        
        if anchors_name not in self.spev['anchors']:

            r = self.r 
            m = self.m
            Km = self.compute_gram(landmarks=True)
            P = self.compute_covariance_centering_matrix(quantization=False,landmarks=True,)
            
            # print('nystrom anchors',r,m,Km.shape,P.shape)
            assert(len(P)==m)
            assert(len(Km)==m)
            
            sp_anchors,ev_anchors = ordered_eigsy(1/m*torch.chain_matmul(P,Km,P))        
            # sp_anchors,ev_anchors = ordered_eigsy(1/r*torch.linalg.multi_dot([P,Km,P]))        

            if sum(sp_anchors>0)<r:
                self.r = sum(sp_anchors>0).item()
                r = self.r
                # ajout suite aux simu univariées ou le spectre était parfois négatif, ce qui provoquait des abérations quand on l'inversait. La solution que j'ai choisie est de tronquer le spectre uniquement aux valeurs positives et considérer les autres comme nulles. 
                if verbose>0:
                    print(f'due to a numerical aberation, the number of anchors is reduced from {self.r} to {sum(sp_anchors>0)}')

            self.spev['anchors'][anchors_name] = {'sp':sp_anchors[:r],'ev':ev_anchors[:,:r]}

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

