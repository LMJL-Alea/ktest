from .utils import ordered_eigsy
import numpy as np
import torch
import pandas as pd
# import apt.kmeans 
from kmeans_pytorch import kmeans


from .base import Base
from .outliers_operations import OutliersOps
from .verbosity import Verbosity


"""
Ces fonctions déterminent permettent de déterminer les landmarks puis les ancres dans le cas de l'utilisation
de la méthode de nystrom. 
"""
class NystromOps(OutliersOps,Verbosity):
    def __init__(self):
        super(NystromOps,self).__init__()

    def init_landmark_method(self,verbose=0):
        if self.landmark_method is None:
            self.landmark_method = 'random'        
            if verbose >0:
                print("landmark_method not specified, by default, landmark_method='random'")
    
    def init_anchor_basis(self,verbose=0):
        if self.anchor_basis is None: 
            self.anchor_basis = 'w'
            if verbose > 0:
                print("\tnystrom anchor_basis not specified, default : w")
      
    def init_n_landmarks(self,verbose=0):
        ntot = self.get_ntot()  

        if self.n_landmarks_initial is None:
            if ntot >500:
                self.n_landmarks_initial = 500
                if verbose >1:
                    print("\tn_landmarks not specified, by default, n_landmarks = 500")
            else:
                self.n_landmarks_initial = ntot//10
                if verbose >1:
                    print("\tn_landmarks not specified, by default, n_landmarks = (n1+n2)//10 (less than 500 obs)")

        try:
            n_landmarks = self.get_n_landmarks(verbose=verbose)
            if n_landmarks != self.n_landmarks_initial:
                if verbose>1:
                    print(f"\tReduced from {self.n_landmarks_initial} to {n_landmarks} landmarks for proportion issues")
            if self.n_anchors is not None and self.n_anchors>n_landmarks:
                old_n_anchors = self.n_anchors
                self.n_anchors = n_landmarks
                if verbose >0:
                    print(f"\tReduced from {old_n_anchors} to {self.n_anchors} anchors") 
                    
        except ZeroDivisionError:
            print(f'ZeroDivisionError : the effectifs in dict_nobs are {self.get_nobs()}')
            # sometimes an approximation error changes the number of landmarks # if n_landmarks_initial != n_landmarks_total:
        

    def init_n_anchors(self,verbose):
        ntot = self.get_ntot()
        if self.n_anchors is None:
            if ntot >500:
                self.n_anchors = 50
                if verbose > 0:
                    print("\tn_anchors not specified, by default, n_anchors = 50" )
            else: 
                self.n_anchors = self.get_n_landmarks()    
                if verbose > 0:
                    print("\tn_anchors not specified, by default, n_anchors = n_landmarks" )
            
        assert(self.n_anchors <= self.get_n_landmarks())
    
    def init_nystrom(self,verbose=0):
        if verbose >0:
            print('- Initialize nystrom parameters')
        if not self.nystrom_initialized:
            self.init_landmark_method(verbose)
            self.init_anchor_basis(verbose)
            self.init_n_landmarks(verbose)
            self.init_n_anchors(verbose)
            self.nystrom_initialized = True
    
    def compute_nystrom_landmarks(self,verbose=0):
        """
        The nystrom landmarks are the vectors of the RKHS from which we determine the anchors in the nystrom method.  
        
        Parameters
        ----------

        """
        self.init_nystrom(verbose=verbose)

        if verbose>0:
            s = '- Compute nystrom landmarks'
            if verbose == 1:
                s += f' ({self.get_n_landmarks()} landmarks)'
            else :
                s+= f'\n\tlandmark_method : {self.landmark_method}\n\tn_landmarks total : {self.get_n_landmarks()} ' 

            print(s)
        if self.landmark_method == 'kmeans':
            self.compute_nystrom_landmarks_kmeans(verbose=verbose)
        elif self.landmark_method == 'random':
            self.compute_nystrom_landmarks_random(verbose=verbose)

    def get_n_landmarks(self,verbose=0):
        return(sum(self.get_dict_n_landmarks().values()))

    def get_dict_n_landmarks(self,verbose=0):
        ntot = self.get_ntot(landmarks=False) 
        dict_nobs  =  self.get_nobs(landmarks=False)
        dict_n_landmarks = {k:int(np.floor(n/ntot*self.n_landmarks_initial)) for k,n in dict_nobs.items() if k!='ntot'}
        samples_list = self.get_samples_list()

        # ajouter 5 landmarks au plus pour les échantillons qui n'en ont pas 
        for sample in samples_list:
            if dict_n_landmarks[sample]==0:
                if dict_nobs[sample]>=5:
                    dict_n_landmarks[sample]=5
                else:
                    dict_n_landmarks[sample] = dict_nobs[sample]

                if verbose >1:
                    print(f'\tAdding 5 landmarks to {sample}')
        return(dict_n_landmarks)



    def compute_nystrom_landmarks_random(self,verbose=0):
        
        # a column of booleans to identify the sampled landmarks to obs 
        #  
        dict_index = self.get_index(landmarks=False)
        dict_nobs  =  self.get_nobs(landmarks=False)
        dict_n_landmarks = self.get_dict_n_landmarks(verbose=verbose)
        landmarks_name = self.get_landmarks_name()
        # dict_data = self.get_dataframes_of_data()
        dict_data = self.get_data(in_dict=True,dataframe=True)

            
        for sample in dict_index.keys():
            ni,index,n_landmarks = dict_nobs[sample],dict_index[sample],dict_n_landmarks[sample]
            data = dict_data[sample]
            if verbose>1:
                
                print(f'\tn_landmarks in {sample} : {n_landmarks}')
            if data.shape[1]==1:
                c = data.columns[0]
                nz = (data[c]!=0).sum() # count non-zero
                
                # if there is no non-zero data, we do not change anything because if there is not non-zero data in the whole dataset it should not have been tested. 
                # otherwise, we force the choice of at least one non-zero observations as a landmark
                # if there are less non-zero observation that the number of landmarks, we chose them all
                if nz != 0:
                    if nz<n_landmarks:
                        if verbose >2:
                            print(f'\tnon-zero {nz} < {n_landmarks} : forcing non-zero obs in landmarks')
                        index_nz = data[data[c]!=0].index
                        index_z = data[data[c]==0].index[:n_landmarks-nz]
                        index_landmarks = index_nz.union(index_z)
                    else:
                        if verbose >2:
                            print(f'\tnon-zero {nz}> {n_landmarks} n_landmarks')
                        z = np.random.choice(ni,size=n_landmarks,replace=False)
                        is_chosen = data.index.isin(index[z])

                        if (data[c][is_chosen]!=0).sum() == 0:
                            nzobs = data[data[c]!=0].index[np.random.choice(nz,size=1)]
                            index_landmarks = index[z][:-1].union(nzobs)
                        else:
                            index_landmarks = index[z]

                    self.mark_observations(observations_to_mark=index_landmarks,
                        marking_name=f'{sample}_{landmarks_name}')
                    if verbose>2:
                        print(f'n_landmarks {sample} final : {len(index_landmarks)}')

                else: 
                    z = np.random.choice(ni,size=n_landmarks,replace=False)
                    self.mark_observations(observations_to_mark=index[z],
                            marking_name=f'{sample}_{landmarks_name}')

            else: 
                z = np.random.choice(ni,size=n_landmarks,replace=False)
                self.mark_observations(observations_to_mark=index[z],
                                    marking_name=f'{sample}_{landmarks_name}')
        self.has_landmarks= True

    def compute_nystrom_landmarks_kmeans(self,verbose=0):
        # a new dataset containing the centroïds is added to self.data 

        dict_index = self.get_index(landmarks=False)
        dict_nobs  =  self.get_nobs(landmarks=False)
        ntot = self.get_ntot(landmarks=False)      

        dict_n_landmarks = {k:int(np.floor(n/ntot*self.n_landmarks_initial)) for k,n in dict_nobs.items() if k!='ntot'}
        
        
        dict_data = self.get_data(landmarks=False) 
        
        # determine kmeans centroids and assignations
        for sample in dict_data.keys():
            # determine centroids
            kmeans_landmarks_name = self.get_kmeans_landmarks_name_for_sample(sample=sample)
            if kmeans_landmarks_name in self.data:
                print(f'kmeans landmarks {kmeans_landmarks_name} already computed')
            else:
                x,index,n_landmarks = dict_data[sample],dict_index[sample],dict_n_landmarks[sample]
                assignations,landmarks = kmeans(X=x, num_clusters=n_landmarks, distance='euclidean', tqdm_flag=False) #cuda:0')
                landmarks = landmarks.double()
                # save results 
                kmeans_landmarks_name = self.get_kmeans_landmarks_name_for_sample(sample=sample)

                self._update_dict_data(landmarks,kmeans_landmarks_name,False)
                self._update_index(n_landmarks,index=None,data_name=kmeans_landmarks_name)
                self.obs[kmeans_landmarks_name] = pd.DataFrame(assignations,index=index)

                
        self.has_landmarks= True

    def compute_nystrom_anchors(self,verbose=0):
        """
        Determines the nystrom anchors using 
        Stores the results as a list of eigenvalues and the 
        
        Parameters
        anchor_basis in ['K','S','W']
        ----------
        n_anchors:      <= n_landmarks (= by default). Number of anchors to determine in total (proportionnaly according to the data)
        
        # le anchor basis est encore en param car je n'ai pas réfléchi à la version pour 1 groupe
        """
        
        if verbose>0:
            print(f'- Compute nystrom anchors ({self.n_anchors} anchors)')

        assert(self.anchor_basis is not None)
        anchors_name = self.get_anchors_name()
        
        if anchors_name not in self.spev['anchors']:

            n_anchors = self.n_anchors 
            n_landmarks = self.get_n_landmarks()
            # m = self.get_ntot(landmarks=True)
            Km = self.compute_gram(landmarks=True)
            P = self.compute_covariance_centering_matrix(landmarks=True,)
            
            if verbose >2:
                print(f'nystrom anchors : n_landmarks = {n_landmarks}, n_anchors = {n_anchors}, P:{len(P)}, Km:{len(Km)})')
            assert(len(P)==n_landmarks)
            assert(len(Km)==n_landmarks)
            
            sp_anchors,ev_anchors = ordered_eigsy(1/n_landmarks*torch.linalg.multi_dot([P,Km,P]))        

            if sum(sp_anchors>0) ==0:
                if verbose>0:
                    print('\tNo anchors found, the dataset may have two many zero data.')

            else: 
                if sum(sp_anchors>0)<n_anchors:
                    old_n_anchors = self.n_anchors
                    self.n_anchors = sum(sp_anchors>0).item()
                    n_anchors = self.n_anchors
                    # ajout suite aux simu univariées ou le spectre était parfois négatif, ce qui provoquait des abérations quand on l'inversait. La solution que j'ai choisie est de tronquer le spectre uniquement aux valeurs positives et considérer les autres comme nulles. 
                    if verbose>1:
                        print(f'\tThe number of anchors is reduced from {old_n_anchors} to {sum(sp_anchors>0)} for numerical stability')

                self.spev['anchors'][anchors_name] = {'sp':sp_anchors[:n_anchors],'ev':ev_anchors[:,:n_anchors]}
                self.has_anchors=True

