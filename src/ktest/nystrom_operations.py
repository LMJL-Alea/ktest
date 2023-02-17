from .utils import ordered_eigsy
import numpy as np
import torch
import pandas as pd
# import apt.kmeans 
from kmeans_pytorch import kmeans


from ktest.base import Base
from ktest.outliers_operations import OutliersOps
from ktest.verbosity import Verbosity


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
      
    def init_nlandmarks(self,verbose=0):
        ntot = self.get_ntot()  

        if self.nlandmarks_initial is None:
            if ntot >500:
                self.nlandmarks_initial = 500
                self.nlandmarks = 500
                if verbose >0:
                    print("\tnlandmarks not specified, by default, nlandmarks = 500")
            else:
                self.nlandmarks_initial = ntot//10
                self.nlandmarks = ntot//10
                if verbose >0:
                    print("\tnlandmarks not specified, by default, nlandmarks = (n1+n2)//10 (less than 500 obs)")

        try:
            nlandmarks = sum([int(np.floor(n/ntot*self.nlandmarks_initial)) for k,n in self.get_nobs().items() if k!='ntot'])
            if nlandmarks != self.nlandmarks:
                self.nlandmarks = nlandmarks   
                if verbose>0:
                    print(f"\tReduced from {self.nlandmarks_initial} to {self.nlandmarks} landmarks for proportion issues")
                if self.nanchors is not None and self.nanchors>nlandmarks:
                    old_nanchors = self.nanchors
                    self.nanchors = nlandmarks
                    if verbose >0:
                       print(f"\tConsequence : reduced from {old_nanchors} to {self.nanchors} anchors") 
                    
        except ZeroDivisionError:
            print(f'ZeroDivisionError : the effectifs in dict_nobs are {self.get_nobs()}')
            # sometimes an approximation error changes the number of landmarks # if nlandmarks_initial != nlandmarks_total:
        

    def init_nanchors(self,verbose):
        ntot = self.get_ntot()
        if self.nanchors is None:
            if ntot >500:
                self.nanchors = 50
                if verbose > 0:
                    print("\tnanchors not specified, by default, nanchors = 50" )
            else: 
                self.nanchors = self.nlandmarks      
                if verbose > 0:
                    print("\tnanchors not specified, by default, nanchors = nlandmarks" )
            
        assert(self.nanchors <= self.nlandmarks)
    
    def init_nystrom(self,verbose=0):
        if verbose >0:
            print('- Initialize nystrom parameters')
        if not self.nystrom_initialized:
            self.init_landmark_method(verbose)
            self.init_anchor_basis(verbose)
            self.init_nlandmarks(verbose)
            self.init_nanchors(verbose)
                
            self.nystrom_initialized = True
    
    def compute_nystrom_landmarks(self,verbose=0):
        """
        The nystrom landmarks are the vectors of the RKHS from which we determine the anchors in the nystrom method.  
        
        Parameters
        ----------

        """
        self.init_nystrom(verbose=verbose)

        if verbose>0:
            print('- Compute nystrom landmarks') 
            print(f'\tlandmark_method : {self.landmark_method}\n\tm (nlandmarks total) : {self.nlandmarks} ')  

        
        if self.landmark_method == 'kmeans':
            self.compute_nystrom_landmarks_kmeans(verbose=verbose)
        elif self.landmark_method == 'random':
            self.compute_nystrom_landmarks_random(verbose=verbose)

    def compute_nystrom_landmarks_random(self,verbose=0):
        
        # a column of booleans to identify the sampled landmarks to obs 
        #  
        dict_index = self.get_index(landmarks=False)
        dict_nobs  =  self.get_nobs(landmarks=False)
        ntot = self.get_ntot(landmarks=False)      
        dict_nlandmarks = {k:int(np.floor(n/ntot*self.nlandmarks_initial)) for k,n in dict_nobs.items() if k!='ntot'}
        landmarks_name = self.get_landmarks_name()
        dict_data = self.get_dataframes_of_data()

            
        for sample in dict_index.keys():
            ni,index,nlandmarks = dict_nobs[sample],dict_index[sample],dict_nlandmarks[sample]
            data = dict_data[sample]
            if verbose>0:
                print(f'\tnlandmarks in {sample} : {nlandmarks}')
            if data.shape[1]==1:
                c = data.columns[0]
                nz = (data[c]!=0).sum() # count non-zero
                
                # if there is no non-zero data, we do not change anything because if there is not non-zero data in the whole dataset it should not have been tested. 
                # otherwise, we force the choice of at least one non-zero observations as a landmark
                # if there are less non-zero observation that the number of landmarks, we chose them all
                if nz != 0:
                    if nz<nlandmarks:
                        if verbose >0:
                            print(f'\tnon-zero {nz} < {nlandmarks} : forcing non-zero obs in landmarks')
                        index_nz = data[data[c]!=0].index
                        index_z = data[data[c]==0].index[:nlandmarks-nz]
                        index_landmarks = index_nz.union(index_z)

                    else:
                        if verbose >0:
                            print(f'\tnon-zero {nz}> {nlandmarks} nlandmarks')
                        z = np.random.choice(ni,size=nlandmarks,replace=False)
                        is_chosen = data.index.isin(index[z])

                        if (data[c][is_chosen]!=0).sum() == 0:
                            nzobs = data[data[c]!=0].index[np.random.choice(nz,size=1)]
                            index_landmarks = index[z][:-1].union(nzobs)
                        else:
                            index_landmarks = index[z]

                    self.mark_observations(observations_to_mark=index_landmarks,
                        marking_name=f'{sample}_{landmarks_name}')


                else: 
                    z = np.random.choice(ni,size=nlandmarks,replace=False)
                    self.mark_observations(observations_to_mark=index[z],
                            marking_name=f'{sample}_{landmarks_name}')

            else: 
                z = np.random.choice(ni,size=nlandmarks,replace=False)
                self.mark_observations(observations_to_mark=index[z],
                                    marking_name=f'{sample}_{landmarks_name}')
        self.has_landmarks= True

    def compute_nystrom_landmarks_kmeans(self,verbose=0):
        # a new dataset containing the centroïds is added to self.data 

        dict_index = self.get_index(landmarks=False)
        dict_nobs  =  self.get_nobs(landmarks=False)
        ntot = self.get_ntot(landmarks=False)      

        dict_nlandmarks = {k:int(np.floor(n/ntot*self.nlandmarks_initial)) for k,n in dict_nobs.items() if k!='ntot'}
        
        
        dict_data = self.get_data(landmarks=False) 
        
        # determine kmeans centroids and assignations
        for sample in dict_data.keys():
            # determine centroids
            kmeans_landmarks_name = self.get_kmeans_landmarks_name_for_sample(sample=sample)
            if kmeans_landmarks_name in self.data:
                print(f'kmeans landmarks {kmeans_landmarks_name} already computed')
            else:
                x,index,nlandmarks = dict_data[sample],dict_index[sample],dict_nlandmarks[sample]
                assignations,landmarks = kmeans(X=x, num_clusters=nlandmarks, distance='euclidean', tqdm_flag=False) #cuda:0')
                landmarks = landmarks.double()
                # save results 
                kmeans_landmarks_name = self.get_kmeans_landmarks_name_for_sample(sample=sample)

                self._update_dict_data(landmarks,kmeans_landmarks_name,False)
                self._update_index(nlandmarks,index=None,data_name=kmeans_landmarks_name)
                self.obs[kmeans_landmarks_name] = pd.DataFrame(assignations,index=index)

                self.quantization_with_landmarks_possible = True
                
        self.has_landmarks= True

    def compute_nystrom_anchors(self,verbose=0):
        """
        Determines the nystrom anchors using 
        Stores the results as a list of eigenvalues and the 
        
        Parameters
        anchor_basis in ['K','S','W']
        ----------
        nanchors:      <= nlandmarks (= by default). Number of anchors to determine in total (proportionnaly according to the data)
        
        # le anchor basis est encore en param car je n'ai pas réfléchi à la version pour 1 groupe
        """
        
        if verbose>0:
            print('- Compute nystrom anchors')
            print(f'\tnanchors : {self.nanchors}')

        assert(self.anchor_basis is not None)
        anchors_name = self.get_anchors_name()
        
        if anchors_name not in self.spev['anchors']:

            nanchors = self.nanchors 
            nlandmarks = self.nlandmarks
            # m = self.get_ntot(landmarks=True)
            Km = self.compute_gram(landmarks=True)
            P = self.compute_covariance_centering_matrix(quantization=False,landmarks=True,)
            
            # print('nystrom anchors',r,m,Km.shape,P.shape)
            assert(len(P)==nlandmarks)
            assert(len(Km)==nlandmarks)
            
            sp_anchors,ev_anchors = ordered_eigsy(1/nlandmarks*torch.linalg.multi_dot([P,Km,P]))        

            if sum(sp_anchors>0) ==0:
                if verbose>0:
                    print('\tNo anchors found, the dataset may have two many zero data.')

            else: 
                if sum(sp_anchors>0)<nanchors:
                    old_nanchors = self.nanchors
                    self.nanchors = sum(sp_anchors>0).item()
                    nanchors = self.nanchors
                    # ajout suite aux simu univariées ou le spectre était parfois négatif, ce qui provoquait des abérations quand on l'inversait. La solution que j'ai choisie est de tronquer le spectre uniquement aux valeurs positives et considérer les autres comme nulles. 
                    if verbose>0:
                        print(f'\tThe number of anchors is reduced from {old_nanchors} to {sum(sp_anchors>0)} for numerical stability')

                self.spev['anchors'][anchors_name] = {'sp':sp_anchors[:nanchors],'ev':ev_anchors[:,:nanchors]}
                self.has_anchors=True


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

