
import pandas as pd
from ktest._testdata import TestData

class Kfdat:

    from .kernel_operations import \
        compute_gram,\
        compute_omega,\
        compute_kmn,\
       compute_within_covariance_centered_gram,\
        compute_covariance_centering_matrix,\
        diagonalize_centered_gram
    

    from .nystrom_operations import \
        compute_nystrom_anchors,\
        compute_nystrom_landmarks,\
        compute_quantization_weights,\
        reinitialize_landmarks,\
        reinitialize_anchors
    
    from .statistics import \
        compute_kfdat,\
        compute_pkm,\
        initialize_kfdat,\
        kfdat,\
        kpca

    from .initializations import \
        init_data,verbosity

    def __init__(self,approximation='standard',m=None,r=None,landmark_method='random',anchors_basis='w',verbose=0,approximation_mmd='standard'):
        
        if approximation == 'standard':
            assert(m is None, r is None)
        if approximation =='quantization':
            assert(r is None)
            m_method = 'kmeans'
        
        self.approximation = approximation
        self.m = m
        self.r = r
        self.landmark_method = landmark_method
        self.anchors_basis = anchors_basis
        self.verbose = 0
        self.approximation_mmd = approximation_mmd

        self.df_statistiques = pd.DataFrame
        self.proj_kpca = {}
        self.proj_kfda = {} 
        self.corr_kpca = {}
        self.corr_kfda = {}
        self.data = {}
        self.spev = {'xy':{},'x':{},'y':{}}


    def initialize_dataset(self,name,x,y,kernel=None,x_index=None,y_index=None,variables=None):
        self.init_data(x,y,kernel,x_index,y_index,variables)   
        
        self.initialize_kfdat(approximation_cov=self.approximation,approximation_mmd=self.approximation_mmd,
        sample='xy',m=self.m,r=self.r,landmark_method=self.landmark_method,anchors_basis=self.anchors_basis,verbose=self.verbose)


