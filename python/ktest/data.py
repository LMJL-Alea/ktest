import torch
import numpy as np
import pandas as pd
from sklearn.cluster import kmeans_plusplus

class Data():
    """
    Attributes: 
    ------------------------
    data: dict with two torch.tensors
        data matrix initialized from data
     
    index: pandas.index 
        indexes of the observations
    
    variables: pandas.index 
        indexes of the variables

    nobs: int
        number of observations 

    nvar: int 
        number of variables
        
    sample_names: 

    """  
    def __init__(self, data, metadata, sample_names=None, nystrom=False,
                 n_landmarks=None, landmark_method='kmeans++', random_state=None):
        """

        Parameters
        ----------
            data : Pandas.DataFrame 
                the data dataframe
            
            metadata : Pandas.DataFrame 
                the metadata dataframe
                
            sample_names: 
                
        """
        self.data = {}
        self.index = {}
        self.variables = {}
        self.nvar = {}
        self.nobs = {}
        self.sample_names = sample_names if sample_names is not None else ['Sample 1',
                                                                           'Sample 2']
        try:
            assert len(data.squeeze().shape) <= 2, 'Data has to be at most 2-dimensional'
            assert len(metadata.squeeze().shape) == 1, 'Metadata has to be 1-dimensional'
            if not (hasattr(data, 'index') or hasattr(metadata, 'index')):
                assrt_str_1 = 'If index is not provided, '
                assrt_str_1 += 'data and metadata have to have the same first dimension'
                assert data.shape[0] == data.shape, assrt_str_1
            if isinstance(metadata, pd.DataFrame):
                meta_fmt = metadata[metadata.columns[0]]
            else:
                meta_fmt = metadata.copy()
            levels = meta_fmt.unique() if hasattr(meta_fmt, 'unique') else np.unique(metadata)
            assrt_str_2 = 'Metadata required to contain exactly 2 levels for two-sample test'
            assert len(levels) == 2, assrt_str_2
            
            self.sample_names = sample_names if sample_names is not None else levels
            self.data[self.sample_names[0]] = data[meta_fmt == levels[0]]
            self.data[self.sample_names[1]] = data[meta_fmt == levels[1]]
            
            for n, data_n in self.data.items():
                self.nobs[n] = data_n.shape[0]
                self.nvar[n] = data_n.shape[1] if len(data_n.squeeze().shape) == 2 else 1
                self.index[n] = data_n.index if hasattr(data_n, 'index') else np.arange(self.nobs[n])
                self.variables[n] = data_n.columns if hasattr(data_n, 'columns') else np.arange(self.var[n])
                
                # Convert data to tensor:
                if isinstance(data_n, pd.Series):
                    self.data[n] = torch.from_numpy(data_n.to_numpy().reshape(-1,1)).double()
                if isinstance(data_n, pd.DataFrame):
                    self.data[n] = torch.from_numpy(data_n.to_numpy()).double()
                if isinstance(data_n, torch.Tensor):
                    self.data[n] = data_n.double()
                else:
                    X = data_n.to_numpy() if not isinstance(data_n, np.ndarray) else data_n.copy()
                    self.data[n] = torch.from_numpy(X).double()
                
                if nystrom:
                    n_landmarks_n = n_landmarks if n_landmarks is not None else self.nobs[n] // 5
                    if landmark_method == 'random':
                        ny_ind = random_state.choice(self.nobs[n],
                                                     size=n_landmarks_n,
                                                     replace=False)
                    if landmark_method == 'kmeans++':
                        _, ny_ind = kmeans_plusplus(self.data[n].numpy(), 
                                                    n_clusters=n_landmarks_n, 
                                                    random_state=random_state)
                    self.data[n] = self.data[n][ny_ind]
                    self.nobs[n] = n_landmarks_n
                    self.index[n] = self.index[n][ny_ind]
                    
        except AttributeError:
            print(f'Unknown data type {type(data_n)}')
        self.ntot = sum(self.nobs.values())

   
