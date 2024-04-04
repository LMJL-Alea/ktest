import torch
import numpy as np
import pandas as pd
from sklearn.cluster import kmeans_plusplus

class Data():
    """
    Class transforming two-sample data provided by the user into the quantities
    compatible with the kernel test statistic calculations performed by the
    class Statistics.
    
    Parameters
    ----------
        data : 1 or 2-dimensional array-like
            The data to test, containing two samples at the same time. 
            Can be either Pandas.DataFrame, Pandas.Series (if univariate data), 
            numpy.ndarray or torch.tensor. Rows correspond to observations,
            and columns correspond to features. If Pandas.DataFrame or
            Pandas.Series, and so is 'data', assigned indices are used to 
            label observations. Otherwise, observations are treated with 
            respect to the provided order.

        metadata : 1-dimensional array-like
            The metadata associated with the data to test. The values should 
            indicate the labels of the samples in 'data'. At the moment, the
            library performs a two sample test only, thus two or more levels 
            are expected. Can be either Pandas.DataFrame, Pandas.Series
            (if univariate data), numpy.ndarray or torch.tensor. If 
            Pandas.DataFrame or Pandas.Series, ans so is 'data', indices should
            coincide. Otherwise, the first dimensions of 'data' and 'metadata',
            corresponding to observations, should coincide.
            
        sample_names : None or 1-dimensional array-like, optional
            If given, should containt exactly two strings, labeling the samples
            (overrides the metadata levels). If None, sample names are derived 
            from the metadata levels (the first two detected levels).
            
        nystrom : bool, optional
            If True, computes the Nystrom landmarks, in which case all 
            attributes correspond to the landmarks and not the original data.
            The default if False.
    
        n_landmarks: int, optional
            Number of landmarks used in the Nystrom method. If unspecified, one
            fifth of the observations are selected as landmarks.
    
        landmark_method : 'random' or 'kmeans++', optional
            Method of the landmarks selection, 'random '(default) corresponds 
            to selecting landmarks among the observations according to the 
            random uniform distribution.
    
        random_state :  int, RandomState instance or None
            Determines random number generation for the landmarks selection. 
            If None (default), the generator is the RandomState instance used 
            by `np.random`. To ensure the results are reproducible, pass an int
            to instanciate the seed, or a RandomState instance (recommended).
            
    Attributes
    ----------
        sample_names : 1-dimensional array-like
            Identical to the corresponding parameter if provided by the user.
            Otherwise, the first two sample labels detected from metadata.
            
        data : dict
            Keys correspond to sample names, values are two torch.tensors
            corresponding to the data extracted for each sample.
         
        index : dict
            Keys correspond to sample names, values are two array-likes 
            corresponding to observation labels either extracted from the data 
            or asigned automatically.
        
        variables : 1-dimensional array-like
            Array-like of size nvar corresponding to variable (feature) labels
            either extracted from the data or asigned automatically.
    
        nobs : dict
            Keys correspond to sample names, values are two integers 
            corresonding to the numbers of observations in each sample.
    
        nvar : int
            Number of variables in the dataset.
            
        ntot : int
            Total numbetr of variables in the dataset (sum of nobs).

    """  
    def __init__(self, data, metadata, sample_names=None, nystrom=False,
                 n_landmarks=None, landmark_method='kmeans++', random_state=None):
        self.data = {}
        self.index = {}
        self.nobs = {}

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
            if sample_names is not None:
                levels = sample_names
                assrt_str_2 = 'Sample names must be present in metadata'
                assert np.isin(levels, meta_fmt.values).all(), assrt_str_2
                assrt_str_3 = 'Exactly 2 sample names are expected'
            else:
                levels = (meta_fmt.unique()[:2] if hasattr(meta_fmt, 'unique')
                          else np.unique(metadata)[:2])
                assrt_str_3 = 'Metadata has to include at least 2 distinct values for 2-sample test'
            assert len(levels) == 2, assrt_str_3
            
            self.sample_names = sample_names if sample_names is not None else levels
            self.data[self.sample_names[0]] = data[meta_fmt == levels[0]]
            self.data[self.sample_names[1]] = data[meta_fmt == levels[1]]
            
            self.nvar = data.shape[1] if len(data.squeeze().shape) == 2 else 1
            self.variables = (data.columns if hasattr(data, 'columns')
                              else np.arange(self.nvar))
            
            for n, data_n in self.data.items():
                self.nobs[n] = data_n.shape[0]
                self.index[n] = (data_n.index if hasattr(data_n, 'index')
                                 else np.arange(self.nobs[n]))
                
                # Convert data to tensor:
                if isinstance(data_n, pd.Series):
                    self.data[n] = (torch.from_numpy(data_n.to_numpy()
                                                     .reshape(-1,1)).double())
                if isinstance(data_n, pd.DataFrame):
                    self.data[n] = torch.from_numpy(data_n.to_numpy()).double()
                if isinstance(data_n, torch.Tensor):
                    self.data[n] = data_n.double()
                else:
                    X = (data_n.to_numpy() if not isinstance(data_n, np.ndarray)
                         else data_n.copy())
                    self.data[n] = torch.from_numpy(X).double()
                
                if nystrom:
                    n_landmarks_n = (n_landmarks * self.nobs[n] // data.shape[0]
                                     if n_landmarks is not None 
                                     else self.nobs[n] // 5)
                    if landmark_method == 'random':
                        ny_ind = random_state.choice(self.nobs[n],
                                                     size=n_landmarks_n,
                                                     replace=False)
                    if landmark_method == 'kmeans++':
                        rnd_st = (random_state if isinstance(random_state,
                                                             (np.random.RandomState, int))
                                  else None)
                        _, ny_ind = kmeans_plusplus(self.data[n].numpy(), 
                                                    n_clusters=n_landmarks_n, 
                                                    random_state=rnd_st)
                    self.data[n] = self.data[n][ny_ind]
                    self.nobs[n] = n_landmarks_n
                    self.index[n] = self.index[n][ny_ind]
                    
        except AttributeError:
            print(f'Unknown data type {type(data_n)}')
        self.ntot = sum(self.nobs.values())
        
    def __str__(self):
        s = f"\n{self.nvar} features across {self.ntot} observations"
        s += "\nComparison: "
        s += f"{self.sample_names[0]} ({self.nobs[self.sample_names[0]]} observations)"
        s+= f" and {self.sample_names[1]} ({self.nobs[self.sample_names[1]]} observations)."
        return(s)

   