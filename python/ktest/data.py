import torch
import numpy as np
import pandas as pd

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
    def __init__(self, data_1, data_2, sample_names=None):
        """

        Parameters
        ----------
            data_1 : Pandas.DataFrame 
                the data dataframe
            
            data_2 : Pandas.DataFrame 
                the data dataframe
                
            sample_names: 
                
        """
        self.data = {}
        self.index = {}
        self.variables = {}
        self.nvar = {}
        self.nobs = {}
        self.sample_names = sample_names if sample_names is not None else ['Sample 1',
                                                                           'Sample 2']
        self.data[self.sample_names[0]] = data_1
        self.data[self.sample_names[1]] = data_2
            
        for n, data_n in self.data.items():
            try:
                assert len(data_n.shape) <= 2, 'Data has to be at most 2-dimensional'
                self.nobs[n] = data_n.shape[0]
                self.nvar[n] = data_n.shape[1] if len(data_n.shape) == 2 else 1
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
            except AttributeError:
                print(f'unknown data type {type(data_n)}')
                break
        self.ntot = sum(self.nobs.values())

   