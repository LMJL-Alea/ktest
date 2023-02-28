import torch
import numpy as np
import pandas as pd
from ktest.kernels import gauss_kernel_mediane,mediane,gauss_kernel,linear_kernel
# from ktest._testdata import TestData
from time import time




"""
Les fonctions de ce scripts initialisent les informations liées au modèle ou aux données dans l'objet Ktest(). 
"""

class Verbosity:


    def verbosity(self,function_name,dict_of_variables=None,start=True,verbose=0):
        '''
        
        Parameters
        ----------

        Returns
        ------- 
        '''
        
        if verbose >0:
            end = ' ' if verbose == 1 else '\n'
            if start:  # pour verbose ==1 on start juste le chrono mais écris rien     
                self.start_times[function_name] = time()
                if verbose >1 : 
                    print(f"Starting {function_name} ...",end= end)
                    if dict_of_variables is not None:
                        for k,v in dict_of_variables.items():
                            if verbose ==2:
                                print(f'\t {k}:{v}', end = '') 
                            else:
                                print(f'\t {k}:{v}')
                    
            else: 
                start_time = self.start_times[function_name]
                print(f"Done {function_name} in  {time() - start_time:.2f}")

