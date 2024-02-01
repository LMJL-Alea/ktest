from scipy.stats import chi2
import pandas as pd
import torch
import numpy as np
from .utils import ordered_eigsy

# to do : hyp['spev'] should be in 'spev'
        # hyp['residuals'] should be in df_proj_residuals


def chi2_LT(L,T):
    trunc = range(1,T+1)
    yas = [chi2.ppf(0.95,t*len(L)) for t in trunc] 
    return(yas)

def compute_design_matrix(x):
    """
    Computes the design matrix of a MANOVA model

    For a model with one effect and p conditions (e.g. p treatments), 
    x is a list of assignations : for instance it may be x = [1,1,1,2,2,2,3,3,3] 

    For a model with k effects and pk conditions per effect (e.g. [[1,1],[1,2],[2,1]] 
    where [1,2] stand for first condition of effect 1 and second condition of effect 2)
    """
    
    df = pd.DataFrame(x)
    ncol = len(df.columns)
    X = pd.get_dummies(df, columns=range(ncol)).to_numpy()
    return(torch.tensor(X,dtype=torch.float64)[:,:])



class Hotelling_Lawley:
                

    def set_design(self,design_cols):
        """
        Defines the design matrix on the basis of the columns of the metadata dataframe 
        that are in the list `design_cols`

        Parameters 
        ----------
            design_cols : list of str
            the columns of the metadata dataframe to take into account to define the design matrix 

        """

        self.design_cols = design_cols
        self._init_design_matrix()
        self._init_XprimeX()
        self._init_XprimeXinv()
        self._init_ProjImX()
        self._init_ProjImXorthogonal()
        self._diagonalize_residual_covariance()

    def set_hypothesis(self,L,hypothesis_name,Tmax):
        '''
        Updates the attribute `self.current_hyp` with `hypothesis_name` and 
        appends an entry to the dict attribute `self.hypotheses` that is a dict 
        stored at key `hypothesis_name` and containing the test matrix `L`, the 
        truncation parameter `Tmax` ant the matrix A=L'(L(X'X)^{-}L')^{-})L, where
        X is the design matrix and M^{-} stands for the pseudo inverse of matrix M.
        A is computed through the function `self._init_L_LXXL_L()`

        Parameters :
        ------------
            L : torch Tensor
            The (l x p) test matrix where the l rows are the hypothesis to test 
            and the p columns correspond to the p parameters of the kernel linear 
            model. 

            hypothesis_name : str
            The name refering to the test matrix and truncation parameter 

            Tmax : int 
            The truncation parameter used to compute the test statistic and 
            project on the discriminant directions
        '''
        self.current_hyp = hypothesis_name
        hyp = self.hypotheses
        if hypothesis_name not in hyp:
            hyp[hypothesis_name] = {'L':L,'Tmax':Tmax}
        self._init_L_LXXL_L(hypothesis_name)
        
    def get_hypothesis(self,hypothesis_name=None):
        ''''
        Returns the dict associated to key `hypothesis_name` in the attribute 
        `self.hypotheses`. Returns the dict associated to key `self.current_hyp`
        when called with no parameter. 

        Parameters : 
        ------------
            hypothesis_name : str (default = None) 
            Key corresponding to the hypothesis of interest in `self.hypotheses`. 
            This key should have been initialized through the function 
            `self.set_hypothesis`
        '''
        if hypothesis_name is None:
            if self.current_hyp is None:
                print("You need to define an hypothesis with function 'self.set_hypothesis' first")
            hypothesis_name = self.current_hyp
        return(self.hypotheses[hypothesis_name])
        
    def _init_design_matrix(self):
        design_cols = self.design_cols
        explanatory = self.get_metadata(samples='all')[design_cols]
        data_index = self.get_index(in_dict=False,samples='all')
        design_matrix = pd.get_dummies(explanatory,columns=design_cols).loc[data_index].to_numpy()
        self.design = torch.tensor(design_matrix,dtype=torch.float64)[:,:]
    
    def _init_XprimeX(self):
        X = self.design
        XX = torch.matmul(X.T,X)
        self.XX = XX
        
    def _init_XprimeXinv(self):
        XX = self.XX
        sp,ev = ordered_eigsy(XX)
        non_zero = sp>10e-14
        sp = sp[non_zero]
        ev = ev[:,non_zero]
        XXinv = torch.linalg.multi_dot([ev,torch.diag(sp**-1),ev.T])
        self.XXinv = XXinv

    def _init_ProjImX(self):
        X = self.design
        XXinv = self.XXinv
        Pi = torch.linalg.multi_dot([X,XXinv,X.T])
        self.ProjImX = Pi

    def _init_ProjImXorthogonal(self):
        n = self.get_ntot(samples='all')
        Pi = self.ProjImX
        In = torch.eye(n)
        Piperp = In - Pi
        self.ProjImXorthogonal = Piperp

    def _init_L_LXXL_L(self,hypothesis_name=None):

        hyp = self.get_hypothesis(hypothesis_name)
        L = hyp['L']
        XXinv = self.XXinv
        LXXL = torch.linalg.multi_dot([L,XXinv,L.T])
        LXXLinv = torch.tensor([[LXXL**-1]]) if len(LXXL.shape) == 0 else LXXL.inverse()
        A = torch.linalg.multi_dot([L.T,LXXLinv,L])
        hyp['A'] = A
        

    def _compute_residual_covariance(self):
        n = self.get_ntot(samples='all')
        kernel = self.kernel
        Y = self.get_data(in_dict=False,dataframe=False,samples='all')
        K = kernel(Y,Y) # to do : call self.compute_gram()
        Piperp = self.ProjImXorthogonal
        Kresidual = 1/n * torch.linalg.multi_dot([Piperp,K,Piperp])
        return(Kresidual)

    def _diagonalize_residual_covariance(self):
        """
        S is the residal covariance operator 
        The eigenvalues \lambda of Kresiduals are the eigenvalues of S.
        If u is an eigenvector of Kresiduals 
        Then f = \Phi(Y) · n^{-1/2}\lambda^{-1/2} Piperp u is an unit eigenfunction of S
        We compute directly ev = n^{-1/2}\lambda^{-1} Piperp u as in every computation of interest we 
        need ev = \lambda^{-1/2} f  
        """
        Piperp = self.ProjImXorthogonal

        Kresidual = self._compute_residual_covariance()
        sp,ev = ordered_eigsy(Kresidual)
        n = self.get_ntot(samples='all')
        
        sp12 = sp**(-1)*n**(-1/2)
        to_ignore = sp12.isnan()
        sp12 = sp12[~to_ignore]
        U = sp12 * ev[:,~to_ignore]
        evnorm = torch.matmul(Piperp,U)
        self.spev['residuals']['sp'] = sp
        self.spev['residuals']['ev'] = evnorm

        
    def _compute_inner_products_thetai_ft(self):
        X = self.design
        XXinv = self.XXinv
        kernel = self.kernel
        Y = self.get_data(in_dict=False,dataframe=False,samples='all')
        K = kernel(Y,Y) 
        ev = self.spev['residuals']['ev']
        K_theta = torch.linalg.multi_dot([XXinv,X.T,K,ev])
        return(K_theta)
    
        
    def compute_D(self,hypothesis_name=None):
        """
        Grosse matrice facile a calculer inutile de la garder en mémoire
        """
        hyp = self.get_hypothesis(hypothesis_name)

        X = self.design
        XXinv = self.XXinv
        A = hyp['A']
        D = torch.linalg.multi_dot([X,XXinv,A,XXinv,X.T])
        return(D)
    
        

    def compute_kernel_Hotelling_Lawley_test_statistic(self,hypothesis_name=None):
        hyp = self.get_hypothesis(hypothesis_name)

        A = hyp['A']
        L = hyp['L']
        Tmax = hyp['Tmax']

        K_theta = self._compute_inner_products_thetai_ft()
        

        matrix = torch.linalg.multi_dot([K_theta.T,A,K_theta])
        stats = []
        pvals = []
        for t in range(1,Tmax):
            stat = torch.trace(matrix[:t,:t]).item()
            stats += [stat]
            pvals += [chi2.sf(np.sum(stat),t*len(L))]

        hyp['Hotellin-Lawley'] = stats
        hyp['p-value'] = pvals

    
    def compute_Kdiscriminant(self,T,hypothesis_name=None):
        U = self.spev['residuals']['ev'] 
        kernel = self.kernel
        Y = self.get_data(in_dict=False,dataframe=False,samples='all')
        K = kernel(Y,Y) 
        D = self.compute_D(hypothesis_name)
    #     print('T',T,'U',U.shape,'R',R.shape,'K',K.shape)

        Kdiscriminant = torch.linalg.multi_dot([U[:,:T].T,K,D,K,U[:,:T]])
        return(Kdiscriminant)
       
    def diagonalize_Kdiscriminant(self,T,hypothesis_name=None):
        hyp = self.get_hypothesis(hypothesis_name)
        Kdiscriminant = self.compute_Kdiscriminant(T=T,hypothesis_name=hypothesis_name)
        sp,ev = ordered_eigsy(Kdiscriminant)
        hyp['spev'] = {'sp':sp,'ev':ev}

    def compute_proj_on_discriminant_directions(self,T,hypothesis_name=None):
        if hypothesis_name is None:
            hypothesis_name = self.current_hyp

        hyp = self.get_hypothesis(hypothesis_name)
        self.diagonalize_Kdiscriminant(T=T,hypothesis_name=hypothesis_name)

        ev = hyp['spev']['ev']
        U = self.spev['residuals']['ev']
        kernel = self.kernel
        Y = self.get_data(in_dict=False,dataframe=False,samples='all')
        index = self.get_index(in_dict=False,samples='all')
        K = kernel(Y,Y) 
        projections = torch.linalg.multi_dot([ev.T,U[:,:T].T,K])
        self.df_proj_residuals[hypothesis_name] = pd.DataFrame(projections.T,index=index)

    def compute_diagnostics(self,Tmax=30):
        """
        The diagnostics are defined similarly to the diagnostic plot of the multivariate linear model. 
        The response plot represents the embeddings with respect to their predictions.
        The residual plot represents the residuals with respect to the predictions. 
        Embeddings, residuals and predictions are $n$-dimensional objects. 
        We would need $n$ diagnostic plots to draw an exhaustive picture of the diagnostics. 
        Instead, we propose to select a few directions of interest of the feature space in order to 
        compute informative but non-exhaustive diagnostic plots. 
        For a direction $h$ of interest in the feature space, we propose to represent the diagnostic plots 
        with respect to the projections of the embeddings, residuals and predisctions on $h$.
        
        
        For `Tmax` such that the first `Tmax` eigenvalues of the residuals covariance operator are >0,
        we choose the first `Tmax` eigendirections of the residual covariance operator 
        as directions of interest to compute the diagnostic plots.

        Parameters
        ----------
            Tmax (default = 30) : int
                Number of directions of interest to draw the diagnostic plot

        Returns 
        -------
            embeddings : pd.DataFrame() of shape ntot x Tmax
                Table of the positions of the embeddings projected on the `Tmax` directions of interest.

            predictions : pd.DataFrame() of shape ntot x Tmax
                Table of the positions of the predictions projected on the `Tmax` directions of interest.

            residuals : pd.DataFrame() of shape ntot x Tmax
                Table of the positions of the residuals projected on the `Tmax` directions of interest.

        """
        ev = self.spev['residuals']['ev']
        kernel = self.kernel
        Y = self.get_data(in_dict=False,dataframe=False,samples='all')
        index= self.get_index(in_dict=False,samples='all')
        K = kernel(Y,Y) 
        Pi = self.ProjImX
        Piperp = self.ProjImXorthogonal
        columns = list(range(1,Tmax+1))
        embeddings = pd.DataFrame(torch.linalg.multi_dot([K,ev[:,:Tmax]]),index=index,columns=columns)
        predictions = pd.DataFrame(torch.linalg.multi_dot([Pi,K,ev[:,:Tmax]]),index=index,columns=columns)
        residuals = pd.DataFrame(torch.linalg.multi_dot([Piperp,K,ev[:,:Tmax]]),index=index,columns=columns)
        return(embeddings,predictions,residuals)
    
    def compute_cook_distances(self,Tmax=30):
        index_data = self.get_index(in_dict=False,samples='all')
        XX = self.XX
        X = self.design
        XXinv = self.XXinv
        XXX = torch.linalg.multi_dot([X,XXinv])
        Pi = self.ProjImX
        one_minus_hii = (1 - torch.diag(Pi))
        wi_by_one_minus_hii = XXX.T/one_minus_hii
        cook_coefs = torch.diag(torch.linalg.multi_dot([wi_by_one_minus_hii.T,XX,wi_by_one_minus_hii]))
        _,_,res = self.compute_diagnostics(Tmax=Tmax)
        torch_res = torch.tensor(res.to_numpy(),dtype=torch.float64)

        cook_distances = {}
        for t in range(1,Tmax+1):
            cook_traces = torch.diag(torch.linalg.multi_dot([torch_res[:,:t],torch_res[:,:t].T]))
            cook_distances[t] = (cook_traces * cook_coefs).numpy()
            
        cook_distances = pd.DataFrame(cook_distances,index=index_data)
        return(cook_distances)
        

