

# import torch
from torch import mv,dot,norm,ger,eye,diag,ones,diag,matmul,chain_matmul,float64
from numpy import sqrt
from .utils import ordered_eigsy
import pandas as pd


        ####
def compute_discriminant_axis_qh(self,t=None):
    '''
    The kernel trick on discriminant axis is a vector qh such that 
    h = kx qh/|| kx qh || = Sigma_t^-1/2 (\mu_2 - \mu_1)/ || Sigma_t^-1/2 (\mu_2 - \mu_1) ||
    where h is the discriminant axis in the RKHS obtained through KFDA. 
    This function is not compatible with the nystrom approach yet. 
    t is the truncation parameter, i.e. the number of eigenfunctions of the within covariance operator on which h is constructed.  
    '''
    
    if t is None:
        t = self.t
    
    n = self.n1+self.n2
    K = self.compute_gram()
    m = self.compute_omega(sample='xy',quantization=False)
    P = self.compute_covariance_centering_matrix(sample='xy',quantization=False,landmarks=False)
    
    cov,mmd = self.approximation_cov,self.approximation_mmd
    suffix_nystrom = self.anchors_basis if 'nystrom' in cov else ''
    assert(cov=='standard')
    Lt32 = diag(self.spev['xy'][cov+suffix_nystrom]['sp'][:t]**(-3/2))
    Ut = self.spev['xy'][cov+suffix_nystrom]['ev'][:t]
#     print(P.shape,Ut.T.shape,Lt32.shape,Ut.shape,K.shape,m.shape)
    qh = 1/n * mv(P,mv(Ut.T,mv(Lt32,mv(Ut,mv(K,m)))))
    return(qh)

def project_on_discriminant_axis(self,t=None):
    '''
    This functions projects the data on the discriminant axis through the formula 
    hh^T kx =  (<kx1,h>h,...,<kxn,h>h) = (kx qh qh^T K )/|| kx qh ||^2 
    Thus with a kernel trick, we compute the matrix 
    qh qh^T K / || kx qh ||^2 
    '''
    qh = compute_discriminant_axis_qh(self,t=t)
    K  = self.compute_gram()
    proj = mv(K,qh)/(dot(mv(K,qh),qh))**(1/2)
    return(proj)

def compute_proj_on_discriminant_orthogonal(self,t=None):
    ''' 
    Compute P_\epsilon which is such that kx P_\epsilon = \epsilon = kx - hh^tkx
     \epsilon is the residual of the observation, orthogonal to the discriminant axis.  
    '''
    n = self.n1+self.n2
    In = eye(n,dtype=float64)
    qh = compute_discriminant_axis_qh(self,t=t)
    K  = self.compute_gram()
    
    P_epsilon = In - matmul(ger(qh,qh),K)/dot(mv(K,qh),qh)
    return(P_epsilon)

def compute_residual_covariance(self,t=None,center = 'W'):
    n = self.n1+self.n2
    if center.lower() == 't':
        In = eye(n, dtype=float64)
        Jn = ones(n, n, dtype=float64)
        P = In - 1/n * Jn
    elif center.lower() =='w':
        P = self.compute_covariance_centering_matrix()
    P_epsilon = compute_proj_on_discriminant_orthogonal(self,t)
    K = self.compute_gram()
    if center.lower() in 'tw':
        K_epsilon = 1/n * chain_matmul(P,P_epsilon.T,K,P_epsilon,P)
    else :
        K_epsilon = 1/n * chain_matmul(P_epsilon.T,K,P_epsilon)
    return(K_epsilon)

def diagonalize_residual_covariance(self,t=None,center='W'):
    if t is None:
        t=self.t
    approximation = 'standard' # pour l'instant 
    suffix_nystrom = '' # pour l'instant 
    
    name = f'{approximation}{suffix_nystrom}{center.lower()}{t}'   
    if name not in self.spev['residuals']:
        K_epsilon = compute_residual_covariance(self,t,center=center)
        P_epsilon = compute_proj_on_discriminant_orthogonal(self,t)
        n = self.n1+self.n2

        # modifier centering matrix pour prendre cette diff en compte 
        if center.lower() == 't':
            In = eye(n, dtype=float64)
            Jn = ones(n, n, dtype=float64)
            P = In - 1/n * Jn
        elif center.lower() =='w':
            P = self.compute_covariance_centering_matrix()
            
        sp,ev = ordered_eigsy(K_epsilon)
        # print('Kw',Kw,'sp',sp,'ev',ev)
    #     suffix_nystrom = self.anchors_basis if 'nystrom' in approximation else ''
        if 'residus' not in self.spev:
            self.spev['residuals'] = {}
        L_12 = diag(sp**-(1/2))
        self.spev['residuals'][name] = {'sp':sp,'ev':1/sqrt(n)* chain_matmul(P_epsilon,P,ev,L_12)}

def proj_residus(self,t = None,ndirections=10,center='w'):
    if t is None:
        t = self.t
    name_residus = f'standard{center.lower()}{t}'
    if name_residus not in self.df_proj_residuals:
        K = self.compute_gram()
        epsilon = self.spev['residuals'][name_residus]['ev'][:,:ndirections]
        proj_residus = matmul(K,epsilon)
        # if not hasattr(self,'df_proj_residus'):
        #     self.df_proj_residus = {}
        self.df_proj_residuals[name_residus] = pd.DataFrame(proj_residus,index= self.index[self.imask],columns=[str(i) for i in range(1,ndirections+1)])
        self.df_proj_residuals[name_residus]['sample'] =['x']*self.n1 + ['y']*self.n2
