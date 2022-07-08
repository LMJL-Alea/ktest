

# import torch
from torch import mv,dot,norm,ger,eye,diag,ones,diag,matmul,chain_matmul,float64,isnan,sort,cat,tensor
from numpy import sqrt
from .utils import ordered_eigsy
import pandas as pd


        ####
def compute_discriminant_axis_qh(self,t=None,outliers_in_obs=None):
    '''
    The kernel trick on discriminant axis is a vector qh such that 
    h = kx qh/|| kx qh || = Sigma_t^-1/2 (\mu_2 - \mu_1)/ || Sigma_t^-1/2 (\mu_2 - \mu_1) ||
    where h is the discriminant axis in the RKHS obtained through KFDA. 
    This function is not compatible with the nystrom approach yet. 
    t is the truncation parameter, i.e. the number of eigenfunctions of the within covariance operator on which h is constructed.  
 
    13/06/2022 : an amelioration would be to scale the projection with n1n2/n as it is done in proj_kfda
    '''
    
    if t is None:
        t = self.t
    
    n1,n2,n = self.get_n1n2n(outliers_in_obs=outliers_in_obs)
    K = self.compute_gram(outliers_in_obs=outliers_in_obs)
    omega = self.compute_omega(sample='xy',quantization=False,outliers_in_obs=outliers_in_obs)
    
    cov,mmd = self.approximation_cov,self.approximation_mmd
    anchors_basis = self.anchors_basis
    suffix_nystrom = anchors_basis if 'nystrom' in cov else ''
    suffix_outliers = outliers_in_obs if outliers_in_obs is not None else ''
    Ut = self.spev['xy'][f'{cov}{suffix_nystrom}{suffix_outliers}']['ev'][:,:t]
    Lt32 = diag(self.spev['xy'][f'{cov}{suffix_nystrom}{suffix_outliers}']['sp'][:t]**(-3/2))
    
    if cov == 'standard':
        P = self.compute_covariance_centering_matrix(sample='xy',quantization=False,landmarks=False,outliers_in_obs=outliers_in_obs)
        qh = 1/n * mv(P,mv(Ut,mv(Lt32,mv(Ut.T,mv(K,omega)))))

    if cov == 'nystrom3':
        # pas totalement sûr de cette partie 
        anchor_name = f'{anchors_basis}{suffix_outliers}'
        m1,m2,m = self.get_n1n2n(landmarks=True,outliers_in_obs=outliers_in_obs)
        Uz = self.spev['xy']['anchors'][anchor_name]['ev']
        Lz = diag(self.spev['xy']['anchors'][anchor_name]['sp']**-1)
        Lz12 = diag(self.spev['xy']['anchors'][anchor_name]['sp']**-(1/2))   
        P = self.compute_covariance_centering_matrix(sample='xy',
            quantization=False,landmarks=True,outliers_in_obs=outliers_in_obs)
        Kmn = self.compute_kmn(outliers_in_obs=outliers_in_obs) 

        qh = 1/m * mv(P,mv(Uz,mv(Lz12,mv(Ut,mv(Lt32,mv(Ut.T,mv(Lz12,mv(Uz.T,mv(P,mv(Kmn,omega))))))))))
      
    return(qh)

def project_on_discriminant_axis(self,t=None,outliers_in_obs=None):
    '''
    This functions projects the data on the discriminant axis through the formula 
    hh^T kx =  (<kx1,h>h,...,<kxn,h>h) = (kx qh qh^T K )/|| kx qh ||^2 
    Thus with a kernel trick, we compute the matrix 
    qh qh^T K / || kx qh ||^2 

    equivalent à proj_kfda ? 
    '''
    cov = self.approximation_cov
    qh = self.compute_discriminant_axis_qh(t=t,outliers_in_obs=outliers_in_obs)
    
    if cov == 'standard': 
        K  = self.compute_gram(outliers_in_obs=outliers_in_obs)
        proj = mv(K,qh)/(dot(mv(K,qh),qh))**(1/2)
    if cov == 'nystrom3':
        # Pas totalement sur de cette partie 
        # Fait à un moment ou c'était urgent 
        # il y a du nystrom dans le dénominateur ?

        K  = self.compute_gram(landmarks=True,outliers_in_obs=outliers_in_obs)
        Kmn = self.compute_kmn(outliers_in_obs=outliers_in_obs) 
        proj = mv(Kmn.T,qh)/(dot(mv(K,qh),qh))**(1/2)
        
    return(proj)

def compute_proj_on_discriminant_orthogonal(self,t=None,outliers_in_obs=None):
    ''' 
    Compute P_\epsilon which is such that kx P_\epsilon = \epsilon = kx - hh^tkx
     \epsilon is the residual of the observation, orthogonal to the discriminant axis.  
    '''
    cov = self.approximation_cov
    n1,n2,n = self.get_n1n2n(outliers_in_obs=outliers_in_obs)
    qh = self.compute_discriminant_axis_qh(t=t,outliers_in_obs=outliers_in_obs)
    In = eye(n,dtype=float64)
        
    if cov == 'standard':    
        K  = self.compute_gram(outliers_in_obs=outliers_in_obs)
        P_epsilon = In - matmul(ger(qh,qh),K)/dot(mv(K,qh),qh)
    if cov == 'nystrom3':
        m1,m2,m = self.get_n1n2n(landmarks=True)
        K  = self.compute_gram(landmarks=True,outliers_in_obs=outliers_in_obs)
        Kmn = self.compute_kmn(outliers_in_obs=outliers_in_obs)
        P_epsilon = In[:m] - matmul(ger(qh,qh),Kmn)/dot(mv(K,qh),qh)

    return(P_epsilon)

def compute_residual_covariance(self,t=None,center = 'W',outliers_in_obs=None):
    cov = self.approximation_cov
    n1,n2,nm = self.get_n1n2n(outliers_in_obs=outliers_in_obs,landmarks=(cov=='nystrom3'))
    if center.lower() == 't':
        In = eye(nm, dtype=float64)
        Jn = ones(nm, nm, dtype=float64)
        P = In - 1/nm * Jn
    elif center.lower() =='w':
        P = self.compute_covariance_centering_matrix(outliers_in_obs=outliers_in_obs,landmarks=(cov=='nystrom3'))
    P_epsilon = self.compute_proj_on_discriminant_orthogonal(t,outliers_in_obs=outliers_in_obs)
    
    if cov == 'standard': 
        n1,n2,n = self.get_n1n2n(outliers_in_obs=outliers_in_obs)
        K = self.compute_gram(outliers_in_obs=outliers_in_obs)
        if center.lower() in 'tw':
            K_epsilon = 1/n * chain_matmul(P,P_epsilon.T,K,P_epsilon,P)
        else :
            K_epsilon = 1/n * chain_matmul(P_epsilon.T,K,P_epsilon)
    if cov == 'nystrom3':
        anchors_basis = self.anchors_basis
        suffix_outliers = outliers_in_obs if outliers_in_obs is not None else ''
        anchor_name = f'{anchors_basis}{suffix_outliers}'
        n1,n2,n = self.get_n1n2n(outliers_in_obs=outliers_in_obs)
        m1,m2,m = self.get_n1n2n(landmarks=True,outliers_in_obs=outliers_in_obs)
        Uz = self.spev['xy']['anchors'][anchor_name]['ev']
        Lz12 = diag(self.spev['xy']['anchors'][anchor_name]['sp']**-(1/2))  
        Pz = self.compute_covariance_centering_matrix(sample='xy',
            quantization=False,landmarks=True,outliers_in_obs=outliers_in_obs)
        Kmn = self.compute_kmn(outliers_in_obs=outliers_in_obs) 

        if center.lower() in 'tw':
            K_epsilon = 1/(n*m) * chain_matmul(Lz12,Uz.T,Pz,Kmn,P_epsilon.T,P,P_epsilon,
            Kmn.T,Pz,Uz,Lz12)
        else : 
            K_epsilon = 1/(n*m) * chain_matmul(Lz12,Uz.T,Pz,Kmn,P_epsilon.T,P_epsilon,
            Kmn.T,Pz,Uz,Lz12)
    return(K_epsilon)

def diagonalize_residual_covariance(self,t=None,center='W',outliers_in_obs=None):
    if t is None:
        t=self.t
    cov = self.approximation_cov    
    
    suffix_center = center.lower() if center.lower() != 'w' else ''
    suffix_outliers = outliers_in_obs if outliers_in_obs is not None else ''
    suffix_nystrom = self.anchors_basis if 'nystrom' in cov else ''
        
    residuals_name = f'{cov}{suffix_center}{suffix_nystrom}{suffix_outliers}{t}' 
       
    if residuals_name not in self.spev['residuals']:
        _,_,nm = self.get_n1n2n(outliers_in_obs=outliers_in_obs,landmarks=(cov=='nystrom3'))
        if center.lower() == 't':
            In = eye(nm, dtype=float64)
            Jn = ones(nm, nm, dtype=float64)
            P = In - 1/nm * Jn
        elif center.lower() =='w':
            P = self.compute_covariance_centering_matrix(outliers_in_obs=outliers_in_obs,landmarks=(cov=='nystrom3'))
        K_epsilon = self.compute_residual_covariance(t,center=center,outliers_in_obs=outliers_in_obs)
        P_epsilon = self.compute_proj_on_discriminant_orthogonal(t,outliers_in_obs=outliers_in_obs)
        n1,n2,n = self.get_n1n2n(outliers_in_obs=outliers_in_obs)

        sp,ev = ordered_eigsy(K_epsilon)
        L_12 = diag(sp**-(1/2))
        if cov == 'standard':
            fv = 1/sqrt(n)* chain_matmul(P_epsilon,P,ev,L_12)
        if cov == 'nystrom3':
            anchors_basis = self.anchors_basis
            anchor_name = f'{anchors_basis}{suffix_outliers}'
            m1,m2,m = self.get_n1n2n(landmarks=True,outliers_in_obs=outliers_in_obs)
            Uz = self.spev['xy']['anchors'][anchor_name]['ev']
            Lz = diag(self.spev['xy']['anchors'][anchor_name]['sp']**-1)
            Pz = self.compute_covariance_centering_matrix(sample='xy',
                landmarks=True,outliers_in_obs=outliers_in_obs)
            Kmn = self.compute_kmn(outliers_in_obs=outliers_in_obs) 
            
            if len(ev) != len(P):
                P = P[:,:len(ev)]
            fv = 1/sqrt(n) * 1/m * chain_matmul(Pz,Uz,Lz,Uz.T,Kmn,P_epsilon.T,P,ev,L_12)
        self.spev['residuals'][residuals_name] = {'sp':sp,'ev':fv}
        return(residuals_name)

        
def proj_residus(self,t = None,ndirections=10,center='w',outliers_in_obs=None):
    if t is None:
        t = self.t
    n1,n2,n = self.get_n1n2n(outliers_in_obs=outliers_in_obs)
    cov = self.approximation_cov    
    suffix_center = center.lower() if center.lower() != 'w' else ''
    suffix_outliers = outliers_in_obs if outliers_in_obs is not None else ''
    suffix_nystrom = self.anchors_basis if 'nystrom' in cov else ''
        
    residuals_name = f'{cov}{suffix_center}{suffix_nystrom}{suffix_outliers}{t}' 
      
    if residuals_name not in self.df_proj_residuals:
        epsilon = self.spev['residuals'][residuals_name]['ev'][:,:ndirections]
        if cov == 'standard':
            K = self.compute_gram(outliers_in_obs=outliers_in_obs)
            proj_residus = matmul(K,epsilon)
        if cov == 'nystrom3':
            Kmn = self.compute_kmn(outliers_in_obs=outliers_in_obs) 
            proj_residus = matmul(Kmn.T,epsilon)

        # if not hasattr(self,'df_proj_residus'):
        #     self.df_proj_residus = {}
        index = self.get_index(outliers_in_obs=outliers_in_obs)
        columns = [str(i) for i in range(1,ndirections+1)]
        self.df_proj_residuals[residuals_name] = pd.DataFrame(proj_residus,
                    index= index,columns=columns)
        self.df_proj_residuals[residuals_name]['sample'] =['x']*n1 + ['y']*n2
    return(residuals_name)

def get_between_covariance_projection_error(self,outliers_in_obs=None,return_total=False):
    '''
    Returns the projection error of the unique eigenvector (\mu_2- \mu_1) of the between covariance operator. 
    The projection error is the percentage of the eigenvector obtained by projecting it 
    onto the ordered eigenvectors (default is by decreasing eigenvalues) of the within covariance operator. 
    
    Parameters
    ----------
        self : Tester, 
        Should contain the eigenvectors and eigenvalues of the within covariance operator in the attribute `spev`
        
        trunc (optionnal) : list,
        The order of the eigenvectors to project (\mu_2 - \mu_1), 
        By default, the eigenvectors are ordered by decreasing eigenvalues. 

    Returns 
    ------
        projection_error : torch.Tensor
        The projection error of (\mu_2- \mu_1) as a percentage. 
    '''
    
    cov = self.approximation_cov
    suffix_outliers = outliers_in_obs if outliers_in_obs is not None else ''
    suffix_nystrom = self.anchors_basis if 'nystrom' in self.approximation_cov else ''
    n1,n2,n = self.get_n1n2n(outliers_in_obs=outliers_in_obs)
    
        
    sp    = self.spev['xy'][f'{cov}{suffix_nystrom}{suffix_outliers}']['sp']
    ev    = self.spev['xy'][f'{cov}{suffix_nystrom}{suffix_outliers}']['ev']
    sp12 = sp**(-1/2)
    ev = ev[:,~isnan(sp12)]
    sp12 = sp12[~isnan(sp12)]
    fv    = n**(-1/2)*sp12*ev if cov == 'standard' else ev 
    
    P = self.compute_covariance_centering_matrix(sample='xy',quantization=(cov=='quantization'),outliers_in_obs=outliers_in_obs)
    K = self.compute_gram(outliers_in_obs=outliers_in_obs)
    om = self.compute_omega(outliers_in_obs=outliers_in_obs)
    
    if cov != 'standard':
        m1,m2,m = self.get_n1n2n(landmarks=True,outliers_in_obs=outliers_in_obs)
        r = self.r
        anchors_basis = self.anchors_basis
        suffix_outliers = '' if outliers_in_obs is None else outliers_in_obs 
        anchor_name = f'{anchors_basis}{suffix_outliers}'
        Uz = self.spev['xy']['anchors'][anchor_name]['ev']
        Lz = diag(self.spev['xy']['anchors'][anchor_name]['sp']**-(1/2))
        Pz = self.compute_covariance_centering_matrix(sample='xy',landmarks=True,outliers_in_obs=outliers_in_obs)
        Kzx = self.compute_kmn(sample='xy',outliers_in_obs=outliers_in_obs)
        mmdt = (m**(-1/2)* mv(fv.T,mv(Lz,mv(Uz.T,mv(Pz,mv(Kzx,om)))))**2).cumsum(0)**(1/2)

    else:
        pkm = self.compute_pkm(outliers_in_obs=outliers_in_obs)
        mmdt =(mv(fv.T,pkm)**2).cumsum(0)**(1/2)


    delta = sqrt(dot(om,mv(K,om))) # || mu2 - mu1 || = wKw

    
    if self.data['x'][f'data_p'] == 1:
        # les abérations numériques semblent ne se produire que sur les données univariées. 
        # print('between reconstruction is not exact')
        delta = mmdt[-1]
    
    projection_error = (mmdt/delta) 
    if return_total:
        return(projection_error,delta)
    else:
        return projection_error


def get_between_covariance_projection_error_associated_to_t(self,t,outliers_in_obs=None):
    pe = self.get_between_covariance_projection_error(outliers_in_obs=outliers_in_obs)
    pe = cat([tensor([0],dtype =float64),pe])
    pe = 1-pe
    return(pe[t].item())
    

def get_ordered_spectrum_wrt_between_covariance_projection_error(self,outliers_in_obs=None):
    '''
    Sorts the eigenvalues of the within covariance operator in order to 
    have the best reconstruction of (\mu_2 - \mu1)
    
    Returns 
    -------
        sorted_projection_error : torch.Tensor,
        The percentage of (\mu_2 - \mu1) captured by the eigenvector capturing the ith largest 
        percentage of (\mu_2 - \mu1) is at the ith position. 
        
        ordered_truncation : torch.Tensor,
        The position of the vector capturing the ith largest percentage of (\mu_2 - \mu1) in the list 
        of eigenvectors of the within covariance operator ordered by decreasing eigenvalues. 
    
    '''
    eB = self.get_between_covariance_projection_error(outliers_in_obs=outliers_in_obs)

    eB = cat((tensor([0],dtype=float64),eB))
    projection_error = eB[1:] - eB[:-1]
    projection_error = projection_error[~isnan(projection_error)]
    sorted_projection_error,ordered_truncations  = sort(projection_error,descending = True)
    ordered_truncations += 1
    
    return(sorted_projection_error,ordered_truncations)
    
    
