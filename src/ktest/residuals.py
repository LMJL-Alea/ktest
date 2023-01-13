

# import torch
from torch import mv,dot,norm,ger,eye,diag,ones,diag,matmul,float64,isnan,sort,cat,tensor
import torch
from numpy import sqrt
from .utils import ordered_eigsy
import pandas as pd

from .kernel_statistics import Statistics

"""
Tout ce qui est en rapport avec le calcul des résidus. 
Les résidus pour la troncature t sont définis comme les directions
 d'une ACP sur l'orthogonal de l'axe discriminant correspondant à la troncature t.  
"""
        ####
class Residuals(Statistics): 

    def __init__(self):
        super(Residuals,self).__init__()


    def compute_discriminant_axis_qh(self,t=None):
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
        
        n = self.get_ntot(landmarks=False)
        K = self.compute_gram()
        omega = self.compute_omega(quantization=False)
        
        cov = self.approximation_cov
        L,U = self.get_spev('covw')
        Ut = U[:,:t]
        Lt32 = diag(L[:t]**(-3/2))
        
        if cov == 'standard':
            P = self.compute_covariance_centering_matrix(quantization=False,landmarks=False)
            qh = 1/n * mv(P,mv(Ut,mv(Lt32,mv(Ut.T,mv(K,omega)))))

        elif cov == 'nystrom3':
            # pas totalement sûr de cette partie 

            m = self.get_ntot(landmarks=True)
            Lz,Uz = self.get_spev(slot='anchors')
            Lz12 = diag(Lz**-(1/2))   
            P = self.compute_covariance_centering_matrix(quantization=False,landmarks=True)
            Kmn = self.compute_kmn() 

            qh = 1/m * mv(P,mv(Uz,mv(Lz12,mv(Ut,mv(Lt32,mv(Ut.T,mv(Lz12,mv(Uz.T,mv(P,mv(Kmn,omega))))))))))
        else:
            print(f'approximation {cov} not computed yet for residuals')
        return(qh)

    def project_on_discriminant_axis(self,t=None):
        '''
        This functions projects the data on the discriminant axis through the formula 
        hh^T kx =  (<kx1,h>h,...,<kxn,h>h) = (kx qh qh^T K )/|| kx qh ||^2 
        Thus with a kernel trick, we compute the matrix 
        qh qh^T K / || kx qh ||^2 

        equivalent à proj_kfda ? 
        '''

        cov = self.approximation_cov
        qh = self.compute_discriminant_axis_qh(t=t)
        
        if cov == 'standard': 
            K  = self.compute_gram(landmarks=False)
            proj = mv(K,qh)/(dot(mv(K,qh),qh))**(1/2)
        if cov == 'nystrom3':
            # Pas totalement sur de cette partie 
            # Fait à un moment ou c'était urgent 
            # il y a du nystrom dans le dénominateur ?

            K  = self.compute_gram(landmarks=True)
            Kmn = self.compute_kmn() 
            proj = mv(Kmn.T,qh)/(dot(mv(K,qh),qh))**(1/2)
            
        return(proj)

    def compute_proj_on_discriminant_orthogonal(self,t=None):
        ''' 
        Compute P_\epsilon which is such that kx P_\epsilon = \epsilon = kx - hh^tkx
         \epsilon is the residual of the observation, orthogonal to the discriminant axis.  
        '''
        cov = self.approximation_cov
        n = self.get_ntot(landmarks=False)
        
        qh = self.compute_discriminant_axis_qh(t=t)
        In = eye(n,dtype=float64)
            
        if cov == 'standard':    
            K  = self.compute_gram()
            P_epsilon = In - matmul(ger(qh,qh),K)/dot(mv(K,qh),qh)
        if cov == 'nystrom3':
            m = self.get_ntot(landmarks=True)
            K  = self.compute_gram(landmarks=True)
            Kmn = self.compute_kmn()
            P_epsilon = In[:m] - matmul(ger(qh,qh),Kmn)/dot(mv(K,qh),qh)

        return(P_epsilon)

    def compute_residual_covariance(self,t=None,center = 'W'):
        
        cov = self.approximation_cov
        nm = self.get_ntot(landmarks=(cov=='nystrom3'))
                
        if center.lower() == 't':
            In = eye(nm, dtype=float64)
            Jn = ones(nm, nm, dtype=float64)
            P = In - 1/nm * Jn
        elif center.lower() =='w':
            P = self.compute_covariance_centering_matrix(quantization=False,landmarks=(cov=='nystrom3'))
        P_epsilon = self.compute_proj_on_discriminant_orthogonal(t)
        
        if cov == 'standard': 
            n = self.get_ntot(landmarks=False)
            K = self.compute_gram()
            if center.lower() in 'tw':
                K_epsilon = 1/n * torch.linalg.multi_dot([P,P_epsilon.T,K,P_epsilon,P])
            else :
                K_epsilon = 1/n * torch.linalg.multi_dot([P_epsilon.T,K,P_epsilon])
        if cov == 'nystrom3':
            
            n = self.get_ntot(landmarks=False)
            m = self.get_ntot(landmarks=True)
            Lz,Uz = self.get_spev(slot='anchors')
            Lz12 = diag(Lz**-(1/2))  
            Pz = self.compute_covariance_centering_matrix(quantization=False,landmarks=True)
            Kmn = self.compute_kmn() 

            if center.lower() in 'tw':
                K_epsilon = 1/(n*m) * torch.linalg.multi_dot([Lz12,Uz.T,Pz,Kmn,P_epsilon.T,P,P_epsilon,
                Kmn.T,Pz,Uz,Lz12])
            else : 
                K_epsilon = 1/(n*m) * torch.linalg.multi_dot([Lz12,Uz.T,Pz,Kmn,P_epsilon.T,P_epsilon,
                Kmn.T,Pz,Uz,Lz12])
        return(K_epsilon)

    def diagonalize_residual_covariance(self,t=None,center='W'):
        if t is None:
            t=self.t
        cov = self.approximation_cov    
        
        residuals_name = self.get_residuals_name(t=t,center=center)
        
        if residuals_name not in self.spev['residuals']:
            nm = self.get_ntot(landmarks=(cov=='nystrom3'))
            if center.lower() == 't':
                In = eye(nm, dtype=float64)
                Jn = ones(nm, nm, dtype=float64)
                P = In - 1/nm * Jn
            elif center.lower() =='w':
                P = self.compute_covariance_centering_matrix(quantization=False,landmarks=(cov=='nystrom3'))
            K_epsilon = self.compute_residual_covariance(t,center=center)
            P_epsilon = self.compute_proj_on_discriminant_orthogonal(t)
            n = self.get_ntot(landmarks=False)

            sp,ev = ordered_eigsy(K_epsilon)
            L_12 = diag(sp**-(1/2))
            if cov == 'standard':
                fv = 1/sqrt(n)* torch.linalg.multi_dot([P_epsilon,P,ev,L_12])
            if cov == 'nystrom3':
                anchors_name = self.get_anchors_name()
                m = self.get_ntot(landmarks=True)
                Lz1,Uz = self.get_spev(slot='anchors')
                Lz = diag(Lz1**-1)
                Pz = self.compute_covariance_centering_matrix(quantization=False,landmarks=True)
                Kmn = self.compute_kmn() 
                
                if len(ev) != len(P):
                    P = P[:,:len(ev)]
                fv = 1/sqrt(n) * 1/m * torch.linalg.multi_dot([Pz,Uz,Lz,Uz.T,Kmn,P_epsilon.T,P,ev,L_12])
            self.spev['residuals'][residuals_name] = {'sp':sp,'ev':fv}
            return(residuals_name)
            
    def proj_residus(self,t = None,ndirections=10,center='w'):
        if t is None:
            t = self.t
        cov = self.approximation_cov    
        
        residuals_name = self.get_residuals_name(t=t,center=center,)
        
        if residuals_name not in self.df_proj_residuals:
            _,Ures = self.get_spev('residuals',t=t,center=center)
            epsilon = Ures[:,:ndirections]

            if cov == 'standard':
                K = self.compute_gram()
                proj_residus = matmul(K,epsilon)
            if cov == 'nystrom3':
                Kmn = self.compute_kmn() 
                proj_residus = matmul(Kmn.T,epsilon)

            index = self.get_xy_index()
            columns = [str(i) for i in range(1,ndirections+1)]
            self.df_proj_residuals[residuals_name] = pd.DataFrame(proj_residus,
                        index= index,columns=columns)
        return(residuals_name)

    def residuals(self,t=None,ndirections=10,center='w'):
        self.diagonalize_residual_covariance(t=t,center=center)
        self.proj_residus(t=t,ndirections=ndirections,center=center)


        




