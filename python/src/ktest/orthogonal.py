

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
class Orthogonal(Statistics): 

    # def __init__(self,data,obs=None,var=None):
    #     super(Orthogonal,self).__init__(data,obs=obs,var=var)


    def compute_discriminant_axis_qh(self,t):
        '''
        The kernel trick on discriminant axis is a vector qh such that 
        h = kx qh/|| kx qh || = Sigma_t^-1/2 (\mu_2 - \mu_1)/ || Sigma_t^-1/2 (\mu_2 - \mu_1) ||
        where h is the discriminant axis in the RKHS obtained through KFDA. 
        This function is not compatible with the nystrom approach yet. 
        t is the truncation parameter, i.e. the number of eigenfunctions of the within covariance operator on which h is constructed.  
    
        13/06/2022 : an amelioration would be to scale the projection with n1n2/n as it is done in proj_kfda
        '''

        
        n = self.get_ntot(landmarks=False)
        K = self.compute_gram()
        omega = self.compute_omega()
        
        nystrom = self.nystrom
        L,U = self.get_spev('covw')
        Ut = U[:,:t]
        Lt32 = diag(L[:t]**(-3/2))
        
        if nystrom:
            # pas totalement sûr de cette partie 
            n_landmarks = self.get_ntot(landmarks=True)
            Lz,Uz = self.get_spev(slot='anchors')
            Lz12 = diag(Lz**-(1/2))   
            P = self.compute_covariance_centering_matrix(landmarks=True)
            Kmn = self.compute_kmn() 
            qh = 1/n_landmarks * mv(P,mv(Uz,mv(Lz12,mv(Ut,mv(Lt32,mv(Ut.T,mv(Lz12,mv(Uz.T,mv(P,mv(Kmn,omega))))))))))
        else:
            P = self.compute_covariance_centering_matrix(landmarks=False)
            qh = 1/n * mv(P,mv(Ut,mv(Lt32,mv(Ut.T,mv(K,omega)))))
        return(qh)

    def project_on_discriminant_axis(self,t=None):
        '''
        This functions projects the data on the discriminant axis through the formula 
        hh^T kx =  (<kx1,h>h,...,<kxn,h>h) = (kx qh qh^T K )/|| kx qh ||^2 
        Thus with a kernel trick, we compute the matrix 
        qh qh^T K / || kx qh ||^2 

        equivalent à proj_kfda ? 
        '''
        nystrom = self.nystrom
        qh = self.compute_discriminant_axis_qh(t=t)
        
        if nystrom:
            # Pas totalement sur de cette partie 
            # Fait à un moment ou c'était urgent 
            # il y a du nystrom dans le dénominateur ?

            K  = self.compute_gram(landmarks=True)
            Kmn = self.compute_kmn() 
            proj = mv(Kmn.T,qh)/(dot(mv(K,qh),qh))**(1/2)
            
        else: 
            K  = self.compute_gram(landmarks=False)
            proj = mv(K,qh)/(dot(mv(K,qh),qh))**(1/2)
        return(proj)

    def compute_proj_on_discriminant_orthogonal(self,t=None):
        ''' 
        Compute P_\epsilon which is such that kx P_\epsilon = \epsilon = kx - hh^tkx
         \epsilon is the orthogonal of the observation, orthogonal to the discriminant axis.  
        '''
        nystrom = self.nystrom
        n = self.get_ntot(landmarks=False)
        
        qh = self.compute_discriminant_axis_qh(t=t)
        In = eye(n,dtype=float64)
            
        if nystrom:
            n_landmarks = self.get_ntot(landmarks=True)
            K  = self.compute_gram(landmarks=True)
            Kmn = self.compute_kmn()
            P_epsilon = In[:n_landmarks] - matmul(ger(qh,qh),Kmn)/dot(mv(K,qh),qh)
        else:
            K  = self.compute_gram()
            P_epsilon = In - matmul(ger(qh,qh),K)/dot(mv(K,qh),qh)

        return(P_epsilon)

    def compute_orthogonal_covariance(self,t=None,center = 'W'):
        
        nystrom = self.nystrom
        nm = self.get_ntot(landmarks=nystrom)
                
        if center.lower() == 't':
            In = eye(nm, dtype=float64)
            Jn = ones(nm, nm, dtype=float64)
            P = In - 1/nm * Jn
        elif center.lower() =='w':
            P = self.compute_covariance_centering_matrix(landmarks=nystrom)
        P_epsilon = self.compute_proj_on_discriminant_orthogonal(t)
        
        if nystrom:
            
            n = self.get_ntot(landmarks=False)
            n_landmarks = self.get_ntot(landmarks=True)
            Lz,Uz = self.get_spev(slot='anchors')
            Lz12 = diag(Lz**-(1/2))  
            Pz = self.compute_covariance_centering_matrix(landmarks=True)
            Kmn = self.compute_kmn() 

            if center.lower() in 'tw':
                K_epsilon = 1/(n*n_landmarks) * torch.linalg.multi_dot([Lz12,Uz.T,Pz,Kmn,P_epsilon.T,P,P_epsilon,
                Kmn.T,Pz,Uz,Lz12])
            else : 
                K_epsilon = 1/(n*n_landmarks) * torch.linalg.multi_dot([Lz12,Uz.T,Pz,Kmn,P_epsilon.T,P_epsilon,
                Kmn.T,Pz,Uz,Lz12])
        else:
            n = self.get_ntot(landmarks=False)
            K = self.compute_gram()
            if center.lower() in 'tw':
                K_epsilon = 1/n * torch.linalg.multi_dot([P,P_epsilon.T,K,P_epsilon,P])
            else :
                K_epsilon = 1/n * torch.linalg.multi_dot([P_epsilon.T,K,P_epsilon])
        return(K_epsilon)

    def diagonalize_orthogonal_covariance(self,t,center='W'):

        nystrom=self.nystrom
        
        orthogonal_name = self.get_orthogonal_name(t=t,center=center)
        
        if orthogonal_name not in self.spev['orthogonal']:
            nm = self.get_ntot(landmarks=nystrom)
            if center.lower() == 't':
                In = eye(nm, dtype=float64)
                Jn = ones(nm, nm, dtype=float64)
                P = In - 1/nm * Jn
            elif center.lower() =='w':
                P = self.compute_covariance_centering_matrix(landmarks=nystrom)
            K_epsilon = self.compute_orthogonal_covariance(t,center=center)
            P_epsilon = self.compute_proj_on_discriminant_orthogonal(t)
            n = self.get_ntot(landmarks=False)

            sp,ev = ordered_eigsy(K_epsilon)
            L_12 = diag(sp**-(1/2))
            
            if nystrom:
                anchors_name = self.get_anchors_name()
                n_landmarks = self.get_ntot(landmarks=True)
                Lz1,Uz = self.get_spev(slot='anchors')
                Lz = diag(Lz1**-1)
                Pz = self.compute_covariance_centering_matrix(landmarks=True)
                Kmn = self.compute_kmn() 
                
                if len(ev) != len(P):
                    P = P[:,:len(ev)]
                fv = 1/sqrt(n) * 1/n_landmarks * torch.linalg.multi_dot([Pz,Uz,Lz,Uz.T,Kmn,P_epsilon.T,P,ev,L_12])
            else:
                fv = 1/sqrt(n)* torch.linalg.multi_dot([P_epsilon,P,ev,L_12])
            self.spev['orthogonal'][orthogonal_name] = {'sp':sp,'ev':fv}
            return(orthogonal_name)
            
    def proj_orthogonal(self,t,ndirections=10,center='w'):

        nystrom = self.nystrom
        orthogonal_name = self.get_orthogonal_name(t=t,center=center,)
        
        if orthogonal_name not in self.df_proj_orthogonal:
            _,Ures = self.get_spev('orthogonal',t=t,center=center)
            epsilon = Ures[:,:ndirections]

            if nystrom:
                Kmn = self.compute_kmn() 
                proj_orthogonal = matmul(Kmn.T,epsilon)
            else:
                K = self.compute_gram()
                proj_orthogonal = matmul(K,epsilon)

            index = self.get_index(in_dict=False)
            if proj_orthogonal.shape[1] == ndirections:
                columns = [str(i) for i in range(1,ndirections+1)]
            else:
                columns = [str(i) for i in range(1,proj_orthogonal.shape[1]+1)]
            self.df_proj_orthogonal[orthogonal_name] = pd.DataFrame(proj_orthogonal,
                        index= index,columns=columns)
        return(orthogonal_name)

    def orthogonal(self,t=None,ndirections=10,center='w'):
        """
        Computes the orthogonal projections, namely the projections of the observations 
        on the principal components of a covariance structure computed on the space orthogonal 
        to the discriminant axis.  

        Parameters 
        ----------
            t : int 
                Truncation from which the orthogonal space is defined

            ndirections : int
                Number of principal components to compute in the orthogonal space 
                (in practice we only focus on the first one)

            center (default = 'w'): str in ['k','s','w']
                Defines the covariance structure to determine the principal components of 
                in the orthogonal space.
                k : second order moment of the embeddings in the orthogonal space
                s : total covariance structure of the embeddings in the orthogonal space
                w : within-group covariance operator
                
        """

        self.diagonalize_orthogonal_covariance(t=t,center=center)
        self.proj_orthogonal(t=t,ndirections=ndirections,center=center)


        




