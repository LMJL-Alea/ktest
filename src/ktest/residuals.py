

# import torch
from torch import mv,dot,norm,ger,eye,diag,ones,diag,matmul,chain_matmul,float64,isnan,sort,cat,tensor
from numpy import sqrt
from .utils import ordered_eigsy
import pandas as pd

from ktest.statistics import Statistics

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
                K_epsilon = 1/n * chain_matmul(P,P_epsilon.T,K,P_epsilon,P)
            else :
                K_epsilon = 1/n * chain_matmul(P_epsilon.T,K,P_epsilon)
        if cov == 'nystrom3':
            
            n = self.get_ntot(landmarks=False)
            m = self.get_ntot(landmarks=True)
            Lz,Uz = self.get_spev(slot='anchors')
            Lz12 = diag(Lz**-(1/2))  
            Pz = self.compute_covariance_centering_matrix(quantization=False,landmarks=True)
            Kmn = self.compute_kmn() 

            if center.lower() in 'tw':
                K_epsilon = 1/(n*m) * chain_matmul(Lz12,Uz.T,Pz,Kmn,P_epsilon.T,P,P_epsilon,
                Kmn.T,Pz,Uz,Lz12)
            else : 
                K_epsilon = 1/(n*m) * chain_matmul(Lz12,Uz.T,Pz,Kmn,P_epsilon.T,P_epsilon,
                Kmn.T,Pz,Uz,Lz12)
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
                fv = 1/sqrt(n)* chain_matmul(P_epsilon,P,ev,L_12)
            if cov == 'nystrom3':
                anchors_name = self.get_anchors_name()
                m = self.get_ntot(landmarks=True)
                Lz1,Uz = self.get_spev(slot='anchors')
                Lz = diag(Lz1**-1)
                Pz = self.compute_covariance_centering_matrix(quantization=False,landmarks=True)
                Kmn = self.compute_kmn() 
                
                if len(ev) != len(P):
                    P = P[:,:len(ev)]
                fv = 1/sqrt(n) * 1/m * chain_matmul(Pz,Uz,Lz,Uz.T,Kmn,P_epsilon.T,P,ev,L_12)
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

    def get_between_covariance_projection_error(self,return_total=False):
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
        
        n = self.get_ntot(landmarks=False)
        sp,ev = self.get_spev('covw')  
        sp12 = sp**(-1/2)
        ev = ev[:,~isnan(sp12)]
        sp12 = sp12[~isnan(sp12)]
        fv    = n**(-1/2)*sp12*ev if cov == 'standard' else ev 
        
        P = self.compute_covariance_centering_matrix(quantization=(cov=='quantization'),landmarks=False)
        K = self.compute_gram(landmarks=False)
        om = self.compute_omega()
        
        if cov != 'standard':
            m = self.get_ntot(landmarks=True)
            Lz,Uz = self.get_spev(slot='anchors')
            Lz12 = diag(Lz**-(1/2))
            Pz = self.compute_covariance_centering_matrix(quantization=False,landmarks=True)
            Kzx = self.compute_kmn()
            # print(f'm{m},fv{fv.shape} Lz12 {Lz12.shape} Uz{Uz.shape} Pz {Pz.shape} Kzx {Kzx.shape} om {om.shape}')
            mmdt = (m**(-1/2)* mv(fv.T,mv(Lz12,mv(Uz.T,mv(Pz,mv(Kzx,om)))))**2).cumsum(0)**(1/2)

        else:
            pkm = self.compute_pkm()
            mmdt =(mv(fv.T,pkm)**2).cumsum(0)**(1/2)

        # cette version crée des erreurs car la norme calculée ainsi
        # est plus petite que la norme reconstruite.
        # Probablement un cumul de l'erreur de la direction exacte des vecteurs propres
        # Mais je n'ai jamais réussi à le montrer clairement donc peut être que l'erreur est ailleurs. 
        delta = sqrt(dot(om,mv(K,om))) # || mu2 - mu1 || = wKw

        delta = mmdt[-1]

        
        
        projection_reconstruction = (mmdt/delta) 
        projection_error = 1 -projection_reconstruction
        if return_total:
            return(projection_error,delta)
        else:
            return projection_error

    def get_between_covariance_projection_error_associated_to_t(self,t):
        pe = self.get_between_covariance_projection_error()
        pe = cat([tensor([1],dtype =float64),pe])
        return(pe[t].item())

    def get_within_covariance_explained_variance_associated_to_t(self,t):
        exv = 1-self.get_explained_variance()
        exv = cat([tensor([1],dtype =float64),exv])
        return(exv[t].item())

    def get_ordered_spectrum_wrt_between_covariance_projection_error(self):
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
        print("attention la fonction get_between_covariance_projection_error a été modifiée mais cette" +\
            "fonction get_ordered_spectrum_wrt_between_covariance_projection_error n'a pas été modifiée" +\
            "car je ne savais pas si elle avait encore un intérêt.")
        eB = self.get_between_covariance_projection_error()

        eB = cat((tensor([1],dtype=float64),eB))
        projection_error = eB[1:] - eB[:-1]
        projection_error = projection_error[~isnan(projection_error)]
        sorted_projection_error,ordered_truncations  = sort(projection_error,descending = True)
        ordered_truncations += 1
        
        return(sorted_projection_error,ordered_truncations)
        
        
