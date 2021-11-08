import torch
import pandas as pd
from torch import mv



def compute_pkm(self,approximation_cov='standard',approximation_mmd='standard',anchors_basis=None,cov_anchors='shared', mmd_anchors='shared'):
    cov,mmd = approximation_cov,approximation_mmd
    # omega = self.compute_m(quantization=(mmd=='quantization'))
    omega = self.compute_omega(quantization=(mmd=='quantization'))
    Pbi = self.compute_centering_matrix(sample='xy',quantization=(cov=='quantization'))
    
    if 'nystrom' in approximation_cov or 'nystrom' in approximation_mmd :
        r = self.r

    if any([ny in [mmd,cov] for ny in ['nystrom1','nystrom2','nystrom3','nystrom']]):
        Uz = self.spev['xy']['anchors'][anchors_basis]['ev']
        Lz = torch.diag(self.spev['xy']['anchors'][anchors_basis]['sp']**-1)
        
    if not (mmd == cov) or mmd == 'nystrom':
        Kzx = self.compute_kmn(sample='xy')
    
    if cov == 'standard':
        if mmd == 'standard':
            Kx = self.compute_gram()
            pkm = mv(Pbi,mv(Kx,omega))

        elif mmd == 'nystrom':
            Pi = self.compute_centering_matrix(sample='xy',landmarks=True,anchors_basis=anchors_basis)
            pkm = 1/r * mv(Pbi,mv(Kzx.T,mv(Pi,mv(Uz,mv(Lz,mv(Uz.T,mv(Pi,mv(Kzx,omega))))))))
            # pkm = mv(Pbi,mv(Kzx.T,mv(Pi,mv(Uz,mv(Lz,mv(Uz.T,mv(Pi,mv(Kzx,omega))))))))

        elif mmd == 'quantization':
            pkm = mv(Pbi,mv(Kzx.T,omega))

    if cov == 'nystrom1' and cov_anchors == 'shared':
        if mmd in ['standard','nystrom']: # c'est exactement la même stat  
            Pi = self.compute_centering_matrix(sample='xy',landmarks=True,anchors_basis=anchors_basis)
            pkm = 1/r**2 * mv(Pbi,mv(Kzx.T,mv(Pi,mv(Uz,mv(Lz,mv(Uz.T,mv(Pi,mv(Kzx,omega))))))))
            # pkm = mv(Pbi,mv(Kzx.T,mv(Pi,mv(Uz,mv(Lz,mv(Uz.T,mv(Pi,mv(Kzx,omega))))))))

        elif mmd == 'quantization':
            Kz = self.compute_gram(landmarks=True)
            pkm = 1/r**2 * mv(Pbi,mv(Kzx.T,mv(Uz,mv(Lz,mv(Uz.T,mv(Kz,omega))))))
            # pkm = mv(Pbi,mv(Kzx.T,mv(Uz,mv(Lz,mv(Uz.T,mv(Kz,omega))))))
    
    if cov == 'nystrom2' and cov_anchors == 'shared':
        Lz12 = torch.diag(self.spev['xy']['anchors'][anchors_basis]['sp']**-(1/2))
        if mmd in ['standard','nystrom']: # c'est exactement la même stat  
            Pi = self.compute_centering_matrix(sample='xy',landmarks=True,anchors_basis=anchors_basis)
            pkm = 1/r**3 * mv(Lz12,mv(Uz.T,mv(Pi,mv(Kzx,mv(Pbi,mv(Kzx.T,mv(Pi,mv(Uz,mv(Lz,mv(Uz.T,mv(Pi,mv(Kzx,omega))))))))))))
            # pkm = mv(Lz12,mv(Uz.T,mv(Pi,mv(Kzx,mv(Pbi,mv(Kzx.T,mv(Pi,mv(Uz,mv(Lz,mv(Uz.T,mv(Pi,mv(Kzx,omega))))))))))))

        elif mmd == 'quantization': # pas à jour
            # il pourrait y avoir la dichotomie anchres centrees ou non ici. 
            Kz = self.compute_gram(landmarks=True)
            pkm = 1/r**3 * mv(Lz12,mv(Uz.T,mv(Kzx,mv(Pbi,mv(Kzx.T,mv(Uz,mv(Lz,mv(Uz.T,mv(Kz,omega)))))))))
            # pkm = mv(Lz12,mv(Uz.T,mv(Kzx,mv(Pbi,mv(Kzx.T,mv(Uz,mv(Lz,mv(Uz.T,mv(Kz,omega)))))))))
    
    if cov == 'nystrom3' and cov_anchors == 'shared':
        Lz12 = torch.diag(self.spev['xy']['anchors'][anchors_basis]['sp']**-(1/2))
        Pi = self.compute_centering_matrix(sample='xy',landmarks=True,anchors_basis=anchors_basis)
        if mmd in ['standard','nystrom']: # c'est exactement la même stat  
            pkm = 1/r * mv(Lz12,mv(Uz.T,mv(Pi,mv(Kzx,omega))))
            # pkm = mv(Lz12,mv(Uz.T,mv(Pi,mv(Kzx,omega))))

        elif mmd == 'quantization': # pas à jour 
            # il faut ajouter Pi ici . 
            Kz = self.compute_gram(landmarks=True)
            pkm = 1/r**2 * mv(Lz12,mv(Uz.T,mv(Pi,mv(Kzx,mv(Pbi,mv(Kzx.T,mv(Pi,mv(Uz,mv(Lz,mv(Uz.T,mv(Kz,omega)))))))))))
            # pkm = mv(Lz12,mv(Uz.T,mv(Kzx,mv(Pbi,mv(Kzx.T,mv(Uz,mv(Lz,mv(Uz.T,mv(Kz,omega)))))))))
    
    if cov == 'nystrom1' and cov_anchors == 'separated':
        if mmd == 'standard':
            x,y = self.get_xy()
            z1,z2 = self.get_xy(landmarks=True)
            Kz1x = self.kerne(z1,x)
            Kz1y = self.kerne(z1,y)
            Kz2x = self.kerne(z2,x)
            Kz2y = self.kerne(z2,y)
            Uz1 = self.spev['x']['anchors'][anchors_basis]['ev']
            Lz1 = torch.diag(self.spev['x']['anchors'][anchors_basis]['sp']**-1)
            Uz2 = self.spev['y']['anchors'][anchors_basis]['ev']
            Lz2 = torch.diag(self.spev['y']['anchors'][anchors_basis]['sp']**-1)
            omega1 = self.compute_omega(sample='x',quantization=False)
            omega2 = self.compute_omega(sample='y',quantization=False)
            Pn1 = self.compute_centering_matrix(sample='x')
            Pn2 = self.compute_centering_matrix(sample='y')
            haut = mv(Lz1,mv(Uz1,mv(Kz1x,mv(Pn1,mv(Kz1x,mv(Uz1,mv(Lz1,mv(Uz1.T,mv(Kz1y,omega2) -mv(Kz1x,omega1)))))))))
            bas = mv(Lz2,mv(Uz2,mv(Kz2y,mv(Pn2,mv(Kz2y,mv(Uz2,mv(Lz2,mv(Uz2.T,mv(Kz2y,omega2) -mv(Kz2x,omega1)))))))))
            

    if cov == 'quantization': # pas à jour 
        A = self.compute_quantization_weights(power=1/2,sample='xy')
        if mmd == 'standard':
            pkm = mv(Pbi,mv(A,mv(Kzx,omega)))

        elif mmd == 'nystrom':
            Pi = self.compute_centering_matrix(sample='xy',landmarks=True,anchors_basis=anchors_basis)
            Kz = self.compute_gram(landmarks=True)
            pkm = 1/r * mv(Pbi,mv(A,mv(Kz,mv(Uz,mv(Lz,mv(Uz.T,mv(Pi,mv(Kzx,omega))))))))

        elif mmd == 'quantization':
            Kz = self.compute_gram(landmarks=True)
            pkm = mv(Pbi,mv(A,mv(Kz,omega)))
    return(pkm)
#
def compute_kfdat(self,t=None,approximation_cov='standard',approximation_mmd='standard',name=None,verbose=0,anchors_basis=None,cov_anchors='shared',mmd_anchors='shared'):
    # je n'ai plus besoin de trunc, seulement d'un t max 
    """ 
    Computes the kfda truncated statistic of [Harchaoui 2009].
    9 methods : 
    approximation_cov in ['standard','nystrom1','quantization']
    approximation_mmd in ['standard','nystrom','quantization']
    
    Stores the result as a column in the dataframe df_kfdat
    """
    
    cov,mmd = approximation_cov,approximation_mmd
    name = name if name is not None else f'{cov}{mmd}' 

    suffix_nystrom = anchors_basis if 'nystrom' in cov else ''
    sp,ev = self.spev['xy'][cov+suffix_nystrom]['sp'],self.spev['xy'][cov+suffix_nystrom]['ev']
    tmax = 2000
    if t is None : 
        t = tmax if (len(sp)>tmax) else len(sp)
    else:
        t = len(sp) if len(sp)<t else t 
    trunc = range(1,t+1)        
    self.verbosity(function_name='compute_kfdat',
            dict_of_variables={
            't':t,
            'approximation_cov':cov,
            'approximation_mmd':mmd,
            'name':name},
            start=True,
            verbose = verbose)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pkm = self.compute_pkm(approximation_cov=cov,approximation_mmd=mmd,anchors_basis=anchors_basis,cov_anchors = cov_anchors,mmd_anchors=mmd_anchors)
    n1,n2 = (self.n1,self.n2) 
    n = n1+n2
    exposant = 2 if cov in ['standard','nystrom1','quantization'] else 3 if cov == 'nystrom2' else 1 if cov == 'nystrom3' else 'erreur exposant'
    kfda = ((n1*n2)/(n**exposant*sp[:t]**exposant)*mv(ev.T[:t],pkm)**2).cumsum(axis=0).numpy()
    name = name if name is not None else f'{cov}{mmd}{suffix_nystrom}' 
    if name in self.df_kfdat:
        print(f"écrasement de {name} dans df_kfdat")
    self.df_kfdat[name] = pd.Series(kfda,index=trunc)

    self.verbosity(function_name='compute_kfdat',
                            dict_of_variables={
            't':t,
            'approximation_cov':cov,
            'approximation_mmd':mmd,
            'name':name},
            start=False,
            verbose = verbose)

def compute_kfdat_old(self,t=None,approximation_cov='standard',approximation_mmd='standard',name=None,verbose=0,anchors_basis=None):
    # je n'ai plus besoin de trunc, seulement d'un t max 
    """ 
    Computes the kfda truncated statistic of [Harchaoui 2009].
    9 methods : 
    approximation_cov in ['standard','nystrom1','quantization']
    approximation_mmd in ['standard','nystrom','quantization']
    
    Stores the result as a column in the dataframe df_kfdat
    """
    
    cov,mmd = approximation_cov,approximation_mmd
    name = name if name is not None else f'{cov}{mmd}' 

    suffix_nystrom = anchors_basis if 'nystrom' in cov else ''
    sp,ev = self.spev['xy'][cov+suffix_nystrom]['sp'],self.spev['xy'][cov+suffix_nystrom]['ev']
    tmax = 200
    t = tmax if (t is None and len(sp)+1>tmax) else len(sp)+1 if (t is None and len(sp)+1<=tmax) else t
    

    trunc = range(1,t)        
    self.verbosity(function_name='compute_kfdat',
            dict_of_variables={
            't':t,
            'approximation_cov':cov,
            'approximation_mmd':mmd,
            'name':name},
            start=True,
            verbose = verbose)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n1,n2 = (self.n1,self.n2) 
    n = n1+n2


    m = self.compute_omega(quantization=(mmd=='quantization'))
    Pbi = self.compute_centering_matrix(sample='xy',quantization=(cov=='quantization'))

    if any([ny in [mmd,cov] for ny in ['nystrom1','nystrom2','nystrom3']]):
        Up = self.spev['xy']['anchors'][anchors_basis]['ev']
        Lp_inv = torch.diag(self.spev['xy']['anchors'][anchors_basis]['sp']**-1)

    if not (mmd == cov) or mmd == 'nystrom':
        Kmn = self.compute_kmn(sample='xy')
    
    if cov == 'standard':
        if mmd == 'standard':
            K = self.compute_gram()
            pkm = mv(Pbi,mv(K,m))
            kfda = ((n1*n2)/(n**2*sp[:t]**2)*mv(ev.T[:t],pkm)**2).cumsum(axis=0).numpy() 

        elif mmd == 'nystrom':
            Pi = self.compute_centering_matrix(sample='xy',landmarks=True,anchors_basis=anchors_basis)
            pkm = mv(Pbi,mv(Kmn.T,mv(Pi,mv(Up,mv(Lp_inv,mv(Up.T,mv(Pi,mv(Kmn,m))))))))
            kfda = ((n1*n2)/(n**2*sp[:t]**2)*mv(ev.T[:t],pkm)**2).cumsum(axis=0).numpy() 

            # else:
            #     pkuLukm = mv(Pbi,mv(Kmn.T,mv(Up,mv(Lp_inv,mv(Up.T,mv(Kmn,m))))))
            #     kfda = ((n1*n2)/(n**2*sp[:t]**2)*mv(ev.T[:t],pkuLukm)**2).cumsum(axis=0).numpy() 

        elif mmd == 'quantization':
            pkm = mv(Pbi,mv(Kmn.T,m))
            kfda = ((n1*n2)/(n**2*sp[:t]**2)*mv(ev.T[:t],pkm)**2).cumsum(axis=0).numpy() 

    if cov == 'nystrom1':
        if mmd in ['standard','nystrom']: # c'est exactement la même stat  
            Pi = self.compute_centering_matrix(sample='xy',landmarks=True,anchors_basis=anchors_basis)
            pkm = mv(Pbi,mv(Kmn.T,mv(Pi,mv(Up,mv(Lp_inv,mv(Up.T,mv(Pi,mv(Kmn,m))))))))
            kfda = ((n1*n2)/(n**2*sp[:t]**2)*mv(ev.T[:t],pkm)**2).cumsum(axis=0).numpy() 

        elif mmd == 'quantization':
            Kmm = self.compute_gram(landmarks=True)
            pkm = mv(Pbi,mv(Kmn.T,mv(Up,mv(Lp_inv,mv(Up.T,mv(Kmm,m))))))
            kfda = ((n1*n2)/(n**2*sp[:t]**2)*mv(ev.T[:t],pkm)**2).cumsum(axis=0).numpy() 
    
    if cov == 'nystrom2':
        Lp12 = torch.diag(self.spev['xy']['anchors'][anchors_basis]['sp']**-(1/2))
        if mmd in ['standard','nystrom']: # c'est exactement la même stat  
            Pi = self.compute_centering_matrix(sample='xy',landmarks=True,anchors_basis=anchors_basis)
            pkm = mv(Lp12,mv(Up.T,mv(Pi,mv(Kmn,mv(Pbi,mv(Kmn.T,mv(Pi,mv(Up,mv(Lp_inv,mv(Up.T,mv(Pi,mv(Kmn,m))))))))))))
            kfda = ((n1*n2)/(n**3*sp[:t]**3)*mv(ev.T[:t],pkm)**2).cumsum(axis=0).numpy() 

            # else:
            #     LupkpkpuLupkm = mv(Lp12,mv(Up.T,mv(Kmn,mv(Pbi,mv(Kmn.T,mv(Up,mv(Lp_inv,mv(Up.T,mv(Kmn,m)))))))))
            #     kfda = ((n1*n2)/(n**3*sp[:t]**3)*mv(ev.T[:t],LupkpkpuLupkm)**2).cumsum(axis=0).numpy() 

        elif mmd == 'quantization': # pas à jour
            # il pourrait y avoir la dichotomie anchres centrees ou non ici. 
            Kmm = self.compute_gram(landmarks=True)
            pkm = mv(Lp12,mv(Up.T,mv(Kmn,mv(Pbi,mv(Kmn.T,mv(Up,mv(Lp_inv,mv(Up.T,mv(Kmm,m)))))))))
            kfda = ((n1*n2)/(n**3*sp[:t]**3)*mv(ev.T[:t],pkm)**2).cumsum(axis=0).numpy() 
    
    if cov == 'nystrom3':
        Lp12 = torch.diag(self.spev['xy']['anchors'][anchors_basis]['sp']**-(1/2))
        if mmd in ['standard','nystrom']: # c'est exactement la même stat  
            Pi = self.compute_centering_matrix(sample='xy',landmarks=True,anchors_basis=anchors_basis)
            pkm = mv(Lp12,mv(Up.T,mv(Pi,mv(Kmn,m))))
                # pkpuLupkm = mv(Pbi,mv(Kmn.T,mv(Pm,mv(Up,mv(Lp_inv,mv(Up.T,mv(Pm,mv(Kmn,m))))))))
            kfda = ((n1*n2)/(n*sp[:t])*mv(ev.T[:t],pkm)**2).cumsum(axis=0).numpy() 

            # else:
            #     LupkpkpuLupkm = mv(Lp12,mv(Up.T,mv(Kmn,m)))
            #     kfda = ((n1*n2)/(n*sp[:t])*mv(ev.T[:t],LupkpkpuLupkm)**2).cumsum(axis=0).numpy() 

        elif mmd == 'quantization': # pas à jour 
            # il faut ajouter Pi ici . 
            Kmm = self.compute_gram(landmarks=True)
            pkm = mv(Lp12,mv(Up.T,mv(Kmn,mv(Pbi,mv(Kmn.T,mv(Up,mv(Lp_inv,mv(Up.T,mv(Kmm,m)))))))))
            kfda = ((n1*n2)/(n**3*sp[:t]**3)*mv(ev.T[:t],pkm)**2).cumsum(axis=0).numpy() 
    
    if cov == 'quantization': # pas à jour 
        A_12 = self.compute_quantization_weights(power=1/2,sample='xy')
        if mmd == 'standard':
            pkm = mv(Pbi,mv(A_12,mv(Kmn,m)))
            # le n n'est pas au carré ici
            kfda = ((n1*n2)/(n**2*sp[:t]**2)*mv(ev.T[:t],pkm)**2).cumsum(axis=0).numpy() 

        elif mmd == 'nystrom':
            Kmm = self.compute_gram(landmarks=True)
            pkm = mv(Pbi,mv(A_12,mv(Kmm,mv(Up,mv(Lp_inv,mv(Up.T,mv(Kmn,m)))))))
            # le n n'est pas au carré ici # en fait si ? 
            kfda = ((n1*n2)/(n**2*sp[:t]**2)*mv(ev.T[:t],pkm)**2).cumsum(axis=0).numpy() 

        elif mmd == 'quantization':
            Kmm = self.compute_gram(landmarks=True)
            pkm = mv(Pbi,mv(A_12,mv(Kmm,m)))
            kfda = ((n1*n2)/(n**2*sp[:t]**2)*mv(ev.T[:t],pkm)**2).cumsum(axis=0).numpy() 

    name = name if name is not None else f'{cov}{mmd}{suffix_nystrom}' 
    if name in self.df_kfdat:
        print(f"écrasement de {name} dans df_kfdat")
    self.df_kfdat[name] = pd.Series(kfda,index=trunc)

    self.verbosity(function_name='compute_kfdat',
                            dict_of_variables={
            't':t,
            'approximation_cov':cov,
            'approximation_mmd':mmd,
            'name':name},
            start=False,
            verbose = verbose)
#
def initialize_kfdat(self,approximation_cov='standard',approximation_mmd=None,sample='xy',m=None,
                            r=None,landmarks_method='random',verbose=0,anchors_basis=None,**kwargs):
    # verbose -1 au lieu de verbose ? 
    cov,mmd = approximation_cov,approximation_mmd
    if 'quantization' in [cov,mmd] and not self.quantization_with_landmarks_possible: # besoin des poids des ancres de kmeans en quantization
        self.compute_nystrom_landmarks(m=m,landmarks_method='kmeans',verbose=verbose)
    
    if any([ny in [cov,mmd] for ny in ['nystrom1','nystrom2','nystrom3']]):
        if not self.has_landmarks:
            self.compute_nystrom_landmarks(m=m,landmarks_method=landmarks_method,verbose=verbose)
        if "anchors" not in self.spev[sample]:
            self.compute_nystrom_anchors(r=r,sample=sample,verbose=verbose,anchors_basis=anchors_basis)
        
    if cov not in self.spev[sample]:
        self.diagonalize_centered_gram(approximation=cov,sample=sample,verbose=verbose,anchors_basis=anchors_basis)
#
def kfdat(self,t=None,approximation_cov='standard',approximation_mmd='standard',
            m=None,r=None,landmarks_method='random',
            name=None,verbose=0,anchors_basis=None):
            
    cov,mmd = approximation_cov,approximation_mmd
    name = name if name is not None else f'{cov}{mmd}' 
    if name in self.df_kfdat :
        if verbose : 
            print(f'kfdat {name} already computed')
    else:
        self.initialize_kfdat(approximation_cov=cov,approximation_mmd=mmd,sample='xy',
                            m=m,r=r,landmarks_method=landmarks_method,
                                    verbose=verbose,anchors_basis=anchors_basis)            
        self.compute_kfdat(t=t,approximation_cov=cov,approximation_mmd=mmd,name=name,verbose=verbose,anchors_basis=anchors_basis)

def initialize_mmd(self,approximation='standard',shared_anchors=True,m=None,
                            r=None,landmarks_method='random',verbose=0,anchors_basis=None):

    """
    Calculs preliminaires pour lancer le MMD.
    approximation: determine les calculs a faire en amont du calcul du mmd
                full : aucun calcul en amont puisque la Gram et m seront calcules dans mmd
                nystrom : 
                        si il n'y a pas de landmarks deja calcules, on calcule nloandmarks avec la methode landmarks_method
                        si shared_anchors = True, alors on calcule un seul jeu d'ancres de taille r pour les deux echantillons
                        si shared_anchors = False, alors on determine un jeu d'ancre par echantillon de taille r//2
                                    attention : le parametre r est divise par 2 pour avoir le meme nombre total d'ancres, risque de poser probleme si les donnees sont desequilibrees
                quantization : m sont determines comme les centroides de l'algo kmeans 
    shared_anchors : si approximation='nystrom' alors shared anchors determine si les ancres sont partagees ou non
    m : nombre de landmarks a calculer si approximation='nystrom' ou 'kmeans'
    landmarks_method : dans ['random','kmeans'] methode de choix des landmarks
    verbose : booleen, vrai si les methodes appellees renvoies des infos sur ce qui se passe.  
    """
        # verbose -1 au lieu de verbose ? 

    if approximation == 'quantization' and not self.quantization_with_landmarks_possible: # besoin des poids des ancres de kmeans en quantization
        self.compute_nystrom_landmarks(m=m,landmarks_method='kmeans',verbose=verbose)
    
    if approximation == 'nystrom':
        if not self.has_landmarks:
                self.compute_nystrom_landmarks(m=m,landmarks_method=landmarks_method,verbose=verbose)
        
        if shared_anchors:
            if "anchors" not in self.spev['xy']:
                self.compute_nystrom_anchors(r=r,sample='xy',verbose=verbose,anchors_basis=anchors_basis)
        else:
            for xy in 'xy':
                if 'anchors' not in self.spev[xy]:
                    assert(r is not None,"r not specified")
                    self.compute_nystrom_anchors(r=r//2,sample=xy,verbose=verbose,anchors_basis=anchors_basis)
#
def mmd(self,approximation='standard',shared_anchors=True,m=None,
                            r=None,landmarks_method='random',name=None,unbiaised=False,verbose=0):
    """
    appelle la fonction initialize mmd puis la fonction compute_mmd si le mmd n'a pas deja ete calcule. 
    """
    if name is None:
        name=f'{approximation}'
        if approximation == 'nystrom':
            name += 'shared' if shared_anchors else 'diff'
    
    if name in self.dict_mmd :
        if verbose : 
            print(f'mmd {name} already computed')
    else:
        self.initialize_mmd(approximation=approximation,shared_anchors=shared_anchors,
                m=m,r=r,landmarks_method=landmarks_method,verbose=verbose)
        self.compute_mmd(approximation=approximation,shared_anchors=shared_anchors,
                        name=name,unbiaised=unbiaised,verbose=0)

def compute_mmd(self,unbiaised=False,approximation='standard',shared_anchors=True,name=None,verbose=0,anchors_basis=None):
    
    self.verbosity(function_name='compute_mmd',
            dict_of_variables={'unbiaised':unbiaised,
                                'approximation':approximation,
                                'shared_anchors':shared_anchors,
                                'name':name},
            start=True,
            verbose = verbose)

    if approximation == 'standard':
        m = self.compute_omega(sample='xy',quantization=False)
        K = self.compute_gram()
        if unbiaised:
            K.masked_fill_(torch.eye(K.shape[0],K.shape[0]).byte(), 0)
        mmd = torch.dot(mv(K,m),m)**2
    
    if approximation == 'nystrom' and shared_anchors:
        m = self.compute_omega(sample='xy',quantization=False)
        Up = self.spev['xy']['anchors'][anchors_basis][anchors_basis]['ev']
        Lp_inv2 = torch.diag(self.spev['xy']['anchors'][anchors_basis]['sp']**-(1/2))
        Pm = self.compute_centering_matrix(sample='xy',landmarks=True)
        Kmn = self.compute_kmn(sample='xy')
        psi_m = mv(Lp_inv2,mv(Up.T,mv(Pm,mv(Kmn,m))))
        mmd = torch.dot(psi_m,psi_m)**2
    
    if approximation == 'nystrom' and not shared_anchors:
        
        mx = self.compute_omega(sample='x',quantization=False)
        my = self.compute_omega(sample='y',quantization=False)
        Upx = self.spev['x']['anchors'][anchors_basis]['ev']
        Upy = self.spev['y']['anchors'][anchors_basis]['ev']
        Lpx_inv2 = torch.diag(self.spev['x']['anchors'][anchors_basis]['sp']**-(1/2))
        Lpy_inv2 = torch.diag(self.spev['y']['anchors'][anchors_basis]['sp']**-(1/2))
        Lpy_inv = torch.diag(self.spev['y']['anchors'][anchors_basis]['sp']**-1)
        Pmx = self.compute_centering_matrix(sample='x',landmarks=True)
        Pmy = self.compute_centering_matrix(sample='y',landmarks=True)
        Kmnx = self.compute_kmn(sample='x')
        Kmny = self.compute_kmn(sample='y')
        
        Km = self.compute_gram(sample='xy',landmarks=True)
        m1 = Kmnx.shape[0]
        m2 = Kmny.shape[0]
        Kmxmy = Km[:m1,m2:]

        psix_mx = mv(Lpx_inv2,mv(Upx.T,mv(Pmx,mv(Kmnx,mx))))
        psiy_my = mv(Lpy_inv2,mv(Upy.T,mv(Pmy,mv(Kmny,my))))
        Cpsiy_my = mv(Lpx_inv2,mv(Upx.T,mv(Pmx,mv(Kmxmy,\
            mv(Pmy,mv(Upy,mv(Lpy_inv,mv(Upy.T,mv(Pmy,mv(Kmny,my))))))))))
        mmd = torch.dot(psix_mx,psix_mx)**2 + torch.dot(psiy_my,psiy_my)**2 - 2*torch.dot(psix_mx,Cpsiy_my)
    
    if approximation == 'quantization':
        mq = self.compute_omega(sample='xy',quantization=True)
        Km = self.compute_gram(sample='xy',landmarks=True)
        mmd = torch.dot(mv(Km,mq),mq)**2


    if name is None:
        name=f'{approximation}'
        if approximation == 'nystrom':
            name += 'shared' if shared_anchors else 'diff'
    
    self.dict_mmd[name] = mmd.item()
    
    self.verbosity(function_name='compute_mmd',
            dict_of_variables={'unbiaised':unbiaised,
                                'approximation':approximation,
                                'shared_anchors':shared_anchors,
                                'name':name},
            start=False,
            verbose = verbose)

def kpca(self,t=None,approximation_cov='standard',sample='xy',name=None,verbose=0):
    
    cov = approximation_cov
    name = name if name is not None else f'{cov}{sample}' 
    if name in self.df_proj_kpca :
        if verbose : 
            print(f'kfdat {name} already computed')
    else:
        self.initialize_kfda(approximation_cov=cov,sample=sample,verbose=verbose)            
        self.compute_proj_kpca(t=t,approximation_cov=cov,sample=sample,name=name,verbose=verbose)


