import torch
import pandas as pd
from torch import mv
from scipy.stats import chi2


def compute_pkm(self):
    """
    pkm is an alias for the product PKomega in the standard KFDA statistic. 
    This functions computes the corresponding block with respect to the model parameters. 
    """
    cov,mmd = self.approximation_cov,self.approximation_mmd
    anchors_basis = self.anchors_basis
    cov_anchors = 'shared' # pas terminé  
    
    if 'nystrom' in cov or 'nystrom' in mmd :
        r = self.r
    
    omega = self.compute_omega(quantization=(mmd=='quantization'))
    Pbi = self.compute_centering_matrix(sample='xy',quantization=(cov=='quantization'))
    

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
            Pi = self.compute_centering_matrix(sample='xy',landmarks=True)
            pkm = 1/r * mv(Pbi,mv(Kzx.T,mv(Pi,mv(Uz,mv(Lz,mv(Uz.T,mv(Pi,mv(Kzx,omega))))))))
            # pkm = mv(Pbi,mv(Kzx.T,mv(Pi,mv(Uz,mv(Lz,mv(Uz.T,mv(Pi,mv(Kzx,omega))))))))

        elif mmd == 'quantization':
            pkm = mv(Pbi,mv(Kzx.T,omega))

    if cov == 'nystrom1' and cov_anchors == 'shared':
        if mmd in ['standard','nystrom']: # c'est exactement la même stat  
            Pi = self.compute_centering_matrix(sample='xy',landmarks=True)
            pkm = 1/r**2 * mv(Pbi,mv(Kzx.T,mv(Pi,mv(Uz,mv(Lz,mv(Uz.T,mv(Pi,mv(Kzx,omega))))))))
            # pkm = mv(Pbi,mv(Kzx.T,mv(Pi,mv(Uz,mv(Lz,mv(Uz.T,mv(Pi,mv(Kzx,omega))))))))

        elif mmd == 'quantization':
            Kz = self.compute_gram(landmarks=True)
            pkm = 1/r**2 * mv(Pbi,mv(Kzx.T,mv(Uz,mv(Lz,mv(Uz.T,mv(Kz,omega))))))
            # pkm = mv(Pbi,mv(Kzx.T,mv(Uz,mv(Lz,mv(Uz.T,mv(Kz,omega))))))
    
    if cov == 'nystrom2' and cov_anchors == 'shared':
        Lz12 = torch.diag(self.spev['xy']['anchors'][anchors_basis]['sp']**-(1/2))
        if mmd in ['standard','nystrom']: # c'est exactement la même stat  
            Pi = self.compute_centering_matrix(sample='xy',landmarks=True)
            pkm = 1/r**3 * mv(Lz12,mv(Uz.T,mv(Pi,mv(Kzx,mv(Pbi,mv(Kzx.T,mv(Pi,mv(Uz,mv(Lz,mv(Uz.T,mv(Pi,mv(Kzx,omega))))))))))))
            # pkm = mv(Lz12,mv(Uz.T,mv(Pi,mv(Kzx,mv(Pbi,mv(Kzx.T,mv(Pi,mv(Uz,mv(Lz,mv(Uz.T,mv(Pi,mv(Kzx,omega))))))))))))

        elif mmd == 'quantization': # pas à jour
            # il pourrait y avoir la dichotomie anchres centrees ou non ici. 
            Kz = self.compute_gram(landmarks=True)
            pkm = 1/r**3 * mv(Lz12,mv(Uz.T,mv(Kzx,mv(Pbi,mv(Kzx.T,mv(Uz,mv(Lz,mv(Uz.T,mv(Kz,omega)))))))))
            # pkm = mv(Lz12,mv(Uz.T,mv(Kzx,mv(Pbi,mv(Kzx.T,mv(Uz,mv(Lz,mv(Uz.T,mv(Kz,omega)))))))))
    
    if cov == 'nystrom3' and cov_anchors == 'shared':
        Lz12 = torch.diag(self.spev['xy']['anchors'][anchors_basis]['sp']**-(1/2))
        # print("statistics pkm: L-1 nan ",(torch.isnan(torch.diag(Lz12))))
        # Lz12 = torch.nan_to_num(Lz12)
        Pi = self.compute_centering_matrix(sample='xy',landmarks=True)

        if mmd in ['standard','nystrom']: # c'est exactement la même stat  
            pkm = 1/r * mv(Lz12,mv(Uz.T,mv(Pi,mv(Kzx,omega))))
            # pkm = mv(Lz12,mv(Uz.T,mv(Pi,mv(Kzx,omega))))
            # print(f'in compute pkm: \n\t\
            #      Lz12{Lz12}\n Uz{Uz}\n Kzx{Kzx}')

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
            Pi = self.compute_centering_matrix(sample='xy',landmarks=True)
            Kz = self.compute_gram(landmarks=True)
            pkm = 1/r * mv(Pbi,mv(A,mv(Kz,mv(Uz,mv(Lz,mv(Uz.T,mv(Pi,mv(Kzx,omega))))))))

        elif mmd == 'quantization':
            Kz = self.compute_gram(landmarks=True)
            pkm = mv(Pbi,mv(A,mv(Kz,omega)))
    return(pkm)

def compute_epk(self,t):
    """
    epk is an alias for the product ePK that appears when projecting the data on the discriminant axis. 
    This functions computes the corresponding block with respect to the model parameters. 
    
    warning: some work remains to be done to :
        - normalize the vecters with respect to r as in pkm 
        - separate the different nystrom approaches 
    """
    
    cov = self.approximation_cov
    anchors_basis = self.anchors_basis
    suffix_nystrom = anchors_basis if 'nystrom' in cov else ''
    sp,ev = self.spev['xy'][cov+suffix_nystrom]['sp'],self.spev['xy'][cov+suffix_nystrom]['ev']
    

    Pbi = self.compute_centering_matrix(sample='xy',quantization=(cov=='quantization'))
      
    if not (cov == 'standard'):
        Kzx = self.compute_kmn(sample='xy')
    
    if cov == 'standard':
        Kx = self.compute_gram()
        epk = torch.chain_matmul(ev.T[:t],Pbi,Kx).T
        # epk = torch.linalg.multi_dot([ev.T[:t],Pbi,Kx]).T
    if 'nystrom' in cov:
        Uz = self.spev['xy']['anchors'][anchors_basis]['ev']
        Lz = torch.diag(self.spev['xy']['anchors'][anchors_basis]['sp']**-1)
        r = self.r
        epk = 1/r*torch.chain_matmul(ev.T[:t],Pbi,Kzx.T,Uz,Lz,Uz.T,Kzx).T
        # epk = 1/r*torch.linalg.multi_dot([ev.T[:t],Pbi,Kzx.T,Uz,Lz,Uz.T,Kzx]).T
    if cov == 'quantization':
        A_12 = self.compute_quantization_weights(power=1/2,sample='xy')
        epk = torch.chain_matmul(ev.T[:t],A_12,Pbi,Kzx).T
        # epk = torch.linalg.multi_dot([ev.T[:t],A_12,Pbi,Kzx]).T
    
    return(epk)
#
def compute_kfdat(self,t=None,name=None,verbose=0,):
    # je n'ai plus besoin de trunc, seulement d'un t max 
    """ 
    Computes the kfda truncated statistic of [Harchaoui 2009].
    9 methods : 
    approximation_cov in ['standard','nystrom1','quantization']
    approximation_mmd in ['standard','nystrom','quantization']
    
    Stores the result as a column in the dataframe df_kfdat
    """
    
    cov,mmd = self.approximation_cov,self.approximation_mmd
    anchors_basis = self.anchors_basis
    cov_anchors='shared'
    mmd_anchors='shared'
    
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
    pkm = self.compute_pkm()
    n1,n2 = (self.n1,self.n2) 
    n = n1+n2
    exposant = 2 if cov in ['standard','nystrom1','quantization'] else 3 if cov == 'nystrom2' else 1 if cov == 'nystrom3' else 'erreur exposant'
    kfda = ((n1*n2)/(n**exposant*sp[:t]**exposant)*mv(ev.T[:t],pkm)**2).cumsum(axis=0)
    kfda = torch.nan_to_num(kfda).numpy()
    
    
    # print('\n\nstat compute kfdat\n\n''sp',sp,'kfda',kfda)
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

def compute_pval(self,t=None):
    """
    Calcul des pvalue asymptotique d'un df_kfdat pour chaque valeur de t. 
    Attention, la présence de Nan augmente considérablement le temps de calcul. 
    """
    pvals = {}
    if t is None:
        t = min(100,len(self.df_kfdat))
    else : 
        t = min(t,len(self.df_kfdat))
    trunc=range(1,t+1)

    # est-ce qu'on peut accelérer cette boucle avec une approche vectorielle ? 
    for t_ in trunc:
        pvals[t_] = self.df_kfdat.T[t_].apply(lambda x: chi2.sf(x,int(t_)))
    self.df_pval = pd.DataFrame(pvals).T 

def correct_BenjaminiHochberg_pval_of_dfcolumn(df,t):
    df = pd.concat([df,df.rank()],keys=['pval','rank'],axis=1)
    df['pvalc'] = df.apply(lambda x: len(df) * x['pval']/x['rank'],axis=1) # correction
    df['rankc'] = df['pvalc'].rank() # calcul du nouvel ordre
    corrected_pvals = []
    # correction des pval qui auraient changé d'ordre
    l = []
    if not df['rankc'].equals(df['rank']):
        first_rank = df['rank'].sort_values().values[0] # égal à 1 sauf si égalité 
        pvalc_prec = df.loc[df['rank']==first_rank,'pvalc'].iat[0]
        df['rank'] = df['rank'].fillna(10000)
        for rank in df['rank'].sort_values().values[1:]: # le 1 est déjà dans rank prec et on prend le dernier 
            # if t >=8:
            #     print(rank,end=' ')
            pvalc = df.loc[df['rank']==rank,'pvalc'].iat[0]
            if pvalc_prec >= 1 : 
                pvalue = 1 # l += [1]
            elif pvalc_prec > pvalc :
                pvalue = pvalc # l += [pvalc]
            elif pvalc_prec <= pvalc:
                pvalue = pvalc_prec # l+= [pvalc_prec]
            else: 
                print('error pval correction',f'rank{rank} pvalc{pvalc} pvalcprec{pvalc_prec}')
                print(df.loc[df['rank']==rank].index)
            pvalc_prec = pvalc
            l += [pvalue]
        # dernier terme 
        pvalue = 1 if pvalc >1 else pvalc
        l += [pvalue]
#             corrected_pvals[t] = pd.Series(l,index=ranks)#df['rank'].sort_values().index)
    if len(l)>0: 
        return(pd.Series(l,index=df['rank'].sort_values().index))
    else: 
        return(pd.Series(df['pvalc'].values,index=df['rank'].sort_values().index))

def correct_BenjaminiHochberg_pval_of_dataframe(df_pval,t=20):
    """
    Benjamini Hochberg correction of a dataframe containing the p-values where the rows are the truncations.    
    """
    trunc = range(1,t+1)
    corrected_pvals = []
    for t in trunc:
        # print(t)
        corrected_pvals += [correct_BenjaminiHochberg_pval_of_dfcolumn(df_pval.T[t],t=t)]   
    return(pd.concat(corrected_pvals,axis=1).T)

def correct_BenjaminiHochberg_pval(self,t=20):
    """
    Correction of the p-values of df_pval according to Benjamini and Hochberg 1995 approach.
    This is to use when the different tests correspond to multiple testing. 
    The results are stored in self.df_BH_corrected_pval 
    The pvalues are adjusted for each truncation lower or equal to t. 
    """
    
    self.df_pval_BH_corrected = correct_BenjaminiHochberg_pval_of_dataframe(self.df_pval,t=t)

def initialize_kfdat(self,sample='xy',verbose=0,**kwargs):
    # verbose -1 au lieu de verbose ? 
    cov,mmd = self.approximation_cov,self.approximation_mmd
    if 'quantization' in [cov,mmd] and not self.quantization_with_landmarks_possible: # besoin des poids des ancres de kmeans en quantization
        self.compute_nystrom_landmarks(verbose=verbose)
    
    if any([ny in [cov,mmd] for ny in ['nystrom1','nystrom2','nystrom3']]):
        if not self.has_landmarks:
            self.compute_nystrom_landmarks(verbose=verbose)
        if "anchors" not in self.spev[sample]:
            self.compute_nystrom_anchors(sample=sample,verbose=verbose)
    
    # if cov not in self.spev[sample]:
    self.diagonalize_centered_gram(approximation=cov,sample=sample,verbose=verbose)
#
def kfdat(self,t=None,name=None,pval=True,verbose=0):
    cov,mmd = self.approximation_cov,self.approximation_mmd
    name = name if name is not None else f'{cov}{mmd}' 
    if name in self.df_kfdat :
        if verbose : 
            print(f'kfdat {name} already computed')
    else:
        self.initialize_kfdat(sample='xy',verbose=verbose)            
        self.compute_kfdat(t=t,name=name,verbose=verbose)
        if pval:
            self.compute_pval(t=t)



def initialize_mmd(self,shared_anchors=True,verbose=0,anchors_basis=None):

    """
    Calculs preliminaires pour lancer le MMD.
    approximation: determine les calculs a faire en amont du calcul du mmd
                full : aucun calcul en amont puisque la Gram et m seront calcules dans mmd
                nystrom : 
                        si il n'y a pas de landmarks deja calcules, on calcule nloandmarks avec la methode landmark_method
                        si shared_anchors = True, alors on calcule un seul jeu d'ancres de taille r pour les deux echantillons
                        si shared_anchors = False, alors on determine un jeu d'ancre par echantillon de taille r//2
                                    attention : le parametre r est divise par 2 pour avoir le meme nombre total d'ancres, risque de poser probleme si les donnees sont desequilibrees
                quantization : m sont determines comme les centroides de l'algo kmeans 
    shared_anchors : si approximation='nystrom' alors shared anchors determine si les ancres sont partagees ou non
    m : nombre de landmarks a calculer si approximation='nystrom' ou 'kmeans'
    landmark_method : dans ['random','kmeans'] methode de choix des landmarks
    verbose : booleen, vrai si les methodes appellees renvoies des infos sur ce qui se passe.  
    """
        # verbose -1 au lieu de verbose ? 

    approx = self.approximation_mmd
    anchors_basis = self.anchors_basis

    if approx == 'quantization' and not self.quantization_with_landmarks_possible: # besoin des poids des ancres de kmeans en quantization
        self.compute_nystrom_landmarks(verbose=verbose)
    
    if approx == 'nystrom':
        if not self.has_landmarks:
                self.compute_nystrom_landmarks(verbose=verbose)
        
        if shared_anchors:
            if "anchors" not in self.spev['xy']:
                self.compute_nystrom_anchors(sample='xy',verbose=verbose,anchors_basis=anchors_basis)
        else:
            for xy in 'xy':
                if 'anchors' not in self.spev[xy]:
                    assert(r is not None,"r not specified")
                    self.compute_nystrom_anchors(sample=xy,verbose=verbose,anchors_basis=anchors_basis)
#
def mmd(self,shared_anchors=True,name=None,unbiaised=False,verbose=0):
    """
    appelle la fonction initialize mmd puis la fonction compute_mmd si le mmd n'a pas deja ete calcule. 
    """
    approx = self.approximation_mmd
    
    if name is None:
        name=f'{approx}'
        if approx == 'nystrom':
            name += 'shared' if shared_anchors else 'diff'
    
    if name in self.dict_mmd :
        if verbose : 
            print(f'mmd {name} already computed')
    else:
        self.initialize_mmd(shared_anchors=shared_anchors,verbose=verbose)
        self.compute_mmd(shared_anchors=shared_anchors,
                        name=name,unbiaised=unbiaised,verbose=0)

def compute_mmd(self,unbiaised=False,shared_anchors=True,name=None,verbose=0,anchors_basis=None):
    
    approx = self.approximation_mmd
    self.verbosity(function_name='compute_mmd',
            dict_of_variables={'unbiaised':unbiaised,
                                'approximation':approx,
                                'shared_anchors':shared_anchors,
                                'name':name},
            start=True,
            verbose = verbose)

    if approx == 'standard':
        m = self.compute_omega(sample='xy',quantization=False)
        K = self.compute_gram()
        if unbiaised:
            K.masked_fill_(torch.eye(K.shape[0],K.shape[0]).byte(), 0)
        mmd = torch.dot(mv(K,m),m)**2
    
    if approx == 'nystrom' and shared_anchors:
        m = self.compute_omega(sample='xy',quantization=False)
        Up = self.spev['xy']['anchors'][anchors_basis][anchors_basis]['ev']
        Lp_inv2 = torch.diag(self.spev['xy']['anchors'][anchors_basis]['sp']**-(1/2))
        Pm = self.compute_centering_matrix(sample='xy',landmarks=True)
        Kmn = self.compute_kmn(sample='xy')
        psi_m = mv(Lp_inv2,mv(Up.T,mv(Pm,mv(Kmn,m))))
        mmd = torch.dot(psi_m,psi_m)**2
    
    if approx == 'nystrom' and not shared_anchors:
        
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
    
    if approx == 'quantization':
        mq = self.compute_omega(sample='xy',quantization=True)
        Km = self.compute_gram(sample='xy',landmarks=True)
        mmd = torch.dot(mv(Km,mq),mq)**2


    if name is None:
        name=f'{approx}'
        if approx == 'nystrom':
            name += 'shared' if shared_anchors else 'diff'
    
    self.dict_mmd[name] = mmd.item()
    
    self.verbosity(function_name='compute_mmd',
            dict_of_variables={'unbiaised':unbiaised,
                                'approximation':approx,
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


