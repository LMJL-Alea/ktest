import numpy as np
import torch
import scipy as sc

# Pour tout plein de distributions clé en main
# https://docs.scipy.org/doc/numpy-1.15.0/reference/routines.random.html

"""
Génération de variables aléatoires multivariées pour des simulations. 
 """

P,D,N,S = 10,50,10,0.01
DI,DN = 5,5
#%% Generation


def generate_covariance(p,spectrum='isotropic'):
    if spectrum == 'isotropic':
        cov = torch.eye(p)
    elif spectrum == 'decreasing_linear':
        cov = torch.diag(torch.tensor([p-i for i in range(p)]))  # list(range(1,p+1))[::-1]))
    elif spectrum == 'decreasing_geometric':
        cov = torch.diag(torch.tensor([p/2**i for i in range(p)]))
    elif spectrum == 'decreasing_exponential':
        cov = torch.diag(torch.tensor([p*np.exp(-i/np.sqrt(p)) for i in range(p)]))
    else:
        print(f"spectrum '{spectrum}' not handled")
    return(cov)        

def generate_mean(p):
    return(torch.zeros(p))

def add_noise(x,sigma):
    s = x.shape()
    x += np.random.normal(0,sigma,s)
    return(x)

def generate_multivariate_gaussian(p,nobs,seed,spectrum='isotropic',sigma_noise=None,d=None):
    np.random.seed(seed=seed)
    mu = generate_mean(p)
    cov = generate_covariance(p,spectrum)
    x = np.random.multivariate_normal(mu,cov,nobs)
    if d is not None:
        x = np.concatenate((x,np.zeros((nobs,d-p))))
    if sigma_noise is not None:
        x = add_noise(x,sigma_noise)
    return(x)

def generate_standard_gaussian_2D(n=300,seed=0):
    return(generate_multivariate_gaussian(p=2, nobs=n, seed=seed, sigma_noise=.01,spectrum='isotropic'))

def generate_gaussian_mixture(means,covs,weights,nobs,seed,return_assignations=False):
    np.random.seed(seed=seed)

    assignations = np.random.multinomial(nobs,weights)
    
    x = np.concatenate([np.random.multivariate_normal(mean,cov,assignation) \
                        for mean,cov,assignation in zip(means,covs,assignations)],axis=0)
    if return_assignations:
        return(x,assignations)
    else:
        return(x)

def generate_sparse_gaussian_mixture(means,covs,weights,nobs,seed,d):
    xp = generate_gaussian_mixture(means,covs,weights,nobs,seed)
    p=len(means[0])
    xd_p = np.zeros((nobs,d-p))
    x = np.concatenate((xp,xd_p),axis=1)
    return(x)

def generate_couple_of_gaussians_H0(p,nobs,seed,spectrum='isotropic',sigma_noise=None,d=None):
    x = generate_multivariate_gaussian(p=p,nobs=nobs,seed=seed,spectrum=spectrum,sigma_noise=sigma_noise,d=d)
    y = generate_multivariate_gaussian(p=p,nobs=nobs,seed=seed+123,spectrum=spectrum,sigma_noise=sigma_noise,d=d)
    return(x,y)

def generate_bimodal_gaussian(p,nobs,seed,shift,spectrum='isotropic'):
    m = np.zeros(p)
    means = [m-shift,m+shift]
    cov = generate_covariance(p,spectrum)
    x = generate_gaussian_mixture(means=means,covs=[cov,cov],weights=[.5,.5],nobs=nobs,seed=seed)
    return(x)

def generate_gaussian_unimodal_and_bimodal(pu,pb,nobs,seed,spectrum='isotropic',shift_bimodal=5):
    x1 = generate_multivariate_gaussian(p=pu,nobs=nobs,seed=seed,spectrum=spectrum)
    x2 = generate_bimodal_gaussian(shift=shift_bimodal,p=pb,nobs=nobs,seed=seed+18,spectrum=spectrum)
    x = np.concatenate((x1,x2),axis=1)
    return(x)

def generate_couple_unimodal_and_bimodal(pu=5,pb=5,nobs=500,seed=1994,spectrum='isotropic',shift_bimodal=2):
    x = generate_gaussian_unimodal_and_bimodal(pu=pu,pb=pb,nobs=nobs,seed=seed,spectrum=spectrum,shift_bimodal=shift_bimodal)    
    y = generate_gaussian_unimodal_and_bimodal(pu=pu,pb=pb,nobs=nobs,seed=seed+200,spectrum=spectrum,shift_bimodal=shift_bimodal)    
    return(x,y)

def generate_gaussian_unidirectional_shift(p,nobs,seed,spectrum,dim_shift,shift):
    x = generate_multivariate_gaussian(p=p,nobs=nobs,seed=seed,spectrum=spectrum)
    x = shift_data(x,ndim=1,shift=shift,dim1=dim_shift)
    return(x)

def generate_couple_gaussian_unidirectional_diff(dim_shift,shift,p,nobs,seed,spectrum='decreasing_linear'):
    x = generate_multivariate_gaussian(p=p,nobs=nobs,seed=seed,spectrum=spectrum)
    y = generate_gaussian_unidirectional_shift(p=p,nobs=nobs,seed=seed+998,spectrum=spectrum,dim_shift=dim_shift,shift=shift)
    return(x,y)


def gen_couple_H1(p,nobs,seed,spectrum='isotropic',sigma_noise=None,d=None,
                 alternative='shift',param=1,ndim=1,dim1=0):
    """

    if alternative in ['GMD','GVD','Blobs'] : 
    Generates a toy dataset of two samples of observations with different distributions 
    for which the difference has already been used to assess the performances of a two sample test.
    Attention: the information contained in key may be ignored in order to fit exactly the distribution       
    """
    
    if alternative in ['shift','rescale','rot_plane']:
        
        x,y = generate_couple_of_gaussians_H0(p,nobs,seed,spectrum,sigma_noise,d)
        

        if alternative == 'shift':
            y = shift_data(y,ndim=ndim,shift=param,dim1=dim1)

        if alternative == 'rescale':
            y = rescale(y,ndim=ndim,coef=param,dim1=dim1)

        if alternative =='rot_plane':
            y = rot_plane(y,angle=param)
    
    else:
        if alternative =='GMD':
            x,y = generate_couple_of_gaussians_H0(p,nobs,seed,spectrum='isotropic',sigma_noise=sigma_noise,d=d)
            y = shift_data(y,ndim=1,shift=1)
        
        if alternative =='GVD':
            x,y = generate_couple_of_gaussians_H0(p,nobs,seed,spectrum='isotropic',sigma_noise=sigma_noise,d=d)
            y =  rescale(y,ndim=1,coef=np.sqrt(2))
            
        if alternative =='Gretton2012b1':
            x,y = Gretton2012b1(p,nobs,seed,mean_shift=.5)
            
        if alternative =='BlobsJitkrittum2016':
            x,y = blobs(nobs,seed,version='Jitkrittum2016')
            
        if alternative =='BlobsChwialkowski2015':
            x,y = blobs(nobs,seed,version='Chwialkowski2015')
            
        if alternative =='BlobsGretton2012b':
            a=0
            # x,y = blobs(nobs,seed,version='Gretton2012b').values()
            
    return(x,y)


#%% Transformation
def shift_data(y,ndim=1,shift=1,dim1=0):
    """
    Shifts the `ndim_to_shift` first dimensions of every observation of dataset `y` by `shift_by` 
    """
    y[:,dim1:dim1+ndim] = y[:,dim1:dim1+ndim] + shift
    return(y)

def rescale(y,ndim=1,coef=2,dim1=0):
    """
    Rescales the `ndim_to_rescale` first dimensions of every observation of dataset `y` by `coef` 
    """
    y[:,dim1:dim1+ndim] = y[:,dim1:dim1+ndim] * coef
    return(y)
   
def rot_plane(y,angle=np.pi/2):
    """
    Rotates the plane composed of dimensions 1 and 2 of `y` by the angle `angle` `y` by `coef` 
    """
    m=np.eye(y.shape[1])
    c,s = np.cos(angle),np.sin(angle)
    e1,e2 = 0,1
    m[e1, e1] = c;    m[e1, e2] = -s;    m[e2, e1] = s;    m[e2, e2] = c
    return(np.matmul(y,m))



#%% Final 


def Gretton2012b1(p,nobs,seed,diff=.5):
    """
    x is a gaussian and y is the mixtrue of two gaussians with different means. 
    """

    x = generate_couple_of_gaussians_H0(p=p,nobs=nobs,seed=seed,spectrum='isotropic')['x']
    

    mean1 = np.zeros(p)
    mean2 = np.zeros(p)
    mean1[0] = mean2[1] = diff
    means=[mean1,mean2]
    covs = [np.eye(p)]*2
    weights = [.5,.5]
    y = generate_gaussian_mixture(means,covs,weights,nobs,seed+1)
    
    return(x,y)

def blobs(nobs,seed,version='Jitkrittum2016',cote=None,param=None):
    """
    Generates a cote * cote square of cote**2 gaussians, the covariance of each gaussian in x is identity
    the covariance of each gaussian in y depends on the version. 
    version: In Jitkrittum2016, the covariance of y is [[2,0],[0,1]], first axis is amplificated
             In Chwialkowski2015, the covariance of y is [[cos(pi/4),sin(pi/4)],[0,1]], first axis is rotated
    """
    cote = 4 if (cote is None and version in ['Chwialkowski2015','Jitkrittum2016']) else cote
    means = [[i*7,j*7] for i in range(cote) for j in range(cote)]
    
    weights = [1/len(means)]*len(means)

    xcovs = [np.eye(2)]*len(means)
    x = generate_gaussian_mixture(means,xcovs,weights,nobs,seed)

    # version = 'Chwialkowski2015'

    if version == 'H0':
        y = generate_gaussian_mixture(means,xcovs,weights,nobs,seed+1)
 

    if version == 'Jitkrittum2016':
        
        param = 2 if param is None else param
        ycov = np.eye(2)
        ycov[0,0] = param
        ycovs = [ycov]*len(means)
        y = generate_gaussian_mixture(means,ycovs,weights,nobs,seed+1)

    if version =='Chwialkowski2015':
        param = np.pi/4 if param is None else param
        ycov = np.eye(2)
        ycov[0,0] = np.cos(np.pi/4)
        ycov[1,0] = np.sin(np.pi/4)
        ycovs = [ycov]*len(means)
        y = generate_gaussian_mixture(means,ycovs,weights,nobs,seed+1)

    if version == 'Gretton2012b':
        print('pas implémenté')
        # a grid of correlated Gaussians with a ratio ε of largest to smallest covariance eigenvalues.
    return(x,y)



# def gen_couple_H1(p=P,d=D,sig=S,noise=False,spectrum='isotropic',seed = 1994,data_nobs=N,data_ynobs=None,
#             alternative = 'shift',alternative_param=1,alternative_ndim=1):
#     """
    
    
#     if alternative in ['GMD','GVD','Blobs'] : 
#     Generates a toy dataset of two samples of observations with different distributions 
#     for which the difference has already been used to assess the performances of a two sample test.
#     Attention: the information contained in key may be ignored in order to fit exactly the distribution       
#     """
#     assert('alternative' in key)
    
#     if alternative in ['shift','rescale','rot_plane']:
        
#         x,y = gen_couple(p=p,d=d,sig=sig,noise=noise,spectrum=spectrum,seed=seed,data_nobs=data_nobs,data_ynobs=data_ynobs).values()
#         # param = key['alternative_param']
#         # if alternative in ['shift','rescale']:
#             # ndim = key['alternative_ndim']

#         if alternative == 'shift':
#             y = shift(y,ndim_to_shift=alternative_ndim,shift_by=alternative_param)

#         if alternative == 'rescale':
#             y = rescale(y,ndim_to_rescale=alternative_ndim,coef=alternative_param)

#         if alternative =='rot_plane':
#             y = rot_plane(y,angle=alternative_param)
    
#     else:
#         if alternative =='GMD':
#             d = key['data_dimg']
#             key['data_dimi'] = d
#             key['data_type'] = 'gaussienne.multivariee'
#             key['spectrum']  = 'isotropic'
#             x,y = gen_couple(d=d,p=p,key).values()
#             y = shift(y,ndim_to_shift=1,shift_by=1)
        
#         if alternative =='GVD':
#             d = key['data_dimg']
#             key['data_dimi'] = d
#             key['data_type'] = 'gaussienne.multivariee'
#             key['spectrum']  = 'isotropic'
#             x,y = gen_couple(key).values()
#             y =  rescale(y,ndim_to_rescale=1,coef=np.sqrt(2))
            
#         if alternative =='Gretton2012b1':
#             d,nobs,seed = key['data_dimg'],key['data_nobs'],key['data_seed']
#             x,y = Gretton2012b1(d,nobs,seed,mean_shift=.5).values()
            
#         if alternative =='BlobsJitkrittum2016':
#             nobs,seed = key['data_nobs'],key['data_seed']
#             x,y = blobs(nobs,seed,version='Jitkrittum2016').values()
            
#         if alternative =='BlobsChwialkowski2015':
#             nobs,seed = key['data_nobs'],key['data_seed']
#             x,y = blobs(nobs,seed,version='Chwialkowski2015').values()
            
#         if alternative =='BlobsGretton2012b':
#             a=0
#             # x,y = blobs(nobs,seed,version='Gretton2012b').values()
            
#     return({'x':x,'y':y})
        


#%% Permutation and bootstrap

def two_sample_permutation(x,y,seed,bootstrap = False):
    if (isinstance(x, np.ndarray)):
        x = torch.from_numpy(x)
    if (isinstance(y, np.ndarray)):
        y = torch.from_numpy(y)
    
    n1=len(x)
    z = torch.cat((x,y),axis=0)
    n=len(z)
    np.random.seed(seed=seed)
    iz = np.random.choice(n,n,replace = bootstrap,)
    return({'x':z[iz][:n1],'y':z[iz][n1:]})
    



if __name__ == "__main__":
    key = {    
    'data_type': "gaussienne.multivariee",
    'data_dimi': 2,     # n intrinsic variables
    'data_dimg': 3,     # n variables
    'data_seed': 1234,  #
    'data_nobs': 5,  # seq(200,2000, by = 100)}
    'alternative_param':10,    
    'alternative_ndim':1
    }

    alternatives = [
    'shift',
    'rescale',
    'rot_plane',
    'GMD',
    'GVD',
    'Gretton2012b1',
    'BlobsJitkrittum2016',
    'BlobsChwialkowski2015']

    for f in gen_couple(key).values():
        print(f)


    for alternative in alternatives:
        key['alternative'] = alternative
        print(alternative)
        for xy,f in gen_couple_H1(key).items():
            print(xy,len(f))



    # for e in ['rescale','shift','rot_plane']:
    #     print('\n **',e,'\n')
    #     for f in gen_transformed_couple(key).values():
    #         print(f)