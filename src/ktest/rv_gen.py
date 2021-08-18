import numpy as np
import torch
import scipy as sc

# Pour tout plein de distributions clé en main
# https://docs.scipy.org/doc/numpy-1.15.0/reference/routines.random.html


P,D,N,S = 10,50,10,0.01
DI,DN = 5,5
#%% Generation

def mvn_pf(p=P, d=D, nobs=N, sig=S, seed=1, noise=True,spectrum='isotropic', **kwargs):
    """ 
    Generate a multivariate gaussian random variable
    """
    #    torch.manual_seed(seed)
    np.random.seed(seed=seed)
    mu = torch.zeros(p)
    if spectrum == 'isotropic':
        cov = torch.eye(p)
    elif spectrum == 'decreasing_linear':
        cov = torch.diag(torch.tensor(list(range(1,p+1))))
    elif spectrum == 'decreasing_geometric':
        cov = torch.diag(torch.tensor([0.9**i for i in range(p)]))
    xp = np.random.multivariate_normal(mu, cov, nobs)
    xd_p = np.zeros((nobs, d-p))
    xd = np.concatenate((xp, xd_p), axis=1)

    if noise:
        # ep et ep_d car on peut vouloir en enlever un
        ep = np.random.normal(0, sig, (nobs, p))
        ep_d = np.random.normal(0, sig, (nobs, d-p))
        e = np.concatenate((ep, ep_d), axis=1)
        return torch.from_numpy(xd + e)

    else:
        return xd



def mixture_gaussienne(means,covs,weights,nobs,seed):
    np.random.seed(seed=seed)
    assignations = np.random.multinomial(nobs,weights)
    x = np.concatenate([np.random.multivariate_normal(mean,cov,assignation) \
                        for mean,cov,assignation in zip(means,covs,assignations)],axis=0)
    return(x)

def mixture_gaussienne_sparse(means,covs,weights,nobs,seed,d):
    xp = mixture_gaussienne(means,covs,weights,nobs,seed)
    p=len(means[0])
    xd_p = np.zeros((nobs,d-p))
    x = np.concatenate((xp,xd_p),axis=1)
    return(x)

def gen_couple(key = {}):
    """
    Generates a couple of sample for which H0 holds
    Example key : 
    {'data_type':'gaussienne.multivariee', #type of distribution
     'data_dimi':1, #intrinsic dimension
     'data_dimg':3, #global dimension
     'data_noise':False, #noise on 
     'spectrum':'isotropic',
     'data_nobs':102,
     'data_seed':1999}
    """
    
    ref_generators = {'gaussienne.multivariee': mvn_pf}
    generator = ref_generators[key['data_type']] if 'data_type' in key else mvn_pf
    
    arg = {arg_name: key[arg_key] if arg_key in key else default \
           for arg_name,arg_key,default in zip(['p','d','sig','noise','spectrum'],
        ['data_dimi','data_dimg','data_noise_sig','data_noise','data_spectrum'],
        [P,D,S,False,'isotropic'])}
    
    seed = key['data_seed'] if 'data_seed' in key else 1994

    # échantillons de taille différente
    if 'data_xnobs' in key:
        argx = arg.copy()
        argy = arg.copy()
        argx['nobs'] = key['data_xnobs'] if 'data_xnobs' in key else N
        argy['nobs'] = key['data_ynobs'] if 'data_ynobs' in key else N
        couple = {'x': generator(seed=seed, **argx),
              'y': generator(seed=seed+1, **argy)}
    # les deux échantillons font la même taille
    else:
        arg['nobs'] =  key['data_nobs'] if 'data_nobs' in key else N
        couple = {'x': generator(seed=seed, **arg),
              'y': generator(seed=seed+1, **arg)}

    return couple

#%% Transformation
def shift(y,ndim_to_shift=1,shift_by=1):
    """
    Shifts the `ndim_to_shift` first dimensions of every observation of dataset `y` by `shift_by` 
    """
    y[:,:ndim_to_shift] = y[:,:ndim_to_shift] + shift_by 
    return(y)

def rescale(y,ndim_to_rescale=1,coef=2):
    """
    Rescales the `ndim_to_rescale` first dimensions of every observation of dataset `y` by `coef` 
    """
    y[:,:ndim_to_rescale] = y[:,:ndim_to_rescale] * coef
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


def Gretton2012b1(d,nobs,seed,mean_shift=.5):
    """
    x is a gaussian and y is the mixtrue of two gaussians with different means. 
    """

    key= {'data_dimg':d,'data_dimi':d,'data_nobs':nobs,'data_seed':seed,'data_type':'gaussienne.multivariee','spectrum':'isotropic'}
    x = gen_couple(key)['x']
    mean1 = np.zeros(d)
    mean2 = np.zeros(d)
    mean1[0] = mean2[1] = mean_shift
    
    means=[mean1,mean2]
    covs = [np.eye(d)]*2
    weights = [.5,.5]
    
    y = mixture_gaussienne(means,covs,weights,nobs,seed+1)
    return({'x':x,'y':y})

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
    x = mixture_gaussienne(means,xcovs,weights,nobs,seed)

    version = 'Chwialkowski2015'

    if version == 'Jitkrittum2016':
        
        param = 2 if param is None else param
        ycov = np.eye(2)
        ycov[0,0] = param
        ycovs = [ycov]*len(means)
        y = mixture_gaussienne(means,ycovs,weights,nobs,seed+1)

    if version =='Chwialkowski2015':
        param = np.pi/4 if param is None else param
        ycov = np.eye(2)
        ycov[0,0] = np.cos(np.pi/4)
        ycov[1,0] = np.sin(np.pi/4)
        ycovs = [ycov]*len(means)
        y = mixture_gaussienne(means,ycovs,weights,nobs,seed+1)

    if version == 'Gretton2012b':
        print('pas implémenté')
        # a grid of correlated Gaussians with a ratio ε of largest to smallest covariance eigenvalues.
    return({'x':x,'y':y})


def gen_couple_H1(key={}):
    """
    
    
    if alternative in ['GMD','GVD','Blobs'] : 
    Generates a toy dataset of two samples of observations with different distributions 
    for which the difference has already been used to assess the performances of a two sample test.
    Attention: the information contained in key may be ignored in order to fit exactly the distribution       
    """
    
    assert('alternative' in key)
    
    alternative = key['alternative'] 
    if alternative in ['shift','rescale','rot_plane']:
        
        x,y = gen_couple(key).values()
        param = key['alternative_param']
        if alternative in ['shift','rescale']:
            ndim = key['alternative_ndim']

        if alternative == 'shift':
            y = shift(y,ndim_to_shift=ndim,shift_by=param)

        if alternative == 'rescale':
            y = rescale(y,ndim_to_rescale=ndim,coef=param)

        if alternative =='rot_plane':
            y = rot_plane(y,angle=param)
    
    else:
        if alternative =='GMD':
            d = key['data_dimg']
            key['data_dimi'] = d
            key['data_type'] = 'gaussienne.multivariee'
            key['spectrum']  = 'isotropic'
            x,y = gen_couple(key).values()
            y = shift(y,ndim_to_shift=1,shift_by=1)
        
        if alternative =='GVD':
            d = key['data_dimg']
            key['data_dimi'] = d
            key['data_type'] = 'gaussienne.multivariee'
            key['spectrum']  = 'isotropic'
            x,y = gen_couple(key).values()
            y =  rescale(y,ndim_to_rescale=1,coef=np.sqrt(2))
            
        if alternative =='Gretton2012b1':
            d,nobs,seed = key['data_dimg'],key['data_nobs'],key['data_seed']
            x,y = Gretton2012b1(d,nobs,seed,mean_shift=.5).values()
            
        if alternative =='BlobsJitkrittum2016':
            nobs,seed = key['data_nobs'],key['data_seed']
            x,y = blobs(nobs,seed,version='Jitkrittum2016').values()
            
        if alternative =='BlobsChwialkowski2015':
            nobs,seed = key['data_nobs'],key['data_seed']
            x,y = blobs(nobs,seed,version='Chwialkowski2015').values()
            
        if alternative =='BlobsGretton2012b':
            a=0
            # x,y = blobs(nobs,seed,version='Gretton2012b').values()
            
    return({'x':x,'y':y})
        


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