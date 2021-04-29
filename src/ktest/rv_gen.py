import numpy as np
import torch
import scipy as sc

# Pour tout plein de distributions clé en main
# https://docs.scipy.org/doc/numpy-1.15.0/reference/routines.random.html


P,D,N,S = 10,50,10,0.01
DI,DN = 5,5
#%% Generation
def mvn_pf(p=P, d=D, nobs=N, sig=S, seed=1, noise=True,dsp = False, **kwargs):
    #    torch.manual_seed(seed)
    np.random.seed(seed=seed)
    mu = torch.zeros(p)
    if dsp:
        cov = torch.diag(torch.tensor(list(range(1,p+1))))
    else:
        cov = torch.eye(p)
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


def mixn_pf(p=P, d=D, nobs=N, sig=S, seed=1, noise=True, **kwargs):
    np.random.seed(seed=seed)

    nc = 3  # ncomponents
    mu = [[5]*p, [-5]*p, ([5, -5]*p)[:p]]
    cov = torch.eye(p)
    w = [1/nc]*nc  # uniform

    c = np.random.multinomial(nobs, w, size=1)
    xp = np.concatenate([np.random.multivariate_normal(
        mu[a], cov, c[0, a]) for a in range(nc)], axis=0)
    xd_p = np.zeros((nobs, d-p))
    xd = np.concatenate((xp, xd_p), axis=1)
    if noise:
        # deux car on peut vouloir en enlever un
        ep = np.random.normal(0, sig, (nobs, p))
        ep_d = np.random.normal(0, sig, (nobs, d-p))
        e = np.concatenate((ep, ep_d), axis=1)
        return(torch.from_numpy(xd + e))
    else:
        return(xd)



#%% Transformation

def rot_plane(p=P, d=D, parameter=np.pi/2, **kwargs):
    m = torch.eye(d, dtype=torch.float64)
    if p != d:

        c = np.cos(parameter)
        s = np.sin(parameter)

        e1 = 0
        e2 = p

        m[e1, e1] = c
        m[e1, e2] = -s
        m[e2, e1] = s
        m[e2, e2] = c

#  m = torch.from_numpy(m)
    return(m)


def rescale(p=P, d=D, dimi=DI, dimn=DN, parameter=0.5, **kwargs):
    # p intrinsic space dim  # d global space dim   # transfo.dimi dimensions scaled in intrinsic space   # transfo.dimn dimensions scaled in noise space

    i1 = np.eye(dimi) * parameter
    i2 = np.eye(p-dimi)

    n1 = np.eye(dimn)*parameter
    n2 = np.eye(d-p-dimn)

    m = sc.linalg.block_diag(i1, i2, n1, n2)

    m = torch.from_numpy(m)
    return(m)


def shift(p=P, d=D, nobs=N, dimi=DI, dimn=DN, parameter=1, **kwargs):
    # p intrinsic space dim  # d global space dim  # nobs nobservations of dataset  # dimi dimensions scaled in intrinsic space   
    # # dimn dimensions scaled in noise space

    m = np.repeat([[parameter]*dimi + [0] * (p-dimi) + [parameter]
                   * dimn + [0] * (d-p-dimn)], nobs, axis=0)
    m = torch.from_numpy(m)

    return(m)


#%% Final 
def gen_transfo(key = {}, params=None):

    ref_transformations = {'shift': shift, 'rot_plane': rot_plane, 'rescale': rescale}
    function_transformation = ref_transformations[key['transfo_type']] if 'transfo_type' in key else shift
    arg = {'p':    key['data_dimi']    if 'data_dimi'    in key else P,
           'd':    key['data_dimg']    if 'data_dimg'    in key else D,
           'nobs': key['data_nobs']    if 'data_nobs'    in key else N,
           'dimi': key['transfo_dimi'] if 'transfo_dimi' in key else DI,
           'dimn': key['transfo_dimn'] if 'transfo_dimn' in key else DN}
    if params == None:
        return function_transformation(**arg)
    return {param: function_transformation(parameter=param,**arg) for param in params}

def gen_couple(key = {}):
    ref_generators = {'gaussienne.multivariee': mvn_pf,  'mixture.gaussienne': mixn_pf}
    generator = ref_generators[key['data_type']] if 'data_type' in key else mvn_pf

    arg = {'p':     key['data_dimi']      if 'data_dimi'      in key else P,
           'd':     key['data_dimg']      if 'data_dimg'      in key else D,
           'sig':   key['data_noise_sig'] if 'data_noise_sig' in key else S,
           'noise': key['data_noise']     if 'data_noise'     in key else True,
           'dsp':   key['dsp']            if 'dsp'            in key else False}
    seed = key['data_seed'] if 'data_seed' in key else 1994
        
    # les deux échantillons font la même taille
    if 'data_xnobs' in key:
        argx = arg.copy()
        argy = arg.copy()
        argx['nobs'] = key['data_xnobs'] if 'data_xnobs' in key else N
        argy['nobs'] = key['data_ynobs'] if 'data_ynobs' in key else N
        couple = {'x': generator(seed=seed, **argx),
              'y': generator(seed=seed+1, **argy)}
    
    else:
        arg['nobs'] =  key['data_nobs'] if 'data_nobs' in key else N
        couple = {'x': generator(seed=seed, **arg),
              'y': generator(seed=seed+1, **arg)}

    # échantillons de taille différente
    
    return couple


def gen_transformed_couple(key={}, params=None):

    x,y = gen_couple(key).values()
    mtype = key['transfo_type'] if 'transfo_type' in key else 'shift'
    couple = {}
    if params == None:
        matrix = gen_transfo(key)
        if mtype in ['rescale', 'rot_plane']:
            y = torch.mm(y, matrix)
        elif mtype == 'shift':
            y = y + matrix
        return{'x':x,'y':y}

    else: 
        matrices = gen_transfo(key, params)            
        for param in params:
            matrix = matrices[param]
            if mtype in ['rescale', 'rot_plane']:
                y = torch.mm(y, matrix)
            elif mtype == 'shift':
                y = y + matrix
            couple[param] = {'x': x, 'y': y}
        return couple


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

    'transfo_type': "shift",  # rescale / shift / rot_plane
    'transfo_dimi': 1,  # n intrinsic dim to change
    'transfo_dimn': 0,  # n noise dim to change

    # mixture.gaussienne / gaussienne.multivariee
    'data_type': "gaussienne.multivariee",
    'data_noise': True,
    'data_noise_sig': S,  # noise ~ N(0,sig)
    'data_dimi': 2,     # n intrinsic variables
    'data_dimg': 3,     # n variables
    'data_seed': 1234,  #
    'data_nobs': 2,  # seq(200,2000, by = 100)
    }

    for e in ['gaussienne.multivariee','mixture.gaussienne']:
        key['data_type'] = e
        print('\n **',e,'\n')
        for f in gen_couple(key).values():
            print(f)

    for e in ['rescale','shift','rot_plane']:
        print('\n **',e,'\n')
        for f in gen_transformed_couple(key).values():
            print(f)