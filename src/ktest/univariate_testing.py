import pandas as pd

from time import time
from joblib import Parallel, delayed, parallel_backend
import os

from .outliers_operations import OutliersOps

'''
La question du test univarié sur des données univariées est difficile notamment par rapport à ce qu'on veut
retenir de chaque test univarié unique. 
Ces fonctions permettent de simplifier l'approche de test univariée à partir de données multivariées et 
la visualisation des résultats.  
'''

class Univariate:
    
    def __init__(self):        
        super(Univariate, self).__init__()

    def variable_eligible_for_univariate_testing(self,variable,max_dropout=.85):
        # on suppose que la fonction add_zero_proportions_to_var_new a deja tourné
        dn = self.data_name
        sl = list(self.get_index().keys())
        elligible = any([self.var[dn][f'{s}_pz'][variable]<=max_dropout for s in sl])
        return(elligible)

    def compute_zero_proportions_of_variable(self,variable):
        
        proj = self.init_df_proj(variable)
        dict_index = self.get_index()
        dict_zp = {}
        for sample,index_sample in dict_index.items():
            df = proj[proj.index.isin(index_sample)][variable]
            n = len(df)
            nz = len(df[df==0])
            dict_zp[f'{sample}_pz'] = nz/n
            dict_zp[f'{sample}_nz'] = nz
            dict_zp[f'{sample}_n'] = n
        return(dict_zp)

    def add_zero_proportions_to_var_new(self):
        '''
        The pd.dataframe attribute var is filled with columns corresponding to the number of zero and the 
        proportion of zero of the genes, with respect to the sample or a condition if informed. 
        '''
        variables = self.data[self.data_name]['variables']
        zp = [self.compute_zero_proportions_of_variable(v) for v in variables]
        dfzp = pd.DataFrame(zp,index=variables)
        self.update_var_from_dataframe(dfzp)

    def get_zero_proportions_of_variable(self,variable):
        dn = self.data_name
        dict_index = self.get_index()
        dict_zp = {sample:self.var[dn][f'{sample}_pz'][variable] for sample in dict_index.keys()}
        return(dict_zp)

    def compute_univariate_kfda(self,variable,common_covariance=False):
        from .tester import Tester
        # récupérer la donnée
        
        # Get testing info from global object 
        dfv = self.init_df_proj(variable)
        data_name,condition,samples,outliers_in_obs = self.get_data_name_condition_samples_outliers()
        cov,mmd,lm,ab,m,r = self.get_model()
        center_by = self.center_by
        
        # Initialize univariate tester with common covariance 
        t = Tester()
        
        t.add_data_to_Tester_from_dataframe(dfv,sample='x',df_meta=self.obs,data_name=data_name)
        t.obs['sample'] = self.obs['sample']
        
        t.set_test_data_info(data_name,condition,samples)
        t.set_outliers_in_obs(outliers_in_obs)
        t.set_center_by(center_by)
        t.init_model(approximation_cov=cov,approximation_mmd=mmd,
                    m=m,r=r,landmark_method=lm,anchors_basis=ab)
        
        # Infos for common covariance

        if common_covariance:
            
            spc,evc = self.get_spev('covw')
            spev_name = self.get_covw_spev_name()
            t.spev['covw'][spev_name] = {'sp':spc,'ev':evc}

            if 'nystrom' in cov or 'nystrom' in mmd:
                spa,eva = self.get_spev('anchors')
                anchors_name = self.get_anchors_name()
                t.spev['anchors'][anchors_name] = {'sp':spa,'ev':eva}

            kernel_bandwith = self.kernel_bandwidth
            p = self.data[data_name]['p']
            t.init_kernel(f'gauss_{kernel_bandwith/p}')


            kfdat_name = t.compute_kfdat_new() # caclul de la stat         
            t.select_trunc() # selection automatique de la troncature 
            t.compute_pval() # calcul des troncatures asymptotiques 
        else: 
            t.init_kernel('gauss_median')
            kfdat_name = t.kfdat()

        # ccovw = 'ccovw' if common_covariance else ''
        t.df_kfdat[variable] = t.df_kfdat[kfdat_name]
        t.df_pval[variable] = t.df_pval[kfdat_name]
        
        return(t)
        
    def add_results_univariate_kfda_in_var(self,vtest,variable):
        
        kfdat_name = self.get_kfdat_name()
        dn = self.data_name
        zp = self.get_zero_proportions_of_variable(variable)
        t1 = 1
        tr1 = vtest.select_trunc_by_between_reconstruction_ressaut(which_ressaut='first')
        tr2 = vtest.select_trunc_by_between_reconstruction_ressaut(which_ressaut='second')
        tr3 = vtest.select_trunc_by_between_reconstruction_ressaut(which_ressaut='third')
        trm = vtest.select_trunc_by_between_reconstruction_ressaut(which_ressaut='max')

        ts = [t1,tr1,tr2,tr3,trm]
        tnames = ['1','r1','r2','r3','rmax']
        for t,tname in zip(ts,tnames):
            if t<100:

                col = f'{kfdat_name}_t{tname}'
                pval = vtest.df_pval[variable][t]
                kfda = vtest.df_kfdat[variable][t]
                errB = 1-vtest.get_between_covariance_projection_error_associated_to_t_new(t)

                self.vard[dn][variable][f'{col}_pval'] = pval
                self.vard[dn][variable][f'{col}_kfda'] = kfda 
                self.vard[dn][variable][f'{col}_errB'] = errB
                if tname != '1':
                    self.vard[dn][variable][f'{col}_t'] = t

        self.vard[dn][variable][f'{kfdat_name}_univariate'] = True
                

        tab = '\t' if len(variable)>6 else '\t\t'

        tr1 = self.vard[dn][variable][f'{kfdat_name}_tr1_t']
        tr2 = self.vard[dn][variable][f'{kfdat_name}_tr2_t']
        errB1 = self.vard[dn][variable][f'{kfdat_name}_tr1_errB']
        errB2 = self.vard[dn][variable][f'{kfdat_name}_tr2_errB']
        pval1 = self.vard[dn][variable][f'{kfdat_name}_tr1_pval']
        pval2 = self.vard[dn][variable][f'{kfdat_name}_tr2_pval']

        zp_string = "zp: "+"".join([f'{s}{zp[s]:.2f}' for s in zp.keys()])
        string = f'{variable} {tab} {zp_string} \t pval{tr1:1.0f}:{pval1:1.0e}  r{tr1:1.0f}:{errB1:.2f} \t pval{tr2:1.0f}:{pval2:1.0e}  r{tr2:1.0f}:{errB2:.2f}'
        print(string)

    def univariate_kfda(self,variable,common_covariance=True,parallel=False):
        univariate_name = f'{self.get_kfdat_name}_univariate'
        dn = self.data_name

        if univariate_name in self.var[dn] and self.var[dn][univariate_name][variable]:
            print(f'{univariate_name} already computed for {variable}')
        else:
            t=self.compute_univariate_kfda(variable,common_covariance=common_covariance)
        self.add_results_univariate_kfda_in_var(t,variable)
        if parallel:
            return({'v':variable,**self.vard[variable]})
        return(t) 
        
    # pas à jour 
    def parallel_univariate_kfda(self,params_model={},center_by=None,name='',n_jobs=1,lots=100,fromto=[0,-1],save_path=None):
        """
        parallel univariate testing.
        Variables are tested by groups of `lots` variables in order to save intermediate results if the 
        procedure is interupted before the end. 
        """
        dname = name
        cb = '' if center_by is None else center_by

        if 'approximation_cov' in params_model:
            dname += params_model['approximation_cov']+cb
        else:
            dname += f'standard{cb}'
        print(dname)
            
        fromto[1] = len(voi) if fromto[1]==-1 else fromto[1]
        variables = self.data[self.data_name]['variables']
        voi = variables[fromto[0]:fromto[1]]
        
        
        ntot = len(variables)
        ntested = 0
        nnot_tested = len(voi)
        if f'{dname}_univariate' in self.var:
            tested = self.var[self.var[f'{dname}_univariate']==1].index
            ntested = len(tested)
            nnot_tested = ntot-ntested

            voi = voi[~voi.isin(tested)]
            print(f'filter {ntested} variables already tested')

        ntotest = len(voi)
        print(f'variables : total:{ntot}  t:{ntested} nt:{nnot_tested} ->{ntotest}')
            

        
        if n_jobs==1:
            a=[]
            for v in voi:
                a+=[self.univariate_kfda(v,
                                        params_model=params_model,
                                        center_by=center_by,
                                        name=name,
                                        visualize_test=False,
                                        return_visualize=False,
                                        parallel=True)]
        elif n_jobs >1:
            # with parallel_backend('loky'):
            
            vss = [voi[h*lots:(h+1)*lots] for h in range(len(voi)//lots)]
            vss += [voi[len(voi)//lots:]]
            
            for vs in vss:  
                print(f'testing {len(vs)}/{ntot}')
                t0 = time()  
                with parallel_backend('loky'):
                    a = Parallel(n_jobs=n_jobs)(delayed(self.univariate_kfda)(
                        v,
                        params_model=params_model,
                        center_by=center_by,
                        name=name,
                        visualize_test=False,
                        return_visualize=False,
                        parallel=True)
                        for v  in  vs)
                
                print(f'Tested in {time() - t0} \n\n')
                for a_,v in zip(a,vs):
                    if a_ is None: 
                        print(f'{v} was not tested')
                    if a_ is not None:
        #                 print(a_)
                        if a_['v'] != v:
                            print(f'pb with {v} different from ', a_['v'])
                            v = a_['v']                    
                        for key,value in a_.items():
                            if key != 'v':
                                self.vard[v][key] = value

                if save_path is not None:
                    print('saving')
                    
                    df = pd.DataFrame(self.vard).T
                    self.update_var_from_dataframe(df)
                    file = f'{dname}_univariate.csv'
                    if file in os.listdir(save_path):
                        self.load_univariate_test_results_in_var(save_path,file,)
                    self.save_univariate_test_results_in_var(save_path)
     
    def update_var_from_dataframe(self,df,verbose = 0):
        dn = self.data_name
        for c in df.columns:
            if verbose>1:
                print(c,end=' ')
            token = False
            if 'univariate' in c and c in self.var[dn]:
                token = True
                nbef = sum(self.var[dn][c]==1)
            if c not in self.var[dn]:
                self.var[dn][c] = df[c].astype('float64').copy()
            else:
                if verbose>1:
                    print('update',end= '|')
                self.var[dn][c].update(df[c].astype('float64'))
            if token:
                naft = sum(self.var[dn][c]==1)
                if verbose >0:
                    print(f'\n tested from {nbef} to {naft}')

    def save_univariate_test_results_in_var(self,path):
        for cu in self.var.columns:
            if 'univariate' in cu:
                print(cu)
                dname = cu.split(sep='_univariate')[0]
                cols = [c for c in self.var.columns if f'{dname}_' in c]
                df = self.var[cols]
                df = df[df[cu]==True]
                print(df.shape)
                df.to_csv(f'{path}{cu}.csv')

    def load_univariate_test_results_in_var(self,path,file,verbose=0):
        df = pd.read_csv(f'{path}{file}',index_col=0)
        if verbose >0:
            print(file,len(df))
        self.update_var_from_dataframe(df)
    

        
"""
Les tests univariés sont couteux, ils nécessitent de faire un test par variable d'intérêt. 
Ces fonctions permettent de paralleliser ces calculs. 
Ce sont d'anciennes fonctions que j'ai codées dans un autre contexte et que je n'ai pas encore pris le 
temps de rendre compatible avec ma nouvelle façon de faire des tests univariés. 
Ce qui change, c'est les infos que je veux garder à la fin, et le fait que je veux que le calcul univarié
soit une fonction liée à la classe Tester.  
"""
# def concat_kfda_and_pval_from_parallel_simu(output_parallel_simu):
#     lkfda,lpval = [],[]
#     for l in output_parallel_simu: 
#         lkfda += [l[0]]
#         lpval += [l[1]]
    
#     return(pd.concat(lkfda,axis=1),pd.concat(lpval,axis=1))

# def parallel_kfda_univariate(x,y,kernel='gauss_median',params_model={},n_jobs=3):
#     """
#     A function to analyse each gene in parallel using the univariate approach
#     n_job=1 : parallel code is not used, it is usefull for debugging
#     """

#     outputs_test = Tester()
#     outputs_test.df_power = pd.DataFrame()
#     outputs_test.infos = {}
#     # non parallel 
#     if n_jobs==1:
#         a=[]
#         for c in x.columns:
#             xc, yc = x[c].to_numpy().reshape(-1,1),y[c].to_numpy().reshape(-1,1)
#             a+=[compute_standard_kfda(xc,yc,c,kernel=kernel,params_model=params_model)]
#     elif n_jobs >1:
#         # with parallel_backend('loky'):
#         with parallel_backend('loky'):
#             a = Parallel(n_jobs=n_jobs)(delayed(compute_standard_kfda)(
#                 x[variable].to_numpy().reshape(-1,1),
#                 y[variable].to_numpy().reshape(-1,1),
#                 variable,
#                 kernel = kernel,
#                 params_model=params_model,
#                 ) for variable  in  x.columns)
#     kfda0,pval0 = concat_kfda_and_pval_from_parallel_simu(a)
#     outputs_test.df_kfdat = kfda0
#     outputs_test.df_pval = pval0
#     return(outputs_test)
