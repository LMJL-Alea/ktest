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

    def univariate_kfda(self,variable,params_model={},visualize_test=True,return_visualize=False,center_by=None,name='',parallel=False,patient=False):
        from .tester import Tester
        zp = self.get_zero_proportions_of_variable(variable)
        # print(zp)
        if zp['x_pz'] == 1 and zp['y_pz'] == 1:
            print("gene not expressed in the dataset")
        else:
            dname = name
            cb = '' if center_by is None else center_by

            if 'approximation_cov' in params_model:
                dname += params_model['approximation_cov']+cb
                print(dname)
            else:
                dname += f'standard{cb}'
                print(dname)

            if f'{dname}_univariate' in self.var and self.var[f'{dname}_univariate'][variable] == True:
                print(f'{dname} already computed for {variable}')

            else : 

                proj = self.init_df_proj(variable)
                cond = self.obs['sample']
                xv = proj[cond=='x'][variable]
                yv = proj[cond=='y'][variable]
            #     print(xv,yv)
                vtest = Tester()
                vtest.init_data(xv,yv,center_by=center_by)
                vtest.init_model(**params_model)
                vtest.obs = self.obs
                vtest.initialize_kfdat()
        #         sp = vtest.spev['xy']['anchors']['w']['sp']
                column = vtest.kfdat(name='kfda')

                t1 = 1
        #         t50 = vtest.select_trunc_by_between_reconstruction_ratio(ratio=.5)
                tr1 = vtest.select_trunc_by_between_reconstruction_ressaut(which_ressaut='first')
                tr2 = vtest.select_trunc_by_between_reconstruction_ressaut(which_ressaut='second')
                tr3 = vtest.select_trunc_by_between_reconstruction_ressaut(which_ressaut='third')
                trm = vtest.select_trunc_by_between_reconstruction_ressaut(which_ressaut='max')

                ts = [t1,tr1,tr2,tr3,trm]
                tnames = ['1','r1','r2','r3','rmax']
                for t,tname in zip(ts,tnames):
                    if t<100:

                        col = f'{dname}_t{tname}'
                        pval = vtest.df_pval[column][t]
                        kfda = vtest.df_kfdat[column][t]
                        errB = 1-vtest.get_between_covariance_projection_error_associated_to_t_new(t)

                        self.vard[variable][f'{col}_pval'] = pval
                        self.vard[variable][f'{col}_kfda'] = kfda 
                        self.vard[variable][f'{col}_errB'] = errB
                        if tname != '1':
                            self.vard[variable][f'{col}_t'] = t

                self.vard[variable][f'{dname}_univariate'] = True
                

                tab = '\t' if len(variable)>6 else '\t\t'
                x_pz = zp['x_pz']
                y_pz = zp['y_pz']
                tr1 = self.vard[variable][f'{dname}_tr1_t']
                tr2 = self.vard[variable][f'{dname}_tr2_t']
                errB1 = self.vard[variable][f'{dname}_tr1_errB']
                errB2 = self.vard[variable][f'{dname}_tr2_errB']
                pval1 = self.vard[variable][f'{dname}_tr1_pval']
                pval2 = self.vard[variable][f'{dname}_tr2_pval']

                string = f'{variable} {tab} pz: x{x_pz:.2f} y{y_pz:.2f} \t pval{tr1:1.0f}:{pval1:1.0e}  r{tr1:1.0f}:{errB1:.2f} \t pval{tr2:1.0f}:{pval2:1.0e}  r{tr2:1.0f}:{errB2:.2f}'
                print(string)
                


                if parallel:
                    return({'v':variable,**self.vard[variable]})
                
                if visualize_test:
                    fig,axes = self.visualize_univariate_test_CRCL(variable,vtest,column=column,patient=patient)
                    if return_visualize:
                        return(vtest,fig,axes)

                return(vtest) 
        
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

    def get_zero_proportions_of_variable(self,variable,condition=None):
        condition = 'sample' if condition is None else condition
        proj = self.init_df_proj(variable,name='counts')
        proj[condition] = self.obs[condition].astype('category')
        output = {}
        for c in proj[condition].cat.categories:
            df = proj[proj[condition]==c][variable]
            n = len(df)
            nz = len(df[df==0])
            pz = nz/n
            output[f'{c}_pz'] = pz
            output[f'{c}_nz'] = nz
            output[f'{c}_n'] = n
        return(output)

    def add_zero_proportions_to_var(self,condition=None):
        '''
        The pd.dataframe attribute var is filled with columns corresponding to the number of zero and the 
        proportion of zero of the genes, with respect to the sample or a condition if informed. 
        '''
        variables = self.data[self.data_name]['variables']
        zp = [self.get_zero_proportions_of_variable(v,condition) for v in variables]
        self.update_var_from_dataframe(pd.DataFrame(zp,index=variables))
    

        
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
