import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .residuals import Residuals
# from .kernel_statistics import Statistics
from .utils_plot import adjusted_xticks
from torch import mv,dot,norm,ger,eye,diag,ones,diag,matmul,float64,isnan,sort,cat,tensor,sum,log
from numpy import sqrt
import torch
class Plot_WBerrors(Residuals):
        
    def __init__(self):        
        super(Plot_WBerrors, self).__init__()

    # Visualizations considérées comme pertinentes


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
        eB = self.get_explained_difference()

        eB = cat((tensor([1],dtype=float64),eB))
        projection_error = eB[1:] - eB[:-1]
        projection_error = projection_error[~isnan(projection_error)]
        sorted_projection_error,ordered_truncations  = sort(projection_error,descending = True)
        ordered_truncations += 1
        
        return(sorted_projection_error,ordered_truncations)
        
    # def get_between_covariance_projection_error(self,return_total=False):


    def get_explained_difference_of_t(self,t):
        pe = self.get_explained_difference()
        return(pe[t-1].item())

    def get_explained_variability_of_t(self,t):
        exv = self.get_explained_variability()
        return(exv[t-1].item())

    def get_explained_difference(self,cumul=False,log=False,decreasing=False):
        '''
        Returns the explained difference with respect to the truncation. 
        Can be cumulated and log. 
        Parameters
        ----------
            self : Ktest, 
            Should contain the eigenvectors and eigenvalues of the within covariance operator in the attribute `spev`
            
        Returns 
        ------
            projection_error : torch.Tensor
            The projection error of (\mu_2- \mu_1) as a percentage. 
        '''
        
        cov = self.approximation_cov
        n = self.get_ntot(landmarks=False)
        sp,ev = self.get_spev('covw')  
        sp12 = sp**(-1/2)
        ev,sp12 = ev[:,~isnan(sp12)],sp12[~isnan(sp12)]
        fv    = n**(-1/2)*sp12*ev if cov == 'standard' else ev         
        # K = self.compute_gram(landmarks=False)
        om = self.compute_omega()
        
        if cov != 'standard':
            n_landmarks = self.get_ntot(landmarks=True)
            Lz,Uz = self.get_spev(slot='anchors')
            Lz12 = diag(Lz**-(1/2))
            Pz = self.compute_covariance_centering_matrix(quantization=False,landmarks=True)
            Kzx = self.compute_kmn()
            # print(f'm{m},fv{fv.shape} Lz12 {Lz12.shape} Uz{Uz.shape} Pz {Pz.shape} Kzx {Kzx.shape} om {om.shape}')
            mmdt = (n_landmarks**(-1/2)* mv(fv.T,mv(Lz12,mv(Uz.T,mv(Pz,mv(Kzx,om)))))**2).cumsum(0)**(1/2) if cumul else \
                (n_landmarks**(-1/2)* mv(fv.T,mv(Lz12,mv(Uz.T,mv(Pz,mv(Kzx,om)))))**2)**(1/2)
            tot = (n_landmarks**(-1/2)* mv(fv.T,mv(Lz12,mv(Uz.T,mv(Pz,mv(Kzx,om)))))**2).sum(0)**(1/2)
            exd = mmdt/tot
        else:
            pkm = self.compute_pkm()
            mmdt =(mv(fv.T,pkm)**2).cumsum(0)**(1/2) if cumul else (mv(fv.T,pkm)**2)**(1/2) 
            tot = (mv(fv.T,pkm)**2).sum(0)**(1/2)
            exd = mmdt/tot
        exd = 1 - exd if decreasing else exd
        exd = np.log(exd) if log else (exd)

        return exd



    def plot_explained_difference(self,t=None,fig=None,ax=None,cumul=False,log=False,decreasing=False,color='xkcd:neon purple',label='difference',legend=True):
        if fig is None:
            fig,ax = plt.subplots(figsize=(12,6))
        exd = self.get_explained_difference(cumul=cumul,log=log,decreasing=decreasing)
        t = len(exd) if t is None else t
        trunc = range(1,t+1)
        
        # ax.plot(trunc,exd[:t],lw=1,alpha=1,label=label,color=color)
        ax.bar(trunc,exd[:t],color=color,alpha=1,label=label)
        # ax1.bar(xp,ypnorm,color='red',alpha=.2)

        ax.set_ylabel(r'Difference',fontsize=30)
        ax.set_xlabel('t',fontsize=30)
        ax.set_xticks(adjusted_xticks(t))
                
        if not log:     
            ax.set_ylim(-.05,1.05)
        if legend:
            ax.legend(fontsize=20)
        return(fig,ax)

    def compute_explained_variability_per_condition(self,cumul=False,log=False,decreasing=False):
        # Sur nystrom on pourrait se demander quelle est la part de la variabilité des conditions capturée par les ancres. 
        
        n=self.get_ntot()    
        upk = self.compute_upk(n)
        if 'nystrom' in self.approximation_cov:
            upk = upk[:,:-1]
        dfepk = pd.DataFrame(upk,index=self.get_index(in_dict=False),columns=[str(t) for t in range(upk.shape[1])])
        
        exvc = {} 
        for k,v in self.get_index().items():
            dfk = dfepk.loc[dfepk.index.isin(v)]
            exvck = dfk.var().cumsum()/dfk.var().sum() if cumul else dfk.var()/dfk.var().sum()
            exvck = 1 - exvck if decreasing else exvck
            exvck = np.log(exvck) if log else exvck
            exvc[k] = exvck
        return(exvc)
        
    def get_explained_variability(self,within=True,cumul=False,log=False,decreasing=False):
        if within: 
            exv = self.get_spectrum(cumul=cumul,part_of_inertia=True,log=log,decreasing=decreasing)
        else: 
            exv = self.compute_explained_variability_per_condition(cumul=cumul,log=log,decreasing=decreasing)
        return(exv)

    def plot_explained_variability(self,t=None,fig=None,ax=None,within=True,conditions=True,cumul=False,log=False,decreasing=False,color_w='xkcd:sea green',colors_g=['xkcd:cerulean','xkcd:light orange'],legend=True):
        if fig is None:
            fig,ax = plt.subplots(figsize=(12,6))
        colors_g = [None,None] if colors_g is None else colors_g
        tmax = 0 if t is None else t 
        if within:
            exv = self.get_explained_variability(cumul=cumul,log=log,decreasing=decreasing)
            tw = len(exv) if t is None else t
            trunc = range(1,tw+1)
            label = 'w-variability'
            ax.bar(trunc,exv[:t],color=color_w,alpha=1,label=label)
            # ax.plot(trunc,exv[:tw],lw=1,alpha=1,color=color_w,label=label)
            tmax = np.max([tmax,tw])
            ax.set_ylabel(r'w-variability',fontsize=30)
        
        if conditions:
            exv = self.get_explained_variability(within=False,cumul=cumul,log=log,decreasing=decreasing)
            if len(exv)==2:
                colors = {k:c for k,c in zip(exv.keys(),colors_g)}
            else:
                colors = {k:None for k in exv.keys()}
            for k,v in exv.items():
                tk = len(v) if t is None else t                   
                trunc = np.arange(1,tk+1)
                label=f'{k}-variability'
                # ax.plot(trunc,v[:tk],label=label,lw=1,alpha=1,color=colors[k])
                ax.bar(trunc,v[:tk],color=colors[k],alpha=.4,label=label)
                tmax = np.max([tmax,tk])
            ax.set_ylabel(r'group variability',fontsize=30)
            
        if within and conditions:
            ax.set_ylabel(r'variability',fontsize=30)
        if legend:
            ax.legend(fontsize=20)

        ax.set_xlabel('t',fontsize=30)
        if not log:
            ax.set_ylim(-.05,1.05)
        ax.set_xlim(-1,tmax)
        ax.set_xticks(adjusted_xticks(tmax))
        
        return(fig,ax)

    def plot_part_log_spectrum(self,t=None,fig=None,ax=None):
        if fig is None:
            fig,ax = plt.subplots(figsize=(12,6))

        sp = self.get_spectrum(cumul=False,part_of_inertia=False,log=True)
        sp = sp[~torch.isnan(sp)]
        sp = (sp+sp.min().abs())/(sp.max()-sp.min())
        tmax = len(sp) if t is None else min(t,len(sp))
        trunc = range(1,tmax)
        ax.plot(trunc,sp[:trunc[-1]],label='log-spectrum')
        ax.set_xlabel('t',fontsize= 20)
        ax.set_ylim(0,1.05)
        return(fig,ax)



