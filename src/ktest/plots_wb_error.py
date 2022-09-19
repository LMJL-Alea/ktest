import numpy as np
import matplotlib.pyplot as plt

from .residuals import Residuals
from .statistics import Statistics
from torch import cat,tensor,float64


class Plot_WBerrors(Residuals,Statistics):
        
    def __init__(self):        
        super(Plot_WBerrors, self).__init__()


    def plot_pval_with_respect_to_within_covariance_reconstruction_error(self,name,fig=None,ax=None,scatter=True,trunc=None,outliers_in_obs=None):
        '''
        Plots the opposite of log10 pvalue with respect to the percentage of reconstruction 
        of the spectral truncation of the within covariance operator 
        compared to the full within covariance operator. 
        
        Parameters
        ----------
            self : Tester,
            Should contain a filled `df_pval` attribute
            
            name : str,
            Should correspond to a column of the attribute `df_pval`

            scatter (default = True) : boolean, 
            If True, a scatter plot is added to the plot in order to visualize each step of the reconstruction.     
            If False, it is only a plot. 
    
            fig (optional) : matplotlib.pyplot.figure 
            The figure of the plot.
            A new one is created if needed.
            
            ax (optional) : matplotlib.pyplot.axis 
            The axis of the plot.
            A new one is created if needed.

        
            trunc (optionnal) : list,
            The order of the eigenvectors to project (\mu_2 - \mu_1), 
            By default, the eigenvectors are ordered by decreasing eigenvalues. 

        '''
        
        if fig is None:
            fig,ax = plt.subplots(figsize=(7,7))
        
        name = outliers_in_obs if outliers_in_obs is not None else name 

        log10pval = self.df_pval[name].apply(lambda x: -np.log(x)/np.log(10))
        log10pval = np.array(log10pval[log10pval<10**10])
        expvar = np.array(self.get_explained_variance()[:len(log10pval)])
        
        threshold = -np.log(0.05)/np.log(10)
        ax.plot(expvar,log10pval,label=name,lw=.8,alpha=.5)
        
        expvar_acc = expvar[log10pval<=threshold]
        log10pval_acc = log10pval[log10pval<=threshold]

        expvar_rej = expvar[log10pval>threshold]
        log10pval_rej = log10pval[log10pval>threshold]
        
        if scatter:
            if len(expvar_acc)>0:
                ax.scatter(expvar_acc,log10pval_acc,color='green')
            if len(expvar_rej)>0:
                ax.scatter(expvar_rej,log10pval_rej,color='red')
            ax.plot(expvar,log10pval,label=name,lw=.8,alpha=.5)
        else:
            ax.plot(expvar_acc,log10pval_acc,lw=1,alpha=1)
            ax.plot(expvar_rej,log10pval_rej,label=name,lw=1,alpha=1)

        ax.set_ylabel('-log10pval',fontsize=30)
        ax.set_xlabel(r'$\Sigma_W$ reconstruction',fontsize=30)
        ax.set_ylim(0,20)
        ax.set_xlim(-.05,1.05)
        ax.axhline(-np.log(0.05)/np.log(10),)
        return(fig,ax)
    
    def plot_pval_with_respect_to_between_covariance_reconstruction_error(self,name,fig=None,ax=None,scatter=True,outliers_in_obs=None):
        if fig is None:
            fig,ax = plt.subplots(figsize=(7,7))
        
        name = outliers_in_obs if outliers_in_obs is not None else name 

        log10pval = self.df_pval[name].apply(lambda x: -np.log(x)/np.log(10))
        log10pval = np.array(log10pval[log10pval<10**10])
        error = np.array(self.get_between_covariance_projection_error()[:len(log10pval)])
        
        threshold = -np.log(0.05)/np.log(10)
        
        error_acc = error[log10pval<=threshold]
        log10pval_acc = log10pval[log10pval<=threshold]

        error_rej = error[log10pval>threshold]
        log10pval_rej = log10pval[log10pval>threshold]
        
        if scatter:
            if len(error_acc)>0:
                ax.scatter(error_acc,log10pval_acc,color='green')
            if len(error_rej)>0:
                ax.scatter(error_rej,log10pval_rej,color='red')
            ax.plot(error,log10pval,label=name,lw=.8,alpha=.5)
        else:
            ax.plot(error_acc,log10pval_acc,lw=1,alpha=1)
            ax.plot(error_rej,log10pval_rej,label=name,lw=1,alpha=1)

        ax.set_ylabel('-log10pval',fontsize=30)
        ax.set_xlabel(r'$\Sigma_B$ reconstruction ',fontsize=30)
        ax.set_ylim(0,20)
        ax.set_xlim(-.05,1.05)
        ax.axhline(-np.log(0.05)/np.log(10),)
        return(fig,ax)
        
    def plot_relative_reconstruction_errors(self,name,fig=None,ax=None,scatter=True,outliers_in_obs=None):
        if fig is None:
            fig,ax = plt.subplots(figsize=(7,7))

        name = outliers_in_obs if outliers_in_obs is not None else name 

        log10pval = self.df_pval[name].apply(lambda x: -np.log(x)/np.log(10))
        log10pval = np.array(log10pval[log10pval<10**10])
        threshold = -np.log(0.05)/np.log(10)

        errorB = np.array(self.get_between_covariance_projection_error(outliers_in_obs=outliers_in_obs))
        errorW = np.array(self.get_explained_variance())

        errorB_acc = errorB[:len(log10pval)][log10pval<=threshold]
        errorW_acc = errorW[:len(log10pval)][log10pval<=threshold]
        
        errorB_rej = errorB[:len(log10pval)][log10pval>threshold]
        errorW_rej = errorW[:len(log10pval)][log10pval>threshold]
        
        if scatter:
            if len(errorB_acc)>0:
                ax.scatter(errorB_acc,errorW_acc,color='green')
            if len(errorB_rej)>0:
                ax.scatter(errorB_rej,errorW_rej,color='red')
            ax.plot(errorB,errorW,label=name,lw=.8,alpha=.5)
            

        else:
            ax.plot(errorB,errorW,lw=1,alpha=1)

        ax.set_ylabel(r'$\Sigma_W$ reconstruction ',fontsize=30)
        ax.set_xlabel(r'$\Sigma_B$ reconstruction ',fontsize=30)
        
        errorB = errorB[~np.isnan(errorB)]
        errorW = errorW[~np.isnan(errorW)]

        mini = np.min([np.min(errorB),np.min(errorW)])
        h = (1 - mini)/20
        ax.plot(np.arange(mini,1,h),np.arange(mini,1,h),c='xkcd:bluish purple',lw=.4,alpha=1)
        
        return(fig,ax)
        
    def plot_ratio_reconstruction_errors(self,name,fig=None,ax=None,scatter=True,outliers_in_obs=None):
        if fig is None:
            fig,ax = plt.subplots(figsize=(7,7))
        
        log10pval = self.df_pval[name].apply(lambda x: -np.log(x)/np.log(10))
        log10pval = np.array(log10pval[log10pval<10**10])
        threshold = -np.log(0.05)/np.log(10)

    #     errorB = np.array(get_between_covariance_projection_error(self))
    #     errorW = np.array(self.get_explained_variance())

        errorB = np.array(self.get_between_covariance_projection_error()[:len(log10pval)])
        errorW = np.array(self.get_explained_variance()[:len(log10pval)])

        
        errorB_acc = errorB[log10pval<=threshold]
        errorW_acc = errorW[log10pval<=threshold]
        
        errorB_rej = errorB[log10pval>threshold]
        errorW_rej = errorW[log10pval>threshold]
        
        
        la = len(errorB_acc)
        lb = len(errorB_rej)
        if scatter:
            if len(errorB_acc)>0:
                ax.scatter(np.arange(1,la+1),errorB_acc/errorW_acc,color='green')
            if len(errorB_rej)>0:
                ax.scatter(np.arange(la+1,la+1+lb),errorB_rej/errorW_rej,color='red')
            ax.plot(np.arange(1,la+1+lb),errorB/errorW,label=name,lw=.8,alpha=.5)
        
        ax.set_xlabel('truncation',fontsize=30)
        ax.set_ylabel('reconstruction ratio',fontsize=30)
        return(fig,ax)

    def plot_within_covariance_reconstruction_error_with_respect_to_t(self,name='explained variance',fig=None,ax=None,scatter=True,t=None,outliers_in_obs=None):
        
        if fig is None:
            fig,ax = plt.subplots(figsize=(7,7))


        label = name # tr($\Sigma_W$) = {trace:.3e}'

        explained_variance = self.get_explained_variance()
        explained_variance = cat([tensor([0],dtype=float64),explained_variance])
        expvar = 1 - explained_variance
        trunc = np.arange(0,len(expvar))
        
        if scatter:
            ax.scatter(trunc,expvar)
            ax.plot(trunc,expvar,label=label,lw=.8,alpha=.5)
        else:
            ax.plot(trunc,expvar,lw=1,alpha=1,label=label)
        

        ax.set_ylabel(r'$\Sigma_W$ reconstruction',fontsize=30)
        ax.set_xlabel('truncation',fontsize=30)
        ax.set_ylim(-.05,1.05)
        xmax = len(expvar) if t is None else t
        
        ax.set_xlim(-1,xmax)
        xticks = np.arange(0,xmax) if xmax<20 else np.arange(0,xmax,2) if xmax<50 else np.arange(0,xmax,5) if xmax<200 else np.arange(0,xmax,10) if xmax < 500 else np.arange(0,xmax,20)
        ax.set_xticks(xticks)
        return(fig,ax)

    def plot_between_covariance_reconstruction_error_with_respect_to_t(self,name='explained difference',fig=None,ax=None,scatter=True,t=None):
        if fig is None:
            fig,ax = plt.subplots(figsize=(7,7))
        projection_error,delta = self.get_between_covariance_projection_error(return_total=True)
        projection_error = cat([tensor([0],dtype=float64),projection_error])
        errorB = 1 - projection_error
        trunc = np.arange(0,len(errorB))
        label = name #of {delta:.3e}'
        if scatter:
            ax.scatter(trunc,errorB)
            ax.plot(trunc,errorB,label=label,lw=.8,alpha=.5)
        else:
            ax.plot(trunc,errorB,lw=1,alpha=1,label=label)
        

        ax.set_ylabel(r'$(\mu_2 - \mu_1)$ projection error',fontsize=30)
        ax.set_xlabel('truncation',fontsize=30)
        ax.set_ylim(-.05,1.05)
        xmax = len(errorB) if t is None else t
        
        ax.set_xlim(-1,xmax)
        xticks = np.arange(0,xmax) if xmax<20 else np.arange(0,xmax,2) if xmax<50 else np.arange(0,xmax,5) if xmax<200 else np.arange(0,xmax,10) if xmax < 500 else np.arange(0,xmax,20)
        ax.set_xticks(xticks)
        return(fig,ax)

