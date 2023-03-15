from scipy.cluster.hierarchy import dendrogram,linkage
from scipy.stats import chi2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_sample_argmin_from_similarities(similarities):
    """
    Returns the two closest samples according to the similarity matrix `similarities`.
    
    Parameters
    ----------
        similarities : pandas.DataFrame
            Entries are similarities between samples, index and columns are named according to the samples. 
            Designed to work with the output of Ktest.kfda_similarity_of_samples()  
            
    Returns
    -------
        (sample1,sample2,similarity) : str,str,float
            samples1 and sample2 are the two closest samples according to the matrix `similarities`. 
            similarity is the similarity between sample1 and sample2. 
    """
    
    s = similarities
    c1 = s[s>0].min().idxmin()
    c2 = s[s>0][c1].idxmin()
    kfda = s[s>0].min().min()
    return(c1,c2,kfda)


def get_leafs_of_dendrogram(d):
    leafs = []
    for icoord,dcoord in zip(d['icoord'],d['dcoord']):
        for x,y in zip(icoord,dcoord):
            if y==0:
                leafs+=[x]
    return(leafs)

class Dendrogram:
    
    def kfda_similarity_of_samples(self,t=1,condition=None,samples='all',verbose=0):
        """
        Compute the similarity matrix of the list `samples` stored in column `condition` of the metadata.
        The similarity is defined as the KFDA statistic associate to `t`. 
        Returns the matrix as a pandas.DataFrame
        
        Parameters
        ----------
            self : Object of class Ktest 
            
            t : int
                Truncation used to defined the similarity. 
                
            condition (default = None) : str 
                column of the metadata table containing the samples
            
            samples (default = 'all') : 'all' or list of str 
                list of samples to be considered in the similarity matrix. 
                
            verbose (default = 0) : int
                The higher, the more verbose
                
        Returns
        -------
            similarities : pandas.DataFrame
                The entries are the similarities, index and columns are named according to the samples. 
                
        """
        
        samples_list = self.get_samples_list(condition=condition,samples=samples)
        n = len(samples_list)
        similarities = {sample:{sample:0} for sample in samples_list}
        names = []
        for s1 in samples_list:
            for s2 in samples_list:
                if s1<s2:
                    self.set_test_data_info(condition=condition,samples=[s1,s2])
                    self.multivariate_test(verbose=verbose)
                    kfda = self.get_kfda(condition=condition,samples=[s1,s2])[t]
                    similarities[s1][s2] = kfda
                    similarities[s2][s1] = kfda

        return(pd.DataFrame(similarities,index=samples_list,columns=samples_list))
    


    def custom_linkage(self,t=1,samples=None,condition=None,concatenate_samples=False,verbose=0):
        """
        Computes a linkage compatible with the function dendrogram fro scipy.cluster.hierarchy using the 
        KFDA statistic associated to the truncation `t` as a similarity. 

        Parameters
        ----------
            self : Object of class Ktest 

            t (default = 1): int
                Truncation used to defined the similarity. 

            condition (default = None) : str 
                column of the metadata table containing the samples

            samples (default = 'all') : 'all' or list of str 
                list of samples to be considered in the similarity matrix. 

            verbose (default = 0) : int
                The higher, the more verbose

        Returns
        -------
            Z : linkage matrix

            ordered_comparisons : List of compared samples. 

        """

        samples_list = self.get_samples_list(samples=samples,condition=condition)
        similarities = self.kfda_similarity_of_samples(t=t,condition=condition,samples=samples,verbose=verbose)
        n = len(samples_list)
        if not concatenate_samples:
            s = similarities.to_numpy()
            Z = linkage(s[np.triu_indices(n,k=1)])
            return(Z,samples_list)
        else:
            id_map = list(range(n))
            Z = np.empty((n - 1, 4))
            x=0
            y=0
            indexes = list(range(n-1))
            ordered_comparisons = []

            for k in range(n - 1):
                # my find two closest clusters x, y (x < y)

                c1,c2,kfda = get_sample_argmin_from_similarities(similarities)
                x,y = samples_list.index(c1),samples_list.index(c2)
                x,y = np.sort([x,y])
                id_x,id_y = id_map.index(x),id_map.index(y)
                catx,caty = samples_list[x],samples_list[y]

                new_condition,new_sample = self.concatenate_samples(samples=[catx,caty])
                nx = 1 if id_x < n else Z[id_x - n, 3]
                ny = 1 if id_y < n else Z[id_y - n, 3]

                samples_list += [new_sample]
                ordered_comparisons += [new_sample]

                similarities = self.kfda_similarity_of_samples(t=t,condition=new_condition,samples='all',verbose=verbose)
                #         # my record the new node
                Z[k, 0] = min(x, y)
                Z[k, 1] = max(y, x)
                Z[k, 2] = kfda
                Z[k, 3] = nx + ny # nombre de clusters ou nombre de cells ?  
                id_map[id_x] = -1  # cluster x will be dropped
                id_map[id_y] = n + k  # cluster y will be replaced with the new cluster

            return(Z,ordered_comparisons)


    def dot_of_test_result_on_dendrogram(self,x,y,ax,t=1):
        """
        Add a dot at position (`x`,`y`) on axis `ax` according to the pvalue associated to `t`

        Parameters
        ----------

        """

        pval = self.get_pvalue()[t]
        c =  'green' if pval >.05 else 'red'
        yaccept = chi2.ppf(.95,t)
        ax.scatter(x,yaccept,s=500,c='red',alpha=.5,marker='_',)
        ax.scatter(x,y,s=300,c=c,edgecolor='black',alpha=1,linewidths=3,)



    def plot_custom_dendrogram(self,t=1,samples=None,condition=None,
                                        y_max=None,fig=None,ax=None,
                                        samples_labels=None,dots=False,
                                        concatenate_samples=False,
                                        verbose=0):
        if fig is None:
            fig,ax = plt.subplots(figsize= (10,6))
        if samples_labels==None:
            samples_labels = self.get_samples_list(samples=samples,condition=condition)
        linkage,comparisons = self.custom_linkage(t=t,samples=samples,condition=condition,concatenate_samples=concatenate_samples,verbose=verbose)
        kfda_max = np.max(linkage[:,2])

        d = dendrogram(linkage,labels=samples_labels,ax=ax)

        abscisses = d['icoord']
        ordinates = d['dcoord']

        if dots:
            icomp = 0
            for x,y in zip(abscisses,ordinates):
                if np.abs(y[1]-kfda_max)<10e-6:
                    self.dot_of_test_result_on_dendrogram(1/2*(x[1]+x[2]),y[1],ax=ax,t=t)
                for i in [0,-1]:
                    if y[i] > 10e-6:
                        self.dot_of_test_result_on_dendrogram(x[i],y[i],ax=ax,t=t)
                        icomp+=1
            print(comparisons[icomp])

        if y_max is not None:
            ax.set_ylim(0,y_max)
        ax.tick_params(axis='x', labelrotation=90,labelsize=20 )
    #     return(d)

        return(fig,ax)
