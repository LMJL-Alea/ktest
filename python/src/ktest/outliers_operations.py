from .base import Base
import pandas as pd



def select_observations_in_df_from_threshold(df,threshold,orientation):
    if orientation == '>':
        observations = df[df>threshold].index
    if orientation == '<':
        observations = df[df<threshold].index
    if orientation == '<>':
        observations = df[df<threshold[0]].index
        observations = observations.append(df[df>threshold[1]].index)
    if orientation == '><':
        df = df[df>threshold[0]]
        df = df[df<threshold[1]]
        observations = df.index
    return(observations)

def select_observations_in_df_from_nobs(df,nobs,orientation):
    if orientation == '>':
        observations=df.sort_values(ascending=False)[:nobs].index
    else:
        observations=df.sort_values(ascending=True)[:nobs].index
    return(observations)

class OutliersOps(Base):
    
    # def __init__(self,data,obs=None,var=None,):
    #     super(OutliersOps,self).__init__(data,obs=obs,var=var,)


    def select_observations_from_condition(self,threshold=None,nobs=None,proj='proj_kfda',already_marked_obs_to_consider=None,t='1',orientation='>',sample=None):
        
        if sample is None:
            sample = self.sample
        
        if proj in ['proj_kfda','proj_kpca']:
            column_in_dataframe = self.get_kfdat_name()
        elif proj in self.variables:
            column_in_dataframe=None
        else:
            print(proj,'not implemented yet in determine outliers from condition')

        df = self.init_df_proj(proj=proj,name=column_in_dataframe)[str(t)]
        df = df[df.index.isin(self.obs[self.obs[self.condition]==sample].index)]
        if nobs is None:
            observations=select_observations_in_df_from_threshold(df,threshold,orientation)
        else:
            observations=select_observations_in_df_from_nobs(df,nobs,orientation)

        if already_marked_obs_to_consider is not None:
            marked_obs_to_consider = self.obs[self.obs[already_marked_obs_to_consider]].index
            observations = observations.append(marked_obs_to_consider)

        # observations = observations[observations.isin(self.obs[self.obs[self.condition]==sample].index)]
        
        return(observations)

    def mark_observations(self,observations_to_mark,marking_name):
        index = self.get_index(samples='all',in_dict=False)
        self.obs[marking_name] = pd.DataFrame(index.isin(observations_to_mark),index=index)

    def split_sample(self,pop,sample,new_condition,new_samples=None,condition=None,verbose=0):
        '''
        Split a sample into two distinct populations in the metadata. 
        The first population contains the observations of sample present in pop
        The second population contains the observations of sample absent from pop

        Parameters 
        ----------
            pop : pandas.Index 
                The observations defining the first subpopulation 

            sample : str 
                The sample to split into two subpopulations

            new_condition : str
                The name of the new column created in the metadata table 

            new_samples (default = [sample+'_1',sample+'_2']) : list of 2 str
                The  names to give to the two samples

            condition (default = ktest.condition): str
                The column of the metadata table containing the sample information

        '''
        
        meta = self.obs
        if new_condition in meta:
            if verbose>0:
                print(f'{new_condition} already exists in metadata table')
        else:
            if condition is None:
                condition = self.condition
            if new_samples is None:
                new_samples = [f'{sample}_1',f'{sample}_2']


            meta_sample = meta.loc[meta[condition]==sample]
            pop1 = meta_sample[meta_sample.index.isin(pop)].index
            pop2 = meta_sample[~meta_sample.index.isin(pop)].index

            meta[new_condition] = meta[condition]
            meta[new_condition] = meta[new_condition].cat.add_categories(new_samples)
            
            meta.loc[meta.index.isin(pop1),new_condition] = new_samples[0]
            meta.loc[meta.index.isin(pop2),new_condition] = new_samples[1]

            meta[new_condition] = meta[new_condition].cat.remove_categories(sample)


    def concatenate_samples(self,samples,new_condition=None,new_sample=None,condition=None,verbose=0):
        '''
        Concatenate several samples into one populations in the metadata. 


        Parameters 
        ----------
            samples : list of str 
                The samples to concatenate into one population

            new_condition : str
                The name of the new column created in the metadata table 

            new_sample (default = "_".join(samples)) : str
                The  names to give to the new sample

            condition (default = ktest.condition): str
                The column of the metadata table containing the samples information

                
        Returns
        -------
            new_condition,new_sample
        '''
        
        meta = self.obs

        if new_condition in meta:
            if verbose>0:
                print(f'{new_condition} already exists in metadata table')
        else:
            if condition is None:
                condition = self.condition

            if new_sample is None:
                new_sample = "_".join(samples)

            if new_condition is None:
                new_condition = f"{condition}_concatenated_{new_sample}"
                
            pop = meta.loc[meta[condition].isin(samples)].index
            # meta_sample = meta.loc[meta[condition]==sample]
            # pop1 = meta_sample[meta_sample.index.isin(pop)].index
            # pop2 = meta_sample[~meta_sample.index.isin(pop)].index

            meta[new_condition] = meta[condition]
            meta[new_condition] = meta[new_condition].cat.add_categories(new_sample)
            
            meta.loc[meta.index.isin(pop),new_condition] = new_sample

            for sample in samples:
                meta[new_condition] = meta[new_condition].cat.remove_categories(sample)
        return(new_condition,new_sample)
    

    def make_groups_from_gene_presence(self,gene):

        dfg = self.init_df_proj(proj=gene)
        self.obs[f'pop{gene}'] = (dfg[gene]>=1).map({True: f'{gene}+', False: f'{gene}-'})
        self.obs[f'pop{gene}'] = self.obs[f'pop{gene}'].astype('category')
