from .base import Base
import pandas as pd

class OutliersOps(Base):
    
    def __init__(self):
        super(OutliersOps,self).__init__()

    def select_observations_from_condition(self,threshold,proj='proj_kfda',already_marked_obs_to_consider=None,t='1',orientation='>'):
        if proj in ['proj_kfda','proj_kpca']:
            column_in_dataframe = self.get_kfdat_name()
        elif proj in self.get_variables():
            column_in_dataframe=None
        else:
            print(proj,'not implemented yet in determine outliers from condition')

        df = self.init_df_proj(proj=proj,name=column_in_dataframe)[str(t)]

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

        if already_marked_obs_to_consider is not None:
            marked_obs_to_consider = self.obs[self.obs[already_marked_obs_to_consider]].index
            observations = observations.append(marked_obs_to_consider)

        return(observations)

    def mark_observations(self,observations_to_mark,marking_name):
        index = self.get_index(samples='all',in_dict=False)
        self.obs[marking_name] = pd.DataFrame(index.isin(observations_to_mark),index=index)

