from ktest.base import Data
import pandas as pd
class OutliersOps(Data):
    
    def __init__(self):
        super(OutliersOps,self).__init__()

    def determine_outliers_from_condition(self,threshold,which='proj_kfda',outliers_in_obs=None,t='1',orientation='>'):
        
        if which in ['proj_kfda','proj_kpca']:
            column_in_dataframe = self.get_kfdat_name()
        else:
            print(which,'not implemented yet in determine outliers from condition')

        df = self.init_df_proj(which=which,name=column_in_dataframe)[str(t)]

        if orientation == '>':
            outliers = df[df>threshold].index
        if orientation == '<':
            outliers = df[df<threshold].index
        if orientation == '<>':
            outliers = df[df<threshold[0]].index
            outliers = outliers.append(df[df>threshold[1]].index)

        if outliers_in_obs is not None:
            df_outliers = self.obs[outliers_in_obs]
            old_outliers    = df_outliers[df_outliers].index
            outliers = outliers.append(old_outliers)

        return(outliers)

    def add_outliers_in_obs(self,outliers,name_outliers):
        index = self.get_xy_index()
        # print(outliers)
        # print(f'out index {len(index)} {len(index.isin(outliers))}')
        self.obs[name_outliers] = pd.DataFrame(index.isin(outliers),index=index)

