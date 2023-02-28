def determine_outliers_from_condition(self,threshold,which='proj_kfda',column_in_dataframe='standardstandard',t='1',orientation='>',outliers_in_obs=None):
    df = self.init_df_proj(which=which,name=column_in_dataframe,outliers_in_obs=outliers_in_obs)[str(t)]


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
    index = self.get_index()
    self.obs[name_outliers] = index.isin(outliers)

