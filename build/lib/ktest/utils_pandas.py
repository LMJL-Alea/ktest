
def pd_select_df_from_index(df,index):
    """select a dataframe given an index"""
    return(df.loc[df.index.isin(index)])
