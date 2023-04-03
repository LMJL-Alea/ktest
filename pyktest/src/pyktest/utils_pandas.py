
def pd_select_df_from_index(df,index):
    """select a dataframe given an index"""
    return(df.loc[df.index.isin(index)])

def split_dataframe_by_column(df,column,keep_columns=None):
    if keep_columns is None:
        keep_columns = df.columns
    dfs = {c:df[df[column]==c][keep_columns] for c in df[column].unique().sort_values()}
    return(dfs)
