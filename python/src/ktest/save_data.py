import pandas as pd
# P: Pas dans le main package
class SaveData:

    def __init__(self):        
        super(SaveData, self).__init__()

    def load_kfdat(self,path):
        df = pd.read_csv(path,header=0,index_col=0)
        for c in df.columns:
            if c not in self.df_kfdat.columns:
                self.df_kfdat[c] = df[c]
            else: 
                print(f'kfdat {c} already here')

    def load_proj_kfda(self,path,name,):
        df = pd.read_csv(path,header=0,index_col=0)
        if name not in self.df_proj_kfda:
            self.df_proj_kfda[name] = df
        else:
            print(f'proj kfda {name} already here')

    def load_proj_kpca(self,path,name):
        df = pd.read_csv(path,header=0,index_col=0)
        if name not in self.df_proj_kpca:
            self.df_proj_kpca[name] = df
        else:
            print(f'proj kpca {name} already here')

    def load_correlations(self,path,name):
        df = pd.read_csv(path,header=0,index_col=0)
        if name not in self.corr:
            self.corr[name] = df
        else:
            print(f'corr {name} already here')

    def save_kfdat(self,path):
        self.df_kfdat.to_csv(path,index=True)

    def save_proj_kfda(self,path,name):
        self.df_proj_kfda[name].to_csv(path,index=True)

    def save_proj_kpca(self,path,name):
        self.df_proj_kpca[name].to_csv(path,index=True)

    def save_correlations(self,path,name):
        self.corr[name].to_csv(path,index=True)

    def load_data(self,to_load):
        """
        to_load ={'kfdat':path_kfdat,
                    'proj_kfda':{name1:path1,name2,path2},
                    'proj_kpca':{name1:path1,name2,path2},
                    'correlations':{name1:path1,name2,path2}}
        """
        types_ref= {'proj_kfda':self.load_proj_kfda,
                    'proj_kpca':self.load_proj_kpca,
                    'correlations':self.load_correlations}
        
        for type in to_load:
            if type == 'kfdat':
                self.load_kfdat(to_load[type]) 
            else:
                for name,path in to_load[type].items():
                    types_ref[type](path,name)
