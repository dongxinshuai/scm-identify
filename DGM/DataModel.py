import numpy as np
import pandas as pd
from math import sqrt, log
from pdb import set_trace
import os
import scipy


class DataModel:
    def __init__(self, df_v, df_x):
        self.df_x = df_x.copy()
        self.df_v = df_v.copy()
        self.xvars = list(df_x.columns)
        self.vars = list(df_v.columns)
        self.N = len(self.df_v)
        
    def generate_data(self, N, normalized=True):
        if N==-1:
            num=len(self.df_x)
        else:
            num = N

        df_x=self.df_x.sample(num,replace=False,random_state=0)
        rest_df_x=self.df_x.drop(df_x.index)
        df_v = self.df_v.loc[df_x.index]
        rest_df_v=self.df_v.drop(df_x.index)

        if normalized:
            df_x=(df_x-df_x.mean())/df_x.std()
            df_v=(df_v-df_v.mean())/df_v.std() 
            rest_df_x=(rest_df_x-rest_df_x.mean())/rest_df_x.std()
            rest_df_v=(rest_df_v-rest_df_v.mean())/rest_df_v.std()

        return df_x, df_v

    def check_gaussian(self, save_path="./check_gaussian"): # jsut for using GIN
        import matplotlib.pyplot as plt
        import seaborn
        data_df = self.df_v.copy()
        data_df=(data_df-data_df.mean())
        data_df=data_df/data_df.std()

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for c in data_df.columns.to_list():
            plt.clf()
            data_df[c].hist()  # s is an instance of Series
            plt.savefig(os.path.join(save_path, f'{c}.png'))
            print(f'{c}: {"Not Gaussian" if scipy.stats.shapiro(data_df[c])[1]<0.05 else "Gaussian"}  {scipy.stats.shapiro(data_df[c])}')

    def scatter_plot(self, save_path, var_list=None):
        import matplotlib.pyplot as plt
        import seaborn
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        if var_list is None:
            var_list = self.df_v.columns.to_list()

        for x in var_list:
            for y in var_list:
                if x!=y:
                    df = pd.DataFrame({x: self.df_v[x]+np.random.normal(0,0.1,size=self.df_v[x].shape), y: self.df[y]+np.random.normal(0,0.1,size=self.df_v[y].shape)})
                    plt.clf()
                    #seaborn.displot(data=penguins, x="flipper_length_mm", hue="species", multiple="stack")
                    #g = seaborn.jointplot(x=x, y=y, data=df)
                    g = seaborn.jointplot(x=x, y=y, data=df, scatter_kws={'alpha': 0.3, 'rasterized': True, 's':1,}, kind='reg', joint_kws={'line_kws':{'color': 'orange', 'linewidth':1}})
        
                    plt.savefig(os.path.join(save_path, f'{x}_{y}.png'))