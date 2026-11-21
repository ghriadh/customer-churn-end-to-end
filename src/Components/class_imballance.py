import os
import numpy as np
import pandas as pd
from sklearn.utils import resample
    
class ClassImballance:
    def __init__(self, y = None):
        self.data_path: str=os.path.join('artifacts',"data.csv")
    
    def class_imballance(self, strategy):
            
            df=pd.read_csv(self.data_path)
            target = 'Churn'
            seed = 1234

            minority = df.loc[df[target] == 1]
            majority = df[df[target] == 0]

            if strategy == 'up' : minority = resample(minority, replace = True, n_samples = len(majority), random_state = seed)
            if strategy == 'down' :majority = resample(majority, replace = False, n_samples = len(minority), random_state = seed)

            df = pd.concat([majority, minority])
            df.reset_index(drop=True, inplace = True)

            os.makedirs(os.path.dirname(self.data_path),exist_ok=True)
            df.to_csv(self.data_path,index=False,header=True)
            print(df.shape)
            return(self.data_path)