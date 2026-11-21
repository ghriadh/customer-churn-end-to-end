import os
import numpy as np
import pandas as pd
from sklearn.utils import resample
from dataclasses import dataclass

@dataclass    
class ClassImballance:
    def __init__(self, strategy):
        self.strategy = strategy
    
    def transform(self, df):
            
            target = 'Churn'
            seed = 1234
            
            minority = df.loc[df[target] == 'Yes']
            majority = df[df[target] == 'No']

            if self.strategy == 'up' : minority = resample(minority, replace = True, n_samples = len(majority), random_state = seed)
            if self.strategy == 'down' :majority = resample(majority, replace = False, n_samples = len(minority), random_state = seed)

            df = pd.concat([majority, minority])
            df.reset_index(drop=True, inplace = True)

            return(df)