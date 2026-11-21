import os
import pandas as pd
import numpy as np


class ScaleData:
    def __init__(self, scaler):
        self.scaler = scaler
        
    def fit(self, df, x = None):
        scaler = self.scaler
        target = 'Churn'
        cols = df.columns.drop(target, errors = 'ignore')
            
        df_array = df[cols].values
        scaler.fit(df_array)
        
        return self
        
    def transform(self, df, x = None):
            scaler = self.scaler
            target = 'Churn'
            cols = df.columns.drop(target, errors = 'ignore')
            
            df_array = df[cols].values
            df_scaled = pd.DataFrame(scaler.transform(df_array), columns = cols)
            
            df = df.drop(cols, axis = 1)
            df = pd.concat([df_scaled, df], axis= 1)

            return(df)