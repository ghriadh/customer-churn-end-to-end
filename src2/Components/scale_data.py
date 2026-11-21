import os
import pandas as pd


class ScaleData:
    def __init__(self, scaler):
        self.scaler = scaler

    def transform(self, df):
            scaler = self.scaler
            target = 'Churn'        
            y = df[target]
            df_array = df.values
            df = pd.DataFrame(scaler.fit_transform(df_array), columns = df.columns)
            df[target] = y

            return(df)