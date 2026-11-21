import os
import numpy as np
import pandas as pd
from scipy import stats, special

class Transformation:
    def __init__(self):
        self.data_path: str=os.path.join('artifacts',"data.csv")
        
    
    def transform_data(self, transformer): 
            df=pd.read_csv(self.data_path)
            
            target = 'Churn'
            num_col = df.describe().columns.drop('SeniorCitizen')

            if transformer == 'boxcox' :
                for feature in num_col:
                    df[feature], _ = stats.boxcox(df[feature])
            if transformer == 'log' :
                for feature in num_col:
                    df[feature] = np.log(df[feature])
            if transformer == 'sqrt' :
                for feature in num_col:
                    df[feature] = np.sqrt(df[feature])
            if transformer == 'boxcox1p' :
                for feature in num_col:
                    df[feature] = special.boxcox1p(df[feature] + 1, stats.boxcox_normmax(df[feature] + 1))


            os.makedirs(os.path.dirname(self.data_path),exist_ok=True)
            df.to_csv(self.data_path,index=False,header=True)
            print(df.head(2))
            return(self.data_path)
    


