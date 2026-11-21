import os
import numpy as np
import pandas as pd
from scipy import stats, special

class Transformation:
    def __init__(self, transformer):
        self.transformer = transformer
    
    def fit(self, df, y = None):
          return(self)

    def transform(self, df, y = None): 

            num_col = df.describe().columns.drop(['SeniorCitizen', 'Churn'], errors = 'ignore')

            if self.transformer == 'boxcox' :
                for feature in num_col:
                    df[feature], _ = stats.boxcox(df[feature])
            if self.transformer == 'log' :
                for feature in num_col:
                    df[feature] = np.log(df[feature])
            if self.transformer == 'sqrt' :
                for feature in num_col:
                    df[feature] = np.sqrt(df[feature])
            if self.transformer == 'boxcox1p' :
                for feature in num_col:
                    df[feature] = special.boxcox1p(df[feature] + 1, stats.boxcox_normmax(df[feature] + 1))

            return(df)
    


