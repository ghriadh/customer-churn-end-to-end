import os
import numpy as np
import pandas as pd
from scipy import stats, special
from dataclasses import dataclass

@dataclass
class Transformation:
    
    def transform_data(df, transformer): 

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

            return(df)
    


