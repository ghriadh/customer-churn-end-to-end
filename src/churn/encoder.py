import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class EncoderData:
    def __init__(self) -> None:
        
        self.encoder = OneHotEncoder(sparse_output=False)
    
    def fit(self, df, y = None) :
        target = 'Churn'   
        self.cat_col = df.describe(include=['O']).columns.union(['SeniorCitizen']).drop(target, errors = 'ignore')
        one_hot_encoded = self.encoder.fit(df[self.cat_col])
        return self
    
    def transform(self, df, y = None):
            target = 'Churn'   
            self.cat_col = df.describe(include=['O']).columns.union(['SeniorCitizen']).drop(target, errors = 'ignore')
          
            one_hot_encoded = self.encoder.transform(df[self.cat_col])
            
            one_hot_df = pd.DataFrame(one_hot_encoded, columns=self.encoder.get_feature_names_out(self.cat_col))
            df_encoded = pd.concat([df, one_hot_df], axis=1)
            df_encoded = df_encoded.drop(self.cat_col, axis=1)

            return(df_encoded)
 