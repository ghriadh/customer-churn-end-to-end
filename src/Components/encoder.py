import os
import numpy as np
import pandas as pd

    
class EncoderData:
    def __init__(self):
        self.data_path: str=os.path.join('artifacts',"data.csv")
    
    def encoder_cat(self):
            df=pd.read_csv(self.data_path)
            target = 'Churn'
            cat_col= cat_col = df.describe(include=['O']).columns.union(['SeniorCitizen']).drop('Churn')
            df=pd.read_csv(self.data_path)
            df[target]= df[target].astype('category').cat.codes
            for feature in cat_col :
                if df[feature].nunique() == 2 :
                    df[feature]= df[feature].astype('category').cat.codes
            cat_vars = df[cat_col].nunique()[df[cat_col].nunique()>2].keys().to_list()
            col_dummies = pd.get_dummies(data = df[cat_vars], drop_first= True, dtype=int, columns = cat_vars)
            df = pd.concat([df, col_dummies], axis = 1).drop(cat_vars, axis = 1)

            os.makedirs(os.path.dirname(self.data_path),exist_ok=True)
            df.to_csv(self.data_path,index=False,header=True)
            print(df.shape)
            return(self.data_path)
   