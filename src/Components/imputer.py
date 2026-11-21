import os
import numpy as np
import pandas as pd


def fill_by_median (df, feature1, feature2):
    feat1 = df[feature1].unique()
    feat2 = df[feature2].unique()
    mean_totalcharges = np.zeros((len(feat1),len(feat2)))
    
    for f1 in feat1 :
        i = np.where(feat1 == f1)[0][0]
        for f2 in feat2:
            j = np.where(feat2 == f2)[0][0]
            mean_df = df[(df[feature1] == f1) & (df[feature2] == f2)]['TotalCharges'].dropna().median() 
            mean_totalcharges[i, j]=mean_df
    # mean_totalcharges    
    for i in range(0, len(feat1)):
        f1 = feat1[i]
        for j in range(0, len(feat2)):
            f2 = feat2[j]
            if not(df.loc[df['TotalCharges'].isnull() & (df[feature1]==f1) & (df[feature2] == f2)].empty):
                df.loc[df['TotalCharges'].isnull() & (df[feature1]==f1) & (df[feature2] == f2), 'TotalCharges'] = mean_totalcharges[i,j]
    return df


class Imputer:
    def __init__(self):
        self.data_path: str=os.path.join('artifacts',"data.csv")
    
    def del_imputer(self): 
            df=pd.read_csv(self.data_path)
            df.dropna(inplace = True)
            df.reset_index(drop=True, inplace = True)
            os.makedirs(os.path.dirname(self.data_path),exist_ok=True)
            df.to_csv(self.data_path,index=False,header=True)
            print(df.shape)
            return(self.data_path)
    
    def median_imputer(self): 
            df=pd.read_csv(self.data_path)
            feature1 = 'Contract'
            feature2 = 'OnlineSecurity'
            df = fill_by_median (df, feature1, feature2)
            df.reset_index(drop=True, inplace = True)
            os.makedirs(os.path.dirname(self.data_path),exist_ok=True)
            df.to_csv(self.data_path,index=False,header=True)
            print(df.shape)
            return(self.data_path)
