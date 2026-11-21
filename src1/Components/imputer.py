import os
import numpy as np
import pandas as pd
from dataclasses import dataclass


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

@dataclass
class Imputer:
    def del_imputer(df): 
            df.dropna(inplace = True)
            df.reset_index(drop=True, inplace = True)
            
            return(df)
    
    def median_imputer(df): 
            
            feature1 = 'Contract'
            feature2 = 'OnlineSecurity'
            df = fill_by_median (df, feature1, feature2)
            df.reset_index(drop=True, inplace = True)
        
            return(df)
