import os
import pandas as pd
from sklearn.model_selection import train_test_split

  
class SplitData:
    def __init__(self, rate):
        
        self.data_path: str=os.path.join('artifacts1',"data.csv")
        self.Xtrain_path: str=os.path.join('artifacts1',"X_train.csv")
        self.Xtest_path: str=os.path.join('artifacts1',"X_test.csv")
        self.Xval_path: str=os.path.join('artifacts1',"X_val.csv")

        self.ytrain_path: str=os.path.join('artifacts1',"y_train.csv")
        self.ytest_path: str=os.path.join('artifacts1',"y_test.csv")
        self.yval_path: str=os.path.join('artifacts1',"y_val.csv")

        self.rate = rate

    def transform(self, df):
            
        target = 'Churn'
        seed = 1234
        
        X = df.drop(target, axis = 1)
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1 - self.rate, random_state = seed, stratify = y)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5, random_state = seed, stratify = y_test)

        os.makedirs(os.path.dirname(self.data_path),exist_ok=True)
        df.to_csv(self.data_path,index=False,header=True)
        X_train.to_csv(self.Xtrain_path,index=False,header=True)
        X_test.to_csv(self.Xtest_path,index=False,header=True)
        X_val.to_csv(self.Xval_path,index=False,header=True)

        y_train.to_csv(self.ytrain_path,index=False,header=True)
        y_test.to_csv(self.ytest_path,index=False,header=True)
        y_val.to_csv(self.yval_path,index=False,header=True)

        return(os.path.dirname(self.data_path))