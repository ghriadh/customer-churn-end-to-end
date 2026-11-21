import os
import pandas as pd
from sklearn.model_selection import train_test_split


      
class SplitData:
    def __init__(self, y = None):
        self.data_path: str=os.path.join('artifacts',"data.csv")

        self.Xtrain_path: str=os.path.join('artifacts',"X_train.csv")
        self.Xtest_path: str=os.path.join('artifacts',"X_test.csv")
        self.Xval_path: str=os.path.join('artifacts',"X_val.csv")

        self.ytrain_path: str=os.path.join('artifacts',"y_train.csv")
        self.ytest_path: str=os.path.join('artifacts',"y_test.csv")
        self.yval_path: str=os.path.join('artifacts',"y_val.csv")

    
    def split_data(self, rate):
            
        df=pd.read_csv(self.data_path)
        target = 'Churn'
        seed = 1234
        X = df.drop(target, axis = 1)
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1 - rate, random_state = seed, stratify = y)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5, random_state = seed, stratify = y_test)

        os.makedirs(os.path.dirname(self.data_path),exist_ok=True)
        df.to_csv(self.data_path,index=False,header=True)
        X_train.to_csv(self.Xtrain_path,index=False,header=True)
        X_test.to_csv(self.Xtest_path,index=False,header=True)
        X_val.to_csv(self.Xval_path,index=False,header=True)

        y_train.to_csv(self.ytrain_path,index=False,header=True)
        y_test.to_csv(self.ytest_path,index=False,header=True)
        y_val.to_csv(self.yval_path,index=False,header=True)

        print(X_train.shape)
        return(self.data_path)