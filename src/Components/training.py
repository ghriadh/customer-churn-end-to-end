import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE


class Training:
    def __init__(self):
        self.data_path: str=os.path.join('artifacts',"data.csv")

        self.Xtrain_path: str=os.path.join('artifacts',"X_train.csv")
        self.Xtest_path: str=os.path.join('artifacts',"X_test.csv")
        self.Xval_path: str=os.path.join('artifacts',"X_val.csv")

        self.ytrain_path: str=os.path.join('artifacts',"y_train.csv")
        self.ytest_path: str=os.path.join('artifacts',"y_test.csv")
        self.yval_path: str=os.path.join('artifacts',"y_val.csv")

    
    def train(self):
            
        X_train=pd.read_csv(self.Xtrain_path)
        y_train=pd.read_csv(self.ytrain_path)
        X_test=pd.read_csv(self.Xtest_path)
        y_test=pd.read_csv(self.ytest_path)
        X_val=pd.read_csv(self.Xval_path)
        y_val=pd.read_csv(self.yval_path)

        models = [SGDClassifier(), RidgeClassifier(), RandomForestClassifier(), DecisionTreeClassifier(), 
                  GradientBoostingClassifier(), AdaBoostClassifier(), LGBMClassifier(verbose = -1), 
                  KNeighborsClassifier(), LogisticRegression(), CatBoostClassifier(verbose = 0)
                  ]
        
        for model in models :
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            y_val_pred = model.predict(X_val)
            print(model, "Accuracy:", accuracy_score(y_test, y_test_pred), 'F1 score:', f1_score(y_test, y_test_pred, average="weighted"))
            print( "Accuracy val:", accuracy_score(y_val, y_val_pred), 'F1 score val:', f1_score(y_val, y_val_pred, average="weighted"))
        
        
        
        return(self.data_path)