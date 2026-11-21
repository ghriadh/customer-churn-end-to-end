import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
# from mlxtend.classifier import StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE

def evaluate_models(X_train, y_train,X_test,y_test,models,param):
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score

        return report

class Training:
    def __init__(self, folder):
        self.data_path: str=os.path.join(folder,"data.csv")

        self.Xtrain_path: str=os.path.join(folder,"X_train.csv")
        self.Xtest_path: str=os.path.join(folder,"X_test.csv")
        self.Xval_path: str=os.path.join(folder,"X_val.csv")

        self.ytrain_path: str=os.path.join(folder,"y_train.csv")
        self.ytest_path: str=os.path.join(folder,"y_test.csv")
        self.yval_path: str=os.path.join(folder,"y_val.csv")

    def train(self):
            
        X_train=pd.read_csv(self.Xtrain_path)
        y_train=pd.read_csv(self.ytrain_path).values.ravel()
        X_test=pd.read_csv(self.Xtest_path)
        y_test=pd.read_csv(self.ytest_path).values.ravel()
        X_val=pd.read_csv(self.Xval_path)
        y_val=pd.read_csv(self.yval_path).values.ravel()

        models = {
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Linear Regression": LinearRegression(),
                "AdaBoost Regressor": AdaBoostClassifier(),
                #"XGBRegressor": XGBClassifier(),
                #"CatBoosting Regressor": CatBoostClassifier(verbose=False), 
            }

        params = {
                "Decision Tree": {'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'gini', 'entropy', 'log_loss'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{ 'n_estimators': [8,16,32,64,128,256], 
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                },
                "Gradient Boosting":{ 'learning_rate':[.1,.01,.05,.001],'subsample':[0.6,0.7,0.75,0.8,0.85,0.9], 'n_estimators': [8,16,32,64,128,256],
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                },
                "Linear Regression":{},
                "AdaBoost Regressor":{'learning_rate':[.1,.01,0.5,.001], 'n_estimators': [8,16,32,64,128,256], 
                                      # 'loss':['linear','square','exponential'],
                                      },
                #"XGBRegressor":{'learning_rate':[.1,.01,.05,.001], 'n_estimators': [8,16,32,64,128,256] },
                #"CatBoosting Regressor":{'depth': [6,8,10], 'learning_rate': [0.01, 0.05, 0.1], 'iterations': [30, 50, 100] },
        }
    
        model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test, 
                                                              models=models,param=params)
        # print(model_report)

        best_model_score = max(sorted(model_report.values()))
        best_model_name = list(model_report.keys())[ list(model_report.values()).index(best_model_score)]
        best_model = models[best_model_name]

        if best_model_score<0.6:  
            #save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
            predicted=best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            print(r2_square)
            print(best_model)
        return r2_square


'''
    for model in models :
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        y_val_pred = model.predict(X_val)
            
        print(model, "Accuracy:", accuracy_score(y_test, y_test_pred), 'F1 score:', f1_score(y_test, y_test_pred, average="weighted"))
        print( "Accuracy val:", accuracy_score(y_val, y_val_pred), 'F1 score val:', f1_score(y_val, y_val_pred, average="weighted"))
        print( "r2 score:", r2_score(y_test, y_test_pred), 'r2 score val:', r2_score(y_val, y_val_pred))
        #print( "matrix:", accuracy_score(y_test, y_test_pred), 'matrix', f1_score(y_val, y_val_pred, average="weighted"))
'''
        