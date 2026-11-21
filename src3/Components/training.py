import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from utils import save_process
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
# from mlxtend.classifier import StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE


def model_training(model,param, X_train, y_train,X_test,y_test):
    model_name = model[0]
    model = model[1]
    param = param[1]
    st.write(f'we train now {model_name}, wait')
    txt = 'we train now ' + model_name + ', wait ....'
    st.text_area(txt)
    gs = GridSearchCV(model,param,cv=3)
    gs.fit(X_train,y_train)
    model.set_params(**gs.best_params_)

    model.fit(X_train, y_train)
    y_train_pred= model.predict(X_train)
    y_test_pred = model.predict(X_test)
   
    accuracy_train = accuracy_score(y_train, y_train_pred).round(3)
    accuracy_test = accuracy_score(y_test, y_test_pred).round(3)
    f1_train = f1_score(y_train, y_train_pred).round(3)
    f1_test = f1_score(y_test, y_test_pred).round(3)
        
    plt.figure(figsize=(15,5))
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion matrix ' + model_name , fontweight="black", size=16,pad=10)
    os.makedirs("artifacts2/confusion_matrix",exist_ok=True)
    fig: str = model_name +".png"
    fig_path: str=os.path.join('artifacts2/confusion_matrix',fig)
    plt.savefig(fig_path)
    txt = 'training ' + model_name + ' was completed'
    st.image(fig_path, width=600, caption= model_name )

    model_result = {'model_name' : model_name, 'model': model,  'accuracy_train':accuracy_train, 'accuracy_test': accuracy_test, 'f1_train': f1_train, 'f1_test': f1_test}
    return model_result
       
class Training:
    def __init__(self, folder):
        self.Xtrain_path: str=os.path.join(folder,"X_train.csv")
        self.Xtest_path: str=os.path.join(folder,"X_test.csv")
        self.Xval_path: str=os.path.join(folder,"X_val.csv")
        self.ytrain_path: str=os.path.join(folder,"y_train.csv")
        self.ytest_path: str=os.path.join(folder,"y_test.csv")
        self.yval_path: str=os.path.join(folder,"y_val.csv")

        self.best_model_path : str=os.path.join('artifacts2/process',"best_model.pkl") 

    def train(self):
           
        X_train=pd.read_csv(self.Xtrain_path)
        y_train=pd.read_csv(self.ytrain_path).values.ravel()
        X_test=pd.read_csv(self.Xtest_path)
        y_test=pd.read_csv(self.ytest_path).values.ravel()
        X_val=pd.read_csv(self.Xval_path)
        y_val=pd.read_csv(self.yval_path).values.ravel()

        models = {
                #"Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "AdaBoost Classifier": AdaBoostClassifier(),
                #"XGB Classifier": XGBClassifier(),
                #"CatBoosting Classifier": CatBoostClassifier(verbose=False), 
                "Ridge Classifier" : RidgeClassifier(),
                "SGD Classifier" : SGDClassifier(),
                #'LGBM Classifier()' : LGBMClassifier(verbose = -1)
            }

        params = {
                # "Decision Tree": {'criterion':['entropy', 'log_loss'], 'splitter':['best','random'], 'max_features':['sqrt','log2']},
                "Random Forest":{ 'n_estimators': [8,128,256], 
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                },
                "Gradient Boosting":{ 'learning_rate':[.1,.05,.001],'subsample':[0.6,0.8,0.9], 'n_estimators': [8,64,256],
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                },
                "Logistic Regression":{'C': [1, 10, 50] },
                "AdaBoost Classifier":{'learning_rate':[.1,.01,0.5,], 'n_estimators': [8,16,32,64,128,256], 
                                      # 'loss':['linear','square','exponential'],
                                      },
                #"XGBClassifier":{'learning_rate':[.1,.01,.05,.001], 'n_estimators': [8,16,32,64,128,256] },
                #"CatBoosting Classifier":{'depth': [6,8,10], 'learning_rate': [0.01, 0.05, 0.1], 'iterations': [30, 50, 100] },
                'Ridge Classifier' : {},
                'SGD Classifier' : {},
        }
    
        results = []
        for model, param in zip(models.items(), params.items()) :
            result = model_training(model, param, X_train, y_train,X_test,y_test)
            results.append(result)
            
        rapport = pd.DataFrame(results).sort_values('f1_test', ascending= False).reset_index(drop = True)
        best_model = rapport.loc[0]['model']
        save_process(process_path=self.best_model_path, process=best_model)
        return(best_model)
