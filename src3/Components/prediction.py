import sys
import os
import pandas as pd
from src3.exception import CustomException
from src3.utils import load_process

class Prediction:
    
    def __init__( self,
        gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines,
        InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, 
        StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges):

        self.gender = gender
        self.SeniorCitizen = SeniorCitizen
        self.Partner = Partner
        self.Dependents = Dependents
        self.tenure = tenure
        self.PhoneService = PhoneService
        self.MultipleLines = MultipleLines

        self.InternetService = InternetService
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.StreamingTV = StreamingTV

        self.StreamingMovies = StreamingMovies
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod = PaymentMethod
        self.MonthlyCharges = MonthlyCharges
        self.TotalCharges = TotalCharges
        
    def get_new_data(self):
        try:
            new_data = {
                'gender': [self.gender],
                'SeniorCitizen' : [self.SeniorCitizen],
                'Partner' : [self.Partner],
                'Dependents' : [self.Dependents],
                'tenure' : [self.tenure],
                'PhoneService' : [self.PhoneService],
                'MultipleLines' : [self.MultipleLines],

                'InternetService' : [self.InternetService],
                'OnlineSecurity' : [self.OnlineSecurity],
                'OnlineBackup' : [self.OnlineBackup],
                'DeviceProtection' : [self.DeviceProtection],
                'TechSupport' : [self.TechSupport],
                'StreamingTV' : [self.StreamingTV],

                'StreamingMovies' : [self.StreamingMovies],
                'Contract' : [self.Contract],
                'PaperlessBilling' : [self.PaperlessBilling],
                'PaymentMethod' : [self.PaymentMethod],
                'MonthlyCharges' : [self.MonthlyCharges], 
                'TotalCharges' : [self.TotalCharges]
            }
            
            return pd.DataFrame(new_data)

        except Exception as e:
             raise CustomException(e, sys)

    def predict(self):
        #try:
            
            model_path=os.path.join("artifacts2/process","best_model.pkl")
            transform_process_path=os.path.join('artifacts2/process','transform_process.pkl')
            print("Before Loading")
            
            model= load_process(process_path=model_path)
            transform_process= load_process(process_path=transform_process_path)
            print("After Loading")
            
            new_data = self.get_new_data()
            transformed_new_data = transform_process.transform(new_data)
            
            preds=model.predict(transformed_new_data)
            return preds
        
        # except Exception as e:
        #      raise CustomException(e,sys)