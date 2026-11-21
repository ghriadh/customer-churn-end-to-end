import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from src3.Components.data_injection import DataIngestion
from src3.Components.data_transform import TransformDataPipe
from src3.Components.split_data import SplitData
from src3.Components.training import Training
from src3.Components.prediction import Prediction
        
if __name__=="__main__":
    
    src_path = 'src3\customer_churn.csv'
    id = 'customerID'
    class_imbalance_deal = 'No'
    missing_value_deal = 'del'
    transformer = 'log'
    scaler = StandardScaler()
    target = 'Churn'
    rate = 0.6

    instance=DataIngestion(src_path,id, missing_value_deal,class_imbalance_deal )
    data_path=instance.deal_data_pipeline()

    instance = TransformDataPipe(data_path, transformer, scaler)
    data_path = instance.transform_data_pipeline()

    df = pd.read_csv(data_path)
    instance = SplitData(target, rate)
    data_dirname = instance.transform(df)

    instance = Training(data_dirname)
    instance.train()

    gender = 'Female'
    SeniorCitizen = 1
    Partner = 'No'
    Dependents = 'No'
    tenure = 4
    PhoneService = 'Yes'
    MultipleLines = 'No'

    InternetService = 'No'
    OnlineSecurity = 'No'
    OnlineBackup = 'No'
    DeviceProtection = 'No'
    TechSupport = 'No'
    StreamingTV = 'No'

    StreamingMovies = 'No'
    Contract = 'Month-to-month'
    PaperlessBilling = 'No'
    PaymentMethod = 'Credit card (automatic)'
    MonthlyCharges = 20
    TotalCharges = 60
    
    
    instance = Prediction(
        gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,  MultipleLines, InternetService, 
        OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, 
        Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
    )
    new_data = instance.get_new_data()
    preds = instance.predict()
    print(preds[0])