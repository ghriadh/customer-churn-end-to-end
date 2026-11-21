import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# from src2.Components.aquire_data import DataIngestion
from src2.Components.deal_data_pipe import DealDataPipe
from src2.Components.transform_data_pipe import TransformDataPipe
from src2.Components.training import Training
        
if __name__=="__main__":
    # obj=DataIngestion()
    # data_path=obj.get_data()

    df=pd.read_csv('src2\customer_churn.csv')
  
    #obj = DealDataPipe('customerID', 'del','No', 'log', StandardScaler(), 0.6)
    obj = DealDataPipe('customerID', 'No')
    df = obj.deal_pipe(df)
    obj = TransformDataPipe( 'del', 'log', StandardScaler())
    df = obj.deal_pipe(df)
    # print(data_path)
    print(df)

    # obj = Training(data_path)
    # data_path = obj.train()


    

