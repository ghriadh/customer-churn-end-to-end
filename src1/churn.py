import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from src1.Components.aquire_data import DataIngestion
from src1.Components.deal_data import DealData
from src1.Components.training import Training
        
if __name__=="__main__":
    obj=DataIngestion()
    data_path=obj.get_data()
  
    obj = DealData('customerID', 'No', 'log', StandardScaler(), 0.6)
    data_path = obj.deal_data()
    
    obj = Training()
    data_path = obj.train()


    

