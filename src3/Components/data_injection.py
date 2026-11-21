import os
import pandas as pd
from sklearn.pipeline import Pipeline
from src3.Components.drop_cols import DropCols
from src3.Components.dealtotalcharges import DealTotalCharges
from src3.Components.drop_duplicate import DropDuplicate 
from src3.Components.imputer import Imputer 
from src3.Components.class_imballance import ClassImballance
from utils import save_process

class DataIngestion:
    def __init__(self, initial_path, col, method, strategy):
        self.data_path: str=os.path.join('artifacts2/data',"data.csv")
        self.process_path: str=os.path.join('artifacts2/process',"deal_data_process.pkl")
        self.initial_path = initial_path
        self.col = col
        self.method = method
        self.strategy = strategy
    
    def get_data(self):
            
            df=pd.read_csv(self.initial_path)

            return df
    
    def build_deal_data_pipeline(self):
            
            my_pipeline = Pipeline(steps = [ 
                  ('drop_customerID', DropCols(self.col)), 
                  ('dealtotalcharges', DealTotalCharges()),
                  ('dropduplicate', DropDuplicate()),
                  ('imputer', Imputer(self.method)),
                  ('class imballance', ClassImballance(self.strategy)),
                  ])

            return(my_pipeline)
            
    def deal_data_pipeline(self): 
            df = self.get_data()
            target = 'Churn'
            
            deal_data_process_pipeline = self.build_deal_data_pipeline()
            df = deal_data_process_pipeline.transform(df)
            df[target]= df[target].astype('category').cat.codes    
            
            os.makedirs(os.path.dirname(self.data_path),exist_ok=True)
            df.to_csv(self.data_path,index=False,header=True)
            save_process(process_path=self.process_path, process=deal_data_process_pipeline)
            return(self.data_path)
   