import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from src2.Components.drop_cols import DropCols
from src2.Components.drop_duplicate import DropDuplicate 
from src2.Components.dealtotalcharges import DealTotalCharges 
from src2.Components.imputer import Imputer 
from src2.Components.transformation import Transformation
from src2.Components.encoder import EncoderData
from src2.Components.scale_data import ScaleData
from src2.Components.split_data import SplitData
from src2.Components.class_imballance import ClassImballance
    
class DealDataPipe:
   # def __init__(self, col, method, strategy, transformer, scaler, rate):
    def __init__(self, col,  strategy):
        self.data_path: str=os.path.join('artifacts1',"data.csv")
        self.col = col 
        #self.method = method
        self.strategy = strategy
        #self.transformer = transformer
        #self.scaler = scaler
        #self.rate = rate
    
    def build_pipe(self):
            
            my_pipeline = Pipeline(steps = [ 
                  ('drop_customerID', DropCols(self.col)), 
                  ('dealtotalcharges', DealTotalCharges()),
                  ('dropduplicate', DropDuplicate()),
                  #('imputer', Imputer(self.method)),
                  ('class imballance', ClassImballance(self.strategy)),
                  #('transformation', Transformation(self.transformer)),
                  #('encoder', EncoderData()),
                  #('scaledata', ScaleData(self.scaler)),
                  #('splitdata', SplitData(self.rate))
                  ])

            return(my_pipeline)
            
    def deal_pipe(self, df): 
            my_pipeline = self.build_pipe()
            df = my_pipeline.transform(df)

            return(df)


if __name__=="__main__":
    df=pd.read_csv('src2\customer_churn.csv')
    # obj = DealDataPipe('customerID', 'del', 'No', 'log', StandardScaler(), 0.6)
    obj = DealDataPipe('customerID',  'up')
    folder = obj.deal_pipe(df)
    print(folder)