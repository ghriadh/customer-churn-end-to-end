import os
import pandas as pd

from src1.Components.drop_cols import DropCols
from src1.Components.aquire_data import DataIngestion
from src1.Components.drop_duplicate import DropDuplicate 
from src1.Components.dealtotalcharges import DealTotalCharges 
from src1.Components.imputer import Imputer 
from src1.Components.transformation import Transformation
from src1.Components.encoder import EncoderData
from src1.Components.scale_data import ScaleData
from src1.Components.split_data import SplitData
from src1.Components.class_imballance import ClassImballance
    
class DealData:
    def __init__(self, col, strategy, transformer, scaler, rate):
        self.data_path: str=os.path.join('artifacts1',"data.csv")
        self.col = col 
        self.strategy = strategy
        self.transformer = transformer
        self.scaler = scaler
        self.rate = rate
    
    def deal_data(self, u = None, t = None, x= None, y = None, z = None):
     
            df=pd.read_csv(self.data_path)
            df = DropCols.drop_col(df, self.col)
            df = DealTotalCharges.dealtotalcharges(df)
            df = DropDuplicate.dropduplicate(df)
            df = Imputer.del_imputer(df)
            df = ClassImballance.class_imballance(df, self.strategy)
            df = Transformation.transform_data(df, self.transformer)
            df = EncoderData.encoder_cat(df)
            df = ScaleData.scale_data(df, self.scaler)
            obj = SplitData()
            folder = obj.split_data(df, self.rate)            
            
            return(folder)