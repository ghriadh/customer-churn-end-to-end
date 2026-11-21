import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from src.Components.drop_cols import DropCols
from src.Components.aquire_data import DataIngestion
from src.Components.drop_duplicate import DropDuplicate 
from src.Components.dealtotalcharges import DealTotalCharges 
from src.Components.imputer import Imputer  
from src.Components.transformation import Transformation
from src.Components.description import Description
from src.Components.encoder import EncoderData
from src.Components.class_imballance import ClassImballance
from src.Components.scale_data import ScaleData
from src.Components.split_data import SplitData
from src.Components.training import Training
        
if __name__=="__main__":
    obj=DataIngestion()
    data_path=obj.get_data()
    
    obj = DropCols()
    data_path = obj.drop_col('customerID')
    
    obj = DropDuplicate()
    data_path = obj.dropduplicate()

    obj = DealTotalCharges()
    data_path = obj.dealtotalcharges()

    obj = Imputer()
    data_path = obj.del_imputer()

    # obj = Description()
    # obj.describe_target()
    # obj.describe_cat()


    obj = Transformation()
    data_path = obj.transform_data('log')

    obj = EncoderData()
    data_path = obj.encoder_cat()

    obj = ClassImballance()
    data_path = obj.class_imballance('up')


    obj = ScaleData()
    data_path = obj.scale_data(StandardScaler())

    obj = SplitData()
    data_path = obj.split_data(0.6)

    obj = Training()
    data_path = obj.train()


    print(data_path)

