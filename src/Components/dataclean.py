import os
import pandas as pd
from src.Components.drop_cols import DropCols
from src.Components.aquire_data import DataIngestion
from src.Components.drop_duplicate import DropDuplicate 
from src.Components.dealtotalcharges import DealTotalCharges 
from src.Components.imputer import Imputer  
from src.Components.transformation import Transformation
from src.Components.description import Description


@dataclass
class DataIngestionConfig(col):
    # os.path.join('artifacts',"train.csv") = 'artifacts\\train.csv'
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

    df=pd.read_csv(self.data_path)
    df = df.drop(col, axis = 1)
    os.makedirs(os.path.dirname(self.data_path),exist_ok=True)

    df.to_csv(self.data_path,index=False,header=True)
            
    print(df.shape)
    return(self.data_path )

class DataClean:
    def __init__(self):
        self.data_path: str=os.path.join('artifacts',"data.csv")
    
    def data_clean(self): 
            df=pd.read_csv(self.data_path)
            df.dropna(inplace = True)
            df.reset_index(drop=True, inplace = True)
            os.makedirs(os.path.dirname(self.data_path),exist_ok=True)
            df.to_csv(self.data_path,index=False,header=True)
            print(df.shape)
            return(self.data_path)
    
    def median_imputer(self): 
            df=pd.read_csv(self.data_path)
            feature1 = 'Contract'
            feature2 = 'OnlineSecurity'
            df = fill_by_median (df, feature1, feature2)
            df.reset_index(drop=True, inplace = True)
            os.makedirs(os.path.dirname(self.data_path),exist_ok=True)
            df.to_csv(self.data_path,index=False,header=True)
            print(df.shape)
            return(self.data_path)

        
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

    print(data_path)
