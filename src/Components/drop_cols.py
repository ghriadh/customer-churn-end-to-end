import os
import pandas as pd
#from src.Components.aquire_data import DataIngestion

    
class DropCols:
    def __init__(self):
        self.data_path: str=os.path.join('artifacts',"data.csv")
    
    def drop_col(self, col):
     
            df=pd.read_csv(self.data_path)
            df = df.drop(col, axis = 1)
            os.makedirs(os.path.dirname(self.data_path),exist_ok=True)

            df.to_csv(self.data_path,index=False,header=True)
            
            print(df.shape)
            return(self.data_path )
   
        
# if __name__=="__main__":
#     aquire_data = DataIngestion()
#     data_path = aquire_data.get_data()
    
#     obj=DropCols()
#     data_path=obj.drop_col('customerID')
   

    
