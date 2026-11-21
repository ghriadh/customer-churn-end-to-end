import os
import pandas as pd
# from src.Components.drop_cols import DropCols

    
class DataIngestion:
    def __init__(self):
        self.data_path: str=os.path.join('artifacts',"data.csv")
    
    def get_data(self):
     
            df=pd.read_csv('src\customer_churn.csv')
            os.makedirs(os.path.dirname(self.data_path),exist_ok=True)
            df.to_csv(self.data_path,index=False,header=True)
            print(df.shape)
            return(self.data_path)
   
        
# if __name__=="__main__":
#     obj=DataIngestion()
#     data_path=obj.get_data()
#     obj = DropCols()
#     data_path = obj.drop_col('customerID')
#     print(data_path)




    


