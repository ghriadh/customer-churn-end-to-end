import os
import pandas as pd

class DropDuplicate:
    def __init__(self):
        self.data_path: str=os.path.join('artifacts',"data.csv")
    
    def dropduplicate(self):
     
            df=pd.read_csv(self.data_path)
            df = df[df.duplicated()== False]
            os.makedirs(os.path.dirname(self.data_path),exist_ok=True)
            df.to_csv(self.data_path,index=False,header=True)
            print(df.shape)
            return(self.data_path)
   

