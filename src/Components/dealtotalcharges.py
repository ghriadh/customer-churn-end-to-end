import os
import numpy as np
import pandas as pd

    
class DealTotalCharges:
    def __init__(self):
        self.data_path: str=os.path.join('artifacts',"data.csv")
    
    def dealtotalcharges(self):
     
            df=pd.read_csv(self.data_path)
            df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
            
            os.makedirs(os.path.dirname(self.data_path),exist_ok=True)
            df.to_csv(self.data_path,index=False,header=True)

            print(df.shape)
            return(self.data_path)
   