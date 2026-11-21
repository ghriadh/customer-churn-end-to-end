import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class ScaleData:
    def __init__(self, y = None):
        self.data_path: str=os.path.join('artifacts',"data.csv")
    
    def scale_data(self, scaler):
            
            df=pd.read_csv(self.data_path)
            target = 'Churn'        
            y = df[target]
            df_array = df.values
            df = pd.DataFrame(scaler.fit_transform(df_array), columns = df.columns)
            df[target] = y

            os.makedirs(os.path.dirname(self.data_path),exist_ok=True)
            df.to_csv(self.data_path,index=False,header=True)
            print(df.shape)
            return(self.data_path)