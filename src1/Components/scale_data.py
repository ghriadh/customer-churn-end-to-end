import os
import pandas as pd
from dataclasses import dataclass

@dataclass
class ScaleData:
    
    def scale_data(df, scaler):
            
            target = 'Churn'        
            y = df[target]
            df_array = df.values
            df = pd.DataFrame(scaler.fit_transform(df_array), columns = df.columns)
            df[target] = y

            return(df)