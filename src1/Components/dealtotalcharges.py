import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
    
@dataclass
class DealTotalCharges:
    
    def dealtotalcharges(df):
            
            df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
           
            return(df)
   