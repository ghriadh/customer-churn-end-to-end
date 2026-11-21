import os
import pandas as pd
from dataclasses import dataclass

@dataclass
class DropDuplicate:
    
    def dropduplicate(df):
     
            df = df[df.duplicated()== False]
            
            return(df)
   

