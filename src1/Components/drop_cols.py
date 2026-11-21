import os
import pandas as pd
from dataclasses import dataclass

@dataclass   
class DropCols:
    
    def drop_col(df, col):
     
            df = df.drop(col, axis = 1)
            
            return(df)
   