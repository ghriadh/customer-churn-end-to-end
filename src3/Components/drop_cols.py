import os
import pandas as pd

class DropCols():
    def __init__(self, col):
        self.col = col


    def transform(self, df):
        df = df.drop(self.col, axis=1)
        return df

   