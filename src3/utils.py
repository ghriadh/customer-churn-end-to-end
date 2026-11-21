import os
import sys

# import numpy as np 
# import pandas as pd
import dill
import pickle


#from src.exception import CustomException

def save_process(process_path, process):
    #try:
        dir_path = os.path.dirname(process_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(process_path, "wb") as file_obj:
            pickle.dump(process, file_obj)

    #except Exception as e:
        #raise CustomException(e, sys)

def load_process(process_path):
    # try:
        with open(process_path, "rb") as process:
            return pickle.load(process)

    # except Exception as e:
    #     raise CustomException(e, sys)    