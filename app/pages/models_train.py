import streamlit as st
from churn.config import ARTIFACTS_DIR
import os
import os as OS
from os import listdir
from os.path import isfile, join
import pandas as pd
from churn.split_data import SplitData
from churn.training import Training
# from churn.prediction import Prediction



def main():
    with st.expander('split data'):
        data_path=os.path.join(ARTIFACTS_DIR, 'data', 'data_injected.csv')
        target = st.session_state['target']
        rate = 0.6

        df = pd.read_csv(data_path)
        instance = SplitData(target, rate)
        data_dirname = instance.transform(df)
        st.write('we have split data in train and test and validation')
    
    with st.expander('training'):
        st.write('we try models, we conserve the best')
        instance = Training(data_dirname)
        best = instance.train()
        st.write('ok')
    
    with st.expander('confusion matrix'):
        mypath = os.path.join(ARTIFACTS_DIR, 'confusion_matrix')
        confusion_matrix = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        st.write(confusion_matrix)
        with st.container():
              i = 0
              for img in confusion_matrix:
                 st.image(os.path.join(ARTIFACTS_DIR, 'confusion_matrix', img), width=600, caption= img )
                 i = i + 1
    
    with st.expander('best model'):
        st.write('The best model is : ')

if __name__ == '__main__' : main()