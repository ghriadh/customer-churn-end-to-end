import streamlit as st
import io, os
import pandas as pd
import numpy as np
from sklearn.utils import resample
from utils_streamlit import transformation
from src3.Components.data_transform import TransformDataPipe
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

st.set_page_config(
    page_title="Customer Churn Explore data",
    page_icon="👋",
)

def load_css(file_name = "src3/style.css"):
    with open(file_name) as f:
        css = f'<style>{f.read()}</style>'
    return css
css = load_css()
st.markdown(css, unsafe_allow_html=True)
seed = 1234

def main():
    # df = st.session_state['df']
    # num_col = st.session_state['num_col']
    # target = st.session_state['target']
  
    # with st.expander(' Transform numeric feature'):
    #     st.markdown('''
    #                 - **we can use many transformation (sqrt, log, boxcox, boxcox1p, ...)**
    #                 - **In our dataset, numeric features have positifs values, so we can use all transformation**
    #                 ''')
        
    #     transformer = np.log
    #     st.code( '''
    #         transformation(df, transformer)
    #         df.head(3)
    #             ''', language = 'python')

    #     transformation(df, num_col, transformer)
    #     st.write (df.head(3))
    #     st.write('numeric features was transformed')
        
    # with st.expander('Encodage') : 
    #     st.write('We convert target')
    #     df[target]= df[target].astype('category').cat.codes
    #     st.write(df.head(2))

    #     st.write('We convert others categorical features')
    #     st.write(df.head(2))

    with st.expander(' Transform numeric feature'):
        obj = TransformDataPipe(data_path='artifacts2/data/data_injected.csv', transformer= 'log', scaler= StandardScaler())
        df = obj.transform_data_pipeline()
        st.write(df)



if __name__ == '__main__' : main()