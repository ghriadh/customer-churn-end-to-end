import streamlit as st
import io
import os as OS
import pandas as pd
import numpy as np
from sklearn.utils import resample

st.set_page_config(
    page_title="Customer Churn Explore data",
    page_icon="👋",
)

def load_css(file_name = 'src3/style.css'):
    with open(file_name) as f:
        css = f'<style>{f.read()}</style>'
    return css
css = load_css()
st.markdown(css, unsafe_allow_html=True)

seed = 1234

def main():

    st.title('Customer Churn Explore Data')
    df = st.session_state['df']
    target = st.session_state['target']

    with st.expander('Missing values'):
        df_del = df.copy()
        df_mean = df.copy()

        strategy = st.radio('Choose your strategy deal of missing values', ['Delete', 'mean'], index = None)
        if strategy == 'Delete' :
            
            df_del.dropna(inplace = True)
            st.write('we delete all row having missing value')
            df = df_del
        if strategy == 'mean' :  
            contract = df_mean['Contract'].unique()
            security = df_mean['OnlineSecurity'].unique()
            mean_totalcharges = np.zeros((len(contract),len(security)))

            for ctr in contract :
                i = np.where(contract == ctr)[0][0]
                for os in security:
                    j = np.where(security == os)[0][0]
                    mean_df = df_mean[(df_mean['Contract'] == ctr) & (df_mean['OnlineSecurity'] == os)]['TotalCharges'].dropna().median() 
                    mean_totalcharges[i, j]=mean_df

            for i in range(0, len(contract)):
                ctr = contract[i]
                for j in range(0, len(security)):
                    os = security[j]
                    if not(df_mean.loc[df_mean['TotalCharges'].isnull() & (df_mean['Contract']==ctr) & (df_mean['OnlineSecurity'] == os)].empty):
                        df_mean.loc[df_mean['TotalCharges'].isnull() & (df_mean['Contract']==ctr) & (df_mean['OnlineSecurity'] == os), 'TotalCharges'] = mean_totalcharges[i,j]
            st.write('we change all missing value by the mean of TotalCharges grouped by contract and security')
            df = df_mean
    
    with st.expander('Convert target'):
            df[target]= df[target].astype('category').cat.codes
            st.write('target was converted')

    with st.expander('Class Imbalance'):
            method = st.radio('Choose your method to deal class imbalance', ['Up', 'Down', 'No'], index = None)
            minority = df.loc[df.Churn == 'Yes']
            majority = df[df.Churn == 'No']
            if method == 'Up' :
                minority = resample(minority, replace = True, n_samples = len(majority), random_state = seed)
                df = pd.concat([majority, minority])
                df.reset_index(drop=True, inplace = True)
                st.write(df.shape)
                st.session_state['df'] = df
            if method == 'Down' : 
                majority = resample(majority, replace = False, n_samples = len(minority), random_state = seed)
                df = pd.concat([majority, minority])
                df.reset_index(drop=True, inplace = True)
                st.write(df.shape)
                st.session_state['df'] = df

            if method == 'No' : 
                st.write('No deal of class imbalance')
                st.session_state['df'] = df
    
    with st.expander('data'):       
        data_path = OS.path.join('artifacts2/data','data_injected.csv')
        OS.makedirs(OS.path.dirname(data_path),exist_ok=True)
        st.session_state['df'].to_csv(data_path,index=False,header=True)
        st.write(st.session_state['df'].shape)
        st.write(df.describe())
            

if __name__ == '__main__' : main()