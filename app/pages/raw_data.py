import streamlit as st
import os
from churn.config import APP_CSS_PATH, DATA_PATH
import io

import pandas as pd
import numpy as np

if 'df' not in st.session_state:
    st.session_state['df'] =pd.DataFrame()

st.set_page_config(
    page_title="Customer Churn Raw data",
    page_icon="👋",
)

def load_css(file_name = APP_CSS_PATH):
    with open(file_name) as f:
        css = f'<style>{f.read()}</style>'
    return css
css = load_css()
st.markdown(css, unsafe_allow_html=True)

def main():

    st.title('Customer Churn : Data')
    st.subheader('Auteur : Ghanmi riadh')
    # fonction d'importation des donnees
    @st.cache_data(persist = True)
    def load_data():
        data = pd.read_csv(DATA_PATH)
        return data
        
    # Affichage de la table des donnees 
    
    # global df 
    df = load_data()
    df_sample = df.sample(10)
    if st.sidebar.checkbox('Afficher les donnees brutes', False) :
        st.subheader("Jeu de donnees 'Customer Churn ' : echantillon de 10 observations")

    with st.expander('RAW DATA'):
        df_sample = df.sample(10)
        st.write(df_sample)
        st.write(f'dataframe shape : {df.shape}')
        st.write(f'We have 7043 costumers in our dataset')
    
    with st.expander("Descripe DATA"):    
        st.write('<p style="font-family:sans-serif; color:blue; font-size: 24px; font-weight:bold">Columns : </p>', unsafe_allow_html=True)
        for col in df.columns.values :
            st.write(f':red[{col}]')
        st.write('We have 21 feautures in dataset includes customerID and the target **Churn**')
        st.write('The target variable is the churn variable which takes two values **Yes** for customers who are churned and **No** for customers who are not churned')
        st.write(':blue[Numeric features : ]')
        st.write(df.describe().round(2).T)
        st.write(':blue[Categorical features : ]')
        st.write(df.describe(include=['O']).T)
    
    with st.expander("INFO DATA"):   
            buffer = io.StringIO()
            df.info(buf=buffer)
            info = buffer.getvalue()
        #with st.popover("DATA Info"):
            st.text(info)
            st.markdown('''
                - **We dont have blank, null or empty values**
                - **Three features are integers or floats : SeniorCitizen, tenure, MonthlyCharges  -- 18 features are objects**
                - **customerID is like index, so it is not usefull, we drop it**
                - **16 features are categoricals**
                - **TotalCharges should be float but it is an object. we deal this column**</b>
                ''', unsafe_allow_html=True
                )
    
    with st.popover("Drop custumerID"):
            df.drop('customerID', axis = 1, inplace=True)
            st.code("df.drop('customerID', axis = 1, inplace=True)", language = 'python')
            st.write("custumerID dropped")
    
    with st.popover("Convert TotalCharges to float"):
            df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
            st.markdown('''
                - **TatalCharges values are string**
                - **Some values in TotalCharges Column are ' '**
                - **Replace there values by nan and convert TotalCharges values to float**
                ''', unsafe_allow_html=True
                )
            st.code('''
                    df['TotalCharges'] = df['TotalCharges'].replace(' ' , np.nan)
                    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
                    ''', language = 'python')
        
            st.write("TotalCharges converted to float")

    with st.expander('Duplicated Rows'):
        
        st.markdown('''
            - **we have 22 duplicated rows**
            - **for example (row 22 and row 1338) and (row 1731 and row 3301)** 
            - **we drop duplicated rows**       
            ''')
        st.write(df.iloc[[22, 1338]])
        st.code('''
                df.duplicated().sum()
                df.iloc[[22, 1338]]
                df = df[df.duplicated()== False] 
                ''', language = 'python')
        df = df[df.duplicated()== False]
        st.write('Duplicated rows was dropped')
        st.write('dataframe shape : ', df.shape)

    st.session_state['df'] = df

if __name__ == '__main__' : main()