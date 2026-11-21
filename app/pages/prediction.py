import streamlit as st
import io
import os as OS
import pandas as pd
import numpy as np
from churn.prediction import Prediction

def main():
   
    st.title('Customer Churn Explore Data')
    st.subheader('Auteur : Ghanmi riadh')
    
    gender =  st.radio('Gender : ',  ['Male', 'Female'], horizontal= True, index = None )
    SeniorCitizen = int(st.checkbox('you are senior Citizen'))

    Partner = st.toggle('Partner', ['Yes', 'No'])
    if Partner : Partner = 'Yes' 
    else :  Partner = 'No'

    Dependents = st.radio('Dependents', ['Yes', 'No'], horizontal= True, index = None)
    
    tenure = st.slider('tenure', 0, 100, 0, 1)

    PhoneService = st.radio('PhoneService', ['Yes', 'No'], horizontal= True, index = None)

    MultipleLines = st.radio('MultipleLines', ['Yes', 'No', 'No internet service' ], horizontal= True, index = None)

    InternetService = st.radio('InternetService', ['DSL', 'No', 'Fiber optic'], horizontal= True, index = None)

    OnlineSecurity = st.radio('OnlineSecurity', ['Yes', 'No', 'No internet service'], horizontal= True, index = None)

    OnlineBackup = st.radio('OnlineBackup', ['Yes', 'No', 'No internet service'], horizontal= True, index = None)

    DeviceProtection = st.radio('DeviceProtection', ['Yes', 'No', 'No internet service'], horizontal= True, index = None)

    TechSupport = st.radio('TechSupport', ['Yes', 'No', 'No internet service'], horizontal= True, index = None)

    StreamingTV = st.radio('StreamingTV', ['Yes', 'No', 'No internet service'], horizontal= True, index = None)

    StreamingMovies = st.radio('StreamingMovies', ['Yes', 'No', 'No internet service'], horizontal= True, index = None)

    Contract = st.radio('Contract', ['Month-to-month', 'Two year', 'One year'], horizontal= True, index = None)

    PaperlessBilling = st.radio('PaperlessBilling', ['Yes', 'No'], horizontal= True, index = None)

    PaymentMethod = st.radio('PaymentMethod', ['Mailed check', 'Electronic check', 'Bank transfer (automatic)', 'Credit card (automatic)'], horizontal= True, index = None)

    MonthlyCharges = st.number_input('Monthly charges', 0, 1000, 0, 1)

    TotalCharges = st.number_input('TotalCharges', 0, 10000, 0, 1)

    predict = st.button('Predict')

    if predict :
        instance = Prediction(
        gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,  MultipleLines, InternetService, 
        OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, 
        Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
        )
        new_data = instance.get_new_data()
        preds = instance.predict()
        st.write(preds[0])
    
    
    txt = st.text_area('','yes')
    st.text_area('label', 'value', label_visibility='visible')
    
    
if __name__ == '__main__' : main()