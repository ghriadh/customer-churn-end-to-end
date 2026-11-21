# app web flask
from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from src3.Components.prediction import Prediction


application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def prediction_data():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data_path : str=os.path.join('artifacts2/data',"pred_data.csv")
        data=Prediction(
           
            gender = request.form.get('gender'),
            SeniorCitizen = int(request.form.get('SeniorCitizen')),
            Partner = request.form.get('Partner'),
            Dependents = request.form.get('Dependents'),
            tenure = int(request.form.get('tenure')),
            PhoneService = request.form.get('PhoneService'),
            MultipleLines = request.form.get('MultipleLines'),

            InternetService = request.form.get('InternetService'),
            OnlineSecurity = request.form.get('OnlineSecurity'),
            OnlineBackup = request.form.get('OnlineBackup'),
            DeviceProtection = request.form.get('DeviceProtection'),
            TechSupport = request.form.get('TechSupport'),
            StreamingTV = request.form.get('StreamingTV'),

            StreamingMovies = request.form.get('StreamingTV'),
            Contract = request.form.get('Contract'),
            PaperlessBilling = request.form.get('PaperlessBilling'),
            PaymentMethod = request.form.get('PaymentMethod'),
            MonthlyCharges = float(request.form.get('MonthlyCharges')),
            TotalCharges = float(request.form.get('TotalCharges')),

        )
        pred_df = data.get_new_data()
        os.makedirs(os.path.dirname(data_path),exist_ok=True)
        pred_df.to_csv(data_path,index=False,header=True)
        results=data.predict()
        print(results[0])
        return render_template('home.html',results= str(results[0]))
    
if __name__=="__main__":
    app.run(host="0.0.0.0", port=int('5000'),debug = True)     


