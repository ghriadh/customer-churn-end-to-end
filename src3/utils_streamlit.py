import streamlit as st
import io

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, Back, Style
from ipywidgets import interact, widgets

from scipy import stats, special
from scipy.stats import norm, skew, shapiro, kstest, pearsonr, spearmanr, kendalltau, chi2_contingency, levene

#from ydata_profiling import ProfileReport
#from streamlit_pandas_profiling import st_profile_report

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
#from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier
#from catboost import CatBoostClassifier
#from mlxtend.classifier import StackingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score, mean_absolute_error, ConfusionMatrixDisplay, classification_report, precision_score, recall_score

from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline

# Functions definition : 
def pie_chart(df,feature) :
    total = df[feature].value_counts().values.sum()
    explode = [0.05]*len(df[feature].unique())
    def func(pct):
        return f"{int(np.round(pct/100*total)):d}"
    fig= plt.figure(figsize=(12, 8))
    piechart = plt.pie(df[feature].value_counts().values, labels = df[feature].value_counts().index, autopct = func, startangle = 90,
               explode = explode, shadow = True, textprops={'size': 12, 'color' : 'black', 'weight': 'bold'}, radius= 1)
    plt.title(label = f"{feature} Customers Distribution", fontsize = 14, color = 'white', loc = 'center', backgroundcolor='black', fontweight='bold')
    st.pyplot(fig)

# Distribution
def bar_plot_distribution(df, target, feature) :
    fig = plt.figure(figsize = (10,6))
    bars = sns.countplot(data = df, x=feature, hue = target)
    
    val0 = bars.containers[0].datavalues; val0[np.isnan(val0)] = 0
    val1 = bars.containers[1].datavalues; val1[np.isnan(val1)] = 0
    
    for i, v in enumerate(val0):
        plt.text(i - 0.2  , v/2, str(int(v)), color='black', ha="center", fontweight = 'bold', fontsize = 12)
    for i, v in enumerate(val1):
        plt.text(i + 0.2  , v/2, str(int(v)), color='black', ha="center", fontweight = 'bold', fontsize = 12)
    
    plt.legend(loc='upper right', title=target, title_fontsize=12, facecolor='white', edgecolor ='black', borderpad=1, labelcolor = 'blue')
    plt.title(label =f'Churned and UnChurned Costumers by {feature}', fontsize = 14, color = 'white', backgroundcolor='black', fontweight='bold')
    st.pyplot(fig)

# Relation
def relation(df, target, feature) :
    cross_tab = pd.crosstab(df[feature], df[target])
    chi2_stat, p_value, _, _ = chi2_contingency(cross_tab)
    n = df.shape[0]
    s = cross_tab.sum().sum()
    min_dim = min(cross_tab.shape) - 1
    cramer_v = np.sqrt(chi2_stat / (s * min_dim)).round(2)
    tschuprow_t = np.sqrt(chi2_stat / (n * min_dim)).round(2)
    st.write(f"The relation between {target} and {feature} : " + '\n' )  
    st.write(f'Chie2 Test : p_value = {p_value}' + '\n' )
    st.write(f'Coefficient de Cramér-V = {cramer_v}'+ '\n' )
    st.write(f'Coefficient T de Tschuprow = {tschuprow_t}'+ '\n' )
# Correlation
def correlation(df, target, feature, test) :
    df1 = df[[target,feature]].dropna()
    df1[target]= df1[target].astype('category').cat.codes
    if test == 'pearsonr' : v, p = pearsonr(df1[target], df1[feature])
    if test == 'spearmanr' : v, p = spearmanr(df1[target], df1[feature])
    if test == 'kendalltau' : v, p = kendalltau(df1[target], df1[feature])
    st.text( f'correlation coefficient: {v}')
    st.text( f'P-value of correlation test: {p}' )
def transformation (df, num_col, transformer) :
     if transformer == stats.boxcox :
         for feature in num_col:
             df[feature], _ = transformer(df[feature])
     if transformer == (np.log or np.sqrt) :
         for feature in num_col:
             df[feature] = transformer(df[feature])
     if transformer == special.boxcox1p :
         for feature in num_col:
             df[feature] = special.boxcox1p(df[feature] + 1, stats.boxcox_normmax(df[feature] + 1))