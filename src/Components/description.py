import os
import numpy as np
import pandas as pd
from IPython import display
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from colorama import Fore, Back, Style
from ipywidgets import interact, widgets
from scipy import stats, special
from scipy.stats import norm, skew, shapiro, kstest, pearsonr, spearmanr, kendalltau, chi2_contingency, levene


target = 'Churn'

def pie_chart(df, feature) :
    total = df[feature].value_counts().values.sum()
    explode = [0.05]*len(df[feature].unique())
    def func(pct):
        return f"{int(np.round(pct/100*total)):d}"

    piechart = plt.pie(df[feature].value_counts().values, labels = df[feature].value_counts().index, autopct = func, startangle = 90,
               explode = explode, shadow = True, textprops={'size': 12, 'color' : 'black', 'weight': 'bold'}, radius= 1)
    plt.title(label = f"{feature} Customers Distribution", fontsize = 14, color = 'white', loc = 'center', backgroundcolor='black', fontweight='bold')
    plt.show()

# Distribution
def bar_plot_distribution(df, target, feature) :
    plt.figure(figsize = (10,6))
    bars = sns.countplot(data = df, x=feature, hue = target)
    
    val0 = bars.containers[0].datavalues; val0[np.isnan(val0)] = 0
    val1 = bars.containers[1].datavalues; val1[np.isnan(val1)] = 0
    
    for i, v in enumerate(val0):
        plt.text(i - 0.2  , v/2, str(int(v)), color='black', ha="center", fontweight = 'bold', fontsize = 12)
    for i, v in enumerate(val1):
        plt.text(i + 0.2  , v/2, str(int(v)), color='black', ha="center", fontweight = 'bold', fontsize = 12)
    
    plt.legend(loc='upper right', title=target, title_fontsize=12, facecolor='white', edgecolor ='black', borderpad=1, labelcolor = 'blue')
    plt.title(label =f'Churned and UnChurned Costumers by {feature}', fontsize = 14, color = 'white', backgroundcolor='black', fontweight='bold')

# Describe feature
def description (df, feature) :
    des = df[feature].describe().to_frame().T
    des['cardinality'] = len(df[feature].unique())
    print(des)
    print(Back.YELLOW +'\033[1m' +f'value of {feature} =  {df[feature].unique()}'+ '\033[0m' + '\n')
    print(df[[feature, target]].groupby([feature]).count().rename(columns={target: "Count"}))
    pie_chart(df, feature)

# Distribution categorical feature
def cat_distribution_bytarget (df, feature) :
    cross_tab = pd.crosstab(df[feature], df[target])
    display(cross_tab)
    bar_plot_distribution(target, feature)
    relation(target, feature)

#Distribution numeric feature  
def num_distributuion(df, feature, n) :
    display(df[feature].describe().to_frame().T)
    plt.figure(figsize = (12, 8))
    bins = (df[feature].max()- df[feature].min())/n
    binf = 0 ; bsup = (df[feature].max() + bins) ; biins = np.arange(binf, bsup, bins)
    feat = sns.displot(df[feature], color="blue", aspect = 2, height = 5, bins = biins)
    plt.title (f'►Customers Distribution by {feature}◄', fontsize=14, fontweight='bold')
    plt.xticks(np.arange(binf, bsup, bins))
    feats = feat.facet_axis(0,0).containers[0].datavalues
    for i, v in enumerate(feats):
        plt.text(i*bins + bins/2 , v + 1, str(v), color='black', ha="center", fontweight = 'bold') 

# Normality
def distributuion_normality(df, feature) :
    fig, ax = plt.subplots(2, 2, figsize=(18, 8), gridspec_kw=dict(width_ratios=[1,1], height_ratios= [4,1]))
    sns.distplot(df[feature], fit=norm, ax = ax[0][0])
    res = stats.probplot(df[feature], plot=ax[0][1])
    
    ax[1][0].text(0.1, 0.7, f'Skewness = {df[feature].skew().round(3)}', fontsize = 14, fontweight = 'bold' )
    ax[1][0].text(0.1, 0.4, f'kurtosis = {df[feature].kurt().round(3)}', fontsize = 14, fontweight = 'bold' )
    ax[1][0].text(0.1, 0.1, f'Test = {stats.normaltest(df[feature]).pvalue}', fontsize = 14, fontweight = 'bold' )
    ax[1][1].text(0.1, 0.2, f'KSTest = {stats.kstest(df[feature], "norm")[1]}', fontsize = 14, fontweight = 'bold')
    ax[1][1].text(0.1, 0.6, f'Shapiro = {shapiro(df[feature])[1]}', fontsize = 14, fontweight = 'bold')
    
    ax[1][0].set_xticks([]); ax[1][1].set_xticks([]); ax[1][0].set_yticks([]); ax[1][1].set_yticks([])

# Boxplot - Outliers
def box(df, feature) :
    sns.boxplot(df[feature],color="purple")
    plt.title(f"{feature} Outliers Detection",fontweight="black",pad=10,size=18)
    plt.tight_layout()
    plt.show()

# feature distribution by target
def feature_distribution_bytarget(df, feature) :
    boxs = sns.boxplot(data = df, x = target, y = feature)
    medians = df.groupby([target])[feature].median()
    counts =  df.groupby([target])[feature].count()
    for xtick in boxs.get_xticks():
        boxs.text(xtick,medians[xtick] ,str(medians[xtick]) + ' ' + '('+str(counts[xtick]) + ')', horizontalalignment='center',size=10,color='w',weight='semibold')

# distribution of numeric feature by target
def distribution_num_feature_bytarget(df, feature, n) :
    
    bins = (df[feature].max()- df[feature].min())/n
    binf = 0 ; bsup = (df[feature].max() + bins) ; biins = np.arange(binf, bsup, bins)
    
    feats = sns.displot(data=df, x=feature, hue=target, multiple="dodge", bins = biins, height=5, aspect=2)

    vals0 = feats.facet_axis(0,0).containers[0].datavalues
    vals1 = feats.facet_axis(0,0).containers[1].datavalues
    plt.xticks(np.arange(binf, bsup, bins))
    for i, v in enumerate(vals0):
        plt.text(i*bins + (3*bins/4), v/2, str(v), color='black', ha="center", fontweight = 'bold')
        plt.text(i*bins + (3*bins/4), v+40, str(round(v*100/1857, 1)) + '%', color='red', ha="center", fontweight = 'bold')
    for i, v in enumerate(vals1):
        plt.text(i*bins + (bins/4), v/2, str(v), color='black', ha="center", fontweight = 'bold')
        plt.text(i*bins + (bins/4), v+40, str(round(v*100/5164, 1))+ '%', color='red', ha="center", fontweight = 'bold')

# Numeric feature VS Target by Categorical feature
def distribution_num_feature_target_cat(df, feature, categoric) :
    plt.figure(figsize=(10, 6))
    sns.boxplot(x = target, y = feature, hue = categoric, data = df)
    plt.legend(loc='upper left')
    plt.show()

# Relation
def relation(df, target, feature) :
    cross_tab = pd.crosstab(df[feature], df[target])
    chi2_stat, p_value, _, _ = chi2_contingency(cross_tab)
    n = df.shape[0]
    s = cross_tab.sum().sum()
    min_dim = min(cross_tab.shape) - 1
    cramer_v = np.sqrt(chi2_stat / (s * min_dim)).round(2)
    tschuprow_t = np.sqrt(chi2_stat / (n * min_dim)).round(2)
    print(Back.YELLOW +'\033[1m' + f"The relation between {target} and {feature} : "+ '\033[0m' )  
    print(Back.YELLOW +'\033[1m' + f'Chie2 Test : p_value = {p_value} - Coefficient de Cramér-V = {cramer_v}   Coefficient T de Tschuprow = {tschuprow_t}'+ '\033[0m' + '\n' )

# Correlation
def correlation(df, feature, test) :
    df1 = df[[target,feature]].dropna()
    df1[target]= df1[target].astype('category').cat.codes
    v, p = test(df1[target], df1[feature])
    print(Back.YELLOW +Fore.BLACK + '\033[1m'+ f'correlation coefficient: {v}'+'\033[0m')
    print(Back.BLACK +Fore.WHITE + '\033[1m'+ f'P-value of correlation test: {p}' +'\033[0m')

class Description:
    def __init__(self):
        self.data_path: str=os.path.join('artifacts',"data.csv")
        
    def describe_target(self): 
            df=pd.read_csv(self.data_path)
            
            target = 'Churn'
            num_col = df.describe().columns.drop('SeniorCitizen')

            print(df[target].describe().to_frame().T)
            print(Back.YELLOW +'\033[1m' +f'value of {target} =  {df[target].unique()}'+ '\033[0m' + '\n')
            pie_chart(df,target)
                        
    def describe_cat(self): 
            df=pd.read_csv(self.data_path)
            
            target = 'Churn'
            num_col = df.describe().columns.drop('SeniorCitizen')
            cat_col = df.describe(include=['O']).columns.union(['SeniorCitizen']).drop('Churn')
            description(df, 'SeniorCitizen')

