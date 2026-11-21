import streamlit as st
import io

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, special
from scipy.stats import norm, skew, shapiro, kstest, pearsonr, spearmanr, kendalltau, chi2_contingency, levene
# from ydata_profiling import ProfileReport
# from streamlit_pandas_profiling import st_profile_report
from utils_streamlit import pie_chart, bar_plot_distribution, relation, correlation, transformation

if 'target' not in st.session_state:
    st.session_state['target'] ='Churn'
if 'num_col' not in st.session_state:
    st.session_state['num_col'] =''
if 'cat_col' not in st.session_state:
    st.session_state['cat_col'] =''

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

def main():

    st.title('Customer Churn Explore Data')
    st.subheader('Auteur : Ghanmi riadh')
    
    df = st.session_state['df']

    with st.expander('Exploring data'):
    #     with tab1 :
        st.write(df.describe().round(2).T)
        st.markdown('''
                * **Continuous features `TotalCharges`,`MonthlyCharges` and `tenure` are having `high deviation` values. So we have to use `transformation and scaling technique` to reduce the deviation**
                * **`SeniorCitizen` is a 0 or 1, like categorical feature**
                '''
            )

        st.write(df.describe(include=['O']).T)
        st.markdown('- **We have not a `heigt cardinality` for categorical features in our dataset [2 , 3 or 4]** ' )

        target  = st.session_state['target']
        st.session_state['num_col'] = df.describe().columns.drop('SeniorCitizen')
        st.session_state['cat_col'] = df.describe(include=['O']).columns.union(['SeniorCitizen']).drop('Churn')
        num_col = st.session_state['num_col']
        cat_col = st.session_state['cat_col']


    with st.expander('Target'):
        st.write(df[target].describe().to_frame().T)
        st.text(f'value of {target} =  {df[target].unique()}'+  '\n')
        pie_chart(df, target)
        st.markdown('- **5164/7021 of customers are not churned (73.5%) - 1857/7021 customers are churned (26.5%)**')
        
    with st.expander('Categorical features'):
            feature = st.selectbox('choose a categorical feature', cat_col)
            des = df[feature].describe().to_frame().T
            des['cardinality'] = len(df[feature].unique())
            st.write(des)
            st.write(f'value of {feature} =  {df[feature].unique()}'+  '\n')
            st.write(df[[feature, target]].groupby([feature]).count().rename(columns={target: "Count"}))
            pie_chart(df, feature)
            if feature == 'PhoneService' : st.markdown('- **PhoneService has a certain class imballance**')

            cross_tab = pd.crosstab(df[feature], df[target])
            st.write(cross_tab)
            bar_plot_distribution(df, target, feature)
            relation(df, target, feature)

            coef = []
            for feature in cat_col :
                cross_tab = pd.crosstab(df[feature], df[target])
                chi2_stat, p_value, _, _ = chi2_contingency(cross_tab)
                n = df.shape[0]
                s = cross_tab.sum().sum()
                min_dim = min(cross_tab.shape) - 1
                cramer_v = np.sqrt(chi2_stat / (s * min_dim)).round(2)
                coef.append(cramer_v) 
            rel = pd.DataFrame({'coefficient': coef}, index = cat_col.to_numpy())
            rel = rel.sort_values(by=['coefficient'], ascending=False)
            st.write('Relation between categorical features and target according to cramer coefficient')
            st.write(rel.T)

            fig = plt.figure()
            sns.barplot(rel, x= rel.index, y = 'coefficient')
            i = 0
            for v in rel['coefficient']:
                plt.text(i, v, f"{v:.2f}", ha="center", va="bottom",size=10,fontweight="black")
                i = i +1

            plt.title("Relation between Target (Churn) and Categorical features",fontweight="black",size=14,pad=5)
            plt.xticks(rotation=90)
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown('''
                        - **features like `Contract`, `OnlineSecurity` and `TechSupport` have a `heigt` relation with the target Churn**
                        - **while features like `gender`, `PhoneService` and `MultipleLines` have a `insignificant` and negligible relation with target Churn**
                        ''')
            

            st.markdown('''
                    - **(StreamingMovies and StreamingTV) have same relation with target. Verify that are not in linearity**
                    - **Also (Partner and SeniorCitizen) have same relation with target. Verify that features are not in linearity**
                    ''')
            
            st.code('''
                    relation(df, 'StreamingMovies', 'StreamingTV') 
                    ''' , language = 'python')
            relation(df, 'StreamingMovies', 'StreamingTV')

            st.code('''
                    relation(df, 'Partner', 'SeniorCitizen')
                    ''' , language = 'python')
            relation(df, 'Partner', 'SeniorCitizen')

            
    with st.expander('Categorical features'):
    #     with tab4 :
            feature = st.selectbox('choose a numeric feature', num_col)
            n = st.slider('Select Number of bins', 1, 12, 6, 1)
            st.write(df[feature].describe().to_frame().T)
            bins = (df[feature].max()- df[feature].min())/n
            binf = 0 ; bsup = (df[feature].max() + bins) ; biins = np.arange(binf, bsup, bins)                           
            fig = sns.displot(df[feature], color="blue", bins = biins)
            plt.title (f'►Customers Distribution by {feature}◄', fontsize=12, fontweight='bold')
            plt.xticks(np.arange(binf, bsup, bins))
            feats = fig.facet_axis(0,0).containers[0].datavalues
            for i, v in enumerate(feats):
                    plt.text(i*bins + bins/2 , v + 1, str(v), color='black', ha="center", fontweight = 'bold') 
            st.pyplot(fig)

            st.write('Normality :')
            fig, ax = plt.subplots(2, 2, figsize=(18, 12), gridspec_kw=dict(width_ratios=[1,1], height_ratios= [4,1]))
            sns.displot(df[feature], ax = ax[0][0])
            res = stats.probplot(df[feature], plot=ax[0][1])
                
            ax[1][0].text(0.1, 0.7, f'Skewness = {df[feature].skew().round(3)}', fontsize = 14, fontweight = 'bold' )
            ax[1][0].text(0.1, 0.4, f'kurtosis = {df[feature].kurt().round(3)}', fontsize = 14, fontweight = 'bold' )
            ax[1][0].text(0.1, 0.1, f'Test = {stats.normaltest(df[feature]).pvalue}', fontsize = 14, fontweight = 'bold' )
            ax[1][1].text(0.1, 0.2, f'KSTest = {stats.kstest(df[feature], "norm")[1]}', fontsize = 14, fontweight = 'bold')
            ax[1][1].text(0.1, 0.6, f'Shapiro = {shapiro(df[feature])[1]}', fontsize = 14, fontweight = 'bold')
                
            ax[1][0].set_xticks([]); ax[1][1].set_xticks([]); ax[1][0].set_yticks([]); ax[1][1].set_yticks([])
            st.pyplot(fig)
            st.markdown('- **All features dont have normal distribution**')
                
            st.write('Outliers')
            plt.figure(figsize=(16,8))
            plot = sns.boxplot(df[feature],color="purple")
            plt.title(f"{feature} Outliers Detection",fontweight="black",pad=10,size=14)
            st.pyplot(plot.figure)
            st.markdown('- **According to the boxplots we do not detect outliers**')

            st.write('Feature Distribution by Target class')
            plt.figure(figsize=(16,8))
            boxs = sns.boxplot(data = df, x = target, y = feature)
            medians = df.groupby([target])[feature].median()
            for xtick in boxs.get_xticks():
                    boxs.text(xtick,medians[xtick] ,str(medians[xtick]) , horizontalalignment='center',size=10,color='w',weight='semibold')
            st.pyplot(boxs.figure)
            st.markdown('''
                            - **We have 5164 No churn costumers, and 1857 churn costumers. The median of tenure of No churn costumers is 38, heigher than the median of tenure of churn customers 10**
                            - **The median of MonthlyCharges of churn costumers is 79.7, heigher than the median of MonthlyCharges of No churn customers  64.5**
                            - **The median of TotalCharges of No churn costumers is 1688.9, heigher than the median of MonthlyCharges of churn customers  713.1**
                             ''')    

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
            st.pyplot(feats)
            st.markdown(''' 
                            - **more than 53% of churn customers have a less then 12 months tenure**
                            - **more than 25% of No churn customers have a monthly charges between 12.6 and 25.1**
                            - **more than 60% of churn customer have a monthly charges between 67 and 100.5**
                            - **46% of no churn customers and 65.8% of churn customers have Total charges less than 1444**
                            ''')
                
            st.write('Feature Distribution by Target class and Categorical feature')
            categoric = st.selectbox('choose one categorical feature', cat_col)
            plt.figure(figsize=(10, 6))
            plot = sns.boxplot(x = target, y = feature, hue = categoric, data = df)
            plt.legend(loc='upper left')
            st.pyplot(plot.figure)

            st.write('Correlation feature/target')
            test = st.selectbox('choose one test of correlation', [ 'pearsonr', 'spearmanr', 'kendalltau' ])
            correlation(df, target, feature, test)
           
            if test == 'pearsonr' : test = 'pearson'
            if test == 'spearmanr' : test = 'spearman'
            if test == 'kendalltau' : test = 'kendall'
            df1 = df[num_col.union([target])]
            df1[target]= df1[target].astype('category').cat.codes
            corr = df1.corr(method=test)[target].loc[num_col]
            plt.figure(figsize=(10,6))
            fig = sns.barplot(x = corr.index, y = corr.values, palette=["lightcoral" if v < 0 else "lightgreen" for v in corr])
            for i, v in enumerate(corr):
                    plt.text(i, v/2, f"{v:.2f}",size=15,fontweight="black")
            plt.title("Coorelation of Discrets features w.r.t SalePrice",fontweight="black",size=20,pad=10)
            plt.xticks(ticks = np.arange(len(num_col)), labels = num_col, color = 'red')
            st.pyplot(fig.figure)   

            st.markdown('''
                    - **tenure and TotalCharges have negatives correlations with target**
                    - **MonthlyCharges have a positive correlation with target**
                    - **tenure have the most imprtant correlation with target -0.35**
                        ''')


if __name__ == '__main__' : main()