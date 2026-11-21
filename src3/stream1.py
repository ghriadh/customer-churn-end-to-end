# streamlit run stream1.py
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns 
import plotly.express as px

# Display text
st.write('hello world write')
st.text ('welcome text')
st.header('Project 1 header')
st.subheader('Problem definition subheader')
st.title('Title')
st.markdown('***markdown***')
st.code("print('code python for example')", language = 'python')
st.latex(r'e^{i\pi} + 1 = 0')
x= 10
'x', x
# interactive
## button 
btn = st.button('Click me', key=None, help=None, on_click=None, args=None, kwargs=None, type="secondary", disabled=False, use_container_width=False)
#st.download_button('dowload', data = csv, file_name='data_df.csv', mime='text/csv', key=None, help=None, on_click=None, type="secondary")
st.write(btn)

btn1 = st.button('Submit')
st.write(btn1)
if btn1 : st.info('info')
## toggle
on = st.toggle(label = 'Activate', value=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
if on : st.write('feature activated')

## checkbox 
agree = st.checkbox('I agree', value=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
if agree : st.exception('Great')

## radio button
radio = st.radio(label = 'club', options=['Club africain', 'esperance', 'etoile', 'stade'], captions=['CA', 'EST', 'ESS', 'ST'],index=0, key=None, help=None, on_change=None, args=None, kwargs=None,  disabled=False, horizontal=False, label_visibility="visible")
if radio == 'Club africain' : st.success('you are champion :)')
elif radio == 'esperance' : st.warning('warning : /')
elif radio == 'etoile' : st.error('error : (')
elif radio == 'stade' : st.write('you are not champion')


## selectbox
choix = st.selectbox('country', ['TUN', 'CAN', 'FR', 'QAT', 'USA'], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder="Choose an option", disabled=False, label_visibility="visible")
st.write('You are from :', choix)

## Slider 
age = st.slider('how old are you', min_value=0, max_value=100, value=30, step=5, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

st.write('I am ', age , 'years old')

from datetime import time
appointment = st.slider(
    "Schedule your appointment:",
    value=(time(11, 30), time(12, 45)))
st.write("You're scheduled for:", appointment)

# dataframe
df = sns.load_dataset('taxis')
# st.write(df)
#btn = st.button('show data')
#if btn : st.dataframe(df.sample(5))
st.dataframe(df.sample(5))
# st.table(df.head(3))

# matplotlib 
st.header('MATPLOTLIB')
st.subheader('Line')
fig= plt.figure(figsize=(12, 8))
plt.plot(df.index, df['total'], c = 'red', lw = 3, marker = '*')
plt.title('titre')
plt.xlabel('index')
plt.ylabel('total')
st.pyplot(fig)

n_bins = st.number_input(label = 'NB bins', min_value = 4, value = 10, max_value = 20)
my_title = st.text_input('donner un titre au graphique')
fig, ax = plt.subplots()
ax.hist(df['total'], bins = n_bins)
plt.title(my_title)
st.pyplot(fig)

# visualisation avec streamlit 
st.line_chart(df['total']) # line
st.bar_chart(df['total']) # bar
# st.map(df[['longitude', 'latitude']]) # carte , necessairement une colonne lat or LAT or latitude

# visualisatin en combinaison avec plotly
fig = px.bar(data_frame = df, x = 'total',y = 'fare', title = 'un titre') # diagramme (bares)
st.plotly_chart(fig)  

features = df.select_dtypes(exclude="object").columns.to_list()
feature = st.selectbox('choisir une variable', features)
colors = df.select_dtypes(include = 'object').columns.to_list()
color = st.selectbox('choisir un critere de coloration', colors)
fig = px.scatter(data_frame = df, x = df.index, y = feature, color = color, title = str(feature) + ' discription') # scatter
st.plotly_chart(fig) 

# visualisatin en combinaison avec seaborn et matplotlib
fig, ax = plt.subplots()
ax = sns.histplot(df['total'])
st.pyplot(fig)  

fig, ax = plt.subplots()
ax = plt.hist(df['total'])
st.pyplot(fig)






