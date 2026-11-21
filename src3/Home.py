import streamlit as st
import io

st.set_page_config(
    page_title="Customer Churn",
    page_icon="👋",
)

def load_css(file_name = "src3/style.css"):
    with open(file_name) as f:
        css = f'<style>{f.read()}</style>'
    return css
css = load_css()
st.markdown(css, unsafe_allow_html=True)

# st.markdown(
#         """
#         <style> .streamlit-expanderHeader {font-size: 20px; font-weight:bold; color : blue;}</style>
#         """,
#         unsafe_allow_html=True,)

def main():

    st.title('Application de machine learning pour la detection de desabonnement des clients')
    st.subheader('Auteur : Ghanmi riadh')
    
    with st.popover("Drop custumerID"):
            st.write("custumerID dropped")

    st.subheader("*Problem Definition :*")
    st.write("**In this project we will explore data from a telecommunications company in order to build a machine learning model to predict the churn of the company’s customers**")
   

if __name__ == '__main__' : main()