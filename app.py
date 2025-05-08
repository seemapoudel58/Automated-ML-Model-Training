import streamlit as st
from src.train import read
from sections.home import show_home
from sections.eda import show_eda

st.set_page_config(page_title="Automate ML", page_icon="📊", layout="wide")

if 'page' not in st.session_state:
    st.session_state.page = "Home"  

def go_to_page(page_name):
    st.session_state.page = page_name
    st.rerun()

if st.session_state.page == "Home":
    show_home()

    if st.button('Next Page'):
        go_to_page("EDA")

elif st.session_state.page == "EDA":
    df = st.session_state.get("df", None)  
    if df is not None:
        show_eda(df)
    else:
        st.warning("Please select a dataset from the Home page.")
    
    col1, col2 = st.columns([0.14, 1 ])  
    with col1:
        if st.button("Previous Page"):
            go_to_page("Home")
    with col2:
        if st.button("Next Page"):
            go_to_page("Model Training")
