import os
import streamlit as st
from src.train import read

def show_home():  
    st.title(" Welcome to Automate ML")
    
    st.markdown("""
    This no-code ML app lets you:
    -  Explore datasets
    -  Perform Exploratory Data Analysis (EDA)
    -  Train and evaluate ML models
    """)

    working_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(working_dir)
    
    data_dir = os.path.join(parent_dir, 'data')
    data_list = os.listdir(data_dir)

    dataset = st.selectbox(f" **Select a Dataset ** ", data_list)
    df = read(dataset)

    st.session_state["df"] = df
    st.session_state.dataset_name = dataset



    if df is not None:
        st.subheader("Dataset Preview")
        st.dataframe(df, use_container_width=True)

        st.subheader("Dataset Info")
        st.markdown(f"- **Rows**: {df.shape[0]}  \n- **Columns**: {df.shape[1]} \n- **Column Names**: {', '.join(df.columns)}")

        with st.expander(f"**Summary Statistics**"):
            st.write(df.describe())