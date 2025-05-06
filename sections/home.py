import os
import streamlit as st
from src.train import read

def show_home(df):  

    if df is not None:
        st.subheader("Dataset Preview")
        st.dataframe(df, use_container_width=True)

        st.subheader("Dataset Info")
        st.markdown(f"- **Rows**: {df.shape[0]}  \n- **Columns**: {df.shape[1]} \n- **Column Names**: {', '.join(df.columns)}")

        with st.expander(f"**Summary Statistics**"):
            st.write(df.describe())