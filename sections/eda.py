import streamlit as st
import seaborn as sns   
import matplotlib.pyplot as plt
import pandas as pd    

def show_eda(df):   
    st.title("Exploratory Data Analysis (EDA)")

    if df is None or df.empty:
        st.warning("No dataset available for EDA. Please select a dataset from the Home page.")
        return
    
    mappings = {
        'sex': {0: 'Female', 1: 'Male'},
        'cp': {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-anginal Pain', 3: 'Asymptomatic'},
        'fbs': {0: '≤ 120 mg/dl', 1: '> 120 mg/dl'},
        'restecg': {0: 'Normal', 1: 'ST-T Abnormality', 2: 'Left Ventricular Hypertrophy'},
        'exang': {0: 'No', 1: 'Yes'},
        'slope': {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'},
        'thal': {1: 'Normal', 2: 'Fixed Defect', 3: 'Reversible Defect'} 
    }

    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).astype('category')

    

    numerical_col = df.select_dtypes(include=['number']).columns
    if len(numerical_col) > 0:
        st.markdown("### Pairplot")
        if len(numerical_col) > 0:
            fig = sns.pairplot(df[numerical_col], diag_kind='kde')
            st.pyplot(fig)
            print(df.columns)

        else:
            st.warning("No numerical columns available for pairplot.")


    
    categorical_cols = df.select_dtypes(include=['category']).columns
    if len(categorical_cols) > 0:
        st.markdown("### Pie Chart for Categorical Features")
        if len(categorical_cols)> 0:
            select_col = st.selectbox("Select a categorical column for pie chart", categorical_cols)
            if select_col:
                pie_data = df[select_col].value_counts()
                fig, ax = plt.subplots()
                ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)  



    