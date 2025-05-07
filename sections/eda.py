import streamlit as st
import seaborn as sns   
import matplotlib.pyplot as plt
import pandas as pd   

def detect_categorical_columns(df, max_unique_values=10, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = ['target', 'label', 'outcome']
    
    exclude_cols = [col.lower() for col in exclude_cols]

    cat_cols = []

    for col in df.select_dtypes(include=['category', 'object']).columns:
        if col.lower() not in exclude_cols:
            cat_cols.append(col)

    for col in df.select_dtypes(include='number').columns:
        if col.lower() in exclude_cols:
            continue
        unique_values = df[col].dropna().unique()
        if len(unique_values) <= max_unique_values and all(float(v).is_integer() for v in unique_values):
            cat_cols.append(col)

    return list(set(cat_cols))

def show_eda(df):   
    st.title("Exploratory Data Analysis (EDA)")

    if df is None or df.empty:
        st.warning("No dataset available for EDA. Please select a dataset from the Home page.")
        return
    

    categorical_cols = detect_categorical_columns(df)

    numerical_col = [col for col in df.select_dtypes(include=['number']).columns if col not in categorical_cols]

    if len(numerical_col) > 0:
        st.markdown("### Pairplot")
        if len(numerical_col) > 0:
            fig = sns.pairplot(df[numerical_col],  diag_kind='kde')
            st.pyplot(fig)
            print(df.columns)

        else:
            st.warning("No numerical columns available for pairplot.")
    for col in categorical_cols:
        df[col] = df[col].astype('category')


    if len(categorical_cols) > 0:
        st.markdown("### Pie Chart for Categorical Features")
        select_col = st.selectbox("Select a categorical column for pie chart", categorical_cols)
        
        if select_col:
            # st.write(f"Column '{select_col}' info:")
            # st.write(f"Data type: {df[select_col].dtype}")
            # st.write(f"Number of non-null values: {df[select_col].count()}")
            # st.write(f"Number of unique values: {df[select_col].nunique()}")
            # st.write(f"First few values: {df[select_col].head().tolist()}")
            
            filtered_col = df[select_col].dropna()
            
            if len(filtered_col) > 0:
                pie_data = filtered_col.value_counts()
                
                if len(pie_data) > 0:
                    fig = plt.figure(figsize=(8, 8))
                    plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
                    plt.axis('equal')
                    plt.title(f"Distribution of {select_col}")
                    st.pyplot(fig)
                else:
                    st.warning(f"No valid categorical data found in '{select_col}' after processing")
            else:
                st.warning(f"No non-null values found in column '{select_col}'")
    else:
        st.warning("No categorical columns available for pie chart.")

