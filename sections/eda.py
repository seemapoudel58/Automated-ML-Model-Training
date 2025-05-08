import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_categorical_columns(df, target_col):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()  
    num_cols = [col for col in df.select_dtypes(include='number').columns
                if df[col].nunique() <= 10 and col != target_col] 
    return list(set(cat_cols + num_cols) - {target_col})

# Initialize session state
def init_session(df):
    if 'target_col' not in st.session_state:
        st.session_state.target_col = df.columns[0]

def show_eda(df):
    st.title("Exploratory Data Analysis")

    if df.empty:
        st.warning("Please upload a dataset on the Home page.")
        return

    init_session(df)

    st.selectbox("Select Target Column", df.columns, key='target_col')

    target = st.session_state.target_col
    cat_cols = get_categorical_columns(df, target)
    num_cols = [col for col in df.select_dtypes(include='number').columns if col not in cat_cols + [target]]

    # Pairplot
    if target and len(num_cols) > 1:
        st.subheader("Pairplot")
        try:
            sns_fig = sns.pairplot(df[num_cols + [target]], hue=target, diag_kind='kde')
            st.pyplot(sns_fig)
        except Exception as e:
            st.error(f"Failed to generate pairplot: {e}")
        
    col1, col2 = st.columns(2)

    # Pie Chart
    with col2:
        st.subheader("Pie Chart")
        if cat_cols:
            pie_col = st.selectbox("Choose categorical column for pie chart", cat_cols)
            pie_data = df[pie_col].value_counts()
            fig, ax = plt.subplots()
            ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
            ax.set_title(f"Distribution of {pie_col}")
            st.pyplot(fig)
        else:
            st.warning("No categorical columns available to display the pie chart.")

    # Correlation Heatmap
    with col1:
        if len(num_cols) > 1:
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots()
            corr = df[num_cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

    # Box Plot
    if cat_cols and num_cols:
        st.subheader("Box Plot")
        with st.form(key='box_plot_form'):
            col1, col2 = st.columns(2)
            with col1:
                box_cat = st.selectbox("Choose categorical column", cat_cols, key='box_cat')
            with col2:
                box_num = st.selectbox("Choose numerical column", num_cols, key='box_num')
            generate_plot = st.form_submit_button("Generate Box Plot")

        if generate_plot:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[box_cat], y=df[box_num], ax=ax)
            ax.set_title(f"{box_num} by {box_cat}")
            st.pyplot(fig)
    else:
        st.subheader("Box Plot")
        if not cat_cols:
            st.warning("No categorical columns available to display the box plot.")
        if not num_cols:
            st.warning("No numerical columns available to display the box plot.")


