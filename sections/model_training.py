import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from src import train

def show_model_training(df):
    st.title("Model Training")

    target_column = st.selectbox("Select Target Column", df.columns)

    test_size = st.slider("Select Test Size", 0.1, 0.5, 0.2)
    split_method = st.radio("Splitting Method", ["Random Splitting", "Stratified Splitting"])
    scaler = st.selectbox("Select Scaler", ["StandardScaler", "MinMaxScaler"])
    model_choice = st.selectbox("Select Model", ["Random Forest", "Logistic Regression"])

    if st.button("Train Model"):
        if target_column not in df.columns:
            st.error("Invalid target column.")
            return

        if split_method == "Stratified Splitting":
            stratify = True
        else:
            stratify = False

        try:
            X_train, X_test, y_train, y_test = train.preprocess(
                df,
                target_column=target_column,
                scalar_type=scaler,
                test_size=test_size,
                stratify=stratify,
                random_state=42,
            )

            if model_choice == "Random Forest":
                model = RandomForestClassifier()
            elif model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)

            trained_model = train.train_model(X_train, y_train, model)
            acc = train.evaluate_model(trained_model, X_test, y_test)

            st.session_state["trained_model"]= trained_model
            st.session_state["X_train"] = X_train

            st.success(f"Model trained! Accuracy: {acc}%")
        except Exception as e:
            st.error(f"Error during training: {str(e)}")
