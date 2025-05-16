import streamlit as st
import shap
import matplotlib.pyplot as plt

def show_feature_importance():
    st.title("Feature Importance")
    
    if "trained_model" not in st.session_state:
        st.warning("Please train a model first")
        return
        
    model = st.session_state.trained_model
    X_train = st.session_state.X_train
    
    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, X_train)

    shap_values = explainer.shap_values(X_train)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] 
        
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    fig = plt.gcf() 
    st.pyplot(fig)