import streamlit as st
import pandas as pd
import numpy as np
import base64
from io import BytesIO

st.set_page_config(page_title="Fairness Audit & Mitigation Dashboard", layout="wide", page_icon="⚖️")

from utils.preprocessing import preprocess_data
from utils.training import train_model
from utils.bias_detection import detect_bias
from utils.mitigation import mitigate_bias
from utils.explainability import generate_shap_plots
from utils.reporting import generate_report
import matplotlib.pyplot as plt

# Custom CSS for fintech green and minimal card styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f4f9f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        text-align: center;
        border-top: 4px solid #00b268;
    }
    .metric-title { font-size: 1rem; color: #64748b; margin-bottom: 0px; margin-top: 0px; font-weight: 500;}
    .metric-value { font-size: 2rem; font-weight: 700; color: #00b268; margin-top: 5px; margin-bottom: 0px;}
    .warning-text { color: #dc2626; font-weight: bold; }
    .safe-text { color: #16a34a; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

def render_metric(title, value):
    st.markdown(f'<div class="metric-card"><p class="metric-title">{title}</p><p class="metric-value">{value}</p></div>', unsafe_allow_html=True)

# State init
state_keys = ['data', 'model', 'metrics', 'bias_metrics', 'mitigated_model', 'mitigated_metrics', 
              'mitigated_bias_metrics', 'X_train', 'X_test', 'y_train', 'y_test', 
              'sensitive_col', 'model_type', 'sf_test', 'sf_train', 'approval_rates', 'mitigated_approval_rates']
for k in state_keys:
    if k not in st.session_state:
        st.session_state[k] = None

# Sidebar Navigation
with st.sidebar:
    st.title("⚖️ LoanGard AI")
    st.markdown("Automated Fairness Audit System")
    st.markdown("---")
    page = st.radio("Navigation", [
        "Dashboard", 
        "Data Overview", 
        "Model Training", 
        "Bias Detection", 
        "Bias Mitigation", 
        "Comparison", 
        "Explainability", 
        "Reports"
    ])

def display_dashboard():
    st.title("AI Fairness Audit Dashboard")
    st.subheader("Bias Detection & Mitigation System")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.session_state.metrics:
            render_metric("Accuracy", f"{st.session_state.metrics['Accuracy']:.2%}")
        else:
            render_metric("Accuracy", "N/A")
            
    with col2:
        if st.session_state.bias_metrics:
            dpd = st.session_state.bias_metrics.get('Demographic Parity Difference', 0)
            render_metric("Demographic Parity", f"{dpd:.3f}")
        else:
            render_metric("Demographic Parity", "N/A")
            
    with col3:
        if st.session_state.bias_metrics:
            di = st.session_state.bias_metrics.get('Disparate Impact', 0)
            render_metric("Disparate Impact", f"{di:.3f}")
        else:
            render_metric("Disparate Impact", "N/A")
            
    with col4:
        if st.session_state.mitigated_metrics:
            render_metric("Compliance Status", "Compliant ✅")
        elif st.session_state.bias_metrics and st.session_state.bias_metrics.get('Disparate Impact', 1) < 0.8:
            render_metric("Compliance Status", "Non-Compliant ❌")
        elif st.session_state.bias_metrics:
            render_metric("Compliance Status", "Compliant ✅")
        else:
            render_metric("Compliance Status", "Pending ⏳")
            
    st.markdown("---")
    st.info("Welcome to LoanGard AI! Upload your dataset to begin. Proceed through Model Training and Bias Detection to evaluate your loan approval logic.")

def display_data_overview():
    st.title("Data Overview")
    uploaded_file = st.file_uploader("Upload Loan Dataset (CSV)", type="csv")
    
    if uploaded_file or st.button("Load Sample Data (if available)"):
        try:
            if uploaded_file:
                st.session_state.data = pd.read_csv(uploaded_file)
            else:
                st.session_state.data = pd.read_csv("data/loan_data.csv")
            st.success("Data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading data: {e}")
            
    if st.session_state.data is not None:
        df = st.session_state.data
        st.write("### Data Preview")
        st.dataframe(df.head())
        
        col1, col2 = st.columns(2)
        with col1:
            render_metric("Total Rows", f"{df.shape[0]}")
        with col2:
            render_metric("Total Columns", f"{df.shape[1]}")
            
        st.write("### Missing Values Chart")
        missing = df.isna().sum()
        if missing.sum() > 0:
            st.bar_chart(missing[missing > 0])
        else:
            st.write("No missing values found.")

def display_model_training():
    st.title("Model Training")
    
    if st.session_state.data is None:
        st.warning("Please upload data in the Data Overview section first.")
        return
        
    df = st.session_state.data
    st.write("Select constraints for model training:")
    
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_columns:
        sensitive_attr = st.selectbox("Select Sensitive Attribute", cat_columns)
    else:
        st.error("No categorical columns found that could serve as sensitive attribute.")
        return
        
    target_col = st.selectbox("Select Target Column", df.columns.tolist(), index=len(df.columns)-1)
    model_type = st.selectbox("Select Model", ["Logistic Regression", "Random Forest"])
    
    if st.button("Train Model"):
        with st.spinner("Preprocessing Data and Training Model..."):
            try:
                # Preprocess
                X, y, sf_proc, encoders, sf_raw = preprocess_data(df, target_col=target_col, sensitive_col=sensitive_attr)
                
                # We need sf_raw for fairlearn metrics
                model, metrics, X_test, y_test, y_pred, X_train, y_train, sf_train, sf_test = train_model(
                    X, y, sensitive_features=sf_raw, model_type=model_type
                )
                
                st.session_state.model = model
                st.session_state.metrics = metrics
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.X_train = X_train
                st.session_state.y_train = y_train
                st.session_state.sf_test = sf_test
                st.session_state.sf_train = sf_train
                st.session_state.model_type = model_type
                st.session_state.sensitive_col = sensitive_attr
                
                # Flush mitigation state to avoid inconsistencies
                st.session_state.mitigated_model = None
                st.session_state.mitigated_metrics = None
                st.session_state.mitigated_bias_metrics = None
                
            except Exception as e:
                st.error(f"Error during training: {e}")
                return
                
    if st.session_state.metrics:
        st.success(f"{st.session_state.model_type} trained successfully!")
        cols = st.columns(4)
        cols[0].metric("Accuracy", f"{st.session_state.metrics['Accuracy']:.2%}")
        cols[1].metric("Precision", f"{st.session_state.metrics['Precision']:.2%}")
        cols[2].metric("Recall", f"{st.session_state.metrics['Recall']:.2%}")
        cols[3].metric("F1 Score", f"{st.session_state.metrics['F1 Score']:.2%}")

def display_bias_detection():
    st.title("Bias Detection")
    
    if st.session_state.model is None or st.session_state.sf_test is None:
        st.warning("Please train a model first before detecting bias.")
        return
        
    if st.button("Run Audit"):
        with st.spinner("Analyzing Bias..."):
            y_pred = st.session_state.model.predict(st.session_state.X_test)
            bias_metrics, approval_rates = detect_bias(st.session_state.y_test, y_pred, st.session_state.sf_test)
            
            st.session_state.bias_metrics = bias_metrics
            st.session_state.approval_rates = approval_rates
            
    if st.session_state.bias_metrics:
        st.subheader("Audit Results")
        
        # Display approval rates
        st.write(f"### Approval Rates by {st.session_state.sensitive_col}")
        rates_df = pd.DataFrame(list(st.session_state.approval_rates.items()), columns=['Group', 'Approval Rate'])
        rates_df.set_index('Group', inplace=True)
        st.bar_chart(rates_df)
        
        # Show bias metrics
        cols = st.columns(3)
        di = st.session_state.bias_metrics['Disparate Impact']
        dpd = st.session_state.bias_metrics['Demographic Parity Difference']
        eod = st.session_state.bias_metrics['Equal Opportunity Difference']
        
        cols[0].metric("Disparate Impact", f"{di:.3f}")
        cols[1].metric("Demographic Parity Difference", f"{dpd:.3f}")
        cols[2].metric("Equal Opportunity Difference", f"{eod:.3f}")
        
        # Compliance Check
        st.write("### Compliance Status")
        if di < 0.8:
            st.markdown(f"Status: <span class='warning-text'>Bias Warning</span> (Disparate Impact < 0.8)", unsafe_allow_html=True)
        else:
            st.markdown(f"Status: <span class='safe-text'>Fair / Compliant</span> (Disparate Impact ≥ 0.8)", unsafe_allow_html=True)

def display_bias_mitigation():
    st.title("Bias Mitigation")
    
    if st.session_state.model is None:
        st.warning("Please train a model first.")
        return
        
    if st.session_state.bias_metrics is None:
        st.warning("Please run Bias Detection first to establish a baseline.")
        return
        
    method = st.selectbox("Select Mitigation Method", ["Exponentiated Gradient (Fairlearn)", "Reweighing (AIF360)"])
    
    if st.button("Apply Mitigation & Retrain"):
        with st.spinner("Applying mitigation algorithms. This might take a moment..."):
            try:
                mit_model = mitigate_bias(
                    st.session_state.X_train, 
                    st.session_state.y_train, 
                    st.session_state.sf_train, 
                    model_type=st.session_state.model_type, 
                    method=method
                )
                st.session_state.mitigated_model = mit_model
                
                # Evaluate Mitigated Model
                y_pred_mitigated = mit_model.predict(st.session_state.X_test)
                
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                new_metrics = {
                    'Accuracy': accuracy_score(st.session_state.y_test, y_pred_mitigated),
                    'Precision': precision_score(st.session_state.y_test, y_pred_mitigated),
                    'Recall': recall_score(st.session_state.y_test, y_pred_mitigated),
                    'F1 Score': f1_score(st.session_state.y_test, y_pred_mitigated)
                }
                st.session_state.mitigated_metrics = new_metrics
                
                # Detect bias again
                new_bias_metrics, new_app_rates = detect_bias(st.session_state.y_test, y_pred_mitigated, st.session_state.sf_test)
                st.session_state.mitigated_bias_metrics = new_bias_metrics
                st.session_state.mitigated_approval_rates = new_app_rates
                
                st.success("Mitigation applied successfully!")
            except Exception as e:
                st.error(f"Error applying mitigation: {e}")
                
    if st.session_state.mitigated_bias_metrics:
        st.subheader("Post-Mitigation Results")
        di = st.session_state.mitigated_bias_metrics['Disparate Impact']
        cols = st.columns(3)
        cols[0].metric("New Disparate Impact", f"{di:.3f}")
        cols[1].metric("New Accuracy", f"{st.session_state.mitigated_metrics['Accuracy']:.2%}")
        
        if di >= 0.8:
            st.markdown(f"Status: <span class='safe-text'>Now Fair / Compliant</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"Status: <span class='warning-text'>Still showing Bias Warning</span>", unsafe_allow_html=True)

def display_comparison():
    st.title("Performance & Fairness Comparison")
    
    if st.session_state.mitigated_metrics is None:
        st.warning("Apply bias mitigation first to compare results.")
        return
        
    st.subheader("Model Validation")
    
    # Tables for metrics
    col1, col2 = st.columns(2)
    
    df_perf = pd.DataFrame([
        st.session_state.metrics,
        st.session_state.mitigated_metrics
    ], index=['Before Mitigation', 'After Mitigation'])
    
    df_bias = pd.DataFrame([
        st.session_state.bias_metrics,
        st.session_state.mitigated_bias_metrics
    ], index=['Before Mitigation', 'After Mitigation'])
    
    with col1:
        st.write("### Accuracy & Quality")
        st.dataframe(df_perf.style.highlight_max(axis=0))
        st.bar_chart(df_perf[['Accuracy']])
        
    with col2:
        st.write("### Fairness Metrics")
        st.dataframe(df_bias.style.highlight_min(axis=0, subset=['Demographic Parity Difference', 'Equal Opportunity Difference']))
        st.line_chart(df_bias[['Disparate Impact']])

def display_explainability():
    st.title("Model Explainability (SHAP)")
    
    if st.session_state.model is None:
        st.warning("Please train a model first.")
        return
        
    st.write("Explainability is crucial for compliance. The plot below illustrates feature importance and directional impact.")
    
    if st.button("Generate SHAP Summary Plot"):
        with st.spinner("Computing SHAP values..."):
            
            # Prefer mitigated model if available, otherwise base model
            model_to_explain = st.session_state.mitigated_model if st.session_state.mitigated_model else st.session_state.model
            
            fig = generate_shap_plots(model_to_explain, st.session_state.X_train, st.session_state.X_test, st.session_state.model_type)
            st.pyplot(fig)

def display_reports():
    st.title("Compliance Report Generation")
    
    if st.session_state.metrics is None or st.session_state.bias_metrics is None:
        st.warning("Please run Model Training and Bias Detection to generate a report.")
        return
        
    st.write("Generate a formal text report intended for regulatory compliance boards.")
    
    if st.button("Generate Report"):
        report_text = generate_report(
            st.session_state.metrics,
            st.session_state.bias_metrics,
            st.session_state.mitigated_metrics,
            st.session_state.mitigated_bias_metrics
        )
        st.session_state.report_text = report_text
        
    if 'report_text' in st.session_state:
        st.text_area("Report Preview", st.session_state.report_text, height=300)
        
        b64 = base64.b64encode(st.session_state.report_text.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="Compliance_Report.md">Download Compliance Report (Markdown)</a>'
        st.markdown(href, unsafe_allow_html=True)


# Router
if page == "Dashboard":
    display_dashboard()
elif page == "Data Overview":
    display_data_overview()
elif page == "Model Training":
    display_model_training()
elif page == "Bias Detection":
    display_bias_detection()
elif page == "Bias Mitigation":
    display_bias_mitigation()
elif page == "Comparison":
    display_comparison()
elif page == "Explainability":
    display_explainability()
elif page == "Reports":
    display_reports()
