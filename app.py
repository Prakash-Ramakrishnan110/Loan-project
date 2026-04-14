import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import base64
from io import BytesIO
st.set_page_config(page_title='Fairness Audit Pipeline | Compliance Platform', layout='wide', initial_sidebar_state='expanded')
from utils.preprocessing import preprocess_data, get_data_profile
from utils.training import train_model, evaluate_model
from utils.bias_detection import detect_bias, classify_risk
from utils.mitigation import mitigate_bias
from utils.explainability import compute_shap_values, get_feature_importance, generate_shap_summary_plot
from utils.reporting import generate_report, generate_pdf_report
PRIMARY = '#0F172A'
ACCENT = '#16A34A'
ACCENT_LIGHT = '#DCFCE7'
BG = '#F8FAFC'
CARD_BG = '#FFFFFF'
TEXT = '#111827'
TEXT_MUTED = '#64748B'
BORDER = '#E2E8F0'
RED = '#DC2626'
AMBER = '#F59E0B'
BLUE = '#3B82F6'
st.markdown(f"""\n    <style>\n    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');\n\n    /* ---- Root overrides ---- */\n    html, body, .stApp {{\n        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;\n    }}\n    .stApp {{\n        background-color: {BG};\n    }}\n\n    /* ---- Sidebar styling ---- */\n    section[data-testid="stSidebar"] {{\n        background-color: {PRIMARY};\n        border-right: none;\n        overflow: hidden;\n    }}\n    section[data-testid="stSidebar"] > div:first-child {{\n        padding-top: 1rem;\n        padding-bottom: 0.5rem;\n        overflow: hidden;\n    }}\n    section[data-testid="stSidebar"] * {{\n        color: #CBD5E1 !important;\n    }}\n    section[data-testid="stSidebar"] .stRadio > div {{\n        gap: 0px;\n    }}\n    section[data-testid="stSidebar"] .stRadio label {{\n        color: #CBD5E1 !important;\n        padding: 5px 12px;\n        border-radius: 6px;\n        transition: all 0.2s ease;\n        margin-bottom: 0px;\n        font-size: 0.85rem;\n    }}\n    section[data-testid="stSidebar"] .stRadio label:hover {{\n        background-color: rgba(255,255,255,0.08);\n        color: #FFFFFF !important;\n    }}\n    section[data-testid="stSidebar"] .stRadio label[data-checked="true"],\n    section[data-testid="stSidebar"] [aria-checked="true"] + label {{\n        background-color: rgba(22, 163, 74, 0.15);\n        color: {ACCENT} !important;\n    }}\n    section[data-testid="stSidebar"] h1,\n    section[data-testid="stSidebar"] h2,\n    section[data-testid="stSidebar"] h3 {{\n        color: #FFFFFF !important;\n    }}\n    section[data-testid="stSidebar"] hr {{\n        border-color: rgba(255,255,255,0.1);\n        margin: 6px 0;\n    }}\n\n    /* ---- Aesthetics & Navigation ---- */\n    #MainMenu, footer {{\n        visibility: hidden;\n    }}\n    header[data-testid="stHeader"] {{\n        background: transparent !important;\n    }}\n    header[data-testid="stHeader"] * {{\n        color: {PRIMARY} !important;\n    }}\n\n    /* ---- KPI Cards ---- */\n    .kpi-card {{\n        background: {CARD_BG};\n        border: 1px solid {BORDER};\n        border-radius: 12px;\n        padding: 24px 20px;\n        text-align: center;\n        transition: box-shadow 0.2s ease, transform 0.2s ease;\n    }}\n    .kpi-card:hover {{\n        box-shadow: 0 4px 20px rgba(15, 23, 42, 0.08);\n        transform: translateY(-2px);\n    }}\n    .kpi-label {{\n        font-size: 0.8rem;\n        font-weight: 600;\n        text-transform: uppercase;\n        letter-spacing: 0.05em;\n        color: {TEXT_MUTED};\n        margin: 0 0 8px 0;\n    }}\n    .kpi-value {{\n        font-size: 1.85rem;\n        font-weight: 800;\n        color: {PRIMARY};\n        margin: 0;\n        line-height: 1.2;\n    }}\n    .kpi-value.green {{ color: {ACCENT}; }}\n    .kpi-value.red {{ color: {RED}; }}\n    .kpi-value.amber {{ color: {AMBER}; }}\n    .kpi-value.blue {{ color: {BLUE}; }}\n\n    /* ---- Section Cards (Targeting st.container border) ---- */\n    div[data-testid="stVerticalBlockBorderWrapper"] {{\n        background: {CARD_BG};\n        border: 1px solid {BORDER} !important;\n        border-radius: 12px !important;\n        padding: 24px !important;\n        margin-bottom: 20px;\n        box-shadow: 0 1px 3px rgba(0,0,0,0.02);\n    }}\n    .section-title {{\n        font-size: 1.1rem;\n        font-weight: 700;\n        color: {PRIMARY};\n        margin: 0 0 16px 0;\n        padding-bottom: 12px;\n        border-bottom: 1px solid {BORDER};\n    }}\n\n    /* ---- Page Header ---- */\n    .page-header {{\n        margin-bottom: 32px;\n    }}\n    .page-title {{\n        font-size: 1.8rem;\n        font-weight: 800;\n        color: {PRIMARY};\n        margin: 0 0 4px 0;\n        line-height: 1.3;\n    }}\n    .page-subtitle {{\n        font-size: 1rem;\n        color: {TEXT_MUTED};\n        margin: 0;\n        font-weight: 400;\n    }}\n\n    /* ---- Status Badges ---- */\n    .badge {{\n        display: inline-block;\n        padding: 4px 12px;\n        border-radius: 50px;\n        font-size: 0.78rem;\n        font-weight: 600;\n        letter-spacing: 0.02em;\n    }}\n    .badge-green {{\n        background: {ACCENT_LIGHT};\n        color: {ACCENT};\n    }}\n    .badge-red {{\n        background: #FEE2E2;\n        color: {RED};\n    }}\n    .badge-amber {{\n        background: #FEF3C7;\n        color: {AMBER};\n    }}\n    .badge-gray {{\n        background: #F1F5F9;\n        color: {TEXT_MUTED};\n    }}\n\n    /* ---- Info box (non-emoji) ---- */\n    .info-panel {{\n        background: #EFF6FF;\n        border: 1px solid #BFDBFE;\n        border-radius: 10px;\n        padding: 16px 20px;\n        color: #1E40AF;\n        font-size: 0.9rem;\n        line-height: 1.6;\n    }}\n\n    /* ---- Risk indicator ---- */\n    .risk-indicator {{\n        display: flex;\n        align-items: center;\n        gap: 8px;\n        padding: 12px 16px;\n        border-radius: 10px;\n        font-weight: 600;\n        font-size: 0.9rem;\n    }}\n    .risk-dot {{\n        width: 10px;\n        height: 10px;\n        border-radius: 50%;\n        display: inline-block;\n    }}\n\n    /* ---- Clean up streamlit defaults ---- */\n    .stMetric {{\n        background: {CARD_BG};\n        border: 1px solid {BORDER};\n        border-radius: 12px;\n        padding: 16px;\n    }}\n    .stDataFrame {{\n        border-radius: 8px;\n        overflow: hidden;\n    }}\n    div[data-testid="stMetricValue"] {{\n        font-weight: 700;\n    }}\n\n    /* ---- Buttons ---- */\n    .stButton > button {{\n        background: {ACCENT};\n        color: white;\n        border: none;\n        border-radius: 8px;\n        font-weight: 600;\n        padding: 10px 28px;\n        font-size: 0.9rem;\n        transition: all 0.2s ease;\n    }}\n    .stButton > button:hover {{\n        background: #15803D;\n        box-shadow: 0 4px 12px rgba(22, 163, 74, 0.3);\n        transform: translateY(-1px);\n    }}\n    .stButton > button:active {{\n        transform: translateY(0);\n    }}\n\n    /* ---- Select/Input ---- */\n    .stSelectbox > div > div,\n    .stFileUploader {{\n        border-radius: 8px !important;\n    }}\n\n    /* ---- Divider ---- */\n    hr {{\n        border-color: {BORDER};\n    }}\n    </style>\n    """, unsafe_allow_html=True)

def render_kpi(label, value, color_class=''):
    st.markdown(f'\n        <div class="kpi-card">\n            <p class="kpi-label">{label}</p>\n            <p class="kpi-value {color_class}">{value}</p>\n        </div>\n        ', unsafe_allow_html=True)

def render_page_header(title, subtitle=''):
    sub_html = f'<p class="page-subtitle">{subtitle}</p>' if subtitle else ''
    st.markdown(f'\n        <div class="page-header">\n            <p class="page-title">{title}</p>\n            {sub_html}\n        </div>\n        ', unsafe_allow_html=True)

def render_section(title, content_fn=None):
    container = st.container(border=True)
    with container:
        st.markdown(f'<p class="section-title">{title}</p>', unsafe_allow_html=True)
        if content_fn:
            content_fn()
    return container

def render_badge(text, variant='gray'):
    return f'<span class="badge badge-{variant}">{text}</span>'

def render_info(text):
    st.markdown(f'<div class="info-panel">{text}</div>', unsafe_allow_html=True)

def plotly_theme(fig, height=400):
    fig.update_layout(template='plotly_white', height=height, font=dict(family='Inter, sans-serif', size=13, color=TEXT), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=40, r=40, t=50, b=40), title_font=dict(size=15, color=PRIMARY, family='Inter, sans-serif'))
    fig.update_xaxes(gridcolor='#F1F5F9', linecolor=BORDER)
    fig.update_yaxes(gridcolor='#F1F5F9', linecolor=BORDER)
    return fig
STATE_KEYS = {'data': None, 'data_profile': None, 'model': None, 'metrics': None, 'bias_metrics': None, 'approval_rates': None, 'mitigated_model': None, 'mitigated_metrics': None, 'mitigated_bias_metrics': None, 'mitigated_approval_rates': None, 'X_train': None, 'X_test': None, 'y_train': None, 'y_test': None, 'sensitive_col': None, 'model_type': None, 'sf_test': None, 'sf_train': None, 'mitigation_method': None, 'report_text': None, 'report_pdf': None}
for key, default in STATE_KEYS.items():
    if key not in st.session_state:
        st.session_state[key] = default
with st.sidebar:
    st.markdown(f'\n        <div style="padding: 4px 4px 10px 4px;">\n            <h2 style="margin:0; font-size:1.15555rem; font-weight:800; color:#FFFFFF !important; letter-spacing:-0.02em; line-height:1.1;">\n                Fairness Audit Pipeline\n            </h2>\n            <p style="margin:2px 0 0 0; font-size:0.75rem; color:{TEXT_MUTED} !important; font-weight:400;">\n                Loan Approval Models\n            </p>\n        </div>\n        ', unsafe_allow_html=True)
    st.divider()
    page = st.radio('Navigation', ['Overview', 'Data Management', 'Model Training', 'Bias Analysis', 'Mitigation Engine', 'Performance Comparison', 'Explainability', 'Compliance Reports'], label_visibility='collapsed')
    st.divider()
    steps = {'Data Loaded': st.session_state.data is not None, 'Model Trained': st.session_state.model is not None, 'Bias Analyzed': st.session_state.bias_metrics is not None, 'Mitigation Applied': st.session_state.mitigated_model is not None}
    status_items = ''
    for step_name, completed in steps.items():
        icon = '&#9679;' if completed else '&#9675;'
        color = ACCENT if completed else '#475569'
        status_items += f'<p style="font-size:0.78rem; margin:2px 0; color:{color} !important;">{icon}&nbsp;&nbsp;{step_name}</p>'
    st.markdown(f'<p style="font-size:0.7rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; color:#94A3B8 !important; margin-bottom:4px;">Pipeline Status</p>{status_items}', unsafe_allow_html=True)

def page_overview():
    render_page_header('AI Fairness Audit Dashboard', 'Loan Model Bias Detection and Compliance Monitoring')
    render_info('Welcome to the <b>Fairness Audit & Bias Mitigation Pipeline</b> -- the enterprise-grade fairness audit platform. Navigate through the pipeline using the sidebar: upload data, train a model, detect bias, apply mitigation, compare results, and generate regulatory compliance reports. All metrics update in real-time across the dashboard.')
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Key Performance Indicators</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.session_state.metrics:
            val = f"{st.session_state.metrics['Accuracy']:.1%}"
            render_kpi('Model Accuracy', val, 'green')
        else:
            render_kpi('Model Accuracy', '--', '')
    with c2:
        if st.session_state.bias_metrics:
            di = st.session_state.bias_metrics['Disparate Impact']
            _, color_hex = classify_risk(di)
            color_map = {ACCENT: 'green', RED: 'red', AMBER: 'amber'}
            render_kpi('Fairness Score', f'{di:.3f}', color_map.get(color_hex, ''))
        else:
            render_kpi('Fairness Score', '--', '')
    with c3:
        if st.session_state.bias_metrics:
            dpd = st.session_state.bias_metrics['Demographic Parity Difference']
            render_kpi('Parity Difference', f'{abs(dpd):.3f}', 'blue')
        else:
            render_kpi('Parity Difference', '--', '')
    with c4:
        if st.session_state.mitigated_model:
            render_kpi('Compliance Status', 'Compliant', 'green')
        elif st.session_state.bias_metrics:
            di = st.session_state.bias_metrics['Disparate Impact']
            label, _ = classify_risk(di)
            clr = 'red' if 'High' in label else 'amber' if 'Moderate' in label else 'green'
            render_kpi('Compliance Status', label, clr)
        else:
            render_kpi('Compliance Status', 'Pending', '')
    st.markdown('<br>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(['Pipeline Architecture', 'Regulatory Framework'])
    with tab1:
        pipeline_steps = [('01', 'Data Ingestion', 'Upload and profile loan datasets'), ('02', 'Model Training', 'Train classification models with validation'), ('03', 'Bias Detection', 'Audit for demographic disparities'), ('04', 'Mitigation', 'Apply fairness-aware retraining'), ('05', 'Comparison', 'Before vs after performance analysis'), ('06', 'Explainability', 'SHAP-based decision interpretation'), ('07', 'Reporting', 'Generate compliance documentation')]
        steps_html = ''.join((f'<div style="padding:12px 10px; border-bottom:1px solid {BORDER};"><p style="margin:0; font-size:1rem; color:{TEXT};"><span style="font-weight:800; color:{ACCENT}; margin-right:12px;">{num}</span><b>{name}</b></p><p style="margin:6px 0 0 35px; font-size:0.88rem; color:{TEXT_MUTED};">{desc}</p></div>' for num, name, desc in pipeline_steps))
        st.markdown(f'<div style="background:{CARD_BG}; border:1px solid {BORDER}; border-radius:12px; padding:20px;">{steps_html}</div>', unsafe_allow_html=True)
    with tab2:
        frameworks = [('ECOA', 'Equal Credit Opportunity Act'), ('FHA', 'Fair Housing Act'), ('EEOC', 'Four-Fifths Rule (Disparate Impact)'), ('EU AI Act', 'High-Risk System Requirements'), ('SR 11-7', 'Federal Reserve Model Risk Guidance'), ('OCC 2011-12', 'Model Risk Management'), ('CFPB', 'Consumer Financial Protection Bureau')]
        fw_html = ''.join((f'<div style="padding:12px 10px; border-bottom:1px solid {BORDER};"><p style="margin:0; font-size:1rem; color:{TEXT};"><span style="font-weight:800; color:{PRIMARY}; margin-right:15px; min-width:80px; display:inline-block;">{abbr}</span>{full_name}</p></div>' for abbr, full_name in frameworks))
        st.markdown(f'<div style="background:{CARD_BG}; border:1px solid {BORDER}; border-radius:12px; padding:20px;">{fw_html}</div>', unsafe_allow_html=True)

def page_data_management():
    render_page_header('Data Management', 'Upload, inspect, and profile loan application datasets')
    col_upload, col_sample, col_kaggle = st.columns([2, 1, 1])
    with col_upload:
        uploaded_file = st.file_uploader('Upload Local Dataset (CSV)', type='csv', label_visibility='visible')
    with col_sample:
        st.markdown('<br>', unsafe_allow_html=True)
        load_sample = st.button('Load Mock Data', width='stretch')
    with col_kaggle:
        st.markdown('<br>', unsafe_allow_html=True)
        fetch_kaggle = st.button('Fetch Real-Time Data', width='stretch', help='Sync direct from Kaggle repository.')
    if uploaded_file or load_sample or fetch_kaggle:
        try:
            if uploaded_file:
                st.session_state.data = pd.read_csv(uploaded_file)
            elif load_sample:
                st.session_state.data = pd.read_csv('data/loan_data.csv')
            elif fetch_kaggle:
                with st.spinner('Downloading high-fidelity Lending Club data (Kaggle)...'):
                    import kagglehub
                    import os
                    path = kagglehub.dataset_download('adarshsng/lending-club-loan-data-csv')
                    files = [f for f in os.listdir(path) if f.endswith('.csv')]
                    if not files:
                        st.error('Kaggle download successful, but no CSV file was found in the archive.')
                        return
                    CSV_PATH = os.path.join(path, files[0])
                    df = pd.read_csv(CSV_PATH, nrows=200000)
                    st.session_state.data = df
                    st.info(f'Synchronized top 200,000 records from the 1.7GB Lending Club repository for memory stability.')
            st.session_state.data_profile = get_data_profile(st.session_state.data)
            st.success('Target environment synchronized with real-time data source.')
        except Exception as e:
            st.error(f'Error loading data: {e}. Check your internet or kagglehub installation.')
    if st.session_state.data is not None:
        df = st.session_state.data
        profile = st.session_state.data_profile or get_data_profile(df)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_kpi('Total Records', f"{profile['row_count']:,}")
        with c2:
            render_kpi('Features', str(profile['col_count']))
        with c3:
            render_kpi('Missing Values', f"{profile['total_missing']:,}")
        with c4:
            completeness = (1 - profile['total_missing'] / (profile['row_count'] * profile['col_count'])) * 100
            render_kpi('Completeness', f'{completeness:.1f}%', 'green')
        st.markdown('<br>', unsafe_allow_html=True)
        col_preview, col_profile = st.columns([3, 2])
        with col_preview:
            with st.container(border=True):
                st.markdown('<p class="section-title">Data Preview</p>', unsafe_allow_html=True)
                st.dataframe(df.head(10), width='stretch', hide_index=True)
        with col_profile:
            with st.container(border=True):
                st.markdown('<p class="section-title">Column Profile</p>', unsafe_allow_html=True)
                type_data = []
                for col_name in df.columns:
                    dtype_str = str(df[col_name].dtype)
                    missing_count = profile['missing_counts'].get(col_name, 0)
                    missing_pct_val = profile['missing_pct'].get(col_name, 0.0)
                    type_data.append({'Column': col_name, 'Type': dtype_str, 'Missing': f'{missing_count} ({missing_pct_val:.1f}%)'})
                st.dataframe(pd.DataFrame(type_data), width='stretch', hide_index=True)
        missing_cols = {k: v for k, v in profile['missing_counts'].items() if v > 0}
        if missing_cols:
            with st.container(border=True):
                st.markdown('<p class="section-title">Missing Values Distribution</p>', unsafe_allow_html=True)
                fig = go.Figure(go.Bar(x=list(missing_cols.keys()), y=list(missing_cols.values()), marker_color=ACCENT, text=list(missing_cols.values()), textposition='auto'))
                fig.update_layout(xaxis_title='Column', yaxis_title='Missing Count')
                st.plotly_chart(plotly_theme(fig, 350), width='stretch')

def page_model_training():
    render_page_header('Model Training', 'Configure and train a classification model for loan approvals')
    if st.session_state.data is None:
        render_info('Please load a dataset in <b>Data Management</b> before training a model.')
        return
    df = st.session_state.data
    with st.container(border=True):
        st.markdown('<p class="section-title">Training Configuration</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        all_cols = df.columns.tolist()
        target_keys_fixed = ['loan amount', 'loan_amount']
        sens_keys_fixed = ['age', 'gender', 'genter']
        target_keys_backup = ['loan_status', 'loan', 'status', 'approve', 'default', 'target', 'y']
        sens_keys_backup = ['sex', 'race', 'ethnicity', 'religion', 'marital', 'citizenship']
        target_col = all_cols[-1]
        found_target = False
        for c in all_cols:
            if c.lower() in target_keys_fixed:
                target_col = c
                found_target = True
                break
        if not found_target:
            for c in all_cols:
                c_low = c.lower().replace('_', ' ')
                if any((k in c_low for k in target_keys_backup)):
                    target_col = c
                    break
        sens_keys_strict = ['age', 'gender', 'genter']
        target_keys_fixed = ['loan amount', 'loan_amount']
        target_keys_backup = ['loan_status', 'loan', 'status', 'approve', 'default', 'target', 'y']
        target_col = all_cols[-1]
        found_target = False
        for c in all_cols:
            if c.lower() in target_keys_fixed:
                target_col = c
                found_target = True
                break
        if not found_target:
            for c in all_cols:
                if any((k in c.lower().replace('_', ' ') for k in target_keys_backup)):
                    target_col = c
                    break
        sensitive_cols = []
        for c in all_cols:
            c_low = c.lower().replace('_', ' ')
            if any((k in c_low for k in sens_keys_strict)):
                sensitive_cols.append(c)
        if not sensitive_cols:
            st.error('No demographic attributes (Age or Gender) found in dataset for auditing.')
            return
        if not target_col or not sensitive_cols:
            st.error('Automated detection failed. Please ensure your dataset has clear column names.')
            return
        selected_features = [c for c in all_cols if c != target_col]
        with st.expander('Custom Configuration (Manual Override)', expanded=False):
            c_ed1, c_ed2 = st.columns(2)
            with c_ed1:
                target_col = st.selectbox('Override Target Column', all_cols, index=all_cols.index(target_col), help='Choose the outcome you want to predict.')
            with c_ed2:
                sensitive_cols = st.multiselect('Override Audited Attributes', options=[c for c in all_cols if c != target_col], default=sensitive_cols, help='Choose one or more columns to check for bias.')
            if not sensitive_cols:
                st.error('Select at least one attribute to audit.')
                st.stop()
        st.markdown(f"""\n            <div style="background:{BG}; border: 1px solid {BORDER}; border-radius: 12px; padding: 25px; text-align: center; margin-bottom: 25px;">\n                <div style="display: flex; justify-content: space-around; align-items: center;">\n                    <div style="text-align: center;">\n                        <p style="font-size: 0.8rem; font-weight: 700; color: {TEXT_MUTED}; margin-bottom: 5px;">STATUS</p>\n                        <p style="font-size: 0.8rem; font-weight: 700; color: {ACCENT}; text-transform: uppercase;">Data Ready</p>\n                    </div>\n                    <div style="text-align: center; border-left: 1px solid {BORDER}; padding-left: 15px;">\n                        <p style="font-size: 0.8rem; font-weight: 700; color: {TEXT_MUTED}; margin-bottom: 5px;">TARGET</p>\n                        <p style="font-size: 0.8rem; font-weight: 700; color: {PRIMARY}; text-transform: uppercase;">{target_col.replace('_', ' ')}</p>\n                    </div>\n                    <div style="text-align: center; border-left: 1px solid {BORDER}; padding-left: 15px;">\n                        <p style="font-size: 0.8rem; font-weight: 700; color: {TEXT_MUTED}; margin-bottom: 5px;">AUDITING</p>\n                        <p style="font-size: 0.8rem; font-weight: 700; color: {PRIMARY}; text-transform: uppercase;">{', '.join([c.replace('_', ' ') for c in sensitive_cols[:3]])}</p>\n                    </div>\n                </div>\n            </div>\n            """, unsafe_allow_html=True)
        c_btn, c_set = st.columns([2, 1])
        with c_btn:
            train_clicked = st.button('LAUNCH TRAINING PIPELINE', width='stretch')
        with c_set:
            with st.popover('Advanced Settings'):
                model_type = st.selectbox('Algorithm Selection', ['Random Forest', 'Logistic Regression'], index=0)
                st.info('Random Forest is recommended for higher precision.')
    if train_clicked:
        with st.spinner('Executing fairness-aware training on all columns...'):
            MAX_ROWS = 100000
            df_to_train = df.copy()
            if len(df_to_train) > MAX_ROWS:
                st.warning(f'Dataset is large ({len(df_to_train):,} rows). Subsampling to {MAX_ROWS:,} for analysis.')
                df_to_train = df_to_train.sample(MAX_ROWS, random_state=42)
            primary_sens = sensitive_cols[0]
            try:
                df_subset = df_to_train[selected_features + [target_col]]
                X, y, sf_proc, encoders, sf_raw = preprocess_data(df_subset, target_col=target_col, sensitive_col=primary_sens)
                with st.spinner('Executing model training...'):
                    model, metrics, X_test, y_test, y_pred, X_train, y_train, sf_train, sf_test = train_model(X, y, sensitive_features=sf_raw, model_type=model_type)
                st.session_state.model = model
                st.session_state.metrics = metrics
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.X_train = X_train
                st.session_state.y_train = y_train
                st.session_state.sf_test = sf_test
                st.session_state.sf_train = sf_train
                st.session_state.model_type = model_type
                st.session_state.sensitive_col = primary_sens
                st.session_state.all_sensitive_cols = sensitive_cols
                st.session_state.bias_metrics = None
                st.session_state.approval_rates = None
                st.session_state.mitigated_model = None
                st.session_state.mitigated_metrics = None
                st.session_state.mitigated_bias_metrics = None
                st.session_state.mitigated_approval_rates = None
                st.session_state.mitigation_method = None
                st.session_state.report_text = None
                st.success(f'{model_type} trained successfully on {len(X_train)} samples.')
            except Exception as e:
                st.error(f'Training failed: {e}')
                return
    if st.session_state.metrics:
        st.markdown('<br>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<p class="section-title">Model Performance Metrics</p>', unsafe_allow_html=True)
            m = st.session_state.metrics
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                render_kpi('Accuracy', f"{m['Accuracy']:.2%}", 'green')
            with c2:
                render_kpi('Precision', f"{m['Precision']:.2%}", 'blue')
            with c3:
                render_kpi('Recall', f"{m['Recall']:.2%}", 'blue')
            with c4:
                render_kpi('F1 Score', f"{m['F1 Score']:.2%}", 'green')
            st.markdown('<br>', unsafe_allow_html=True)
            metric_names = list(m.keys())
            metric_vals = list(m.values())
            fig = go.Figure(go.Bar(x=metric_names, y=metric_vals, marker_color=[ACCENT, BLUE, BLUE, ACCENT], text=[f'{v:.2%}' for v in metric_vals], textposition='auto'))
            fig.update_layout(title='Performance Summary', xaxis_title='Metric', yaxis_title='Score', yaxis=dict(range=[0, 1]))
            st.plotly_chart(plotly_theme(fig), width='stretch')

def page_bias_analysis():
    render_page_header('Bias Analysis', 'Audit model predictions for demographic disparities')
    if st.session_state.model is None or st.session_state.sf_test is None:
        render_info('Please train a model in <b>Model Training</b> before running bias analysis.')
        return
    all_sens_raw = st.session_state.get('all_sensitive_cols', [st.session_state.sensitive_col])
    priority_keys = ['age', 'gender', 'genter']
    all_sens = [c for c in all_sens_raw if any((k in c.lower() for k in priority_keys))]
    if not all_sens:
        all_sens = all_sens_raw
    with st.container(border=True):
        st.markdown('<p class="section-title">Audit Dimensionality</p>', unsafe_allow_html=True)
        selected_attrs = st.multiselect('Select Dimensions to Audit', options=all_sens, default=[all_sens[0]] if all_sens else [], format_func=lambda x: x.replace('_', ' ').upper())
        if not selected_attrs:
            st.error('Please select at least one dimension (Age or Gender) to perform the audit.')
            return
        run_audit = st.button('RUN FAIRNESS AUDIT', width='stretch')
    if run_audit:
        with st.spinner('Analyzing model disparities...'):
            audit_results = {}
            df_full = st.session_state.data
            X_test_indices = st.session_state.X_test.index
            y_pred = st.session_state.model.predict(st.session_state.X_test)
            for attr in selected_attrs:
                sf_raw_active = df_full.loc[X_test_indices, attr]
                bias_metrics, approval_rates = detect_bias(st.session_state.y_test, y_pred, sf_raw_active)
                audit_results[attr] = {'metrics': bias_metrics, 'rates': approval_rates}
            st.session_state.audit_results = audit_results
            st.session_state.bias_metrics = audit_results[selected_attrs[0]]['metrics']
            st.session_state.approval_rates = audit_results[selected_attrs[0]]['rates']
            st.session_state.active_audit_col = selected_attrs[0]
    if st.session_state.get('audit_results'):
        results = st.session_state.audit_results
        for idx, (attr_name, data) in enumerate(results.items()):
            attr_label = attr_name.replace('_', ' ').upper()
            st.markdown(f'---')
            st.markdown(f'<h3 style="color:{PRIMARY}; margin-bottom:25px;">AUDIT REPORT: {attr_label}</h3>', unsafe_allow_html=True)
            bm = data['metrics']
            apr = data['rates']
            di = bm['Disparate Impact']
            risk_label, risk_color = classify_risk(di)
            badge_variant = 'green' if 'Fair' in risk_label else 'red' if 'High' in risk_label else 'amber'
            st.markdown(f'<div style="background:{CARD_BG}; border: 1px solid {BORDER}; border-left: 4px solid {risk_color}; border-radius: 12px; padding: 20px; margin-bottom: 20px;"><p style="margin:0; font-size:1rem; font-weight:700; color:{PRIMARY};">Risk Assessment ({attr_label}): {render_badge(risk_label, badge_variant)}</p></div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            color_map = {ACCENT: 'green', RED: 'red', AMBER: 'amber'}
            with c1:
                render_kpi('Disparate Impact', f'{di:.3f}', color_map.get(risk_color, ''))
            with c2:
                render_kpi('Demographic Parity Diff', f"{abs(bm['Demographic Parity Difference']):.3f}", 'blue')
            with c3:
                render_kpi('Equal Opportunity Diff', f"{abs(bm['Equal Opportunity Difference']):.3f}", 'blue')
            st.markdown('<br>', unsafe_allow_html=True)
            if apr:
                with st.container(border=True):
                    st.markdown(f'<p class="section-title">Group-wise Selection Rates by {attr_label}</p>', unsafe_allow_html=True)
                    groups = list(apr.keys())
                    values = list(apr.values())
                    colors = []
                    max_rate = max(values) if values else 1
                    for v in values:
                        ratio = v / max_rate if max_rate > 0 else 1
                        if ratio < 0.8:
                            colors.append(RED)
                        elif ratio < 0.9:
                            colors.append(AMBER)
                        else:
                            colors.append(ACCENT)
                    fig = go.Figure(go.Bar(x=groups, y=values, marker_color=colors, text=[f'{v:.1%}' for v in values], textposition='auto'))
                    fig.update_layout(title=f'Approval Rate by {attr_label}', xaxis_title='Group', yaxis_title='Approval Rate', yaxis=dict(range=[0, 1], tickformat='.0%'))
                    fig.add_hline(y=max_rate * 0.8, line_dash='dash', line_color=RED, annotation_text='80% Threshold (Four-Fifths Rule)', annotation_position='top left')
                    st.plotly_chart(plotly_theme(fig), width='stretch', key=f'chart_{attr_name}_{idx}')
        with st.container(border=True):
            st.markdown('<p class="section-title">Compliance Decision Logic</p>', unsafe_allow_html=True)
            logic_data = [{'Range': 'DI < 0.80', 'Classification': 'High Risk', 'Action': 'Mitigation required'}, {'Range': '0.80 - 0.90', 'Classification': 'Moderate Risk', 'Action': 'Monitoring advised'}, {'Range': 'DI > 0.90', 'Classification': 'Fair', 'Action': 'Within compliance bounds'}]
            st.dataframe(pd.DataFrame(logic_data), width='stretch', hide_index=True)

def page_mitigation():
    render_page_header('Mitigation Engine', 'Apply fairness-aware retraining techniques to reduce model bias')
    if st.session_state.model is None:
        render_info('Please train a model in <b>Model Training</b> first.')
        return
    if st.session_state.bias_metrics is None:
        render_info('Please run <b>Bias Analysis</b> first to establish a baseline.')
        return
    with st.container(border=True):
        st.markdown('<p class="section-title">Mitigation Configuration</p>', unsafe_allow_html=True)
        c1, c2 = st.columns([2, 1])
        with c1:
            method = st.selectbox('Mitigation Technique', ['Exponentiated Gradient', 'Reweighing'])
        with c2:
            st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        apply_clicked = st.button('Apply Mitigation', width='content')
    if apply_clicked:
        with st.spinner('Applying mitigation and retraining model... This may take a moment.'):
            try:
                mit_model = mitigate_bias(st.session_state.X_train, st.session_state.y_train, st.session_state.sf_train, model_type=st.session_state.model_type, method=method)
                st.session_state.mitigated_model = mit_model
                st.session_state.mitigation_method = method
                y_pred_mit = mit_model.predict(st.session_state.X_test)
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                new_metrics = {'Accuracy': accuracy_score(st.session_state.y_test, y_pred_mit), 'Precision': precision_score(st.session_state.y_test, y_pred_mit, zero_division=0), 'Recall': recall_score(st.session_state.y_test, y_pred_mit, zero_division=0), 'F1 Score': f1_score(st.session_state.y_test, y_pred_mit, zero_division=0)}
                st.session_state.mitigated_metrics = new_metrics
                new_bias, new_rates = detect_bias(st.session_state.y_test, y_pred_mit, st.session_state.sf_test)
                st.session_state.mitigated_bias_metrics = new_bias
                st.session_state.mitigated_approval_rates = new_rates
                st.success('Mitigation applied successfully.')
            except Exception as e:
                st.error(f'Mitigation failed: {e}')
    if st.session_state.mitigated_bias_metrics:
        mbm = st.session_state.mitigated_bias_metrics
        mm = st.session_state.mitigated_metrics
        di_new = mbm['Disparate Impact']
        risk_label, risk_color = classify_risk(di_new)
        badge_variant = 'green' if 'Fair' in risk_label else 'red' if 'High' in risk_label else 'amber'
        st.markdown(f'<div style="background:{CARD_BG}; border: 1px solid {BORDER}; border-left: 4px solid {risk_color}; border-radius: 12px; padding: 20px; margin-bottom: 20px;"><p style="margin:0; font-size:1rem; font-weight:700; color:{PRIMARY};">Post-Mitigation Status: {render_badge(risk_label, badge_variant)}</p></div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            color_map = {ACCENT: 'green', RED: 'red', AMBER: 'amber'}
            render_kpi('New Disparate Impact', f'{di_new:.3f}', color_map.get(risk_color, ''))
        with c2:
            render_kpi('New Accuracy', f"{mm['Accuracy']:.2%}", 'blue')
        with c3:
            render_kpi('New Precision', f"{mm['Precision']:.2%}", 'blue')
        with c4:
            render_kpi('New F1 Score', f"{mm['F1 Score']:.2%}", 'green')

def page_comparison():
    render_page_header('Performance Comparison', 'Before vs after mitigation: Performance and fairness trade-off analysis')
    if st.session_state.mitigated_metrics is None:
        render_info('Apply mitigation in the <b>Mitigation Engine</b> to view comparison results.')
        return
    m_before = st.session_state.metrics
    m_after = st.session_state.mitigated_metrics
    b_before = st.session_state.bias_metrics
    b_after = st.session_state.mitigated_bias_metrics
    col_left, col_right = st.columns(2)
    with col_left:
        with st.container(border=True):
            st.markdown('<p class="section-title">Before Mitigation</p>', unsafe_allow_html=True)
            for metric_name, val in m_before.items():
                st.markdown(f'<p style="margin:8px 0; font-size:0.9rem; color:{TEXT};"><b>{metric_name}</b>: {val:.4f}</p>', unsafe_allow_html=True)
            st.divider()
            for metric_name, val in b_before.items():
                st.markdown(f'<p style="margin:8px 0; font-size:0.9rem; color:{TEXT};"><b>{metric_name}</b>: {val:.4f}</p>', unsafe_allow_html=True)
    with col_right:
        with st.container(border=True):
            st.markdown('<p class="section-title">After Mitigation</p>', unsafe_allow_html=True)
            for metric_name, val in m_after.items():
                st.markdown(f'<p style="margin:8px 0; font-size:0.9rem; color:{TEXT};"><b>{metric_name}</b>: {val:.4f}</p>', unsafe_allow_html=True)
            st.divider()
            for metric_name, val in b_after.items():
                st.markdown(f'<p style="margin:8px 0; font-size:0.9rem; color:{TEXT};"><b>{metric_name}</b>: {val:.4f}</p>', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<p class="section-title">Performance Metrics Comparison</p>', unsafe_allow_html=True)
        perf_metrics = list(m_before.keys())
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Before Mitigation', x=perf_metrics, y=[m_before[k] for k in perf_metrics], marker_color='#94A3B8', text=[f'{m_before[k]:.2%}' for k in perf_metrics], textposition='auto'))
        fig.add_trace(go.Bar(name='After Mitigation', x=perf_metrics, y=[m_after[k] for k in perf_metrics], marker_color=ACCENT, text=[f'{m_after[k]:.2%}' for k in perf_metrics], textposition='auto'))
        fig.update_layout(barmode='group', title='Model Performance: Before vs After', xaxis_title='Metric', yaxis_title='Score', yaxis=dict(range=[0, 1]), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
        st.plotly_chart(plotly_theme(fig, 420), width='stretch')
    with st.container(border=True):
        st.markdown('<p class="section-title">Fairness Metrics Comparison</p>', unsafe_allow_html=True)
        bias_metric_names = list(b_before.keys())
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name='Before Mitigation', x=bias_metric_names, y=[abs(b_before[k]) for k in bias_metric_names], marker_color='#94A3B8', text=[f'{abs(b_before[k]):.3f}' for k in bias_metric_names], textposition='auto'))
        fig2.add_trace(go.Bar(name='After Mitigation', x=bias_metric_names, y=[abs(b_after[k]) for k in bias_metric_names], marker_color=ACCENT, text=[f'{abs(b_after[k]):.3f}' for k in bias_metric_names], textposition='auto'))
        fig2.update_layout(barmode='group', title='Fairness Metrics: Before vs After', xaxis_title='Metric', yaxis_title='Value', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
        fig2.add_hline(y=0.8, line_dash='dash', line_color=RED, annotation_text='Compliance Threshold (0.8)', annotation_position='top left')
        st.plotly_chart(plotly_theme(fig2, 420), width='stretch')
    with st.container(border=True):
        st.markdown('<p class="section-title">Impact Summary</p>', unsafe_allow_html=True)
        acc_delta = m_after['Accuracy'] - m_before['Accuracy']
        di_delta = b_after['Disparate Impact'] - b_before['Disparate Impact']
        dpd_delta = abs(b_after['Demographic Parity Difference']) - abs(b_before['Demographic Parity Difference'])
        impact_data = [{'Metric': 'Accuracy Change', 'Delta': f'{acc_delta:+.4f}', 'Direction': 'Improved' if acc_delta >= 0 else 'Decreased'}, {'Metric': 'Disparate Impact Change', 'Delta': f'{di_delta:+.4f}', 'Direction': 'Improved' if di_delta > 0 else 'Decreased'}, {'Metric': 'Parity Difference Change', 'Delta': f'{dpd_delta:+.4f}', 'Direction': 'Improved' if dpd_delta < 0 else 'Increased'}]
        st.dataframe(pd.DataFrame(impact_data), width='stretch', hide_index=True)

def page_explainability():
    render_page_header('Model Explainability', 'SHAP-based feature importance and decision interpretation')
    if st.session_state.model is None:
        render_info('Please train a model in <b>Model Training</b> first.')
        return
    model_to_explain = st.session_state.mitigated_model if st.session_state.mitigated_model else st.session_state.model
    model_label = 'Mitigated Model' if st.session_state.mitigated_model else 'Baseline Model'
    with st.container(border=True):
        st.markdown('<p class="section-title">SHAP Analysis Configuration</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:0.9rem; color:{TEXT_MUTED};">Analyzing: <b style="color:{PRIMARY};">{model_label}</b> ({st.session_state.model_type})</p>', unsafe_allow_html=True)
        compute_clicked = st.button('Compute SHAP Values', width='content')
    if compute_clicked:
        with st.spinner('Computing SHAP values... This may take a moment.'):
            try:
                shap_values, X_sample = compute_shap_values(model_to_explain, st.session_state.X_train, st.session_state.X_test, st.session_state.model_type)
                feature_names = list(st.session_state.X_test.columns) if hasattr(st.session_state.X_test, 'columns') else [f'Feature {i}' for i in range(st.session_state.X_test.shape[1])]
                importance_df = get_feature_importance(shap_values, feature_names)
                st.session_state['shap_importance'] = importance_df
                st.markdown('<br>', unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown('<p class="section-title">Feature Importance (Mean |SHAP|)</p>', unsafe_allow_html=True)
                    top_n = min(15, len(importance_df))
                    top_features = importance_df.head(top_n).iloc[::-1]
                    fig = go.Figure(go.Bar(x=top_features['Importance'], y=top_features['Feature'], orientation='h', marker_color=ACCENT, text=[f'{v:.4f}' for v in top_features['Importance']], textposition='auto'))
                    fig.update_layout(title='Top Feature Contributions to Model Decisions', xaxis_title='Mean |SHAP Value|', yaxis_title='Feature')
                    st.plotly_chart(plotly_theme(fig, 500), width='stretch')
                with st.container(border=True):
                    st.markdown('<p class="section-title">SHAP Summary Plot</p>', unsafe_allow_html=True)
                    summary_fig = generate_shap_summary_plot(model_to_explain, st.session_state.X_train, st.session_state.X_test, st.session_state.model_type)
                    st.pyplot(summary_fig)
            except Exception as e:
                st.error(f'SHAP computation failed: {e}')

def page_reports():
    render_page_header('Compliance Reports', 'Generate structured regulatory compliance documentation')
    if st.session_state.metrics is None or st.session_state.bias_metrics is None:
        render_info('Complete <b>Model Training</b> and <b>Bias Analysis</b> to generate a compliance report.')
        return
    with st.container(border=True):
        st.markdown('<p class="section-title">Report Configuration</p>', unsafe_allow_html=True)
        report_sections = ['Model performance metrics', 'Bias detection results', 'Mitigation outcomes (if applied)', 'Compliance interpretation', 'Regulatory references']
        st.markdown(f'<p style="font-size:0.9rem; color:{TEXT_MUTED}; margin-bottom:8px;">Report will include:</p>', unsafe_allow_html=True)
        for section in report_sections:
            st.markdown(f'<p style="margin:4px 0; font-size:0.88rem; color:{TEXT};"><span style="color:{ACCENT}; font-weight:700;">&#8226;</span>&nbsp;&nbsp;{section}</p>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        generate_clicked = st.button('Generate Report', width='content')
    if generate_clicked:
        with st.spinner('Generating compliance reports...'):
            report_text = generate_report(st.session_state.metrics, st.session_state.bias_metrics, st.session_state.mitigated_metrics, st.session_state.mitigated_bias_metrics, sensitive_col=st.session_state.sensitive_col, model_type=st.session_state.model_type, mitigation_method=st.session_state.mitigation_method)
            st.session_state.report_text = report_text
            report_pdf = generate_pdf_report(st.session_state.metrics, st.session_state.bias_metrics, st.session_state.mitigated_metrics, st.session_state.mitigated_bias_metrics, sensitive_col=st.session_state.sensitive_col, model_type=st.session_state.model_type, mitigation_method=st.session_state.mitigation_method)
            st.session_state.report_pdf = report_pdf
    if st.session_state.report_text:
        st.markdown('<br>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<p class="section-title">Report Preview</p>', unsafe_allow_html=True)
            st.text_area('Report content', st.session_state.report_text, height=400, label_visibility='collapsed')
        st.markdown('<br>', unsafe_allow_html=True)
        c1, c2, c3, _ = st.columns([1.2, 1.2, 1.2, 2])
        with c1:
            if st.session_state.report_pdf:
                st.download_button(label='Download Official PDF', data=st.session_state.report_pdf, file_name='LoanGuard_Compliance_Report.pdf', mime='application/pdf', width='stretch')
            else:
                st.info('PDF Generation requires fpdf2')
        with c2:
            st.download_button(label='Download Markdown', data=st.session_state.report_text, file_name='LoanGuard_Compliance_Report.md', mime='text/markdown', width='stretch')
        with c3:
            st.download_button(label='Download Plain Text', data=st.session_state.report_text, file_name='LoanGuard_Compliance_Report.txt', mime='text/plain', width='stretch')
PAGES = {'Overview': page_overview, 'Data Management': page_data_management, 'Model Training': page_model_training, 'Bias Analysis': page_bias_analysis, 'Mitigation Engine': page_mitigation, 'Performance Comparison': page_comparison, 'Explainability': page_explainability, 'Compliance Reports': page_reports}
PAGES[page]()
