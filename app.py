"""
Automated Fairness Audit & Bias Mitigation Pipeline for Loan Approval Models
Enterprise-grade loan model compliance monitoring platform.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import base64
from io import BytesIO

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Fairness Audit Pipeline | Compliance Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- IMPORTS ----------
from utils.preprocessing import preprocess_data, get_data_profile
from utils.training import train_model, evaluate_model
from utils.bias_detection import detect_bias, classify_risk
from utils.mitigation import mitigate_bias
from utils.explainability import (
    compute_shap_values,
    get_feature_importance,
    generate_shap_summary_plot,
)
from utils.reporting import generate_report, generate_pdf_report

# ---------- DESIGN TOKENS ----------
PRIMARY = "#0F172A"
ACCENT = "#16A34A"
ACCENT_LIGHT = "#DCFCE7"
BG = "#F8FAFC"
CARD_BG = "#FFFFFF"
TEXT = "#111827"
TEXT_MUTED = "#64748B"
BORDER = "#E2E8F0"
RED = "#DC2626"
AMBER = "#F59E0B"
BLUE = "#3B82F6"

# ---------- GLOBAL STYLES ----------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* ---- Root overrides ---- */
    html, body, .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    .stApp {{
        background-color: {BG};
    }}

    /* ---- Sidebar styling ---- */
    section[data-testid="stSidebar"] {{
        background-color: {PRIMARY};
        border-right: none;
        overflow: hidden;
    }}
    section[data-testid="stSidebar"] > div:first-child {{
        padding-top: 1rem;
        padding-bottom: 0.5rem;
        overflow: hidden;
    }}
    section[data-testid="stSidebar"] * {{
        color: #CBD5E1 !important;
    }}
    section[data-testid="stSidebar"] .stRadio > div {{
        gap: 0px;
    }}
    section[data-testid="stSidebar"] .stRadio label {{
        color: #CBD5E1 !important;
        padding: 5px 12px;
        border-radius: 6px;
        transition: all 0.2s ease;
        margin-bottom: 0px;
        font-size: 0.85rem;
    }}
    section[data-testid="stSidebar"] .stRadio label:hover {{
        background-color: rgba(255,255,255,0.08);
        color: #FFFFFF !important;
    }}
    section[data-testid="stSidebar"] .stRadio label[data-checked="true"],
    section[data-testid="stSidebar"] [aria-checked="true"] + label {{
        background-color: rgba(22, 163, 74, 0.15);
        color: {ACCENT} !important;
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: #FFFFFF !important;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.1);
        margin: 6px 0;
    }}

    /* ---- Hide default elements ---- */
    #MainMenu, header, footer {{
        visibility: hidden;
    }}

    /* ---- KPI Cards ---- */
    .kpi-card {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 24px 20px;
        text-align: center;
        transition: box-shadow 0.2s ease, transform 0.2s ease;
    }}
    .kpi-card:hover {{
        box-shadow: 0 4px 20px rgba(15, 23, 42, 0.08);
        transform: translateY(-2px);
    }}
    .kpi-label {{
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {TEXT_MUTED};
        margin: 0 0 8px 0;
    }}
    .kpi-value {{
        font-size: 1.85rem;
        font-weight: 800;
        color: {PRIMARY};
        margin: 0;
        line-height: 1.2;
    }}
    .kpi-value.green {{ color: {ACCENT}; }}
    .kpi-value.red {{ color: {RED}; }}
    .kpi-value.amber {{ color: {AMBER}; }}
    .kpi-value.blue {{ color: {BLUE}; }}

    /* ---- Section Cards (Targeting st.container border) ---- */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: {CARD_BG};
        border: 1px solid {BORDER} !important;
        border-radius: 12px !important;
        padding: 24px !important;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }}
    .section-title {{
        font-size: 1.1rem;
        font-weight: 700;
        color: {PRIMARY};
        margin: 0 0 16px 0;
        padding-bottom: 12px;
        border-bottom: 1px solid {BORDER};
    }}

    /* ---- Page Header ---- */
    .page-header {{
        margin-bottom: 32px;
    }}
    .page-title {{
        font-size: 1.8rem;
        font-weight: 800;
        color: {PRIMARY};
        margin: 0 0 4px 0;
        line-height: 1.3;
    }}
    .page-subtitle {{
        font-size: 1rem;
        color: {TEXT_MUTED};
        margin: 0;
        font-weight: 400;
    }}

    /* ---- Status Badges ---- */
    .badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 50px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }}
    .badge-green {{
        background: {ACCENT_LIGHT};
        color: {ACCENT};
    }}
    .badge-red {{
        background: #FEE2E2;
        color: {RED};
    }}
    .badge-amber {{
        background: #FEF3C7;
        color: {AMBER};
    }}
    .badge-gray {{
        background: #F1F5F9;
        color: {TEXT_MUTED};
    }}

    /* ---- Info box (non-emoji) ---- */
    .info-panel {{
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 10px;
        padding: 16px 20px;
        color: #1E40AF;
        font-size: 0.9rem;
        line-height: 1.6;
    }}

    /* ---- Risk indicator ---- */
    .risk-indicator {{
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 12px 16px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.9rem;
    }}
    .risk-dot {{
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }}

    /* ---- Clean up streamlit defaults ---- */
    .stMetric {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 16px;
    }}
    .stDataFrame {{
        border-radius: 8px;
        overflow: hidden;
    }}
    div[data-testid="stMetricValue"] {{
        font-weight: 700;
    }}

    /* ---- Buttons ---- */
    .stButton > button {{
        background: {ACCENT};
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 28px;
        font-size: 0.9rem;
        transition: all 0.2s ease;
    }}
    .stButton > button:hover {{
        background: #15803D;
        box-shadow: 0 4px 12px rgba(22, 163, 74, 0.3);
        transform: translateY(-1px);
    }}
    .stButton > button:active {{
        transform: translateY(0);
    }}

    /* ---- Select/Input ---- */
    .stSelectbox > div > div,
    .stFileUploader {{
        border-radius: 8px !important;
    }}

    /* ---- Divider ---- */
    hr {{
        border-color: {BORDER};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
#  HELPER COMPONENTS
# ============================================================


def render_kpi(label, value, color_class=""):
    """Render a single KPI card."""
    st.markdown(
        f"""
        <div class="kpi-card">
            <p class="kpi-label">{label}</p>
            <p class="kpi-value {color_class}">{value}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title, subtitle=""):
    """Render a page header block."""
    sub_html = f'<p class="page-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div class="page-header">
            <p class="page-title">{title}</p>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section(title, content_fn=None):
    """Render a section card with a title and optional content callback."""
    container = st.container(border=True)
    with container:
        st.markdown(f'<p class="section-title">{title}</p>', unsafe_allow_html=True)
        if content_fn:
            content_fn()
    return container


def render_badge(text, variant="gray"):
    """Render an inline status badge."""
    return f'<span class="badge badge-{variant}">{text}</span>'


def render_info(text):
    """Render a non-emoji info panel."""
    st.markdown(f'<div class="info-panel">{text}</div>', unsafe_allow_html=True)


def plotly_theme(fig, height=400):
    """Apply consistent Plotly styling."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(family="Inter, sans-serif", size=13, color=TEXT),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=50, b=40),
        title_font=dict(size=15, color=PRIMARY, family="Inter, sans-serif"),
    )
    fig.update_xaxes(gridcolor="#F1F5F9", linecolor=BORDER)
    fig.update_yaxes(gridcolor="#F1F5F9", linecolor=BORDER)
    return fig


# ============================================================
#  SESSION STATE INITIALIZATION
# ============================================================

STATE_KEYS = {
    "data": None,
    "data_profile": None,
    "model": None,
    "metrics": None,
    "bias_metrics": None,
    "approval_rates": None,
    "mitigated_model": None,
    "mitigated_metrics": None,
    "mitigated_bias_metrics": None,
    "mitigated_approval_rates": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "sensitive_col": None,
    "model_type": None,
    "sf_test": None,
    "sf_train": None,
    "mitigation_method": None,
    "report_text": None,
    "report_pdf": None,
}

for key, default in STATE_KEYS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
#  SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown(
        f"""
        <div style="padding: 4px 4px 10px 4px;">
            <h2 style="margin:0; font-size:1.15555rem; font-weight:800; color:#FFFFFF !important; letter-spacing:-0.02em; line-height:1.1;">
                Fairness Audit Pipeline
            </h2>
            <p style="margin:2px 0 0 0; font-size:0.75rem; color:{TEXT_MUTED} !important; font-weight:400;">
                Loan Approval Models
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Data Management",
            "Model Training",
            "Bias Analysis",
            "Mitigation Engine",
            "Performance Comparison",
            "Explainability",
            "Compliance Reports",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # Pipeline status in sidebar -- single HTML block to save space
    steps = {
        "Data Loaded": st.session_state.data is not None,
        "Model Trained": st.session_state.model is not None,
        "Bias Analyzed": st.session_state.bias_metrics is not None,
        "Mitigation Applied": st.session_state.mitigated_model is not None,
    }

    status_items = ""
    for step_name, completed in steps.items():
        icon = "&#9679;" if completed else "&#9675;"
        color = ACCENT if completed else "#475569"
        status_items += (
            f'<p style="font-size:0.78rem; margin:2px 0; color:{color} !important;">'
            f'{icon}&nbsp;&nbsp;{step_name}</p>'
        )

    st.markdown(
        f'<p style="font-size:0.7rem; font-weight:600; text-transform:uppercase; '
        f'letter-spacing:0.06em; color:#94A3B8 !important; margin-bottom:4px;">Pipeline Status</p>'
        f'{status_items}',
        unsafe_allow_html=True,
    )


# ============================================================
#  PAGE 1: OVERVIEW DASHBOARD
# ============================================================

def page_overview():
    render_page_header(
        "AI Fairness Audit Dashboard",
        "Loan Model Bias Detection and Compliance Monitoring",
    )

    # KPI row
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if st.session_state.metrics:
            val = f"{st.session_state.metrics['Accuracy']:.1%}"
            render_kpi("Model Accuracy", val, "green")
        else:
            render_kpi("Model Accuracy", "--", "")

    with c2:
        if st.session_state.bias_metrics:
            di = st.session_state.bias_metrics["Disparate Impact"]
            _, color_hex = classify_risk(di)
            color_map = {ACCENT: "green", RED: "red", AMBER: "amber"}
            render_kpi("Fairness Score", f"{di:.3f}", color_map.get(color_hex, ""))
        else:
            render_kpi("Fairness Score", "--", "")

    with c3:
        if st.session_state.bias_metrics:
            dpd = st.session_state.bias_metrics["Demographic Parity Difference"]
            render_kpi("Parity Difference", f"{abs(dpd):.3f}", "blue")
        else:
            render_kpi("Parity Difference", "--", "")

    with c4:
        if st.session_state.mitigated_model:
            render_kpi("Compliance Status", "Compliant", "green")
        elif st.session_state.bias_metrics:
            di = st.session_state.bias_metrics["Disparate Impact"]
            label, _ = classify_risk(di)
            clr = "red" if "High" in label else ("amber" if "Moderate" in label else "green")
            render_kpi("Compliance Status", label, clr)
        else:
            render_kpi("Compliance Status", "Pending", "")

    st.markdown("<br>", unsafe_allow_html=True)

    # System status panel
    render_info(
        "Welcome to the <b>Fairness Audit & Bias Mitigation Pipeline</b> -- the enterprise-grade fairness audit platform. "
        "Navigate through the pipeline using the sidebar: upload data, train a model, "
        "detect bias, apply mitigation, compare results, and generate regulatory compliance reports. "
        "All metrics update in real-time across the dashboard."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline overview
    col_a, col_b = st.columns(2)

    with col_a:
        pipeline_steps = [
            ("01", "Data Ingestion", "Upload and profile loan datasets"),
            ("02", "Model Training", "Train classification models with validation"),
            ("03", "Bias Detection", "Audit for demographic disparities"),
            ("04", "Mitigation", "Apply fairness-aware retraining"),
            ("05", "Comparison", "Before vs after performance analysis"),
            ("06", "Explainability", "SHAP-based decision interpretation"),
            ("07", "Reporting", "Generate compliance documentation"),
        ]
        steps_html = "".join(
            f'<p style="margin:6px 0; font-size:0.88rem; color:{TEXT};">'
            f'<span style="font-weight:700; color:{ACCENT};">{num}</span>'
            f'&nbsp;&nbsp;<b>{name}</b> -- {desc}</p>'
            for num, name, desc in pipeline_steps
        )
        with st.container(border=True):
            st.markdown('<p class="section-title">Pipeline Architecture</p>', unsafe_allow_html=True)
            st.markdown(steps_html, unsafe_allow_html=True)

    with col_b:
        frameworks = [
            ("ECOA", "Equal Credit Opportunity Act"),
            ("FHA", "Fair Housing Act"),
            ("EEOC", "Four-Fifths Rule (Disparate Impact)"),
            ("EU AI Act", "High-Risk System Requirements"),
            ("SR 11-7", "Federal Reserve Model Risk Guidance"),
            ("OCC 2011-12", "Model Risk Management"),
            ("CFPB", "Consumer Financial Protection Bureau"),
        ]
        fw_html = "".join(
            f'<p style="margin:6px 0; font-size:0.88rem; color:{TEXT};">'
            f'<span style="font-weight:700; color:{PRIMARY};">{abbr}</span>'
            f'&nbsp;&nbsp;{full_name}</p>'
            for abbr, full_name in frameworks
        )
        with st.container(border=True):
            st.markdown('<p class="section-title">Regulatory Framework</p>', unsafe_allow_html=True)
            st.markdown(fw_html, unsafe_allow_html=True)


# ============================================================
#  PAGE 2: DATA MANAGEMENT
# ============================================================

def page_data_management():
    render_page_header(
        "Data Management",
        "Upload, inspect, and profile loan application datasets",
    )

    col_upload, col_sample = st.columns([3, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Loan Dataset (CSV)", type="csv", label_visibility="visible"
        )

    with col_sample:
        st.markdown("<br>", unsafe_allow_html=True)
        load_sample = st.button("Load Sample Data", width="stretch")

    if uploaded_file or load_sample:
        try:
            if uploaded_file:
                st.session_state.data = pd.read_csv(uploaded_file)
            else:
                st.session_state.data = pd.read_csv("data/loan_data.csv")
            st.session_state.data_profile = get_data_profile(st.session_state.data)
            st.success("Dataset loaded and profiled successfully.")
        except Exception as e:
            st.error(f"Error loading data: {e}")

    if st.session_state.data is not None:
        df = st.session_state.data
        profile = st.session_state.data_profile or get_data_profile(df)

        # KPI row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_kpi("Total Records", f"{profile['row_count']:,}")
        with c2:
            render_kpi("Features", str(profile["col_count"]))
        with c3:
            render_kpi("Missing Values", f"{profile['total_missing']:,}")
        with c4:
            completeness = (1 - profile["total_missing"] / (profile["row_count"] * profile["col_count"])) * 100
            render_kpi("Completeness", f"{completeness:.1f}%", "green")

        st.markdown("<br>", unsafe_allow_html=True)

        # Data preview
        col_preview, col_profile = st.columns([3, 2])

        with col_preview:
            with st.container(border=True):
                st.markdown('<p class="section-title">Data Preview</p>', unsafe_allow_html=True)
                st.dataframe(df.head(10), width="stretch", hide_index=True)

        with col_profile:
            with st.container(border=True):
                st.markdown('<p class="section-title">Column Profile</p>', unsafe_allow_html=True)

                type_data = []
                for col_name in df.columns:
                    dtype_str = str(df[col_name].dtype)
                    missing_count = profile["missing_counts"].get(col_name, 0)
                    missing_pct_val = profile["missing_pct"].get(col_name, 0.0)
                    type_data.append({
                        "Column": col_name,
                        "Type": dtype_str,
                        "Missing": f"{missing_count} ({missing_pct_val:.1f}%)",
                    })
                st.dataframe(
                    pd.DataFrame(type_data),
                    width="stretch",
                    hide_index=True,
                )

        # Missing values chart
        missing_cols = {k: v for k, v in profile["missing_counts"].items() if v > 0}
        if missing_cols:
            with st.container(border=True):
                st.markdown('<p class="section-title">Missing Values Distribution</p>', unsafe_allow_html=True)
                fig = go.Figure(
                    go.Bar(
                        x=list(missing_cols.keys()),
                        y=list(missing_cols.values()),
                        marker_color=ACCENT,
                        text=list(missing_cols.values()),
                        textposition="auto",
                    )
                )
                fig.update_layout(
                    xaxis_title="Column",
                    yaxis_title="Missing Count",
                )
                st.plotly_chart(plotly_theme(fig, 350), width="stretch")


# ============================================================
#  PAGE 3: MODEL TRAINING
# ============================================================

def page_model_training():
    render_page_header(
        "Model Training",
        "Configure and train a classification model for loan approvals",
    )

    if st.session_state.data is None:
        render_info("Please load a dataset in <b>Data Management</b> before training a model.")
        return

    df = st.session_state.data

    # Configuration
    with st.container(border=True):
        st.markdown('<p class="section-title">Training Configuration</p>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        cat_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

        with c1:
            if cat_columns:
                sensitive_attr = st.selectbox("Sensitive Attribute", cat_columns)
            else:
                st.warning("No categorical columns found for sensitive attribute.")
                return

        with c2:
            target_col = st.selectbox(
                "Target Column", df.columns.tolist(), index=len(df.columns) - 1
            )

        with c3:
            model_type = st.selectbox("Algorithm", ["Logistic Regression", "Random Forest"])

        st.markdown("<br>", unsafe_allow_html=True)
        train_clicked = st.button("Train Model", width="content")

    if train_clicked:
        with st.spinner("Preprocessing data and training model..."):
            try:
                X, y, sf_proc, encoders, sf_raw = preprocess_data(
                    df, target_col=target_col, sensitive_col=sensitive_attr
                )
                model, metrics, X_test, y_test, y_pred, X_train, y_train, sf_train, sf_test = (
                    train_model(X, y, sensitive_features=sf_raw, model_type=model_type)
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

                # Reset downstream state
                st.session_state.bias_metrics = None
                st.session_state.approval_rates = None
                st.session_state.mitigated_model = None
                st.session_state.mitigated_metrics = None
                st.session_state.mitigated_bias_metrics = None
                st.session_state.mitigated_approval_rates = None
                st.session_state.mitigation_method = None
                st.session_state.report_text = None

                st.success(f"{model_type} trained successfully on {len(X_train)} samples.")

            except Exception as e:
                st.error(f"Training failed: {e}")
                return

    # Display results
    if st.session_state.metrics:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<p class="section-title">Model Performance Metrics</p>', unsafe_allow_html=True)

            m = st.session_state.metrics
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                render_kpi("Accuracy", f"{m['Accuracy']:.2%}", "green")
            with c2:
                render_kpi("Precision", f"{m['Precision']:.2%}", "blue")
            with c3:
                render_kpi("Recall", f"{m['Recall']:.2%}", "blue")
            with c4:
                render_kpi("F1 Score", f"{m['F1 Score']:.2%}", "green")

            # Plotly bar chart of metrics
            st.markdown("<br>", unsafe_allow_html=True)
            metric_names = list(m.keys())
            metric_vals = list(m.values())
            fig = go.Figure(
                go.Bar(
                    x=metric_names,
                    y=metric_vals,
                    marker_color=[ACCENT, BLUE, BLUE, ACCENT],
                    text=[f"{v:.2%}" for v in metric_vals],
                    textposition="auto",
                )
            )
            fig.update_layout(
                title="Performance Summary",
                xaxis_title="Metric",
                yaxis_title="Score",
                yaxis=dict(range=[0, 1]),
            )
            st.plotly_chart(plotly_theme(fig), width="stretch")


# ============================================================
#  PAGE 4: BIAS ANALYSIS
# ============================================================

def page_bias_analysis():
    render_page_header(
        "Bias Analysis",
        "Audit model predictions for demographic disparities",
    )

    if st.session_state.model is None or st.session_state.sf_test is None:
        render_info("Please train a model in <b>Model Training</b> before running bias analysis.")
        return

    with st.container(border=True):
        st.markdown('<p class="section-title">Audit Configuration</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size:0.9rem; color:{TEXT_MUTED};">Protected attribute: '
            f'<b style="color:{PRIMARY};">{st.session_state.sensitive_col}</b></p>',
            unsafe_allow_html=True,
        )
        run_audit = st.button("Run Fairness Audit", width="content")

    if run_audit:
        with st.spinner("Analyzing model predictions for bias..."):
            y_pred = st.session_state.model.predict(st.session_state.X_test)
            bias_metrics, approval_rates = detect_bias(
                st.session_state.y_test, y_pred, st.session_state.sf_test
            )
            st.session_state.bias_metrics = bias_metrics
            st.session_state.approval_rates = approval_rates

    if st.session_state.bias_metrics:
        bm = st.session_state.bias_metrics
        di = bm["Disparate Impact"]
        risk_label, risk_color = classify_risk(di)
        color_map = {ACCENT: "green", RED: "red", AMBER: "amber"}
        badge_variant = "green" if "Fair" in risk_label else ("red" if "High" in risk_label else "amber")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            f'<div style="background:{CARD_BG}; border: 1px solid {BORDER}; border-left: 4px solid {risk_color}; border-radius: 12px; padding: 20px; margin-bottom: 20px;">'
            f'<p style="margin:0; font-size:1rem; font-weight:700; color:{PRIMARY};">'
            f'Risk Assessment: {render_badge(risk_label, badge_variant)}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Fairness KPIs
        c1, c2, c3 = st.columns(3)
        with c1:
            render_kpi("Disparate Impact", f"{di:.3f}", color_map.get(risk_color, ""))
        with c2:
            render_kpi("Demographic Parity Diff", f"{abs(bm['Demographic Parity Difference']):.3f}", "blue")
        with c3:
            render_kpi("Equal Opportunity Diff", f"{abs(bm['Equal Opportunity Difference']):.3f}", "blue")

        st.markdown("<br>", unsafe_allow_html=True)

        # Approval rates chart
        if st.session_state.approval_rates:
            with st.container(border=True):
                st.markdown(
                    f'<p class="section-title">Group-wise Selection Rates by {st.session_state.sensitive_col}</p>',
                    unsafe_allow_html=True,
                )

                rates = st.session_state.approval_rates
                groups = list(rates.keys())
                values = list(rates.values())

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

                fig = go.Figure(
                    go.Bar(
                        x=groups,
                        y=values,
                        marker_color=colors,
                        text=[f"{v:.1%}" for v in values],
                        textposition="auto",
                    )
                )
                fig.update_layout(
                    title=f"Approval Rate by {st.session_state.sensitive_col}",
                    xaxis_title="Group",
                    yaxis_title="Approval Rate",
                    yaxis=dict(range=[0, 1], tickformat=".0%"),
                )
                # Threshold line
                fig.add_hline(
                    y=max_rate * 0.8,
                    line_dash="dash",
                    line_color=RED,
                    annotation_text="80% Threshold (Four-Fifths Rule)",
                    annotation_position="top left",
                )
                st.plotly_chart(plotly_theme(fig), width="stretch")

        # Decision logic explanation
        with st.container(border=True):
            st.markdown('<p class="section-title">Compliance Decision Logic</p>', unsafe_allow_html=True)
            logic_data = [
                {"Range": "DI < 0.80", "Classification": "High Risk", "Action": "Mitigation required"},
                {"Range": "0.80 - 0.90", "Classification": "Moderate Risk", "Action": "Monitoring advised"},
                {"Range": "DI > 0.90", "Classification": "Fair", "Action": "Within compliance bounds"},
            ]
            st.dataframe(pd.DataFrame(logic_data), width="stretch", hide_index=True)


# ============================================================
#  PAGE 5: MITIGATION ENGINE
# ============================================================

def page_mitigation():
    render_page_header(
        "Mitigation Engine",
        "Apply fairness-aware retraining techniques to reduce model bias",
    )

    if st.session_state.model is None:
        render_info("Please train a model in <b>Model Training</b> first.")
        return

    if st.session_state.bias_metrics is None:
        render_info("Please run <b>Bias Analysis</b> first to establish a baseline.")
        return

    with st.container(border=True):
        st.markdown('<p class="section-title">Mitigation Configuration</p>', unsafe_allow_html=True)

        c1, c2 = st.columns([2, 1])
        with c1:
            method = st.selectbox(
                "Mitigation Technique",
                ["Exponentiated Gradient", "Reweighing"],
            )
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        apply_clicked = st.button("Apply Mitigation", width="content")

    if apply_clicked:
        with st.spinner("Applying mitigation and retraining model... This may take a moment."):
            try:
                mit_model = mitigate_bias(
                    st.session_state.X_train,
                    st.session_state.y_train,
                    st.session_state.sf_train,
                    model_type=st.session_state.model_type,
                    method=method,
                )
                st.session_state.mitigated_model = mit_model
                st.session_state.mitigation_method = method

                # Evaluate
                y_pred_mit = mit_model.predict(st.session_state.X_test)
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

                new_metrics = {
                    "Accuracy": accuracy_score(st.session_state.y_test, y_pred_mit),
                    "Precision": precision_score(st.session_state.y_test, y_pred_mit, zero_division=0),
                    "Recall": recall_score(st.session_state.y_test, y_pred_mit, zero_division=0),
                    "F1 Score": f1_score(st.session_state.y_test, y_pred_mit, zero_division=0),
                }
                st.session_state.mitigated_metrics = new_metrics

                new_bias, new_rates = detect_bias(
                    st.session_state.y_test, y_pred_mit, st.session_state.sf_test
                )
                st.session_state.mitigated_bias_metrics = new_bias
                st.session_state.mitigated_approval_rates = new_rates

                st.success("Mitigation applied successfully.")

            except Exception as e:
                st.error(f"Mitigation failed: {e}")

    if st.session_state.mitigated_bias_metrics:
        mbm = st.session_state.mitigated_bias_metrics
        mm = st.session_state.mitigated_metrics
        di_new = mbm["Disparate Impact"]
        risk_label, risk_color = classify_risk(di_new)
        badge_variant = "green" if "Fair" in risk_label else ("red" if "High" in risk_label else "amber")

        st.markdown(
            f'<div style="background:{CARD_BG}; border: 1px solid {BORDER}; border-left: 4px solid {risk_color}; border-radius: 12px; padding: 20px; margin-bottom: 20px;">'
            f'<p style="margin:0; font-size:1rem; font-weight:700; color:{PRIMARY};">'
            f'Post-Mitigation Status: {render_badge(risk_label, badge_variant)}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            color_map = {ACCENT: "green", RED: "red", AMBER: "amber"}
            render_kpi("New Disparate Impact", f"{di_new:.3f}", color_map.get(risk_color, ""))
        with c2:
            render_kpi("New Accuracy", f"{mm['Accuracy']:.2%}", "blue")
        with c3:
            render_kpi("New Precision", f"{mm['Precision']:.2%}", "blue")
        with c4:
            render_kpi("New F1 Score", f"{mm['F1 Score']:.2%}", "green")


# ============================================================
#  PAGE 6: PERFORMANCE COMPARISON
# ============================================================

def page_comparison():
    render_page_header(
        "Performance Comparison",
        "Before vs after mitigation: Performance and fairness trade-off analysis",
    )

    if st.session_state.mitigated_metrics is None:
        render_info("Apply mitigation in the <b>Mitigation Engine</b> to view comparison results.")
        return

    m_before = st.session_state.metrics
    m_after = st.session_state.mitigated_metrics
    b_before = st.session_state.bias_metrics
    b_after = st.session_state.mitigated_bias_metrics

    # Two-column layout
    col_left, col_right = st.columns(2)

    with col_left:
        with st.container(border=True):
            st.markdown('<p class="section-title">Before Mitigation</p>', unsafe_allow_html=True)

            for metric_name, val in m_before.items():
                st.markdown(
                    f'<p style="margin:8px 0; font-size:0.9rem; color:{TEXT};">'
                    f'<b>{metric_name}</b>: {val:.4f}</p>',
                    unsafe_allow_html=True,
                )
            st.divider()
            for metric_name, val in b_before.items():
                st.markdown(
                    f'<p style="margin:8px 0; font-size:0.9rem; color:{TEXT};">'
                    f'<b>{metric_name}</b>: {val:.4f}</p>',
                    unsafe_allow_html=True,
                )

    with col_right:
        with st.container(border=True):
            st.markdown('<p class="section-title">After Mitigation</p>', unsafe_allow_html=True)

            for metric_name, val in m_after.items():
                st.markdown(
                    f'<p style="margin:8px 0; font-size:0.9rem; color:{TEXT};">'
                    f'<b>{metric_name}</b>: {val:.4f}</p>',
                    unsafe_allow_html=True,
                )
            st.divider()
            for metric_name, val in b_after.items():
                st.markdown(
                    f'<p style="margin:8px 0; font-size:0.9rem; color:{TEXT};">'
                    f'<b>{metric_name}</b>: {val:.4f}</p>',
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # Plotly grouped bar: Performance
    with st.container(border=True):
        st.markdown('<p class="section-title">Performance Metrics Comparison</p>', unsafe_allow_html=True)

        perf_metrics = list(m_before.keys())
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Before Mitigation",
            x=perf_metrics,
            y=[m_before[k] for k in perf_metrics],
            marker_color="#94A3B8",
            text=[f"{m_before[k]:.2%}" for k in perf_metrics],
            textposition="auto",
        ))
        fig.add_trace(go.Bar(
            name="After Mitigation",
            x=perf_metrics,
            y=[m_after[k] for k in perf_metrics],
            marker_color=ACCENT,
            text=[f"{m_after[k]:.2%}" for k in perf_metrics],
            textposition="auto",
        ))
        fig.update_layout(
            barmode="group",
            title="Model Performance: Before vs After",
            xaxis_title="Metric",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(plotly_theme(fig, 420), width="stretch")

    # Plotly grouped bar: Fairness
    with st.container(border=True):
        st.markdown('<p class="section-title">Fairness Metrics Comparison</p>', unsafe_allow_html=True)

        bias_metric_names = list(b_before.keys())
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name="Before Mitigation",
            x=bias_metric_names,
            y=[abs(b_before[k]) for k in bias_metric_names],
            marker_color="#94A3B8",
            text=[f"{abs(b_before[k]):.3f}" for k in bias_metric_names],
            textposition="auto",
        ))
        fig2.add_trace(go.Bar(
            name="After Mitigation",
            x=bias_metric_names,
            y=[abs(b_after[k]) for k in bias_metric_names],
            marker_color=ACCENT,
            text=[f"{abs(b_after[k]):.3f}" for k in bias_metric_names],
            textposition="auto",
        ))
        fig2.update_layout(
            barmode="group",
            title="Fairness Metrics: Before vs After",
            xaxis_title="Metric",
            yaxis_title="Value",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        # Threshold line for DI
        fig2.add_hline(
            y=0.8,
            line_dash="dash",
            line_color=RED,
            annotation_text="Compliance Threshold (0.8)",
            annotation_position="top left",
        )
        st.plotly_chart(plotly_theme(fig2, 420), width="stretch")

    # Delta summary
    with st.container(border=True):
        st.markdown('<p class="section-title">Impact Summary</p>', unsafe_allow_html=True)

        acc_delta = m_after["Accuracy"] - m_before["Accuracy"]
        di_delta = b_after["Disparate Impact"] - b_before["Disparate Impact"]
        dpd_delta = abs(b_after["Demographic Parity Difference"]) - abs(b_before["Demographic Parity Difference"])

        impact_data = [
            {
                "Metric": "Accuracy Change",
                "Delta": f"{acc_delta:+.4f}",
                "Direction": "Improved" if acc_delta >= 0 else "Decreased",
            },
            {
                "Metric": "Disparate Impact Change",
                "Delta": f"{di_delta:+.4f}",
                "Direction": "Improved" if di_delta > 0 else "Decreased",
            },
            {
                "Metric": "Parity Difference Change",
                "Delta": f"{dpd_delta:+.4f}",
                "Direction": "Improved" if dpd_delta < 0 else "Increased",
            },
        ]
        st.dataframe(pd.DataFrame(impact_data), width="stretch", hide_index=True)


# ============================================================
#  PAGE 7: EXPLAINABILITY
# ============================================================

def page_explainability():
    render_page_header(
        "Model Explainability",
        "SHAP-based feature importance and decision interpretation",
    )

    if st.session_state.model is None:
        render_info("Please train a model in <b>Model Training</b> first.")
        return

    model_to_explain = (
        st.session_state.mitigated_model
        if st.session_state.mitigated_model
        else st.session_state.model
    )
    model_label = "Mitigated Model" if st.session_state.mitigated_model else "Baseline Model"

    with st.container(border=True):
        st.markdown('<p class="section-title">SHAP Analysis Configuration</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size:0.9rem; color:{TEXT_MUTED};">'
            f'Analyzing: <b style="color:{PRIMARY};">{model_label}</b> '
            f'({st.session_state.model_type})</p>',
            unsafe_allow_html=True,
        )
        compute_clicked = st.button("Compute SHAP Values", width="content")

    if compute_clicked:
        with st.spinner("Computing SHAP values... This may take a moment."):
            try:
                shap_values, X_sample = compute_shap_values(
                    model_to_explain,
                    st.session_state.X_train,
                    st.session_state.X_test,
                    st.session_state.model_type,
                )

                feature_names = (
                    list(st.session_state.X_test.columns)
                    if hasattr(st.session_state.X_test, "columns")
                    else [f"Feature {i}" for i in range(st.session_state.X_test.shape[1])]
                )

                importance_df = get_feature_importance(shap_values, feature_names)
                st.session_state["shap_importance"] = importance_df

                st.markdown("<br>", unsafe_allow_html=True)

                # Plotly feature importance bar chart
                with st.container(border=True):
                    st.markdown('<p class="section-title">Feature Importance (Mean |SHAP|)</p>', unsafe_allow_html=True)

                    top_n = min(15, len(importance_df))
                    top_features = importance_df.head(top_n).iloc[::-1]

                    fig = go.Figure(
                        go.Bar(
                            x=top_features["Importance"],
                            y=top_features["Feature"],
                            orientation="h",
                            marker_color=ACCENT,
                            text=[f"{v:.4f}" for v in top_features["Importance"]],
                            textposition="auto",
                        )
                    )
                    fig.update_layout(
                        title="Top Feature Contributions to Model Decisions",
                        xaxis_title="Mean |SHAP Value|",
                        yaxis_title="Feature",
                    )
                    st.plotly_chart(plotly_theme(fig, 500), width="stretch")

                # SHAP summary plot (matplotlib)
                with st.container(border=True):
                    st.markdown('<p class="section-title">SHAP Summary Plot</p>', unsafe_allow_html=True)

                    summary_fig = generate_shap_summary_plot(
                        model_to_explain,
                        st.session_state.X_train,
                        st.session_state.X_test,
                        st.session_state.model_type,
                    )
                    st.pyplot(summary_fig)

            except Exception as e:
                st.error(f"SHAP computation failed: {e}")


# ============================================================
#  PAGE 8: COMPLIANCE REPORTS
# ============================================================

def page_reports():
    render_page_header(
        "Compliance Reports",
        "Generate structured regulatory compliance documentation",
    )

    if st.session_state.metrics is None or st.session_state.bias_metrics is None:
        render_info(
            "Complete <b>Model Training</b> and <b>Bias Analysis</b> to generate a compliance report."
        )
        return

    with st.container(border=True):
        st.markdown('<p class="section-title">Report Configuration</p>', unsafe_allow_html=True)

        report_sections = [
            "Model performance metrics",
            "Bias detection results",
            "Mitigation outcomes (if applied)",
            "Compliance interpretation",
            "Regulatory references",
        ]
        st.markdown(
            f'<p style="font-size:0.9rem; color:{TEXT_MUTED}; margin-bottom:8px;">Report will include:</p>',
            unsafe_allow_html=True,
        )
        for section in report_sections:
            st.markdown(
                f'<p style="margin:4px 0; font-size:0.88rem; color:{TEXT};">'
                f'<span style="color:{ACCENT}; font-weight:700;">&#8226;</span>&nbsp;&nbsp;{section}</p>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        generate_clicked = st.button("Generate Report", width="content")

    if generate_clicked:
        with st.spinner("Generating compliance reports..."):
            # Markdown Report
            report_text = generate_report(
                st.session_state.metrics,
                st.session_state.bias_metrics,
                st.session_state.mitigated_metrics,
                st.session_state.mitigated_bias_metrics,
                sensitive_col=st.session_state.sensitive_col,
                model_type=st.session_state.model_type,
                mitigation_method=st.session_state.mitigation_method,
            )
            st.session_state.report_text = report_text
            
            # PDF Report
            report_pdf = generate_pdf_report(
                st.session_state.metrics,
                st.session_state.bias_metrics,
                st.session_state.mitigated_metrics,
                st.session_state.mitigated_bias_metrics,
                sensitive_col=st.session_state.sensitive_col,
                model_type=st.session_state.model_type,
                mitigation_method=st.session_state.mitigation_method,
            )
            st.session_state.report_pdf = report_pdf

    if st.session_state.report_text:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<p class="section-title">Report Preview</p>', unsafe_allow_html=True)
            st.text_area(
                "Report content",
                st.session_state.report_text,
                height=400,
                label_visibility="collapsed",
            )

        # Download Options
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3, _ = st.columns([1.2, 1.2, 1.2, 2])
        
        with c1:
            if st.session_state.report_pdf:
                st.download_button(
                    label="Download Official PDF",
                    data=st.session_state.report_pdf,
                    file_name="LoanGuard_Compliance_Report.pdf",
                    mime="application/pdf",
                    width="stretch",
                )
            else:
                st.info("PDF Generation requires fpdf2")

        with c2:
            st.download_button(
                label="Download Markdown",
                data=st.session_state.report_text,
                file_name="LoanGuard_Compliance_Report.md",
                mime="text/markdown",
                width="stretch",
            )
            
        with c3:
            st.download_button(
                label="Download Plain Text",
                data=st.session_state.report_text,
                file_name="LoanGuard_Compliance_Report.txt",
                mime="text/plain",
                width="stretch",
            )


# ============================================================
#  PAGE ROUTER
# ============================================================

PAGES = {
    "Overview": page_overview,
    "Data Management": page_data_management,
    "Model Training": page_model_training,
    "Bias Analysis": page_bias_analysis,
    "Mitigation Engine": page_mitigation,
    "Performance Comparison": page_comparison,
    "Explainability": page_explainability,
    "Compliance Reports": page_reports,
}

PAGES[page]()
