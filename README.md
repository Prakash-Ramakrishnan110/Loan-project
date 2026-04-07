# LoanGuard AI

**Automated Fairness Audit and Bias Mitigation Pipeline for Loan Approval Models with Regulatory Compliance Reporting**

LoanGuard AI is an enterprise-grade, production-ready web application that functions as a professional fintech SaaS platform for auditing machine learning models used in loan approval decisions. The system detects demographic bias, applies algorithmic mitigation techniques, provides SHAP-based model explainability, and generates structured regulatory compliance reports.

---

## Features

### Overview Dashboard
- Real-time KPI grid displaying Model Accuracy, Fairness Score, Parity Difference, and Compliance Status
- Pipeline architecture overview with 7-stage audit workflow
- Live pipeline status tracking in the sidebar
- Regulatory framework reference (ECOA, FHA, EEOC, EU AI Act, SR 11-7, OCC 2011-12, CFPB)

### Data Management
- CSV file upload with drag-and-drop support
- Preloaded sample dataset with synthetic bias for demonstration
- Automated data profiling: row/column counts, missing value analysis, column type detection
- Interactive missing values distribution chart (Plotly)

### Model Training
- Algorithm selection: Logistic Regression or Random Forest
- Automated preprocessing pipeline (imputation, encoding, scaling)
- Performance metrics: Accuracy, Precision, Recall, F1 Score
- Interactive Plotly performance summary chart

### Bias Analysis
- Sensitive attribute selection for protected group analysis
- Fairness metrics computation: Demographic Parity Difference, Equal Opportunity Difference, Disparate Impact
- Group-wise selection rate visualization with Four-Fifths Rule threshold line
- Risk classification: High Risk (DI < 0.80), Moderate Risk (0.80-0.90), Fair (DI > 0.90)
- Compliance decision logic reference table

### Mitigation Engine
- Exponentiated Gradient (Fairlearn) -- in-processing demographic parity constraint
- Reweighing (AIF360) -- pre-processing sample weight adjustment
- Post-mitigation fairness re-evaluation with updated risk classification

### Performance Comparison
- Side-by-side before/after metric cards
- Grouped Plotly bar charts for performance and fairness metrics
- Compliance threshold reference line on fairness comparison
- Delta impact summary table (accuracy change, DI change, parity difference change)

### Explainability
- SHAP (SHapley Additive exPlanations) integration
- Feature importance bar chart (Plotly) ranked by mean absolute SHAP value
- SHAP summary beeswarm plot showing feature-level directional impact
- Supports both baseline and mitigated model analysis

### Compliance Reports
- Structured Markdown report with executive summary
- Tabulated model performance and fairness metrics
- Pre/post-mitigation comparison with compliance assessment
- Regulatory references: ECOA, FHA, EEOC Four-Fifths Rule, EU AI Act
- Download as Markdown or plain text

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | Streamlit (advanced UI structuring, custom CSS) |
| **Backend** | Python, Pandas, NumPy, Scikit-learn |
| **Fairness** | Fairlearn, AIF360 |
| **Explainability** | SHAP |
| **Visualization** | Plotly (primary), Matplotlib (SHAP plots) |

---

## Project Structure

```
loan-project/
├── .streamlit/
│   └── config.toml              # Theme configuration (fintech palette)
├── data/
│   └── loan_data.csv            # Synthetic loan dataset with embedded bias
├── utils/
│   ├── preprocessing.py         # Data loading, imputation, encoding, scaling, profiling
│   ├── training.py              # Model training pipeline and evaluation helpers
│   ├── bias_detection.py        # Fairness metrics computation and risk classification
│   ├── mitigation.py            # Exponentiated Gradient and Reweighing implementations
│   ├── explainability.py        # SHAP value computation and feature importance extraction
│   ├── reporting.py             # Structured compliance report generation
│   └── generate_data.py         # Synthetic biased dataset generator
├── app.py                       # Main Streamlit application (8-page dashboard)
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Installation and Usage

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Prakash-Ramakrishnan110/Loan-project.git
   cd Loan-project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the application**
   ```bash
   python -m streamlit run app.py
   ```

4. Open your browser at **http://localhost:8501**

### Recommended Workflow

1. Navigate to **Data Management** and load the sample dataset (or upload your own CSV)
2. Go to **Model Training**, select a sensitive attribute, target column, and algorithm, then train
3. Run **Bias Analysis** to audit the model for demographic disparities
4. Apply a mitigation technique in the **Mitigation Engine**
5. Review the **Performance Comparison** to analyze accuracy-fairness trade-offs
6. Examine **Explainability** for SHAP-based feature impact analysis
7. Generate a downloadable **Compliance Report** for regulatory documentation

---

## Design

- **Color palette**: Primary `#0F172A` (dark navy), Accent `#16A34A` (green), Background `#F8FAFC` (light gray)
- **Typography**: Inter font family via Google Fonts
- **Layout**: Wide layout with fixed dark sidebar navigation, card-based content sections
- **Charts**: All interactive charts built with Plotly for consistent styling and interactivity

---

## Compliance Standards Referenced

| Standard | Description |
|----------|-------------|
| ECOA | Equal Credit Opportunity Act |
| FHA | Fair Housing Act |
| EEOC | Uniform Guidelines on Employee Selection Procedures (Four-Fifths Rule) |
| EU AI Act | European Union Artificial Intelligence Act (High-Risk Systems) |
| SR 11-7 | Federal Reserve Guidance on Model Risk Management |
| OCC 2011-12 | OCC Supervisory Guidance on Model Risk Management |
| CFPB | Consumer Financial Protection Bureau Oversight |

---

## License

This project is developed for academic and demonstration purposes.

---

*Built with LoanGuard AI -- Automated Fairness Audit Pipeline v2.0*
