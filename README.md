# ⚖️ LoanGard AI: Fairness Audit & Bias Mitigation Pipeline

LoanGard AI is a complete, production-ready AI-powered web application that serves as an **Automated Fairness Audit and Bias Mitigation Pipeline for Loan Approval Models**. The system functions as a professional fintech SaaS dashboard that detects bias, mitigates it, explains model decisions, and generates regulatory compliance reports.

## 🚀 Features

- **📊 Data Overview**: Upload custom loan datasets, automatically handle missing values, and analyze row/column statistics.
- **🤖 Model Training**: Train Scikit-learn models like **Logistic Regression** and **Random Forest** dynamically while extracting real-time metrics (Accuracy, Precision, Recall, F1-Score).
- **🔎 Bias Detection**: Uncover Hidden Bias against sensitive attributes (like Gender or Age) using Fairlearn. Computes *Demographic Parity*, *Equal Opportunity Difference*, and *Disparate Impact*.
- **🛠️ Bias Mitigation**: Automatically apply sophisticated debiasing algorithms like **Exponentiated Gradient (Demographic Parity)** and **Reweighing (AIF360 Wrapper)** algorithms and compare the results to ensure your models reach standard compliance (Disparate Impact ≥ 0.8).
- **💡 Explainability (SHAP)**: Fully transparent AI using SHAP feature importance charts.
- **📄 Regulatory Reporting**: Ensure workflow compliance by generating and downloading Markdown audit reports automatically comparing standard and mitigated models.

## 💻 Tech Stack

- **Frontend**: Streamlit 
- **Backend Algorithms**: Python, Pandas, NumPy, Scikit-learn
- **Fairness Frameworks**: Fairlearn, AIF360
- **AI Explainability**: SHAP (SHapley Additive exPlanations)
- **Data Visualizations**: Streamlit Native Charts, Matplotlib

## 📂 Project Structure

```
/
├── .streamlit/
│   └── config.toml           # UI Theming configs (Fintech Palette)
├── data/
│   └── loan_data.csv         # Synthetically generated biased sample dataset
├── utils/
│   ├── generate_data.py      # Script that generated the mock loan dataset
│   ├── preprocessing.py      # Features encoding, missing value handlers
│   ├── training.py           # Logistic Regression and Random Forest Pipelines
│   ├── bias_detection.py     # Bias calculation metrics
│   ├── mitigation.py         # Exponentiated Gradient model abstractions
│   ├── explainability.py     # SHAP tree/linear explainers integration
│   └── reporting.py          # Auto-generated comprehensive Compliance reports
├── app.py                    # Main Streamlit Frontend Dashboard
└── requirements.txt          # App modules and package dependencies
```

## 🛠️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Prakash-Ramakrishnan110/Loan-project.git
   cd Loan-project
   ```

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Dashboard locally:**
   ```bash
   streamlit run app.py
   ```

4. The application will launch in your default web browser (default url: http://localhost:8501). Use the side-navigation menu to jump between workflows!

---
*Created as part of the Loan Approval Audit pipelines project*
